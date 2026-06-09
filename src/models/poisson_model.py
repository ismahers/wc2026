"""
src/models/poisson_model.py
============================
Modelo Poisson para predicción de goles y mercados derivados.

Arquitectura:
  - Dos XGBoost con objective=count:poisson:
      lambda_home: goles esperados del equipo home
      lambda_away: goles esperados del equipo away
  - Distribución de marcadores P(home=i, away=j) para i,j en 0..8
  - Derivación coherente de 1X2, Over/Under y BTTS

Ventaja sobre XGBoost directo:
  - Las probabilidades de 1X2, Over/Under y BTTS son coherentes
    (salen de la misma distribución, no de tres modelos independientes)
  - Permite calcular marcadores exactos y hándicap asiático

Uso:
    python -m src.models.poisson_model
    python -m src.models.poisson_model --features data/processed/features_train.csv
                                       --wc-features data/processed/features_wc2026.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson
from sklearn.metrics import mean_absolute_error, mean_squared_error

log = logging.getLogger(__name__)

MAX_GOALS = 8  # máximo de goles por equipo en la simulación (cubre 99.9% de casos)

# ---------------------------------------------------------------------------
# Features por perspectiva
# ---------------------------------------------------------------------------

# Features que describen la capacidad goleadora del equipo home
# y la debilidad defensiva del equipo away
HOME_ATTACK_FEATURES = [
    "home_form_gf_10",
    "home_form_ppg_10",
    "home_form_wins_10",
    "diff_form_gf_10",
    "diff_form_gc_10",
    "diff_form_ppg_10",
    "elo_diff",
    "is_neutral",
    "home_is_host",
    "effective_home_adv",
    "same_confederation",
    "stage_ordinal",
    "home_rest_days",
    "rest_days_diff",
    "altitude_m",
    # WC2026 específicas
    "home_goals_per_90",
    "away_defense_norm",
    "home_attack_norm",
    "rating_diff",
    "home_avg_caps",
]

# Features que describen la capacidad goleadora del equipo away
AWAY_ATTACK_FEATURES = [
    "away_form_gf_10",
    "away_form_ppg_10",
    "away_form_wins_10",
    "diff_form_gf_10",
    "diff_form_gc_10",
    "diff_form_ppg_10",
    "elo_diff",
    "is_neutral",
    "away_is_host",
    "effective_home_adv",
    "same_confederation",
    "stage_ordinal",
    "away_rest_days",
    "rest_days_diff",
    "altitude_m",
    # WC2026 específicas
    "away_goals_per_90",
    "home_defense_norm",
    "away_attack_norm",
    "rating_diff",
    "away_avg_caps",
]

XGB_POISSON_PARAMS = {
    "objective":        "count:poisson",
    "max_depth":        4,
    "learning_rate":    0.05,
    "n_estimators":     300,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
}


# ---------------------------------------------------------------------------
# Distribución de marcadores desde dos lambdas Poisson
# ---------------------------------------------------------------------------

def score_matrix(lambda_home: float, lambda_away: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """
    Calcula la matriz de probabilidades de marcadores P(home=i, away=j).

    Retorna array (max_goals+1, max_goals+1) donde [i, j] = P(home=i, away=j).
    Asume independencia entre goles home y away (Poisson bivariado independiente).
    """
    home_probs = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    away_probs = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    return np.outer(home_probs, away_probs)


def derive_markets(matrix: np.ndarray) -> dict:
    """
    Deriva probabilidades de mercados desde la matriz de marcadores.

    Retorna dict con prob_H, prob_D, prob_A, prob_over25, prob_btts,
    expected_home_goals, expected_away_goals y top5_scores.
    """
    n = matrix.shape[0]

    prob_H = float(np.sum(np.tril(matrix, -1)))   # home > away
    prob_D = float(np.trace(matrix))               # home == away
    prob_A = float(np.sum(np.triu(matrix, 1)))     # away > home

    # Over 2.5: suma de probabilidades donde i+j > 2.5
    prob_over25 = 0.0
    for i in range(n):
        for j in range(n):
            if i + j > 2.5:
                prob_over25 += matrix[i, j]

    # BTTS: ambos equipos marcan al menos 1
    prob_btts = float(1 - matrix[0, :].sum() - matrix[:, 0].sum() + matrix[0, 0])

    # Goles esperados
    goals_range = np.arange(n)
    exp_home = float(np.sum(goals_range * matrix.sum(axis=1)))
    exp_away = float(np.sum(goals_range * matrix.sum(axis=0)))

    # Top 5 marcadores más probables
    flat = [(matrix[i, j], i, j) for i in range(n) for j in range(n)]
    flat.sort(reverse=True)
    top5 = [{"score": f"{i}-{j}", "prob": round(p, 4)} for p, i, j in flat[:5]]

    return {
        "prob_H":              round(prob_H,    4),
        "prob_D":              round(prob_D,    4),
        "prob_A":              round(prob_A,    4),
        "prob_over25":         round(prob_over25, 4),
        "prob_btts":           round(float(prob_btts), 4),
        "expected_home_goals": round(exp_home,  3),
        "expected_away_goals": round(exp_away,  3),
        "top5_scores":         top5,
    }


# ---------------------------------------------------------------------------
# Modelo Poisson
# ---------------------------------------------------------------------------

class PoissonGoalModel:
    """
    Dos modelos XGBoost Poisson: uno para lambda_home y otro para lambda_away.
    """

    def __init__(self, params: dict = XGB_POISSON_PARAMS):
        self.params      = params.copy()
        self.model_home  = None
        self.model_away  = None
        self.feats_home: list[str] = []
        self.feats_away: list[str] = []
        self.fill_home: dict[str, float] = {}
        self.fill_away: dict[str, float] = {}

    def _select_features(self, df: pd.DataFrame, candidates: list[str]) -> list[str]:
        """Selecciona features disponibles con al menos 10% de cobertura."""
        return [
            f for f in candidates
            if f in df.columns and df[f].notna().mean() > 0.10
        ]

    def _prepare(
        self,
        df: pd.DataFrame,
        target: str,
        feature_candidates: list[str],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        feats = self._select_features(df, feature_candidates)
        mask  = df[target].notna() & (df[target] >= 0)
        clean = df[mask].copy()

        if len(clean) < 100:
            raise ValueError(f"Solo {len(clean)} muestras con target '{target}'")

        X = clean[feats].copy()
        fill_values = {}
        for col in feats:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            fill_value = X[col].median()
            if pd.isna(fill_value):
                fill_value = 0.0
            fill_values[col] = float(fill_value)
            if X[col].isna().any():
                X[col] = X[col].fillna(fill_value)

        y = clean[target].values.astype(np.float32)
        return X.values.astype(np.float32), y, feats, fill_values

    def fit(self, df_train: pd.DataFrame) -> "PoissonGoalModel":
        log.info("Entrenando modelo Poisson — lambda_home...")
        X_h, y_h, self.feats_home, self.fill_home = self._prepare(
            df_train, "target_home_goals", HOME_ATTACK_FEATURES
        )
        params = self.params.copy()
        n_est = params.pop("n_estimators", 300)

        self.model_home = xgb.XGBRegressor(n_estimators=n_est, **params)
        self.model_home.fit(X_h, y_h)
        log.info("  ✓ lambda_home: %d muestras, %d features", len(X_h), len(self.feats_home))

        log.info("Entrenando modelo Poisson — lambda_away...")
        X_a, y_a, self.feats_away, self.fill_away = self._prepare(
            df_train, "target_away_goals", AWAY_ATTACK_FEATURES
        )
        self.model_away = xgb.XGBRegressor(n_estimators=n_est, **params)
        self.model_away.fit(X_a, y_a)
        log.info("  ✓ lambda_away: %d muestras, %d features", len(X_a), len(self.feats_away))

        return self

    @staticmethod
    def _build_feature_matrix(df: pd.DataFrame, feats: list[str], fill_values: dict[str, float]) -> np.ndarray:
        """Build a numeric matrix using training-time feature order and medians."""
        X = pd.DataFrame(index=df.index)
        for col in feats:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce")
            else:
                values = pd.Series(np.nan, index=df.index)
            X[col] = values.fillna(fill_values.get(col, 0.0))
        return X.values.astype(np.float32)

    def predict_lambdas(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predice lambda_home y lambda_away para cada partido."""
        X_h = self._build_feature_matrix(df, self.feats_home, self.fill_home)
        X_a = self._build_feature_matrix(df, self.feats_away, self.fill_away)

        lambda_home = np.clip(self.model_home.predict(X_h), 0.01, 15)
        lambda_away = np.clip(self.model_away.predict(X_a), 0.01, 15)
        return lambda_home, lambda_away

    def predict_markets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predice todos los mercados para un DataFrame de partidos."""
        lambda_home, lambda_away = self.predict_lambdas(df)

        rows = []
        for i, (lh, la) in enumerate(zip(lambda_home, lambda_away)):
            mat     = score_matrix(lh, la)
            markets = derive_markets(mat)
            rows.append({
                "lambda_home": round(float(lh), 3),
                "lambda_away": round(float(la), 3),
                **markets,
                "top5_scores": json.dumps(markets["top5_scores"]),
            })

        return pd.DataFrame(rows, index=df.index)

    def evaluate(self, df_test: pd.DataFrame) -> dict:
        """Evalúa el modelo en test. Requiere home_score y away_score reales."""
        required = ["target_home_goals", "target_away_goals"]
        mask = df_test[required[0]].notna() & df_test[required[1]].notna()
        df_clean = df_test[mask].copy()

        if df_clean.empty:
            log.warning("Sin datos de test con scores reales")
            return {}

        lambda_home, lambda_away = self.predict_lambdas(df_clean)
        y_h = df_clean["target_home_goals"].values
        y_a = df_clean["target_away_goals"].values

        mae_h  = mean_absolute_error(y_h, lambda_home)
        mae_a  = mean_absolute_error(y_a, lambda_away)
        rmse_h = np.sqrt(mean_squared_error(y_h, lambda_home))
        rmse_a = np.sqrt(mean_squared_error(y_a, lambda_away))

        # Precisión de resultado derivado
        pred_result = []
        actual_result = []
        for lh, la, hs, as_ in zip(lambda_home, lambda_away, y_h, y_a):
            mat = score_matrix(lh, la)
            m   = derive_markets(mat)
            probs = [m["prob_H"], m["prob_D"], m["prob_A"]]
            pred_result.append(["H", "D", "A"][np.argmax(probs)])
            if hs > as_:
                actual_result.append("H")
            elif hs < as_:
                actual_result.append("A")
            else:
                actual_result.append("D")

        accuracy = np.mean([p == a for p, a in zip(pred_result, actual_result)])

        metrics = {
            "n_test":       len(df_clean),
            "mae_home":     round(mae_h,  4),
            "mae_away":     round(mae_a,  4),
            "rmse_home":    round(rmse_h, 4),
            "rmse_away":    round(rmse_a, 4),
            "mae_avg":      round((mae_h + mae_a) / 2, 4),
            "accuracy_1x2": round(accuracy, 4),
            "mean_lambda_home": round(float(lambda_home.mean()), 3),
            "mean_lambda_away": round(float(lambda_away.mean()), 3),
        }

        log.info("Evaluación Poisson:")
        for k, v in metrics.items():
            log.info("  %-22s %s", k, v)

        return metrics

    def feature_importance(self, top_n: int = 10) -> dict:
        """Top features por importancia para cada modelo."""
        result = {}
        for name, model, feats in [
            ("lambda_home", self.model_home, self.feats_home),
            ("lambda_away", self.model_away, self.feats_away),
        ]:
            fi = pd.DataFrame({
                "feature":    feats,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False).head(top_n)
            result[name] = fi.reset_index(drop=True)
        return result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class PoissonRunConfig:
    features_path:    str = "data/processed/features_train.csv"
    wc_features_path: str = "data/processed/features_wc2026.csv"
    output_dir:       str = "outputs"
    train_cutoff:     int = 2018
    val_cutoff:       int = 2023


def _add_goal_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Añade target_home_goals y target_away_goals si no existen."""
    df = df.copy()
    if "target_home_goals" not in df.columns:
        if "home_score" in df.columns:
            df["target_home_goals"] = pd.to_numeric(df["home_score"], errors="coerce")
        else:
            df["target_home_goals"] = np.nan
    if "target_away_goals" not in df.columns:
        if "away_score" in df.columns:
            df["target_away_goals"] = pd.to_numeric(df["away_score"], errors="coerce")
        else:
            df["target_away_goals"] = np.nan
    return df


def run_poisson_pipeline(config: PoissonRunConfig) -> tuple[dict, pd.DataFrame]:
    """
    Entrena el modelo Poisson y genera predicciones WC2026.

    Retorna (metrics, predictions_df).
    """
    os.makedirs(config.output_dir, exist_ok=True)

    # Cargar features
    if not os.path.exists(config.features_path):
        raise FileNotFoundError(f"No se encontró {config.features_path}")

    log.info("Cargando %s...", config.features_path)
    df = pd.read_csv(config.features_path, parse_dates=["date"])
    df = _add_goal_targets(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # Split temporal
    df_train = df[df["date"].dt.year < config.train_cutoff].copy()
    df_val   = df[
        (df["date"].dt.year >= config.train_cutoff) &
        (df["date"].dt.year < config.val_cutoff)
    ].copy()

    log.info("=" * 60)
    log.info("Poisson Goal Model Pipeline")
    log.info("=" * 60)
    log.info("Train: %d partidos (< %d)", len(df_train), config.train_cutoff)
    log.info("Val:   %d partidos (%d-%d)", len(df_val), config.train_cutoff, config.val_cutoff - 1)

    # Entrenar
    model = PoissonGoalModel()
    model.fit(df_train)

    # Evaluar
    metrics = model.evaluate(df_val)

    # Feature importance
    fi = model.feature_importance(top_n=10)
    for name, df_fi in fi.items():
        log.info("\nTop features — %s:", name)
        for _, row in df_fi.iterrows():
            log.info("  %-35s %.4f", row["feature"], row["importance"])

    # Guardar métricas
    metrics_path = os.path.join(config.output_dir, "poisson_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Métricas guardadas en %s", metrics_path)

    # Predicciones WC2026
    predictions = pd.DataFrame()
    if os.path.exists(config.wc_features_path):
        log.info("Generando predicciones WC2026...")
        df_wc = pd.read_csv(config.wc_features_path, parse_dates=["date"])
        df_wc = _add_goal_targets(df_wc)

        # Excluir placeholders (slots sin equipos reales)
        placeholder_mask = (
            df_wc["elo_diff"].isna() &
            df_wc["home_form_ppg_10"].isna() &
            df_wc["away_form_ppg_10"].isna()
        )
        df_wc_real = df_wc[~placeholder_mask].copy()
        df_wc_placeholder = df_wc[placeholder_mask].copy()

        preds = model.predict_markets(df_wc_real)

        predictions = pd.concat([
            df_wc_real[["date", "home_team", "away_team"]].reset_index(drop=True),
            preds.reset_index(drop=True),
        ], axis=1)
        predictions["is_placeholder"] = False

        if not df_wc_placeholder.empty:
            ph = df_wc_placeholder[["date", "home_team", "away_team"]].copy()
            ph["is_placeholder"] = True
            predictions = pd.concat([predictions, ph], ignore_index=True)

        predictions = predictions.sort_values("date").reset_index(drop=True)

        out_path = os.path.join(config.output_dir, "wc2026_poisson_predictions.csv")
        predictions.to_csv(out_path, index=False)
        log.info("Predicciones guardadas en %s", out_path)

        _print_predictions(predictions[~predictions["is_placeholder"]])

    return metrics, predictions


def _print_predictions(df: pd.DataFrame) -> None:
    print("\n" + "=" * 85)
    print("PREDICCIONES WC2026 — MODELO POISSON")
    print(f"{'Partido':<40} {'λH':>5} {'λA':>5} {'P(H)':>7} {'P(D)':>7} {'P(A)':>7} {'O2.5':>7} {'BTTS':>7}")
    print("-" * 85)
    for _, row in df.iterrows():
        if pd.isna(row.get("lambda_home")):
            continue
        partido = f"{row['home_team']} vs {row['away_team']}"[:38]
        print(
            f"  {partido:<38} "
            f"{row['lambda_home']:>5.2f} "
            f"{row['lambda_away']:>5.2f} "
            f"{row['prob_H']:>7.3f} "
            f"{row['prob_D']:>7.3f} "
            f"{row['prob_A']:>7.3f} "
            f"{row['prob_over25']:>7.3f} "
            f"{row['prob_btts']:>7.3f}"
        )
    print("=" * 85)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modelo Poisson de goles para WC2026.")
    parser.add_argument("--features",    default="data/processed/features_train.csv")
    parser.add_argument("--wc-features", default="data/processed/features_wc2026.csv")
    parser.add_argument("--output-dir",  default="outputs")
    parser.add_argument("--train-cutoff", type=int, default=2018)
    parser.add_argument("--val-cutoff",   type=int, default=2023)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args()
    config = PoissonRunConfig(
        features_path    = args.features,
        wc_features_path = args.wc_features,
        output_dir       = args.output_dir,
        train_cutoff     = args.train_cutoff,
        val_cutoff       = args.val_cutoff,
    )
    run_poisson_pipeline(config)


if __name__ == "__main__":
    main()
