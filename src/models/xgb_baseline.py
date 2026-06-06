"""
src/models/xgb_baseline.py
===========================
XGBoost baseline para todos los mercados del Mundial 2026.

Mercados implementados:
  - Resultado 1X2 (clasificación multiclase)
  - Over/Under 2.5 goles (clasificación binaria)
  - Ambos equipos marcan (clasificación binaria)
  - Total de goles (regresión Poisson)
  - Córners totales (regresión Poisson)
  - Tarjetas amarillas totales (tasa analítica: propensión equipos × factor árbitro)

Uso:
    python src/models/xgb_baseline.py
"""

import os
import json
import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    roc_auc_score, f1_score, mean_absolute_error,
    mean_squared_error,
)

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _metric_float(value, ndigits: int = 4) -> float:
    return round(float(value), ndigits)


# ---------------------------------------------------------------------------
# Predicción analítica de amarillas
# ---------------------------------------------------------------------------

GLOBAL_YELLOW_MEAN = 4.27  # media real en partidos internacionales (StatsBomb)


def predict_yellows_analytical(
    home_yellow_per_90: float,
    away_yellow_per_90: float,
    ref_yellow_per_match: float,
    minutes: float = 95.0,
) -> float:
    """
    Amarillas esperadas = propensión_equipos × factor_árbitro.

    Usa las stats de tarjetas por 90min de cada equipo (calculadas desde
    miles de partidos de club en Transfermarkt) y el perfil histórico del
    árbitro del partido. Mucho más fiable que el XGBoost entrenado con
    solo ~116 partidos de StatsBomb.

    Fallbacks:
      - Sin datos de árbitro: usa solo la tasa de equipos.
      - Sin datos de equipos: usa directamente el promedio del árbitro.
      - Sin ninguno: devuelve la media global.
    """
    has_teams = not (np.isnan(home_yellow_per_90) or np.isnan(away_yellow_per_90))
    has_ref   = not np.isnan(ref_yellow_per_match)

    if has_teams and has_ref:
        team_rate  = (home_yellow_per_90 + away_yellow_per_90) / 90.0 * minutes
        ref_factor = ref_yellow_per_match / GLOBAL_YELLOW_MEAN
        return round(team_rate * ref_factor, 2)
    elif has_teams:
        return round((home_yellow_per_90 + away_yellow_per_90) / 90.0 * minutes, 2)
    elif has_ref:
        return round(ref_yellow_per_match, 2)
    else:
        return round(GLOBAL_YELLOW_MEAN, 2)


# ---------------------------------------------------------------------------
# Configuración de features por mercado
# ---------------------------------------------------------------------------

FORM_FEATURES = [
    "home_form_wins_10", "home_form_draws_10", "home_form_losses_10",
    "home_form_gf_10", "home_form_gc_10", "home_form_ppg_10",
    "home_form_streak",
    "away_form_wins_10", "away_form_draws_10", "away_form_losses_10",
    "away_form_gf_10", "away_form_gc_10", "away_form_ppg_10",
    "away_form_streak",
    "diff_form_wins_10", "diff_form_gf_10", "diff_form_gc_10", "diff_form_ppg_10",
    "diff_form_streak",
]

CONTEXT_FEATURES = [
    "is_neutral", "home_is_host", "away_is_host", "effective_home_adv",
    "same_confederation", "stage_ordinal",
    "home_rest_days", "away_rest_days", "rest_days_diff",
]

EXTRA_FEATURES = [
    "altitude_m", "is_indoor",
    "home_travel_km", "away_travel_km", "travel_km_diff",
    "elo_diff",
    "rating_diff",
    "home_form_neutral_wins_10", "away_form_neutral_wins_10",
    "diff_form_neutral_wins_10",
    "ref_yellow_per_match", "ref_red_per_match",
    "home_squad_size", "away_squad_size", "squad_size_diff",
    "home_goalkeepers", "away_goalkeepers", "goalkeepers_diff",
    "home_defenders", "away_defenders", "defenders_diff",
    "home_midfielders", "away_midfielders", "midfielders_diff",
    "home_forwards", "away_forwards", "forwards_diff",
    "home_unique_clubs", "away_unique_clubs", "unique_clubs_diff",
]

# Córners: añadidas features de ataque que ya generamos y antes no usábamos
CORNERS_FEATURES = [
    "home_corners_avg_all", "away_corners_avg_all", "corners_avg_all_sum",   # ← NUEVAS
    "home_form_corners_for_10", "away_form_corners_for_10",
    "diff_form_corners_for_10",
    "home_attack_norm", "away_attack_norm", "attack_diff",
    "home_goals_per_90", "away_goals_per_90", "goals_per_90_diff",
    "elo_diff", "rating_diff",
    "effective_home_adv", "stage_ordinal",
]

# Tarjetas: añadidas propensión de equipos desde stats de club (Transfermarkt)
CARDS_FEATURES = [
    "home_form_yellows_10", "away_form_yellows_10",
    "diff_form_yellows_10",
    "ref_yellow_per_match", "ref_red_per_match",
    "home_yellow_per_90", "away_yellow_per_90", "yellow_per_90_diff",
]


def _get_features_for_market(market: str) -> list[str]:
    base = FORM_FEATURES + CONTEXT_FEATURES + EXTRA_FEATURES
    if market in ("corners", "over85c"):
        return list(dict.fromkeys(base + CORNERS_FEATURES))
    if market in ("yellows", "over35y"):
        return list(dict.fromkeys(base + CARDS_FEATURES))
    return list(dict.fromkeys(base))


# ---------------------------------------------------------------------------
# Definición de mercados
# ---------------------------------------------------------------------------

@dataclass
class MarketConfig:
    name: str
    target_col: str
    task: str
    eval_metric: str
    xgb_params: dict
    min_samples: int = 100
    split_strategy: str = "global_temporal"


MARKETS: dict[str, MarketConfig] = {
    "result_1x2": MarketConfig(
        name="Resultado 1X2",
        target_col="target_result",
        task="multiclass",
        eval_metric="mlogloss",
        xgb_params={
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": 5,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
    ),
    "over25": MarketConfig(
        name="Over 2.5 Goles",
        target_col="target_over25",
        task="binary",
        eval_metric="logloss",
        xgb_params={
            "objective": "binary:logistic",
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 250,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "scale_pos_weight": 1.0,
        },
    ),
    "btts": MarketConfig(
        name="Ambos Marcan (BTTS)",
        target_col="target_btts",
        task="binary",
        eval_metric="logloss",
        xgb_params={
            "objective": "binary:logistic",
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 250,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
        },
    ),
    "total_goals": MarketConfig(
        name="Total Goles",
        target_col="target_total_goals",
        task="poisson",
        eval_metric="poisson-nloglik",
        xgb_params={
            "objective": "count:poisson",
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 250,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
        },
    ),
    "corners": MarketConfig(
        name="Córners Totales",
        target_col="target_corners",
        task="poisson",
        eval_metric="poisson-nloglik",
        xgb_params={
            "objective": "count:poisson",
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
        },
        min_samples=50,
        split_strategy="latest_target_year",
    ),
    "yellows": MarketConfig(
        name="Tarjetas Amarillas",
        target_col="target_yellows",
        task="poisson",
        eval_metric="poisson-nloglik",
        xgb_params={
            "objective": "count:poisson",
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
        },
        min_samples=50,
        split_strategy="latest_target_year",
    ),
}


# ---------------------------------------------------------------------------
# Modelo por mercado
# ---------------------------------------------------------------------------

class MarketModel:
    """Wrapper de XGBoost para un mercado específico."""

    def __init__(self, config: MarketConfig):
        self.config = config
        self.feature_cols: list[str] = []
        self.model: Optional[xgb.XGBClassifier | xgb.XGBRegressor] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.metrics: dict = {}

    def _select_features(self, df: pd.DataFrame) -> list[str]:
        market_key = self.config.target_col.replace("target_", "")
        pool = _get_features_for_market(market_key)
        available = []
        for f in pool:
            if f in df.columns:
                if df[f].notna().mean() > 0.10:
                    available.append(f)
        return available

    def _prepare_data(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
        target = self.config.target_col

        if target not in df.columns:
            raise ValueError(
                f"Columna '{target}' no encontrada en el dataset."
            )

        mask = df[target].notna()
        df_clean = df[mask].copy()

        if len(df_clean) < self.config.min_samples:
            raise ValueError(
                f"Solo {len(df_clean)} muestras con target '{target}'. "
                f"Mínimo requerido: {self.config.min_samples}"
            )

        self.feature_cols = self._select_features(df_clean)
        if not self.feature_cols:
            raise ValueError(f"No hay features disponibles para '{self.config.name}'")

        X = df_clean[self.feature_cols].copy()
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

        y = df_clean[target].copy()

        if self.config.task == "multiclass":
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.array(["A", "D", "H"])
            y = self.label_encoder.transform(y.values)

        return X.values.astype(np.float32), np.array(y, dtype=np.float32), self.feature_cols

    def fit(self, df_train: pd.DataFrame) -> "MarketModel":
        log.info("─── %s ───", self.config.name)

        X, y, features = self._prepare_data(df_train)
        log.info("  Train: %d muestras, %d features", X.shape[0], X.shape[1])

        params = self.config.xgb_params.copy()
        n_estimators = params.pop("n_estimators", 200)

        if self.config.task in ("multiclass", "binary"):
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                random_state=42,
                use_label_encoder=False,
                eval_metric=self.config.eval_metric,
                **params,
            )
        else:
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                random_state=42,
                eval_metric=self.config.eval_metric,
                **params,
            )

        self.model.fit(X, y)
        log.info("  ✓ Modelo entrenado")
        return self

    def evaluate(self, df_test: pd.DataFrame) -> dict:
        target = self.config.target_col
        mask = df_test[target].notna()
        df_clean = df_test[mask].copy()

        if df_clean.empty:
            log.warning("  Sin datos de test con target '%s'", target)
            return {}

        X = df_clean[self.feature_cols].copy()
        for col in X.columns:
            if col not in self.feature_cols:
                X[col] = 0
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        X = X.values.astype(np.float32)

        y_true = df_clean[target].values

        metrics = {
            "market": self.config.name,
            "target_col": target,
            "n_test": len(y_true),
        }

        if self.config.task == "multiclass":
            y_true_enc = self.label_encoder.transform(y_true)
            proba = self.model.predict_proba(X)
            preds = self.label_encoder.inverse_transform(np.argmax(proba, axis=1))

            metrics["accuracy"] = _metric_float(accuracy_score(y_true, preds))
            metrics["log_loss"] = _metric_float(
                log_loss(y_true_enc, proba, labels=np.arange(len(self.label_encoder.classes_))),
            )
            y_onehot = np.zeros_like(proba)
            y_onehot[np.arange(len(y_true_enc)), y_true_enc] = 1
            metrics["brier"] = _metric_float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))

            for i, cls in enumerate(self.label_encoder.classes_):
                metrics[f"pred_mean_{cls}"] = _metric_float(proba[:, i].mean())
                metrics[f"actual_pct_{cls}"] = _metric_float((y_true == cls).mean())

        elif self.config.task == "binary":
            y_true_binary = y_true.astype(int)
            proba_matrix = self.model.predict_proba(X)
            proba = proba_matrix[:, 1] if proba_matrix.ndim == 2 else proba_matrix
            preds = (proba >= 0.5).astype(int)

            metrics["accuracy"] = _metric_float(accuracy_score(y_true_binary, preds))
            metrics["log_loss"] = _metric_float(log_loss(y_true_binary, proba, labels=[0, 1]))
            metrics["brier"] = _metric_float(brier_score_loss(y_true_binary, proba))
            if len(np.unique(y_true_binary)) == 2:
                metrics["auc"] = _metric_float(roc_auc_score(y_true_binary, proba))
            else:
                metrics["auc"] = float("nan")
            metrics["f1"] = _metric_float(f1_score(y_true_binary, preds, zero_division=0))
            metrics["pred_mean"] = _metric_float(proba.mean())
            metrics["pred_std"] = _metric_float(proba.std())
            metrics["actual_mean"] = _metric_float(y_true_binary.mean())
            metrics["positive_count"] = int(y_true_binary.sum())

        else:
            y_true_float = y_true.astype(float)
            preds = self.model.predict(X)
            metrics["mae"] = _metric_float(mean_absolute_error(y_true_float, preds))
            metrics["rmse"] = _metric_float(np.sqrt(mean_squared_error(y_true_float, preds)))
            metrics["pred_mean"] = _metric_float(preds.mean())
            metrics["actual_mean"] = _metric_float(y_true_float.mean())

        self.metrics = metrics
        return metrics

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].copy()
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        X = X.values.astype(np.float32)

        if self.config.task in ("multiclass", "binary"):
            return self.model.predict_proba(X)
        else:
            return self.model.predict(X)

    def feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        imp = self.model.feature_importances_
        fi = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": imp,
        }).sort_values("importance", ascending=False).head(top_n)
        return fi.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

class XGBBaselinePipeline:
    """
    Pipeline que entrena y evalúa un modelo XGBoost por mercado.
    Las amarillas usan predicción analítica en WC2026 (propensión equipos × árbitro).
    """

    def __init__(
        self,
        train_cutoff: int = 2018,
        val_cutoff: int = 2023,
        markets: Optional[list[str]] = None,
    ):
        self.train_cutoff = train_cutoff
        self.val_cutoff = val_cutoff
        self.market_names = markets or list(MARKETS.keys())
        self.models: dict[str, MarketModel] = {}
        self.all_metrics: list[dict] = []

    def _split_for_market(self, df: pd.DataFrame, config: MarketConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
        if config.split_strategy == "latest_target_year":
            target_df = df[df[config.target_col].notna()].copy()
            if target_df.empty:
                return target_df, target_df
            latest_year = int(target_df["date"].dt.year.max())
            df_train = target_df[target_df["date"].dt.year < latest_year].copy()
            df_val = target_df[target_df["date"].dt.year == latest_year].copy()
            return df_train, df_val

        df_train = df[df["date"].dt.year < self.train_cutoff].copy()
        df_val = df[
            (df["date"].dt.year >= self.train_cutoff) &
            (df["date"].dt.year < self.val_cutoff)
        ].copy()
        return df_train, df_val

    def run(self, df: pd.DataFrame, output_dir: str = "./outputs") -> pd.DataFrame:
        os.makedirs(output_dir, exist_ok=True)

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

        df_train_global = df[df["date"].dt.year < self.train_cutoff].copy()
        df_val_global = df[
            (df["date"].dt.year >= self.train_cutoff) &
            (df["date"].dt.year < self.val_cutoff)
        ].copy()

        log.info("=" * 60)
        log.info("XGBoost Baseline Pipeline")
        log.info("=" * 60)
        log.info("Train global:  %d partidos  (< %d)", len(df_train_global), self.train_cutoff)
        log.info("Val global:    %d partidos  (%d-%d)", len(df_val_global), self.train_cutoff, self.val_cutoff - 1)
        log.info("=" * 60)

        for market_key in self.market_names:
            config = MARKETS.get(market_key)
            if config is None:
                log.warning("Mercado desconocido: %s", market_key)
                continue

            model = MarketModel(config)
            df_train, df_val = self._split_for_market(df, config)
            train_target_n = df_train[config.target_col].notna().sum() if config.target_col in df_train else 0
            val_target_n = df_val[config.target_col].notna().sum() if config.target_col in df_val else 0
            log.info(
                "Split %s: train=%d (%d target), val=%d (%d target)",
                config.split_strategy, len(df_train), train_target_n, len(df_val), val_target_n,
            )

            try:
                model.fit(df_train)
                metrics = model.evaluate(df_val)
                self.models[market_key] = model

                if metrics:
                    self.all_metrics.append(metrics)
                    self._print_metrics(metrics)

                    fi = model.feature_importance(top_n=10)
                    log.info("  Top features:")
                    for _, row in fi.iterrows():
                        log.info("    %-35s  %.4f", row["feature"], row["importance"])

            # ── NUEVO: reentrenar con TODOS los datos para predecir WC2026 ──
                full_df = df[df[config.target_col].notna()].copy()
                refit_model = MarketModel(config)
                refit_model.fit(full_df)
                self.models[market_key] = refit_model
                log.info(
                    "  ↻ %s reentrenado con %d partidos (todos) para prediccion",
                    config.name, len(full_df),
                )

            except ValueError as e:
                log.warning("  ⚠ %s: %s", config.name, e)
                continue

        summary = pd.DataFrame(self.all_metrics)
        if not summary.empty:
            self._print_summary(summary)
            summary_path = os.path.join(output_dir, "xgb_baseline_metrics.json")
            with open(summary_path, "w") as f:
                json.dump(self.all_metrics, f, indent=2, default=str)
            log.info("Métricas guardadas en %s", summary_path)

        return summary

    def predict_wc2026(self, df_wc: pd.DataFrame, output_dir: str = "./outputs") -> pd.DataFrame:
        """
        Genera predicciones para los partidos del WC2026.
        Las amarillas usan la tasa analítica (propensión equipos × árbitro).
        """
        if not self.models:
            log.error("No hay modelos entrenados. Ejecuta run() primero.")
            return pd.DataFrame()

        results = df_wc[["date", "home_team", "away_team"]].copy()
        placeholder_mask = self._placeholder_match_mask(df_wc)
        results["is_placeholder_match"] = placeholder_mask

        for market_key, model in self.models.items():
            config = model.config

            try:
                pred_rows = df_wc[~placeholder_mask].copy()
                if pred_rows.empty:
                    continue

                # ── Amarillas: tasa analítica en vez del XGBoost ──────────────
                if market_key == "yellows":
                    analytical_preds = []
                    for _, row in pred_rows.iterrows():
                        y = predict_yellows_analytical(
                            float(row.get("home_yellow_per_90", np.nan)),
                            float(row.get("away_yellow_per_90", np.nan)),
                            float(row.get("ref_yellow_per_match", np.nan)),
                        )
                        analytical_preds.append(y)
                    results["pred_yellows"] = pd.NA
                    results.loc[~placeholder_mask, "pred_yellows"] = np.array(analytical_preds).round(2)
                    log.info("  ✓ Amarillas: tasa analítica aplicada (%d partidos)", len(analytical_preds))
                    continue

                preds = model.predict_proba(pred_rows)

                if config.task == "multiclass":
                    for i, cls in enumerate(model.label_encoder.classes_):
                        results[f"prob_{cls}"] = pd.NA
                        results.loc[~placeholder_mask, f"prob_{cls}"] = preds[:, i].round(4)
                    results["pred_result"] = pd.NA
                    results.loc[~placeholder_mask, "pred_result"] = model.label_encoder.inverse_transform(
                        np.argmax(preds, axis=1)
                    )

                elif config.task == "binary":
                    col_name = config.target_col.replace("target_", "")
                    results[f"prob_{col_name}"] = pd.NA
                    results.loc[~placeholder_mask, f"prob_{col_name}"] = preds[:, 1].round(4)

                else:
                    col_name = config.target_col.replace("target_", "")
                    results[f"pred_{col_name}"] = pd.NA
                    results.loc[~placeholder_mask, f"pred_{col_name}"] = preds.round(2)

            except Exception as e:
                log.warning("  ⚠ Predicción %s falló: %s", config.name, e)

        n_placeholders = int(placeholder_mask.sum())
        if n_placeholders:
            log.info(
                "WC2026: %d partidos de eliminatorias/slots sin predicción hasta resolver equipos reales",
                n_placeholders,
            )

        out_path = os.path.join(output_dir, "wc2026_predictions.csv")
        results.to_csv(out_path, index=False)
        log.info("Predicciones WC2026 guardadas en %s", out_path)
        return results

    @staticmethod
    def _placeholder_match_mask(df: pd.DataFrame) -> pd.Series:
        required = ["elo_diff", "home_form_ppg_10", "away_form_ppg_10"]
        if not all(col in df.columns for col in required):
            return pd.Series(False, index=df.index)
        return (
            df["elo_diff"].isna()
            & df["home_form_ppg_10"].isna()
            & df["away_form_ppg_10"].isna()
        )

    def _print_metrics(self, metrics: dict) -> None:
        for k, v in metrics.items():
            if k in ("market", "n_test"):
                continue
            log.info("  %-20s %s", k, v)

    def _print_summary(self, summary: pd.DataFrame) -> None:
        log.info("")
        log.info("=" * 60)
        log.info("RESUMEN — XGBoost Baseline")
        log.info("=" * 60)
        for _, row in summary.iterrows():
            market = row.get("market", "?")
            n = row.get("n_test", 0)
            line = f"  {market:<25} n={n}"
            if "accuracy" in row and pd.notna(row["accuracy"]):
                line += f"  acc={row['accuracy']:.3f}"
            if "log_loss" in row and pd.notna(row["log_loss"]):
                line += f"  ll={row['log_loss']:.3f}"
            if "brier" in row and pd.notna(row["brier"]):
                line += f"  brier={row['brier']:.3f}"
            if "auc" in row and pd.notna(row["auc"]):
                line += f"  auc={row['auc']:.3f}"
            if "mae" in row and pd.notna(row["mae"]):
                line += f"  mae={row['mae']:.3f}"
            if "rmse" in row and pd.notna(row["rmse"]):
                line += f"  rmse={row['rmse']:.3f}"
            log.info(line)
        log.info("=" * 60)


# ---------------------------------------------------------------------------
# Script standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_DIR = "./data"
    OUTPUT_DIR = "./outputs"

    features_path = os.path.join(DATA_DIR, "processed", "features_train.csv")
    wc_path = os.path.join(DATA_DIR, "processed", "features_wc2026.csv")

    if not os.path.exists(features_path):
        features_path = os.path.join(DATA_DIR, "features.csv")

    if not os.path.exists(features_path):
        print("No se encuentra features_train.csv. Ejecuta feature_engineering.py primero.")
        exit(1)

    print(f"Cargando {features_path}...")
    df = pd.read_csv(features_path, parse_dates=["date"])
    print(f"  → {len(df)} partidos")

    pipeline = XGBBaselinePipeline(train_cutoff=2018, val_cutoff=2023)
    summary = pipeline.run(df, output_dir=OUTPUT_DIR)

    if os.path.exists(wc_path):
        print(f"\nCargando {wc_path}...")
        df_wc = pd.read_csv(wc_path, parse_dates=["date"])
        print(f"  → {len(df_wc)} partidos WC2026")
        predictions = pipeline.predict_wc2026(df_wc, output_dir=OUTPUT_DIR)
        if not predictions.empty:
            print("\nPredicciones WC2026 (muestra):")
            print(predictions.head(10).to_string())
    else:
        print(f"\n{wc_path} no encontrado — omitiendo predicciones WC2026")
        