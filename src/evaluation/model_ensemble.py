"""
src/evaluation/model_ensemble.py
================================
Combina XGBoost y Poisson en una unica prediccion/cuota final por mercado.

Idea de negocio:
  - XGBoost directo suele capturar mejor partidos parejos.
  - Poisson de goles suele ser mas coherente cuando hay mucha diferencia
    entre selecciones, porque fuerza una distribucion de marcador consistente.

El peso de Poisson aumenta con la desigualdad del partido, medida por
abs(rating_diff) y, si no existe, por abs(elo_diff).

Uso:
    python -m src.evaluation.model_ensemble
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss


ONE_X_TWO = ("H", "D", "A")
ONE_X_TWO_INDEX = {side: idx for idx, side in enumerate(ONE_X_TWO)}
PROB_MARKETS = {
    "over25": "prob_over25",
    "btts": "prob_btts",
}
DEFAULT_WEIGHT_CONFIG = "outputs/ensemble_weight_params.json"
DEFAULT_CALIBRATION_CONFIG = "outputs/ensemble_calibration.json"


@dataclass(frozen=True)
class EnsembleWeightParams:
    close_rating: float = 0.15
    strong_rating: float = 0.45
    min_weight: float = 0.25
    max_weight: float = 0.80


DEFAULT_WEIGHT_PARAMS = EnsembleWeightParams()


def _to_date_key(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date.astype(str)


def _safe_prob(value) -> float:
    if pd.isna(value):
        return np.nan
    return float(np.clip(float(value), 0.0001, 0.9999))


def _safe_odds(prob: float) -> float:
    if pd.isna(prob) or prob <= 0:
        return np.nan
    return round(1.0 / max(float(prob), 0.0001), 2)


def _coerce_weight_params(value: dict | EnsembleWeightParams | None) -> EnsembleWeightParams:
    if value is None:
        return DEFAULT_WEIGHT_PARAMS
    if isinstance(value, EnsembleWeightParams):
        return value
    return EnsembleWeightParams(
        close_rating=float(value.get("close_rating", DEFAULT_WEIGHT_PARAMS.close_rating)),
        strong_rating=float(value.get("strong_rating", DEFAULT_WEIGHT_PARAMS.strong_rating)),
        min_weight=float(value.get("min_weight", DEFAULT_WEIGHT_PARAMS.min_weight)),
        max_weight=float(value.get("max_weight", DEFAULT_WEIGHT_PARAMS.max_weight)),
    )


def load_weight_params(path: str = DEFAULT_WEIGHT_CONFIG) -> tuple[EnsembleWeightParams, str]:
    """Load optimized ensemble weights when present; otherwise use defaults."""
    if not path or not os.path.exists(path):
        return DEFAULT_WEIGHT_PARAMS, "default"
    with open(path) as f:
        payload = json.load(f)
    params = _coerce_weight_params(payload.get("params", payload))
    return params, path


def save_weight_params(
    params: EnsembleWeightParams,
    metrics: dict,
    output_path: str = DEFAULT_WEIGHT_CONFIG,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = {
        "params": asdict(params),
        "metrics": metrics,
        "objective": "1x2_log_loss",
        "validation_window": {
            "train": "date.year < train_cutoff",
            "validation": "train_cutoff <= date.year < val_cutoff",
        },
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def load_calibration_config(path: str = DEFAULT_CALIBRATION_CONFIG) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_calibration_config(payload: dict, output_path: str = DEFAULT_CALIBRATION_CONFIG) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def _poisson_weight(
    rating_diff,
    elo_diff,
    *,
    params: EnsembleWeightParams | None = None,
    close_rating: float = 0.15,
    strong_rating: float = 0.45,
    min_weight: float = 0.25,
    max_weight: float = 0.80,
) -> tuple[float, float, str]:
    """
    Return (poisson_weight, strength, regime).

    close_rating: partidos parejos, favorece XGBoost.
    strong_rating: partidos muy desiguales, favorece Poisson.
    """
    if params is not None:
        close_rating = params.close_rating
        strong_rating = params.strong_rating
        min_weight = params.min_weight
        max_weight = params.max_weight

    if pd.notna(rating_diff):
        strength = abs(float(rating_diff))
    elif pd.notna(elo_diff):
        strength = min(abs(float(elo_diff)) / 600.0, 1.0)
    else:
        strength = close_rating

    scale = (strength - close_rating) / (strong_rating - close_rating)
    scale = float(np.clip(scale, 0.0, 1.0))
    weight = min_weight + scale * (max_weight - min_weight)

    if weight <= 0.35:
        regime = "xgb_lean"
    elif weight >= 0.70:
        regime = "poisson_lean"
    else:
        regime = "blend"
    return round(weight, 4), round(strength, 4), regime


def _confidence_from_probs(xgb_prob: float, poisson_prob: float) -> tuple[float, str]:
    if pd.isna(xgb_prob) or pd.isna(poisson_prob):
        return np.nan, "N/A"
    diff = abs(float(xgb_prob) - float(poisson_prob))
    if diff < 0.05:
        return round(diff, 4), "high"
    if diff < 0.10:
        return round(diff, 4), "medium"
    return round(diff, 4), "low"


def _weighted_prob(xgb_prob, poisson_prob, poisson_weight: float) -> float:
    xgb_prob = _safe_prob(xgb_prob)
    poisson_prob = _safe_prob(poisson_prob)
    if pd.isna(xgb_prob):
        return poisson_prob
    if pd.isna(poisson_prob):
        return xgb_prob
    return float((1.0 - poisson_weight) * xgb_prob + poisson_weight * poisson_prob)


def _weighted_numeric(xgb_value, poisson_value, poisson_weight: float) -> float:
    xgb_value = np.nan if pd.isna(xgb_value) else float(xgb_value)
    poisson_value = np.nan if pd.isna(poisson_value) else float(poisson_value)
    if pd.isna(xgb_value):
        return poisson_value
    if pd.isna(poisson_value):
        return xgb_value
    return float((1.0 - poisson_weight) * xgb_value + poisson_weight * poisson_value)


def _weight_vector(df: pd.DataFrame, params: EnsembleWeightParams) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized version of _poisson_weight for optimization."""
    rating = pd.to_numeric(df.get("rating_diff"), errors="coerce")
    elo = pd.to_numeric(df.get("elo_diff"), errors="coerce")
    rating_values = rating.to_numpy(dtype=float) if rating is not None else np.full(len(df), np.nan)
    elo_values = elo.to_numpy(dtype=float) if elo is not None else np.full(len(df), np.nan)

    strength = np.where(
        np.isfinite(rating_values),
        np.abs(rating_values),
        np.where(np.isfinite(elo_values), np.minimum(np.abs(elo_values) / 600.0, 1.0), params.close_rating),
    )
    denom = max(params.strong_rating - params.close_rating, 1e-6)
    scale = np.clip((strength - params.close_rating) / denom, 0.0, 1.0)
    weight = params.min_weight + scale * (params.max_weight - params.min_weight)
    return weight.astype(float), strength.astype(float)


def _blend_validation_probs(df: pd.DataFrame, params: EnsembleWeightParams) -> np.ndarray:
    weights, _ = _weight_vector(df, params)
    xgb_probs = df[[f"xgb_prob_{side}" for side in ONE_X_TWO]].to_numpy(dtype=float)
    poisson_probs = df[[f"poisson_prob_{side}" for side in ONE_X_TWO]].to_numpy(dtype=float)
    probs = (1.0 - weights[:, None]) * xgb_probs + weights[:, None] * poisson_probs
    probs = np.clip(probs, 1e-6, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def _one_x_two_metrics(labels: pd.Series | np.ndarray, probs: np.ndarray) -> dict:
    labels = np.asarray(labels)
    encoded = np.asarray([ONE_X_TWO_INDEX[label] for label in labels], dtype=int)
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-6, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    preds = np.asarray(ONE_X_TWO)[np.argmax(probs, axis=1)]
    return {
        "log_loss": round(float(log_loss(encoded, probs, labels=np.arange(len(ONE_X_TWO)))), 6),
        "accuracy": round(float(accuracy_score(labels, preds)), 6),
        "pred_draw_mean": round(float(probs[:, 1].mean()), 6),
        "actual_draw_rate": round(float(np.mean(labels == "D")), 6),
    }


def _isotonic_payload(model: IsotonicRegression) -> dict:
    return {
        "x_thresholds": [float(x) for x in model.X_thresholds_],
        "y_thresholds": [float(y) for y in model.y_thresholds_],
    }


def _apply_isotonic(values: np.ndarray, payload: dict) -> np.ndarray:
    x = np.asarray(payload["x_thresholds"], dtype=float)
    y = np.asarray(payload["y_thresholds"], dtype=float)
    return np.interp(np.asarray(values, dtype=float), x, y, left=y[0], right=y[-1])


def _apply_calibration_matrix(probs: np.ndarray, calibration_payload: dict | None) -> np.ndarray:
    if not calibration_payload:
        return probs
    calibrators = calibration_payload.get("calibrators", {})
    calibrated = np.asarray(probs, dtype=float).copy()
    for idx, side in enumerate(ONE_X_TWO):
        payload = calibrators.get(side)
        if payload:
            calibrated[:, idx] = _apply_isotonic(calibrated[:, idx], payload)
    calibrated = np.clip(calibrated, 1e-6, 1.0)
    calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
    return calibrated


def fit_ensemble_calibration(
    validation: pd.DataFrame,
    weight_params: EnsembleWeightParams,
    output_path: str = DEFAULT_CALIBRATION_CONFIG,
) -> tuple[dict, dict]:
    """Fit one-vs-rest isotonic calibration for final ensemble 1X2 probabilities."""
    labels = validation["target_result"].to_numpy()
    encoded = np.asarray([ONE_X_TWO_INDEX[label] for label in labels], dtype=int)
    raw_probs = _blend_validation_probs(validation, weight_params)

    calibrators = {}
    calibrated_columns = []
    for idx, side in enumerate(ONE_X_TWO):
        y_binary = (encoded == idx).astype(int)
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(raw_probs[:, idx], y_binary)
        calibrators[side] = _isotonic_payload(model)
        calibrated_columns.append(model.predict(raw_probs[:, idx]))

    calibrated_probs = np.column_stack(calibrated_columns)
    calibrated_probs = np.clip(calibrated_probs, 1e-6, 1.0)
    calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)

    raw_metrics = _one_x_two_metrics(labels, raw_probs)
    calibrated_metrics = _one_x_two_metrics(labels, calibrated_probs)
    payload = {
        "method": "one_vs_rest_isotonic",
        "target": "ensemble_1x2",
        "classes": list(ONE_X_TWO),
        "calibrators": calibrators,
        "metrics": {
            "n_calibration": int(len(validation)),
            "raw": raw_metrics,
            "calibrated": calibrated_metrics,
            "log_loss_delta": round(calibrated_metrics["log_loss"] - raw_metrics["log_loss"], 6),
            "brier_like_note": "Use calibration_report.py for one-vs-rest bin curves.",
        },
        "weight_params_used": asdict(weight_params),
        "caution": (
            "Metrics are measured on the same validation window used to fit the "
            "post-hoc calibrator. Treat them as calibration diagnostics, not a "
            "fully independent betting backtest."
        ),
    }
    save_calibration_config(payload, output_path)
    return payload, {"raw": raw_metrics, "calibrated": calibrated_metrics}


def _build_validation_predictions(
    features_path: str,
    *,
    train_cutoff: int = 2018,
    val_cutoff: int = 2023,
    use_dixon_coles: bool = True,
) -> pd.DataFrame:
    """Train validation-only XGB/Poisson models and return common 1X2 probabilities."""
    from src.models.poisson_model import PoissonGoalModel, _add_goal_targets
    from src.models.xgb_baseline import MARKETS, MarketModel

    df = pd.read_csv(features_path, parse_dates=["date"])
    df = _add_goal_targets(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    train = df[df["date"].dt.year < train_cutoff].copy()
    val = df[
        (df["date"].dt.year >= train_cutoff)
        & (df["date"].dt.year < val_cutoff)
        & df["target_result"].notna()
        & df["target_home_goals"].notna()
        & df["target_away_goals"].notna()
    ].copy()

    if train.empty or val.empty:
        raise ValueError("No hay datos suficientes para optimizar pesos del ensemble.")

    xgb_model = MarketModel(MARKETS["result_1x2"]).fit(train)
    xgb_proba_raw = xgb_model.predict_proba(val)
    xgb_class_idx = {cls: i for i, cls in enumerate(xgb_model.label_encoder.classes_)}

    poisson_model = PoissonGoalModel().fit(train)
    if use_dixon_coles:
        poisson_model.fit_dixon_coles_rho(train)
    poisson_preds = poisson_model.predict_markets(val)

    out = val[["date", "home_team", "away_team", "target_result", "rating_diff", "elo_diff"]].copy()
    for side in ONE_X_TWO:
        out[f"xgb_prob_{side}"] = xgb_proba_raw[:, xgb_class_idx[side]]
        out[f"poisson_prob_{side}"] = poisson_preds[f"prob_{side}"].to_numpy(dtype=float)

    out["dixon_coles_rho"] = poisson_model.dixon_coles_rho
    return out.reset_index(drop=True)


def optimize_ensemble_weights(
    features_path: str = "data/processed/features_train.csv",
    output_path: str = DEFAULT_WEIGHT_CONFIG,
    *,
    train_cutoff: int = 2018,
    val_cutoff: int = 2023,
    maxiter: int = 80,
) -> tuple[EnsembleWeightParams, dict, pd.DataFrame]:
    """Optimize the XGB/Poisson weight curve against validation 1X2 log-loss."""
    validation = _build_validation_predictions(
        features_path,
        train_cutoff=train_cutoff,
        val_cutoff=val_cutoff,
        use_dixon_coles=True,
    )
    labels = validation["target_result"].to_numpy()
    encoded_labels = np.asarray([ONE_X_TWO_INDEX[label] for label in labels], dtype=int)
    xgb_probs = validation[[f"xgb_prob_{side}" for side in ONE_X_TWO]].to_numpy(dtype=float)
    poisson_probs = validation[[f"poisson_prob_{side}" for side in ONE_X_TWO]].to_numpy(dtype=float)

    def objective(values: np.ndarray) -> float:
        close_rating, strong_rating, min_weight, max_weight = map(float, values)
        if strong_rating <= close_rating + 0.02 or max_weight < min_weight:
            return 10.0 + abs(strong_rating - close_rating) + abs(max_weight - min_weight)
        params = EnsembleWeightParams(close_rating, strong_rating, min_weight, max_weight)
        return float(
            log_loss(
                encoded_labels,
                _blend_validation_probs(validation, params),
                labels=np.arange(len(ONE_X_TWO)),
            )
        )

    bounds = [
        (0.02, 0.35),  # close_rating
        (0.20, 1.00),  # strong_rating
        (0.00, 0.80),  # min_weight
        (0.05, 1.00),  # max_weight
    ]
    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=42,
        maxiter=maxiter,
        popsize=12,
        tol=1e-5,
        polish=True,
        updating="immediate",
        workers=1,
    )
    close_rating, strong_rating, min_weight, max_weight = map(float, result.x)
    if strong_rating <= close_rating + 0.02 or max_weight < min_weight:
        params = DEFAULT_WEIGHT_PARAMS
    else:
        params = EnsembleWeightParams(
            close_rating=round(close_rating, 6),
            strong_rating=round(strong_rating, 6),
            min_weight=round(min_weight, 6),
            max_weight=round(max_weight, 6),
        )

    default_probs = _blend_validation_probs(validation, DEFAULT_WEIGHT_PARAMS)
    optimized_probs = _blend_validation_probs(validation, params)
    metrics = {
        "n_validation": int(len(validation)),
        "train_cutoff": int(train_cutoff),
        "val_cutoff": int(val_cutoff),
        "dixon_coles_rho": round(float(validation["dixon_coles_rho"].iloc[0]), 6),
        "xgb_only": _one_x_two_metrics(labels, xgb_probs),
        "poisson_only": _one_x_two_metrics(labels, poisson_probs),
        "default_blend": _one_x_two_metrics(labels, default_probs),
        "optimized_blend": _one_x_two_metrics(labels, optimized_probs),
        "optimizer_success": bool(result.success),
        "optimizer_fun": round(float(result.fun), 6),
    }
    weights, strength = _weight_vector(validation, params)
    metrics["optimized_weight_summary"] = {
        "mean_poisson_weight": round(float(weights.mean()), 6),
        "min_poisson_weight": round(float(weights.min()), 6),
        "max_poisson_weight": round(float(weights.max()), 6),
        "mean_strength": round(float(np.mean(strength)), 6),
    }
    save_weight_params(params, metrics, output_path)
    return params, metrics, validation


def _read_predictions(xgb_path: str, poisson_path: str, features_path: str) -> pd.DataFrame:
    xgb = pd.read_csv(xgb_path)
    poisson = pd.read_csv(poisson_path)
    features = pd.read_csv(features_path)

    xgb = xgb[xgb.get("is_placeholder_match", pd.Series(False, index=xgb.index)) == False].copy()
    poisson = poisson[poisson.get("is_placeholder", pd.Series(False, index=poisson.index)) == False].copy()

    for df in (xgb, poisson, features):
        df["date"] = _to_date_key(df["date"])

    context_cols = [
        "date", "home_team", "away_team",
        "rating_diff", "elo_diff", "stage_ordinal",
        "corners_avg_all_sum",
    ]
    features = features[[c for c in context_cols if c in features.columns]].copy()

    merged = xgb.merge(
        poisson,
        on=["date", "home_team", "away_team"],
        how="inner",
        suffixes=("_xgb", "_poisson"),
    )
    merged = merged.merge(features, on=["date", "home_team", "away_team"], how="left")
    return merged


def build_ensemble_predictions(
    xgb_path: str = "outputs/wc2026_predictions.csv",
    poisson_path: str = "outputs/wc2026_poisson_predictions.csv",
    features_path: str = "data/processed/features_wc2026.csv",
    output_path: str = "outputs/wc2026_ensemble_predictions.csv",
    weight_config_path: str = DEFAULT_WEIGHT_CONFIG,
    calibration_config_path: str = DEFAULT_CALIBRATION_CONFIG,
    weight_params: EnsembleWeightParams | None = None,
    calibration_payload: dict | None = None,
    apply_calibration: bool = False,
) -> pd.DataFrame:
    df = _read_predictions(xgb_path, poisson_path, features_path)
    if df.empty:
        raise ValueError("No hay partidos comunes entre XGBoost, Poisson y features WC2026.")

    if weight_params is None:
        weight_params, weight_source = load_weight_params(weight_config_path)
    else:
        weight_source = "argument"

    if calibration_payload is None and apply_calibration:
        calibration_payload = load_calibration_config(calibration_config_path)

    rows = []
    for _, row in df.iterrows():
        poisson_w, strength, regime = _poisson_weight(
            row.get("rating_diff"),
            row.get("elo_diff"),
            params=weight_params,
        )
        xgb_w = round(1.0 - poisson_w, 4)

        entry = {
            "date": row["date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "rating_diff": row.get("rating_diff"),
            "elo_diff": row.get("elo_diff"),
            "match_strength_gap": strength,
            "xgb_weight": xgb_w,
            "poisson_weight": poisson_w,
            "model_regime": regime,
            "ensemble_weight_source": weight_source,
            "weight_close_rating": weight_params.close_rating,
            "weight_strong_rating": weight_params.strong_rating,
            "weight_min_poisson": weight_params.min_weight,
            "weight_max_poisson": weight_params.max_weight,
            "lambda_home": row.get("lambda_home"),
            "lambda_away": row.get("lambda_away"),
            "dixon_coles_rho": row.get("dixon_coles_rho"),
            "poisson_top_score": _top_score(row.get("top5_scores", "")),
            "xgb_pred_result": row.get("pred_result"),
            "xgb_pred_total_goals": row.get("pred_total_goals"),
            "poisson_expected_goals": _sum_numeric(row.get("expected_home_goals"), row.get("expected_away_goals")),
            "final_pred_corners": row.get("pred_corners"),
        }

        # 1X2: mezclar y renormalizar para que H+D+A = 1.
        final_1x2 = {}
        raw_sum = 0.0
        for side in ONE_X_TWO:
            xgb_prob = row.get(f"prob_{side}_xgb")
            poi_prob = row.get(f"prob_{side}_poisson")
            final = _weighted_prob(xgb_prob, poi_prob, poisson_w)
            final_1x2[side] = final
            raw_sum += final if pd.notna(final) else 0.0

            diff, conf = _confidence_from_probs(_safe_prob(xgb_prob), _safe_prob(poi_prob))
            entry[f"xgb_prob_{side}"] = round(_safe_prob(xgb_prob), 4)
            entry[f"poisson_prob_{side}"] = round(_safe_prob(poi_prob), 4)
            entry[f"model_diff_{side}"] = diff
            entry[f"confidence_{side}"] = conf

        if raw_sum > 0:
            for side in ONE_X_TWO:
                final_1x2[side] = final_1x2[side] / raw_sum

        for side in ONE_X_TWO:
            prob = final_1x2[side]
            entry[f"final_prob_{side}"] = round(prob, 4)
            entry[f"final_odds_{side}"] = _safe_odds(prob)

        entry["final_pred_result"] = max(ONE_X_TWO, key=lambda side: entry[f"final_prob_{side}"])

        # Mercados binarios derivados de goles.
        for market, prob_col in PROB_MARKETS.items():
            xgb_prob = row.get(f"{prob_col}_xgb")
            poi_prob = row.get(f"{prob_col}_poisson")
            final = _weighted_prob(xgb_prob, poi_prob, poisson_w)
            diff, conf = _confidence_from_probs(_safe_prob(xgb_prob), _safe_prob(poi_prob))

            entry[f"xgb_{prob_col}"] = round(_safe_prob(xgb_prob), 4)
            entry[f"poisson_{prob_col}"] = round(_safe_prob(poi_prob), 4)
            entry[f"final_{prob_col}"] = round(final, 4)
            entry[f"final_odds_{market}"] = _safe_odds(final)
            entry[f"model_diff_{market}"] = diff
            entry[f"confidence_{market}"] = conf

        final_goals = _weighted_numeric(
            row.get("pred_total_goals"),
            _sum_numeric(row.get("expected_home_goals"), row.get("expected_away_goals")),
            poisson_w,
        )
        entry["final_expected_goals"] = round(final_goals, 3)

        rows.append(entry)

    result = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if apply_calibration and calibration_payload:
        result = apply_ensemble_calibration(result, calibration_payload, source=calibration_config_path)
    elif apply_calibration:
        result["calibration_source"] = "none"
    else:
        result["calibration_source"] = "not_applied"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Ensemble guardado en {output_path} ({len(result)} partidos)")
    _print_summary(result, weight_params, weight_source)
    return result


def apply_ensemble_calibration(
    df: pd.DataFrame,
    calibration_payload: dict,
    *,
    source: str = DEFAULT_CALIBRATION_CONFIG,
) -> pd.DataFrame:
    """Apply saved 1X2 calibration to final probabilities and fair odds."""
    out = df.copy()
    prob_cols = [f"final_prob_{side}" for side in ONE_X_TWO]
    raw_probs = out[prob_cols].to_numpy(dtype=float)
    calibrated = _apply_calibration_matrix(raw_probs, calibration_payload)

    for idx, side in enumerate(ONE_X_TWO):
        raw_col = f"raw_final_prob_{side}"
        if raw_col not in out.columns:
            out[raw_col] = out[f"final_prob_{side}"]
        out[f"final_prob_{side}"] = np.round(calibrated[:, idx], 4)
        out[f"final_odds_{side}"] = [_safe_odds(prob) for prob in calibrated[:, idx]]

    out["final_pred_result"] = [
        ONE_X_TWO[int(np.argmax(row))]
        for row in calibrated
    ]
    out["calibration_source"] = source
    out["calibration_method"] = calibration_payload.get("method", "unknown")
    return out


def _sum_numeric(left, right) -> float:
    if pd.isna(left) or pd.isna(right):
        return np.nan
    return float(left) + float(right)


def _top_score(top5_json: str) -> str:
    try:
        import json
        scores = json.loads(top5_json)
        if scores:
            return scores[0]["score"]
    except Exception:
        pass
    return "N/A"


def _print_summary(
    df: pd.DataFrame,
    weight_params: EnsembleWeightParams | None = None,
    weight_source: str = "default",
) -> None:
    print("\nResumen ensemble WC2026")
    if weight_params is not None:
        print(
            "Pesos Poisson:",
            f"source={weight_source}",
            f"close={weight_params.close_rating:.4f}",
            f"strong={weight_params.strong_rating:.4f}",
            f"min={weight_params.min_weight:.4f}",
            f"max={weight_params.max_weight:.4f}",
        )
    print(df["model_regime"].value_counts().to_string())
    print("\nMedias finales:")
    for col in ["final_prob_H", "final_prob_D", "final_prob_A", "final_prob_over25", "final_prob_btts"]:
        if col in df.columns:
            print(f"  {col:<18} {df[col].mean():.4f}")

    print("\nPartidos con mas peso Poisson:")
    cols = [
        "home_team", "away_team", "poisson_weight",
        "final_prob_H", "final_prob_D", "final_prob_A",
        "final_pred_result", "poisson_top_score",
    ]
    print(df.sort_values("poisson_weight", ascending=False).head(8)[cols].to_string(index=False))

    print("\nPartidos mas XGBoost:")
    print(df.sort_values("poisson_weight").head(8)[cols].to_string(index=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combina XGBoost y Poisson en cuotas finales WC2026.")
    parser.add_argument("--xgb", default="outputs/wc2026_predictions.csv")
    parser.add_argument("--poisson", default="outputs/wc2026_poisson_predictions.csv")
    parser.add_argument("--features", default="data/processed/features_wc2026.csv")
    parser.add_argument("--output", default="outputs/wc2026_ensemble_predictions.csv")
    parser.add_argument("--weight-config", default=DEFAULT_WEIGHT_CONFIG)
    parser.add_argument("--calibration-config", default=DEFAULT_CALIBRATION_CONFIG)
    parser.add_argument("--optimize-weights", action="store_true")
    parser.add_argument("--fit-calibration", action="store_true")
    parser.add_argument(
        "--apply-calibration",
        action="store_true",
        help="Apply the saved/fitted 1X2 calibration to final probabilities. Off by default for staking.",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Deprecated compatibility flag; calibration is already off unless --apply-calibration is passed.",
    )
    parser.add_argument("--optimization-features", default="data/processed/features_train.csv")
    parser.add_argument("--train-cutoff", type=int, default=2018)
    parser.add_argument("--val-cutoff", type=int, default=2023)
    parser.add_argument("--optimizer-maxiter", type=int, default=80)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    optimized_params = None
    weight_params_for_build = None
    validation_predictions = None
    if args.optimize_weights:
        for path in [args.optimization_features]:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
        optimized_params, metrics, validation_predictions = optimize_ensemble_weights(
            args.optimization_features,
            args.weight_config,
            train_cutoff=args.train_cutoff,
            val_cutoff=args.val_cutoff,
            maxiter=args.optimizer_maxiter,
        )
        print(f"Pesos optimizados guardados en {args.weight_config}")
        print(json.dumps({"params": asdict(optimized_params), "metrics": metrics}, indent=2))
        weight_params_for_build = optimized_params

    calibration_weight_params = optimized_params
    if calibration_weight_params is None:
        calibration_weight_params, _ = load_weight_params(args.weight_config)

    calibration_payload = None
    if args.fit_calibration:
        if validation_predictions is None:
            validation_predictions = _build_validation_predictions(
                args.optimization_features,
                train_cutoff=args.train_cutoff,
                val_cutoff=args.val_cutoff,
                use_dixon_coles=True,
            )
        calibration_payload, calibration_metrics = fit_ensemble_calibration(
            validation_predictions,
            calibration_weight_params,
            args.calibration_config,
        )
        print(f"Calibración ensemble guardada en {args.calibration_config}")
        print(json.dumps(calibration_metrics, indent=2))

    for path in [args.xgb, args.poisson, args.features]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    build_ensemble_predictions(
        args.xgb,
        args.poisson,
        args.features,
        args.output,
        weight_config_path=args.weight_config,
        calibration_config_path=args.calibration_config,
        weight_params=weight_params_for_build,
        calibration_payload=calibration_payload,
        apply_calibration=args.apply_calibration and not args.no_calibration,
    )


if __name__ == "__main__":
    main()
