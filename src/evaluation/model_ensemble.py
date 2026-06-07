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
import os

import numpy as np
import pandas as pd


ONE_X_TWO = ("H", "D", "A")
PROB_MARKETS = {
    "over25": "prob_over25",
    "btts": "prob_btts",
}


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


def _poisson_weight(
    rating_diff,
    elo_diff,
    *,
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
) -> pd.DataFrame:
    df = _read_predictions(xgb_path, poisson_path, features_path)
    if df.empty:
        raise ValueError("No hay partidos comunes entre XGBoost, Poisson y features WC2026.")

    rows = []
    for _, row in df.iterrows():
        poisson_w, strength, regime = _poisson_weight(row.get("rating_diff"), row.get("elo_diff"))
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
            "lambda_home": row.get("lambda_home"),
            "lambda_away": row.get("lambda_away"),
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
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Ensemble guardado en {output_path} ({len(result)} partidos)")
    _print_summary(result)
    return result


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


def _print_summary(df: pd.DataFrame) -> None:
    print("\nResumen ensemble WC2026")
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
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    for path in [args.xgb, args.poisson, args.features]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    build_ensemble_predictions(args.xgb, args.poisson, args.features, args.output)


if __name__ == "__main__":
    main()
