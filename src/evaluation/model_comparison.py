"""
src/evaluation/model_comparison.py
=====================================
Compara las predicciones del XGBoost y el modelo Poisson para cada partido
del WC2026 y genera un indicador de confianza por mercado.

Lógica de confianza:
  - Para cada mercado (1X2, Over/Under, BTTS) calcula la diferencia absoluta
    entre las cuotas de los dos modelos
  - Si la diferencia es pequeña → los modelos coinciden → más confianza
  - Si la diferencia es grande → los modelos divergen → menos confianza

Niveles de confianza:
  ALTA    → diferencia cuotas < 0.20
  MEDIA   → diferencia cuotas entre 0.20 y 0.50
  BAJA    → diferencia cuotas > 0.50

Output:
  outputs/model_comparison.csv  → tabla completa con cuotas y confianza
  Tabla resumen en consola

Uso:
  python -m src.evaluation.model_comparison
  python -m src.evaluation.model_comparison --xgb outputs/wc2026_predictions.csv
                                            --poisson outputs/wc2026_poisson_predictions.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Umbrales de confianza (diferencia absoluta entre cuotas)
# ---------------------------------------------------------------------------

CONF_HIGH   = 0.10   # diferencia relativa < 10% → ALTA
CONF_MEDIUM = 0.25   # diferencia relativa < 25% → MEDIA, si no → BAJA

MARKETS = {
    "1X2_H":   ("prob_H",      "C(H)"),
    "1X2_D":   ("prob_D",      "C(D)"),
    "1X2_A":   ("prob_A",      "C(A)"),
    "over25":  ("prob_over25", "C(O2.5)"),
    "btts":    ("prob_btts",   "C(BTTS)"),
}


def _safe_odds(prob: float, min_prob: float = 0.01) -> float:
    if pd.isna(prob) or prob <= 0:
        return np.nan
    return round(1 / max(prob, min_prob), 2)


def _confidence_pct(xgb_odds: float, poi_odds: float) -> tuple[float, str]:
    """Calcula diferencia porcentual relativa entre dos cuotas y su nivel de confianza."""
    if pd.isna(xgb_odds) or pd.isna(poi_odds) or xgb_odds <= 0 or poi_odds <= 0:
        return np.nan, "N/A"
    avg = (xgb_odds + poi_odds) / 2
    diff_pct = abs(xgb_odds - poi_odds) / avg
    if diff_pct < CONF_HIGH:
        return diff_pct, "ALTA"
    if diff_pct < CONF_MEDIUM:
        return diff_pct, "MEDIA"
    return diff_pct, "BAJA"


def _overall_confidence(row: pd.Series) -> str:
    """Confianza global del partido: la peor confianza entre los mercados principales."""
    scores = {"ALTA": 3, "MEDIA": 2, "BAJA": 1, "N/A": 0}
    levels = [
        row.get("conf_1X2_H", "N/A"),
        row.get("conf_1X2_D", "N/A"),
        row.get("conf_1X2_A", "N/A"),
        row.get("conf_over25", "N/A"),
    ]
    min_score = min(scores.get(l, 0) for l in levels)
    return {3: "ALTA", 2: "MEDIA", 1: "BAJA", 0: "N/A"}[min_score]


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def compare_models(
    xgb_path: str,
    poisson_path: str,
    output_path: str,
) -> pd.DataFrame:

    xgb = pd.read_csv(xgb_path)
    poi = pd.read_csv(poisson_path)

    # Filtrar placeholders
    xgb = xgb[xgb.get("is_placeholder_match", pd.Series(False, index=xgb.index)) == False].copy()
    poi = poi[poi.get("is_placeholder", pd.Series(False, index=poi.index)) == False].copy()

    # Merge por fecha + equipos
    xgb["date"] = pd.to_datetime(xgb["date"]).dt.date.astype(str)
    poi["date"] = pd.to_datetime(poi["date"]).dt.date.astype(str)

    df = xgb.merge(
        poi,
        on=["date", "home_team", "away_team"],
        how="inner",
        suffixes=("_xgb", "_poi"),
    )

    if df.empty:
        print("⚠ No se encontraron partidos comunes entre XGBoost y Poisson")
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        entry = {
            "date":      row["date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            # Lambdas Poisson
            "lambda_home": row.get("lambda_home"),
            "lambda_away": row.get("lambda_away"),
            # Marcador más probable (Poisson)
            "top_score":   _top_score(row.get("top5_scores", "")),
            # Goles esperados
            "exp_home_goals": row.get("expected_home_goals"),
            "exp_away_goals": row.get("expected_away_goals"),
            # Predicción XGBoost
            "xgb_pred_result": row.get("pred_result"),
            "xgb_pred_goals":  row.get("pred_total_goals"),
            "xgb_pred_corners": row.get("pred_corners"),
            "xgb_pred_yellows": row.get("pred_yellows"),
        }

        for market, (prob_col, _) in MARKETS.items():
            xgb_prob = row.get(f"{prob_col}_xgb", row.get(prob_col + "_xgb"))
            poi_prob  = row.get(f"{prob_col}_poi", row.get(prob_col + "_poi"))

            # Fallback si no hay sufijo
            if pd.isna(xgb_prob):
                xgb_prob = row.get(prob_col)
            if pd.isna(poi_prob):
                poi_prob = row.get(prob_col)

            xgb_odds = _safe_odds(xgb_prob)
            poi_odds  = _safe_odds(poi_prob)

            avg_odds = round((xgb_odds + poi_odds) / 2, 2) if pd.notna(xgb_odds) and pd.notna(poi_odds) else np.nan
            diff_pct, conf = _confidence_pct(xgb_odds, poi_odds)

            entry[f"xgb_{market}"]  = xgb_odds
            entry[f"poi_{market}"]  = poi_odds
            entry[f"avg_{market}"]  = avg_odds
            entry[f"diff_{market}"] = round(diff_pct * 100, 1) if pd.notna(diff_pct) else np.nan  # en %
            entry[f"conf_{market}"] = conf

        entry["confidence"] = _overall_confidence(pd.Series(entry))
        rows.append(entry)

    result = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"✓ Comparación guardada en {output_path}")

    _print_summary(result)
    return result


def _top_score(top5_json: str) -> str:
    """Extrae el marcador más probable del JSON de top5."""
    try:
        import json
        scores = json.loads(top5_json)
        if scores:
            return scores[0]["score"]
    except Exception:
        pass
    return "N/A"


def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("COMPARACIÓN XGBoost vs Poisson — WC2026")
    print(f"{'Partido':<35} {'Res':>4} {'C(H)XGB':>8} {'C(H)POI':>8} {'C(D)XGB':>8} {'C(D)POI':>8} {'C(A)XGB':>8} {'C(A)POI':>8} {'O2.5XGB':>8} {'O2.5POI':>8} {'CONF':>6} {'Top Score':>10}")
    print("-" * 100)

    for _, row in df.iterrows():
        partido = f"{row['home_team']} vs {row['away_team']}"[:33]
        conf_emoji = {"ALTA": "✅", "MEDIA": "⚠", "BAJA": "❌", "N/A": "?"}.get(row.get("confidence", "N/A"), "?")

        print(
            f"  {partido:<33} "
            f"{str(row.get('xgb_pred_result', '?')):>4} "
            f"{_fmt(row.get('xgb_1X2_H')):>8} "
            f"{_fmt(row.get('poi_1X2_H')):>8} "
            f"{_fmt(row.get('xgb_1X2_D')):>8} "
            f"{_fmt(row.get('poi_1X2_D')):>8} "
            f"{_fmt(row.get('xgb_1X2_A')):>8} "
            f"{_fmt(row.get('poi_1X2_A')):>8} "
            f"{_fmt(row.get('xgb_over25')):>8} "
            f"{_fmt(row.get('poi_over25')):>8} "
            f"{conf_emoji} {row.get('confidence', 'N/A'):>5} "
            f"{str(row.get('top_score', 'N/A')):>10}"
        )

    print("=" * 100)

    # Resumen por confianza
    conf_counts = df["confidence"].value_counts()
    print(f"\nResumen de confianza:")
    print(f"  ✅ ALTA:  {conf_counts.get('ALTA',  0)} partidos")
    print(f"  ⚠  MEDIA: {conf_counts.get('MEDIA', 0)} partidos")
    print(f"  ❌ BAJA:  {conf_counts.get('BAJA',  0)} partidos")

    # Partidos con mayor divergencia
    if "diff_1X2_H" in df.columns:
        print(f"\nPartidos con mayor divergencia en 1X2:")
        top_div = df.nlargest(5, "diff_1X2_H")[["home_team", "away_team", "xgb_1X2_H", "poi_1X2_H", "diff_1X2_H"]]
        for _, r in top_div.iterrows():
            print(f"  {r['home_team']} vs {r['away_team']}: XGB={r['xgb_1X2_H']:.2f} POI={r['poi_1X2_H']:.2f} diff={r['diff_1X2_H']:.2f}")


def _fmt(val) -> str:
    if pd.isna(val) or val is None:
        return "  N/A"
    return f"{float(val):.2f}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compara XGBoost vs Poisson para WC2026.")
    parser.add_argument("--xgb",     default="outputs/wc2026_predictions.csv")
    parser.add_argument("--poisson", default="outputs/wc2026_poisson_predictions.csv")
    parser.add_argument("--output",  default="outputs/model_comparison.csv")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    for path in [args.xgb, args.poisson]:
        if not os.path.exists(path):
            print(f"✗ No se encontró {path}")
            return
    compare_models(args.xgb, args.poisson, args.output)


if __name__ == "__main__":
    main()
    