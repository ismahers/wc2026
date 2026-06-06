"""
src/evaluation/odds_converter.py
==================================
Convierte probabilidades del modelo a cuotas decimales limpias (sin margen).

Lee cualquier CSV de predicciones con columnas prob_* y genera un CSV
con las cuotas correspondientes cuota_* = 1 / prob_*.

Uso:
    python -m src.evaluation.odds_converter
    python -m src.evaluation.odds_converter --input outputs/wc2026_poisson_predictions.csv
    python -m src.evaluation.odds_converter --input outputs/wc2026_predictions.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


PROB_TO_ODDS = {
    "prob_H":      "cuota_H",
    "prob_D":      "cuota_D",
    "prob_A":      "cuota_A",
    "prob_over25": "cuota_over25",
    "prob_btts":   "cuota_btts",
}


def convert(
    input_path: str,
    output_path: str | None = None,
    min_prob: float = 0.01,  # evitar cuotas absurdas por probabilidades casi 0
) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    converted = []
    for prob_col, odds_col in PROB_TO_ODDS.items():
        if prob_col in df.columns:
            df[odds_col] = (1 / df[prob_col].clip(lower=min_prob)).round(2)
            converted.append(odds_col)

    if not converted:
        print(f"⚠ No se encontraron columnas prob_* en {input_path}")
        return df

    if output_path is None:
        stem   = Path(input_path).stem
        parent = Path(input_path).parent
        output_path = str(parent / f"{stem}_with_odds.csv")

    df.to_csv(output_path, index=False)
    print(f"✓ Cuotas añadidas: {converted}")
    print(f"✓ Guardado en {output_path}")

    # Mostrar tabla resumen
    display_cols = (
        ["home_team", "away_team"] +
        [c for c in ["cuota_H", "cuota_D", "cuota_A", "cuota_over25", "cuota_btts"] if c in df.columns]
    )
    display_cols = [c for c in display_cols if c in df.columns]
    mask = df.get("is_placeholder", pd.Series(False, index=df.index)) == False
    print("\n" + "=" * 80)
    print("CUOTAS DEL MODELO (sin margen)")
    print(f"{'Partido':<40} {'C(H)':>6} {'C(D)':>6} {'C(A)':>6} {'O2.5':>6} {'BTTS':>6}")
    print("-" * 80)
    for _, row in df[mask].iterrows():
        if pd.isna(row.get("cuota_H")):
            continue
        partido = f"{row['home_team']} vs {row['away_team']}"[:38]
        ch   = f"{row['cuota_H']:.2f}"      if "cuota_H"      in row and pd.notna(row["cuota_H"])      else "  N/A"
        cd   = f"{row['cuota_D']:.2f}"      if "cuota_D"      in row and pd.notna(row["cuota_D"])      else "  N/A"
        ca   = f"{row['cuota_A']:.2f}"      if "cuota_A"      in row and pd.notna(row["cuota_A"])      else "  N/A"
        co   = f"{row['cuota_over25']:.2f}" if "cuota_over25" in row and pd.notna(row["cuota_over25"]) else "  N/A"
        cb   = f"{row['cuota_btts']:.2f}"   if "cuota_btts"   in row and pd.notna(row["cuota_btts"])   else "  N/A"
        print(f"  {partido:<38} {ch:>6} {cd:>6} {ca:>6} {co:>6} {cb:>6}")
    print("=" * 80)

    return df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convierte probabilidades del modelo a cuotas decimales.")
    parser.add_argument("--input",   default="outputs/wc2026_poisson_predictions.csv",
                        help="CSV de predicciones con columnas prob_*")
    parser.add_argument("--output",  default=None,
                        help="CSV de salida. Por defecto: <input>_with_odds.csv")
    parser.add_argument("--min-prob", type=float, default=0.01,
                        help="Probabilidad mínima para evitar cuotas absurdas")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not os.path.exists(args.input):
        print(f"✗ No se encontró {args.input}")
        return
    convert(args.input, args.output, args.min_prob)


if __name__ == "__main__":
    main()
    