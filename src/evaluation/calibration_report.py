"""
src/evaluation/calibration_report.py
=====================================
Calibración del modelo sobre el histórico (sin API, sin cuotas).

Responde a "cuando el modelo dice 70%, ¿pasa el 70% de las veces?".
Reentrena los modelos de xgb_baseline en train (< train_cutoff), predice en
validación (train_cutoff..val_cutoff) y compara las probabilidades contra el
resultado real. Saca, por mercado:
  - Brier score (menor = mejor)
  - ECE (Expected Calibration Error; menor = mejor calibrado)
  - tabla por bins de probabilidad (confianza media vs acierto real)

Mercados evaluados: 1X2 desglosado en H / D / A (one-vs-rest), Over 2.5 y BTTS.

NO usa los partidos del Mundial (no se pueden calibrar partidos sin jugar).
Mide cómo de fiable ha sido el modelo en partidos históricos parecidos.

Uso:
    python -m src.evaluation.calibration_report
    python -m src.evaluation.calibration_report --features data/processed/features_train.csv
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.models.xgb_baseline import MARKETS, MarketModel


N_BINS = 10


# ---------------------------------------------------------------------------
# Calibración binaria
# ---------------------------------------------------------------------------

def _bin_stats(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS):
    """Devuelve (brier, ece, filas_por_bin)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(p) & ~np.isnan(y)
    p, y = p[mask], y[mask]

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    n = len(p)

    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        cnt = int(m.sum())
        if cnt == 0:
            rows.append({"bin_lo": round(lo, 2), "bin_hi": round(hi, 2),
                         "n": 0, "conf": np.nan, "acc": np.nan, "gap": np.nan})
            continue
        conf = float(p[m].mean())
        acc = float(y[m].mean())
        gap = abs(conf - acc)
        ece += cnt / n * gap
        rows.append({"bin_lo": round(lo, 2), "bin_hi": round(hi, 2),
                     "n": cnt, "conf": round(conf, 4),
                     "acc": round(acc, 4), "gap": round(gap, 4)})

    brier = float(np.mean((p - y) ** 2)) if n else float("nan")
    return brier, float(ece), rows


# ---------------------------------------------------------------------------
# Entreno + predicción en validación para un mercado binario
# ---------------------------------------------------------------------------

def _fit_predict_binary(df_train, df_val, market_key):
    """Devuelve (proba_positiva, y_true) en validación para un mercado binario."""
    model = MarketModel(MARKETS[market_key])
    model.fit(df_train)
    target = MARKETS[market_key].target_col
    val = df_val[df_val[target].notna()].copy()
    if val.empty:
        return np.array([]), np.array([])
    proba = model.predict_proba(val)
    p = proba[:, 1] if proba.ndim == 2 else proba
    y = val[target].astype(float).values
    return p, y


def _fit_predict_1x2(df_train, df_val):
    """Devuelve dict {clase: (proba, y_bin)} para H/D/A en validación."""
    model = MarketModel(MARKETS["result_1x2"])
    model.fit(df_train)
    val = df_val[df_val["target_result"].notna()].copy()
    if val.empty:
        return {}
    proba = model.predict_proba(val)  # columnas en orden A, D, H
    classes = list(model.label_encoder.classes_)
    actual = val["target_result"].values
    out = {}
    for i, cls in enumerate(classes):
        out[cls] = (proba[:, i], (actual == cls).astype(float))
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(features_path: str, output_dir: str, train_cutoff: int, val_cutoff: int):
    if not os.path.exists(features_path):
        raise FileNotFoundError(features_path)

    df = pd.read_csv(features_path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    df_train = df[df["date"].dt.year < train_cutoff].copy()
    df_val = df[(df["date"].dt.year >= train_cutoff) &
                (df["date"].dt.year < val_cutoff)].copy()

    print("=" * 72)
    print("CALIBRACIÓN DEL MODELO (histórico)")
    print(f"Train: {len(df_train)} (< {train_cutoff})   "
          f"Val: {len(df_val)} ({train_cutoff}-{val_cutoff - 1})")
    print("=" * 72)

    curves = {}  # nombre -> (proba, y)

    # 1X2 desglosado
    for cls, (p, y) in _fit_predict_1x2(df_train, df_val).items():
        label = {"H": "1X2 Local (H)", "D": "1X2 Empate (D)", "A": "1X2 Visitante (A)"}[cls]
        curves[label] = (p, y)

    # Binarios
    curves["Over 2.5"] = _fit_predict_binary(df_train, df_val, "over25")
    curves["BTTS"] = _fit_predict_binary(df_train, df_val, "btts")

    summary = []
    bin_rows = []

    for name, (p, y) in curves.items():
        if len(p) == 0:
            continue
        brier, ece, rows = _bin_stats(p, y)
        summary.append({"market": name, "n": int(len(p)),
                        "brier": round(brier, 4), "ece": round(ece, 4)})

        print(f"\n── {name} ──   n={len(p)}   Brier={brier:.4f}   ECE={ece:.4f}")
        print(f"  {'bin':<13}{'n':>6}{'conf':>9}{'acierto':>10}{'gap':>9}")
        for r in rows:
            if r["n"] == 0:
                continue
            print(f"  [{r['bin_lo']:.1f}-{r['bin_hi']:.1f}]"
                  f"{r['n']:>7}{r['conf']:>9.3f}{r['acc']:>10.3f}{r['gap']:>9.3f}")
            bin_rows.append({"market": name, **r})

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(bin_rows).to_csv(os.path.join(output_dir, "calibration_bins.csv"), index=False)
    with open(os.path.join(output_dir, "calibration_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("RESUMEN  (gap = |confianza - acierto real|; cuanto menor, más fiable ese rango)")
    print(f"{'Mercado':<20}{'n':>7}{'Brier':>9}{'ECE':>9}")
    for s in summary:
        print(f"{s['market']:<20}{s['n']:>7}{s['brier']:>9.4f}{s['ece']:>9.4f}")
    print("=" * 72)
    print(f"Guardado: {output_dir}/calibration_bins.csv y calibration_metrics.json")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibración del modelo sobre el histórico.")
    parser.add_argument("--features", default="data/processed/features_train.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--train-cutoff", type=int, default=2018)
    parser.add_argument("--val-cutoff", type=int, default=2023)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args.features, args.output_dir, args.train_cutoff, args.val_cutoff)


if __name__ == "__main__":
    main()
    