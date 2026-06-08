"""
src/evaluation/match_reliability.py
====================================
Fiabilidad de los INPUTS por partido del WC2026.

⚠ IMPORTANTE: fiabilidad_pct mide la CALIDAD DE LOS DATOS que alimentan la
predicción, NO la probabilidad de acertar el partido. Un 82% significa
"los ingredientes de esta predicción son buenos", no "82% de acierto".
Úsalo como filtro de dónde arriesgar, no como probabilidad de ganar.

Combina tres señales (todas de archivos ya en disco, sin API):
  - coverage   : jugadores casados en Transfermarkt de ambas selecciones
                 (n_players_matched / n_players_total). Resuelve lo de Saudi 1/26.
  - caps       : experiencia internacional media del plantel (avg_*caps*),
                 normalizada contra el percentil alto del propio set. Plantel
                 veterano = más predecible que uno de debutantes.
  - agreement  : se HEREDA de model_comparison.csv (ALTA=1, MEDIA=0.5, BAJA=0),
                 para que esta herramienta NO contradiga al comparador de modelos.

Cada señal va a [0,1]; si falta su fuente, se descarta y se renormalizan los pesos.

Uso:
    python -m src.evaluation.match_reliability
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from src.data.team_names import canonicalize


# Pesos de cada señal (editables). Se renormalizan si alguna señal falta.
WEIGHTS = {
    "coverage":  0.40,
    "caps":      0.25,
    "agreement": 0.35,
}

# Constantes de las señales
CAPS_ANCHOR_PCTL = 90   # percentil de caps medios que se considera "plantel veterano" (=1.0)
AGREEMENT_MAP = {"ALTA": 1.0, "MEDIA": 0.5, "BAJA": 0.0}
_AGREEMENT_LEVELS = set(AGREEMENT_MAP)


def band(pct: float) -> str:
    if pct >= 80: return "MUY ALTA"
    if pct >= 65: return "ALTA"
    if pct >= 50: return "MEDIA"
    if pct >= 35: return "BAJA"
    return "MUY BAJA"


def _coverage_lookup(stats_path: str) -> dict:
    if not os.path.exists(stats_path):
        return {}
    df = pd.read_csv(stats_path)
    if not {"team_canonical", "n_players_matched", "n_players_total"}.issubset(df.columns):
        return {}
    df["team_canonical"] = df["team_canonical"].map(canonicalize)
    out = {}
    for _, r in df.iterrows():
        total = r["n_players_total"] or 0
        out[r["team_canonical"]] = (r["n_players_matched"] / total) if total else np.nan
    return out


def _caps_lookup(stats_path: str) -> dict:
    """Devuelve {team_canonical: caps_norm in [0,1]} usando la columna *caps* que exista."""
    if not os.path.exists(stats_path):
        return {}
    df = pd.read_csv(stats_path)
    if "team_canonical" not in df.columns:
        return {}
    caps_cols = [c for c in df.columns if "caps" in c.lower()]
    if not caps_cols:
        return {}
    col = caps_cols[0]
    print(f"  caps: usando columna '{col}'")
    df = df[["team_canonical", col]].copy()
    df["team_canonical"] = df["team_canonical"].map(canonicalize)
    df[col] = pd.to_numeric(df[col], errors="coerce")
    valid = df[col].dropna()
    if valid.empty:
        return {}
    anchor = float(np.percentile(valid, CAPS_ANCHOR_PCTL))
    if anchor <= 0:
        return {}
    out = {}
    for _, r in df.iterrows():
        v = r[col]
        out[r["team_canonical"]] = float(np.clip(v / anchor, 0.0, 1.0)) if pd.notna(v) else np.nan
    return out


def _agreement_lookup(comparison_path: str) -> dict:
    """Devuelve {(home_canon, away_canon): score} desde model_comparison.csv.

    Detecta automáticamente la columna de confianza (la que tenga valores
    ALTA/MEDIA/BAJA) sin depender de su nombre.
    """
    if not os.path.exists(comparison_path):
        return {}
    df = pd.read_csv(comparison_path)
    if not {"home_team", "away_team"}.issubset(df.columns):
        return {}

    # Buscar primero por nombre conocido, luego por contenido como fallback
    conf_col = None
    if "confidence" in df.columns:
        conf_col = "confidence"
    else:
        for c in df.columns:
            if df[c].dtype == object:
                vals = set(df[c].dropna().astype(str).str.strip().str.upper().unique())
                if vals and vals.issubset(_AGREEMENT_LEVELS):
                    conf_col = c
                    break
    if conf_col is None:
        return {}
    print(f"  agreement: usando columna '{conf_col}' de model_comparison")

    out = {}
    for _, r in df.iterrows():
        h = canonicalize(r["home_team"])
        a = canonicalize(r["away_team"])
        score = AGREEMENT_MAP.get(str(r[conf_col]).strip().upper())
        if score is not None:
            out[(h, a)] = score
    return out


def _signal_coverage(home, away, cov):
    if not cov:
        return np.nan
    h = cov.get(canonicalize(home), np.nan)
    a = cov.get(canonicalize(away), np.nan)
    vals = [v for v in (h, a) if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan


def _signal_caps(home, away, caps):
    if not caps:
        return np.nan
    h = caps.get(canonicalize(home), np.nan)
    a = caps.get(canonicalize(away), np.nan)
    vals = [v for v in (h, a) if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan


def _signal_agreement(home, away, agree):
    if not agree:
        return np.nan
    h, a = canonicalize(home), canonicalize(away)
    if (h, a) in agree:
        return agree[(h, a)]
    if (a, h) in agree:   # por si el orden difiere
        return agree[(a, h)]
    return np.nan


def run(ensemble_path, stats_path, comparison_path, output_path):
    if not os.path.exists(ensemble_path):
        raise FileNotFoundError(
            f"{ensemble_path} no existe. Ejecuta antes: python -m src.evaluation.model_ensemble"
        )

    df = pd.read_csv(ensemble_path)
    cov = _coverage_lookup(stats_path)
    caps = _caps_lookup(stats_path)
    agree = _agreement_lookup(comparison_path)

    if not cov:
        print(f"⚠ Sin cobertura de jugadores ({stats_path}) — se ignora esa señal.")
    if not caps:
        print(f"⚠ Sin columna de caps en {stats_path} — se ignora esa señal.")
    if not agree:
        print(f"⚠ Sin acuerdo de modelos ({comparison_path}) — se ignora esa señal.")

    rows = []
    for _, r in df.iterrows():
        signals = {
            "coverage":  _signal_coverage(r["home_team"], r["away_team"], cov),
            "caps":      _signal_caps(r["home_team"], r["away_team"], caps),
            "agreement": _signal_agreement(r["home_team"], r["away_team"], agree),
        }
        # Renormalizar pesos sobre las señales disponibles
        avail = {k: v for k, v in signals.items() if pd.notna(v)}
        if not avail:
            pct = np.nan
        else:
            wsum = sum(WEIGHTS[k] for k in avail)
            pct = 100.0 * sum(WEIGHTS[k] * v for k, v in avail.items()) / wsum

        rows.append({
            "date": r.get("date"),
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "fiabilidad_pct": round(pct, 1) if pd.notna(pct) else np.nan,
            "fiabilidad_nivel": band(pct) if pd.notna(pct) else "N/A",
            "sig_coverage":  round(signals["coverage"], 3)  if pd.notna(signals["coverage"])  else np.nan,
            "sig_caps":      round(signals["caps"], 3)      if pd.notna(signals["caps"])      else np.nan,
            "sig_agreement": round(signals["agreement"], 3) if pd.notna(signals["agreement"]) else np.nan,
        })

    out = pd.DataFrame(rows).sort_values("fiabilidad_pct", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)

    print("\n" + "=" * 78)
    print("FIABILIDAD DE INPUTS POR PARTIDO  (NO es probabilidad de acierto)")
    print(f"{'Partido':<40}{'Fiab%':>8}  {'Nivel':<10}")
    print("-" * 78)
    for _, r in out.iterrows():
        partido = f"{r['home_team']} vs {r['away_team']}"[:38]
        pct = f"{r['fiabilidad_pct']:.1f}" if pd.notna(r["fiabilidad_pct"]) else " N/A"
        print(f"  {partido:<38}{pct:>8}  {r['fiabilidad_nivel']:<10}")
    print("=" * 78)
    print(f"Guardado en {output_path}")
    if pd.notna(out["fiabilidad_pct"]).any():
        print("\nReparto por nivel:")
        print(out["fiabilidad_nivel"].value_counts().to_string())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fiabilidad de inputs por partido WC2026.")
    parser.add_argument("--ensemble", default="outputs/wc2026_ensemble_predictions.csv")
    parser.add_argument("--stats", default="data/processed/team_player_stats_wc2026.csv")
    parser.add_argument("--comparison", default="outputs/model_comparison.csv")
    parser.add_argument("--output", default="outputs/wc2026_reliability.csv")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args.ensemble, args.stats, args.comparison, args.output)


if __name__ == "__main__":
    main()
    