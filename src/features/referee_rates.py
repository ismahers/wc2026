"""
src/features/referee_rates.py
=============================
Tasas disciplinarias de árbitro listas para modelo.

Resuelve dos problemas a la vez:

1) "Rellenar los 52 árbitros" de forma honesta.
   Solo ~12 árbitros tienen tasas medidas fiables; el resto no tiene datos
   públicos comparables. En vez de inventar números, esos huecos se rellenan
   con un PRIOR (media de su confederación, o media global si su confederación
   no tiene datos aún) mediante shrinkage empírico-Bayes. Así los 52 tienen un
   valor defendible, y el peso de cada uno depende de cuántos partidos tenga.

2) "Que cuente que son partidos internacionales" (quitar el sesgo de liga).
   La función compute_international_rates() toma datos a nivel de partido
   (árbitro × competición × tarjetas) y neutraliza el nivel de cada competición
   (efecto fijo de competición) antes de estimar la tendencia del árbitro. Así
   un árbitro de una liga muy tarjetera no parece "estricto" solo por su liga.

Shrinkage empírico-Bayes
------------------------
    rate_shrunk = (n * rate_obs + k * prior) / (n + k)

  - n     = nº de partidos del árbitro
  - rate  = su tasa observada (si falta, n se trata como 0 -> queda en el prior)
  - prior = media (de confederación o global), ponderada por partidos
  - k     = pseudo-cuenta (cuántos "partidos de prior" pesan; 10-15 razonable)

Un árbitro con 500 partidos se queda casi en su tasa; uno con 5 (o sin datos)
se va hacia el prior. Esto NO arregla el sesgo de liga (eso lo hace la
neutralización por competición), pero sí estabiliza muestras pequeñas.

Uso rápido (rellenar los 52 desde el CSV actual)
------------------------------------------------
    python -m src.features.referee_rates \
        --referees data/processed/referees_with_stats.csv \
        --output   data/processed/referees_with_stats.csv

Genera columnas yellow_per_match_model / red_per_match_model con TODOS los
árbitros rellenos (medido si lo hay, prior si no).
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_RATE_COLS = ("yellow_per_match", "red_per_match")
DEFAULT_PSEUDO_COUNT = 12.0


# ---------------------------------------------------------------------------
# Priors ponderados (global y por grupo)
# ---------------------------------------------------------------------------

def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        # Sin pesos válidos: media simple de lo que haya
        return float(values.dropna().mean()) if values.notna().any() else float("nan")
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def _build_priors(
    df: pd.DataFrame,
    rate_col: str,
    count_col: str,
    group_col: Optional[str],
) -> tuple[float, dict]:
    """Devuelve (prior_global, {grupo: prior_grupo}) ponderados por nº de partidos."""
    global_prior = _weighted_mean(df[rate_col], df[count_col])
    group_priors: dict = {}
    if group_col and group_col in df.columns:
        for g, sub in df.groupby(group_col, dropna=True):
            gp = _weighted_mean(sub[rate_col], sub[count_col])
            if not np.isnan(gp):
                group_priors[g] = gp
    return global_prior, group_priors


# ---------------------------------------------------------------------------
# Shrinkage sobre el CSV de árbitros (rellena los 52)
# ---------------------------------------------------------------------------

def shrink_referee_rates(
    df: pd.DataFrame,
    *,
    rate_cols: Sequence[str] = DEFAULT_RATE_COLS,
    count_col: str = "matches",
    group_col: Optional[str] = "confederation",
    pseudo_count: float = DEFAULT_PSEUDO_COUNT,
    out_suffix: str = "_model",
) -> pd.DataFrame:
    """
    Aplica shrinkage empírico-Bayes a cada tasa y rellena TODOS los árbitros.

    Para cada fila:
      - prior = media de su confederación si esa confederación tiene datos,
        si no la media global.
      - si la tasa observada falta -> el resultado es el prior (n efectivo = 0).

    Crea columnas <rate><out_suffix> con la tasa final para el modelo, y una
    columna 'rate_is_measured' que marca si venía de dato medido o solo prior.
    """
    out = df.copy()
    out[count_col] = pd.to_numeric(out.get(count_col), errors="coerce")

    measured_any = pd.Series(False, index=out.index)

    for rate_col in rate_cols:
        if rate_col not in out.columns:
            log.warning("Columna de tasa '%s' no encontrada; se omite", rate_col)
            continue
        out[rate_col] = pd.to_numeric(out[rate_col], errors="coerce")
        global_prior, group_priors = _build_priors(out, rate_col, count_col, group_col)

        def _prior_for(row) -> float:
            if group_col and group_col in out.columns:
                g = row.get(group_col)
                if g in group_priors:
                    return group_priors[g]
            return global_prior

        priors = out.apply(_prior_for, axis=1).astype(float)
        rate_obs = out[rate_col]
        n_eff = out[count_col].where(rate_obs.notna(), 0.0).fillna(0.0)
        rate_for_calc = rate_obs.fillna(priors)

        shrunk = (n_eff * rate_for_calc + pseudo_count * priors) / (n_eff + pseudo_count)
        out[f"{rate_col}{out_suffix}"] = shrunk.round(3)
        measured_any = measured_any | rate_obs.notna()

    out["rate_is_measured"] = measured_any.astype(int)
    n_meas = int(measured_any.sum())
    log.info("Shrinkage: %d/%d árbitros con tasa medida, %d rellenos con prior",
             n_meas, len(out), len(out) - n_meas)
    return out


# ---------------------------------------------------------------------------
# Tasas internacionales neutralizadas por competición (el fix del sesgo de liga)
# ---------------------------------------------------------------------------

def compute_international_rates(
    matches: pd.DataFrame,
    *,
    referee_col: str = "referee_name",
    competition_col: str = "competition",
    card_cols: Sequence[str] = ("yellow_total", "red_total"),
    pseudo_count: float = DEFAULT_PSEUDO_COUNT,
    keep_competitions: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Estima la tendencia disciplinaria de cada árbitro NETA del nivel de su liga.

    Entrada: un DataFrame a nivel de partido con, como mínimo:
        referee_col, competition_col, y las columnas de tarjetas (card_cols),
        siendo cada card_col el total del partido (local+visitante).

    Método (efecto fijo de competición + shrinkage):
      1. baseline_c   = media de tarjetas por partido en la competición c
      2. residuo_r    = media sobre los partidos de r de (tarjetas - baseline_c)
      3. residuo_shrunk = n_r * residuo_r / (n_r + k)     # hacia 0 si poca muestra
      4. tasa_modelo  = media_global + residuo_shrunk

    Para quedarte SOLO con internacionales, filtra antes o pasa keep_competitions
    con las competiciones internacionales (World Cup, qualifiers, Nations League,
    Copa América, Euro, Confederations Cup, etc.).

    Devuelve un DataFrame por árbitro con: referee_name, matches, y una columna
    <card>_per_match por cada card_col, ya neutralizada y suavizada.
    """
    df = matches.copy()
    if keep_competitions is not None:
        df = df[df[competition_col].isin(set(keep_competitions))].copy()

    if df.empty:
        log.warning("Sin partidos tras el filtro de competición")
        return pd.DataFrame(columns=[referee_col, "matches"])

    result = pd.DataFrame({referee_col: sorted(df[referee_col].dropna().unique())})
    counts = df.groupby(referee_col).size()
    result["matches"] = result[referee_col].map(counts).fillna(0).astype(int)

    for card in card_cols:
        if card not in df.columns:
            log.warning("Columna de tarjetas '%s' no encontrada; se omite", card)
            continue
        df[card] = pd.to_numeric(df[card], errors="coerce")
        global_mean = float(df[card].mean())
        comp_baseline = df.groupby(competition_col)[card].transform("mean")
        df[f"_resid_{card}"] = df[card] - comp_baseline

        resid_mean = df.groupby(referee_col)[f"_resid_{card}"].mean()
        n = df.groupby(referee_col)[card].count()
        resid_shrunk = (n * resid_mean) / (n + pseudo_count)

        rate_name = card.replace("_total", "") + "_per_match"
        result[rate_name] = result[referee_col].map(
            global_mean + resid_shrunk
        ).round(3)

    log.info("Tasas internacionales calculadas para %d árbitros (global yellow≈%.2f)",
             len(result), float(df[card_cols[0]].mean()) if card_cols[0] in df else float("nan"))
    return result


# ---------------------------------------------------------------------------
# CLI: rellenar los 52 desde el CSV actual
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description="Rellena y suaviza tasas de árbitro.")
    parser.add_argument("--referees", default="data/processed/referees_with_stats.csv")
    parser.add_argument("--output", default="data/processed/referees_with_stats.csv")
    parser.add_argument("--pseudo-count", type=float, default=DEFAULT_PSEUDO_COUNT)
    parser.add_argument("--no-group", action="store_true",
                        help="Usar solo prior global, ignorando la confederación.")
    args = parser.parse_args()

    df = pd.read_csv(args.referees)
    out = shrink_referee_rates(
        df,
        group_col=None if args.no_group else "confederation",
        pseudo_count=args.pseudo_count,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Guardado {len(out)} árbitros en {args.output}")
    cols = [c for c in out.columns if c.endswith("_model")]
    print("Columnas para el modelo:", cols)


if __name__ == "__main__":
    main()