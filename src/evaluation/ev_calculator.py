"""
src/evaluation/ev_calculator.py
================================
Calcula el Valor Esperado (EV) de cada apuesta cruzando las probabilidades del
modelo (ensemble) con las cuotas REALES de The Odds API.

    EV = p_modelo * cuota - 1

Un EV > 0 significa que, segun el modelo, la cuota paga mas de lo que deberia
(value bet). Ojo: esto vale lo que valga el modelo; un EV alto en un partido de
baja fiabilidad de inputs es sospechoso, no una orden de compra.

Mercados soportados (se mapean a las columnas del ensemble):
    h2h    -> final_prob_H / final_prob_D / final_prob_A   (seleccion = equipo / "Draw")
    totals -> final_prob_over25 / 1-final_prob_over25      (linea 2.5, "Over"/"Under")
    btts   -> final_prob_btts / 1-final_prob_btts          ("Yes"/"No")

Uso:
    python -m src.evaluation.ev_calculator
    python -m src.evaluation.ev_calculator --min-ev 0.05 --markets h2h,totals
    python -m src.evaluation.ev_calculator --markets h2h,totals,btts   (si tu plan lo da)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from src.data.team_names import canonicalize
from src.data.historical_odds_collector import HistoricalOddsCollector


# Etiquetas legibles por (market, lo que sea)
def _resolve_model_prob(market, selection, line, ens_home_canon, ens_away_canon, sel_canon, row):
    """Devuelve (etiqueta_mercado, prob_modelo) o None si no se puede mapear."""
    m = str(market).strip().lower()
    s = str(selection).strip().lower()

    if m == "h2h":
        if sel_canon and sel_canon == ens_home_canon:
            return ("1X2 Local (H)", row.get("final_prob_H"))
        if sel_canon and sel_canon == ens_away_canon:
            return ("1X2 Visitante (A)", row.get("final_prob_A"))
        if s in ("draw", "empate", "tie"):
            return ("1X2 Empate (D)", row.get("final_prob_D"))
        return None

    if m == "totals":
        if line is None or pd.isna(line):
            return None
        try:
            if abs(float(line) - 2.5) > 1e-6:
                return None
        except (TypeError, ValueError):
            return None
        p_over = row.get("final_prob_over25")
        if p_over is None or pd.isna(p_over):
            return None
        if s == "over":
            return ("Over 2.5", float(p_over))
        if s == "under":
            return ("Under 2.5", 1.0 - float(p_over))
        return None

    if m == "btts":
        p_btts = row.get("final_prob_btts")
        if p_btts is None or pd.isna(p_btts):
            return None
        if s in ("yes", "si", "sí"):
            return ("BTTS Si", float(p_btts))
        if s in ("no",):
            return ("BTTS No", 1.0 - float(p_btts))
        return None

    return None


def _build_ensemble_lookup(ensemble_path: str) -> dict:
    """{frozenset({home_canon, away_canon}): row_dict}. Robusto al orden de equipos."""
    df = pd.read_csv(ensemble_path)
    lookup = {}
    for _, r in df.iterrows():
        h = canonicalize(r["home_team"])
        a = canonicalize(r["away_team"])
        lookup[frozenset((h, a))] = r.to_dict()
    return lookup


def _build_reliability_lookup(reliability_path: str) -> dict:
    """{frozenset({home_canon, away_canon}): (fiabilidad_pct, fiabilidad_nivel)}."""
    if not os.path.exists(reliability_path):
        return {}
    df = pd.read_csv(reliability_path)
    if not {"home_team", "away_team"}.issubset(df.columns):
        return {}
    lookup = {}
    for _, r in df.iterrows():
        h = canonicalize(r["home_team"])
        a = canonicalize(r["away_team"])
        lookup[frozenset((h, a))] = (r.get("fiabilidad_pct"), r.get("fiabilidad_nivel"))
    return lookup


def run(ensemble_path, reliability_path, min_ev, markets, bookmakers, regions, sport, output_path, env_path, data_dir):
    if not os.path.exists(ensemble_path):
        raise FileNotFoundError(
            f"{ensemble_path} no existe. Ejecuta antes: python -m src.evaluation.model_ensemble"
        )

    ens = _build_ensemble_lookup(ensemble_path)
    fiab = _build_reliability_lookup(reliability_path)
    if not fiab:
        print(f"⚠ Sin fiabilidad ({reliability_path}) — se omite esa columna. "
              f"Ejecuta antes: python -m src.evaluation.match_reliability")

    collector = HistoricalOddsCollector(data_dir=data_dir, env_path=env_path)
    print(f"Pidiendo cuotas reales a The Odds API (markets={markets}, regions={regions})...")
    odds = collector.get_current_odds_flat(
        sport=sport,
        markets=markets,
        bookmakers=bookmakers,
        regions=regions,
    )

    if odds.empty:
        print("⚠ The Odds API no devolvio cuotas. Posibles causas: aun no hay mercado abierto, "
              "mercado no soportado para este sport, o region sin libros. Revisa --markets/--regions.")
        return

    rows = []
    no_match = set()
    for _, o in odds.iterrows():
        h = canonicalize(o.get("home_team"))
        a = canonicalize(o.get("away_team"))
        key = frozenset((h, a))
        row = ens.get(key)
        if row is None:
            no_match.add(f"{o.get('home_team')} vs {o.get('away_team')}")
            continue

        ens_home = canonicalize(row["home_team"])
        ens_away = canonicalize(row["away_team"])
        sel_canon = canonicalize(o.get("selection")) if o.get("selection") else None

        resolved = _resolve_model_prob(
            o.get("market"), o.get("selection"), o.get("line"),
            ens_home, ens_away, sel_canon, row,
        )
        if resolved is None:
            continue
        market_label, p_model = resolved
        if p_model is None or pd.isna(p_model) or p_model <= 0:
            continue

        cuota = o.get("odds_decimal")
        if cuota is None or pd.isna(cuota) or float(cuota) <= 1.0:
            continue
        cuota = float(cuota)

        ev = p_model * cuota - 1.0
        fiab_pct, fiab_nivel = fiab.get(key, (np.nan, "N/A"))
        rows.append({
            "date": row.get("date"),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "mercado": market_label,
            "seleccion": o.get("selection"),
            "bookmaker": o.get("bookmaker"),
            "cuota": round(cuota, 3),
            "prob_modelo": round(float(p_model), 4),
            "cuota_justa": round(1.0 / p_model, 3),
            "ev_pct": round(ev * 100, 2),
            "fiabilidad_pct": fiab_pct,
            "fiabilidad_nivel": fiab_nivel,
        })

    if not rows:
        print("No se pudo cruzar ninguna cuota con el modelo. ¿Coinciden los nombres de equipo?")
        if no_match:
            print("Partidos en cuotas sin match en ensemble (muestra):")
            for p in list(no_match)[:10]:
                print(f"  - {p}")
        return

    out = pd.DataFrame(rows).sort_values("ev_pct", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)

    value = out[out["ev_pct"] >= min_ev * 100]

    print("\n" + "=" * 104)
    print(f"VALUE BETS  (EV >= {min_ev*100:.1f}%)   [EV = prob_modelo x cuota - 1]")
    print(f"{'Partido':<34}{'Mercado':<18}{'Sel':<10}{'Book':<12}{'Cuota':>7}{'pMod':>7}{'EV%':>8}  {'Fiabilidad':<10}")
    print("-" * 104)
    if value.empty:
        print("  (ninguna apuesta supera el umbral)")
    else:
        for _, r in value.iterrows():
            partido = f"{r['home_team']} vs {r['away_team']}"[:32]
            sel = str(r["seleccion"])[:9]
            book = str(r["bookmaker"])[:11]
            print(f"  {partido:<32}{r['mercado']:<18}{sel:<10}{book:<12}"
                  f"{r['cuota']:>7.2f}{r['prob_modelo']:>7.2f}{r['ev_pct']:>8.2f}  {str(r['fiabilidad_nivel']):<10}")
    print("=" * 104)
    print(f"Guardado TODO (con y sin value) en {output_path}   [{len(out)} filas, {len(value)} con value]")
    if no_match:
        print(f"\n⚠ {len(no_match)} partidos de las cuotas no casaron con el ensemble "
              f"(nombres distintos). Muestra:")
        for p in list(no_match)[:8]:
            print(f"  - {p}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calcula EV cruzando ensemble con cuotas reales (The Odds API).")
    parser.add_argument("--ensemble", default="outputs/wc2026_ensemble_predictions.csv")
    parser.add_argument("--reliability", default="outputs/wc2026_reliability.csv")
    parser.add_argument("--min-ev", type=float, default=0.08, help="Umbral de EV para mostrar (0.08 = 8%%).")
    parser.add_argument("--markets", default="h2h,totals", help="Mercados: h2h,totals[,btts].")
    parser.add_argument("--bookmakers", default="winamax_fr", help="Bookmakers (coma). Vacio = todos los de la region.")
    parser.add_argument("--regions", default="eu", help="Regiones (coma). 'eu' = libros europeos, ahorra creditos.")
    parser.add_argument("--sport", default="soccer_fifa_world_cup", help="Sport key de The Odds API.")
    parser.add_argument("--output", default="outputs/wc2026_ev.csv")
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--data-dir", default="data")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    bookmakers = args.bookmakers if args.bookmakers else ()
    run(
        ensemble_path=args.ensemble,
        reliability_path=args.reliability,
        min_ev=args.min_ev,
        markets=args.markets,
        bookmakers=bookmakers,
        regions=args.regions,
        sport=args.sport,
        output_path=args.output,
        env_path=args.env_path,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
    