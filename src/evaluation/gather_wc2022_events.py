"""
src/evaluation/gather_wc2022_events.py
=======================================
Recolecta TODOS los event_id del Mundial 2022 consultando el endpoint historico
de eventos en varias fechas del torneo y deduplicando por event_id.

Guarda un unico CSV (por defecto data/processed/odds_events_2022.csv) que luego
usa backtest_2022.py con --events.

El endpoint de eventos es barato (no devuelve cuotas). Aun asi usa cache.

Uso:
    python -m src.evaluation.gather_wc2022_events
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.data.historical_odds_collector import HistoricalOddsCollector

log = logging.getLogger(__name__)

# Una consulta por dia de partidos del Mundial 2022 (a las 11:00 UTC, cuando los
# partidos del dia ya tienen cuotas). Dias sin partido devuelven vacio, no pasa nada.
DEFAULT_DATES = [
    # Fase de grupos (20 nov - 2 dic)
    "2022-11-20T11:00:00Z", "2022-11-21T11:00:00Z", "2022-11-22T11:00:00Z",
    "2022-11-23T11:00:00Z", "2022-11-24T11:00:00Z", "2022-11-25T11:00:00Z",
    "2022-11-26T11:00:00Z", "2022-11-27T11:00:00Z", "2022-11-28T11:00:00Z",
    "2022-11-29T11:00:00Z", "2022-11-30T11:00:00Z", "2022-12-01T11:00:00Z",
    "2022-12-02T11:00:00Z",
    # Octavos (3-6 dic)
    "2022-12-03T11:00:00Z", "2022-12-04T11:00:00Z", "2022-12-05T11:00:00Z",
    "2022-12-06T11:00:00Z",
    # Cuartos (9-10 dic)
    "2022-12-09T11:00:00Z", "2022-12-10T11:00:00Z",
    # Semis (13-14 dic)
    "2022-12-13T11:00:00Z", "2022-12-14T11:00:00Z",
    # 3er puesto y final (17-18 dic)
    "2022-12-17T11:00:00Z", "2022-12-18T11:00:00Z",
]


def run(dates, sport, output_path, env_path, data_dir, use_cache):
    collector = HistoricalOddsCollector(data_dir=data_dir, env_path=env_path)
    by_id = {}
    for d in dates:
        snap = collector.get_historical_events(d, sport=sport, use_cache=use_cache)
        for e in snap.events:
            eid = e.get("id")
            if eid and eid not in by_id:
                by_id[eid] = {
                    "event_id": eid,
                    "commence_time": e.get("commence_time"),
                    "home_team": e.get("home_team"),
                    "away_team": e.get("away_team"),
                }
        log.info("  %s -> %d eventos (acumulado unico: %d)", d, len(snap.events), len(by_id))

    df = pd.DataFrame(by_id.values())
    if df.empty:
        print("No se recolecto ningun evento. Revisa el sport key o las fechas.")
        return
    df = df.sort_values("commence_time").reset_index(drop=True)
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print(f"Eventos unicos WC2022 recolectados: {len(df)}")
    print(f"Rango: {df['commence_time'].min()}  ->  {df['commence_time'].max()}")
    print(f"Guardado: {output_path}")
    print("=" * 60)
    if len(df) < 64:
        print(f"⚠ Solo {len(df)}/64. Faltan fechas; añade dias al barrido si quieres el torneo completo.")
    else:
        print("OK: tienes los 64 (o mas con repesca). Listo para backtest.")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Recolecta event_id del Mundial 2022.")
    p.add_argument("--sport", default="soccer_fifa_world_cup")
    p.add_argument("--output", default="data/processed/odds_events_2022.csv")
    p.add_argument("--env-path", default=".env")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--no-cache", action="store_true")
    return p


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = build_arg_parser().parse_args()
    run(DEFAULT_DATES, a.sport, a.output, a.env_path, a.data_dir, not a.no_cache)


if __name__ == "__main__":
    main()
    