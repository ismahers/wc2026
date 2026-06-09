"""
src/evaluation/multi_market_ev.py
=================================
Cross the multi-market probability radar with a flat odds CSV.

This module does not call The Odds API. It expects an odds snapshot already
saved by historical_odds_collector.py / get_current_odds_flat and calculates EV
for every market it can map.

Supported odds markets in v1:
  - h2h -> h2h
  - totals -> total_goals
  - btts -> btts
  - draw_no_bet -> draw_no_bet (future-proof if the API returns it)
  - double_chance -> double_chance (future-proof for simple outcome labels)

Usage:
    python -m src.evaluation.multi_market_ev
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd

from src.data.team_names import canonicalize


DEFAULT_PROBABILITIES_INPUT = "outputs/wc2026_market_probabilities.csv"
DEFAULT_ODDS_INPUT = "data/processed/odds_current_worldcup_flat.csv"
DEFAULT_OUTPUT = "outputs/wc2026_multi_market_ev.csv"


def _line_key(value: object) -> str:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value).strip().lower()


def _text_key(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _match_key(home_team: object, away_team: object) -> frozenset[str]:
    return frozenset((canonicalize(home_team), canonicalize(away_team)))


def _prob_selection_key(market: str, selection: object) -> str:
    selection_text = str(selection or "").strip()
    if market in {"h2h", "draw_no_bet"}:
        if selection_text.casefold() == "draw":
            return "draw"
        return canonicalize(selection_text)
    if market in {"total_goals", "total_corners", "btts"}:
        return selection_text.casefold()
    return _text_key(selection_text)


def _build_probability_lookup(probabilities: pd.DataFrame) -> dict[tuple[frozenset[str], str, str, str], dict[str, object]]:
    lookup = {}
    for _, row in probabilities.iterrows():
        market = str(row["market"]).strip()
        key = (
            _match_key(row["home_team"], row["away_team"]),
            market,
            _prob_selection_key(market, row["selection"]),
            _line_key(row.get("line")),
        )
        lookup[key] = row.to_dict()
    return lookup


def _map_double_chance_selection(selection: object, home_team: object, away_team: object) -> str:
    text = _text_key(selection)
    home = _text_key(home_team)
    away = _text_key(away_team)
    has_draw = "draw" in text or "empate" in text
    has_home = home in text
    has_away = away in text

    if has_home and has_draw:
        return _text_key(f"{home_team} or Draw")
    if has_away and has_draw:
        return _text_key(f"{away_team} or Draw")
    if has_home and has_away:
        return _text_key(f"{home_team} or {away_team}")
    return text


def _map_odds_row(row: pd.Series) -> tuple[str, str, str] | None:
    market = str(row.get("market") or "").strip().lower()
    selection = row.get("selection")

    if market == "h2h":
        if str(selection).strip().casefold() == "draw":
            return ("h2h", "draw", "")
        return ("h2h", canonicalize(selection), "")

    if market == "totals":
        return ("total_goals", str(selection or "").strip().casefold(), _line_key(row.get("line")))

    if market == "btts":
        return ("btts", str(selection or "").strip().casefold(), "")

    if market in {"draw_no_bet", "h2h_dnb"}:
        return ("draw_no_bet", canonicalize(selection), "")

    if market == "double_chance":
        return (
            "double_chance",
            _map_double_chance_selection(selection, row.get("home_team"), row.get("away_team")),
            "",
        )

    return None


def _operational_scope(market: str, selection: object) -> str:
    if market == "h2h" and str(selection or "").strip().casefold() == "draw":
        return "excluded_draw"
    if market == "h2h":
        return "use_1x2_v1_filters"
    return "radar_only_not_backtested"


def calculate_multi_market_ev(
    probabilities_path: str = DEFAULT_PROBABILITIES_INPUT,
    odds_path: str = DEFAULT_ODDS_INPUT,
    output_path: str = DEFAULT_OUTPUT,
    min_ev_pct: float | None = None,
) -> pd.DataFrame:
    if not os.path.exists(probabilities_path):
        raise FileNotFoundError(
            f"{probabilities_path} no existe. Ejecuta antes: python -m src.evaluation.market_probabilities"
        )
    if not os.path.exists(odds_path):
        raise FileNotFoundError(f"{odds_path} no existe.")

    probabilities = pd.read_csv(probabilities_path)
    odds = pd.read_csv(odds_path)
    lookup = _build_probability_lookup(probabilities)

    rows = []
    skipped_by_market: dict[str, int] = {}
    for _, odds_row in odds.iterrows():
        mapped = _map_odds_row(odds_row)
        if mapped is None:
            market_name = str(odds_row.get("market") or "unknown")
            skipped_by_market[market_name] = skipped_by_market.get(market_name, 0) + 1
            continue

        market, selection_key, line = mapped
        key = (_match_key(odds_row.get("home_team"), odds_row.get("away_team")), market, selection_key, line)
        prob_row = lookup.get(key)
        if prob_row is None:
            skipped_by_market[market] = skipped_by_market.get(market, 0) + 1
            continue

        odds_decimal = pd.to_numeric(odds_row.get("odds_decimal"), errors="coerce")
        probability = pd.to_numeric(prob_row.get("model_probability"), errors="coerce")
        if pd.isna(odds_decimal) or odds_decimal <= 1 or pd.isna(probability) or probability <= 0:
            continue

        ev_pct = float(probability) * float(odds_decimal) * 100.0 - 100.0
        if min_ev_pct is not None and ev_pct < min_ev_pct:
            continue

        rows.append({
            "date": prob_row.get("date"),
            "home_team": prob_row.get("home_team"),
            "away_team": prob_row.get("away_team"),
            "market": prob_row.get("market"),
            "selection": prob_row.get("selection"),
            "line": prob_row.get("line"),
            "bookmaker": odds_row.get("bookmaker"),
            "odds_decimal": round(float(odds_decimal), 3),
            "model_probability": round(float(probability), 4),
            "fair_odds": prob_row.get("fair_odds"),
            "ev_pct": round(ev_pct, 2),
            "registry_market_key": prob_row.get("registry_market_key"),
            "registry_display_name": prob_row.get("registry_display_name"),
            "market_family": prob_row.get("market_family"),
            "model_status": prob_row.get("model_status"),
            "trained_directly": prob_row.get("trained_directly"),
            "validation_status": prob_row.get("validation_status"),
            "betting_status": prob_row.get("betting_status"),
            "stake_allowed": prob_row.get("stake_allowed"),
            "default_tracking_action": prob_row.get("default_tracking_action"),
            "source_model": prob_row.get("source_model"),
            "model_confidence": prob_row.get("model_confidence"),
            "registry_probability_source": prob_row.get("registry_probability_source"),
            "fiabilidad_pct": prob_row.get("fiabilidad_pct"),
            "fiabilidad_nivel": prob_row.get("fiabilidad_nivel"),
            "operational_scope": _operational_scope(str(prob_row.get("market")), prob_row.get("selection")),
            "registry_notes": prob_row.get("registry_notes"),
            "next_step": prob_row.get("next_step"),
            "odds_market": odds_row.get("market"),
            "odds_selection": odds_row.get("selection"),
            "snapshot_type": odds_row.get("snapshot_type"),
            "timestamp": odds_row.get("timestamp"),
        })

    if not rows:
        raise ValueError(
            "No se pudo cruzar ninguna cuota con el radar de probabilidades. "
            f"Mercados saltados: {skipped_by_market}"
        )

    out = pd.DataFrame(rows).sort_values("ev_pct", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def summarize_ev(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("market", dropna=False)
        .agg(
            rows=("market", "size"),
            avg_ev_pct=("ev_pct", "mean"),
            max_ev_pct=("ev_pct", "max"),
            avg_odds=("odds_decimal", "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values("market")
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross multi-market probabilities with a flat odds CSV.")
    parser.add_argument("--probabilities", default=DEFAULT_PROBABILITIES_INPUT)
    parser.add_argument("--odds", default=DEFAULT_ODDS_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--min-ev-pct", type=float, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    df = calculate_multi_market_ev(
        probabilities_path=args.probabilities,
        odds_path=args.odds,
        output_path=args.output,
        min_ev_pct=args.min_ev_pct,
    )
    print(f"Guardado en {args.output} ({len(df)} filas)")
    print(summarize_ev(df).to_string(index=False))
    print()
    cols = [
        "home_team", "away_team", "market", "selection", "line",
        "bookmaker", "odds_decimal", "model_probability", "fair_odds",
        "ev_pct", "fiabilidad_nivel", "operational_scope",
    ]
    print(df.head(20)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
