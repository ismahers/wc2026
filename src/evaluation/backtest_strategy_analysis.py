"""
Strategy diagnostics for historical value-betting backtests.

This module does not call odds APIs. It reads an already materialized backtest
CSV, applies configurable strategy filters, and writes:
  - a strategy grid over EV / odds bands
  - an overall/per-market summary with uncertainty and drawdown
  - the filtered bet rows for the selected default strategy

It is intentionally model-agnostic: it works with the legacy WC2022 XGB
backtest and with the ensemble comparison output if `model_variant` exists.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.evaluation.value_filters import is_allowed_market_label


DEFAULT_INPUT = "outputs/backtest2022_ensemble_bets.csv"
DEFAULT_OUTPUT = "outputs/backtest_strategy_filtered_bets.csv"
DEFAULT_SUMMARY = "outputs/backtest_strategy_summary.csv"
DEFAULT_GRID = "outputs/backtest_strategy_grid.csv"

DEFAULT_MIN_EV = 10.0
DEFAULT_MAX_EV = 40.0
DEFAULT_MIN_ODDS = 1.50
DEFAULT_MAX_ODDS = 2.50


@dataclass(frozen=True)
class Strategy:
    min_ev_pct: float
    max_ev_pct: float
    min_odds: float
    max_odds: float
    include_draws: bool = False


EV_BANDS = [
    (5.0, 20.0),
    (5.0, 30.0),
    (5.0, 40.0),
    (10.0, 25.0),
    (10.0, 30.0),
    (10.0, 40.0),
    (15.0, 30.0),
    (15.0, 40.0),
]

ODDS_BANDS = [
    ("core_1.50_2.50", 1.50, 2.50),
    ("mid_1.50_3.50", 1.50, 3.50),
    ("review_2.50_4.00", 2.50, 4.00),
    ("all_1.01_99", 1.01, 99.0),
]


def _normalise_input(df: pd.DataFrame) -> pd.DataFrame:
    required = {"mercado", "cuota", "ev_pct", "win", "profit"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    out = df.copy()
    out["cuota"] = pd.to_numeric(out["cuota"], errors="coerce")
    out["ev_pct"] = pd.to_numeric(out["ev_pct"], errors="coerce")
    out["profit"] = pd.to_numeric(out["profit"], errors="coerce")
    if "model_variant" not in out.columns:
        out["model_variant"] = "legacy_model"
    if "home_team" in out.columns and "away_team" in out.columns:
        out["match_key"] = out["home_team"].astype(str) + " vs " + out["away_team"].astype(str)
    else:
        out["match_key"] = ""
    out = out.dropna(subset=["cuota", "ev_pct", "profit"]).copy()
    return out


def _allowed_market(market: object, include_draws: bool) -> bool:
    label = str(market or "").strip()
    if include_draws and label in {"1X2-D", "1X2 Empate (D)"}:
        return True
    return is_allowed_market_label(label)


def apply_strategy(df: pd.DataFrame, strategy: Strategy) -> pd.DataFrame:
    out = _normalise_input(df)
    out["strategy_market_ok"] = out["mercado"].map(lambda m: _allowed_market(m, strategy.include_draws))
    out["strategy_ev_ok"] = out["ev_pct"].between(strategy.min_ev_pct, strategy.max_ev_pct, inclusive="both")
    out["strategy_odds_ok"] = out["cuota"].between(strategy.min_odds, strategy.max_odds, inclusive="both")
    out["strategy_bet_allowed"] = (
        out["strategy_market_ok"] & out["strategy_ev_ok"] & out["strategy_odds_ok"]
    )
    out["strategy_reason"] = "allowed"
    out.loc[~out["strategy_market_ok"], "strategy_reason"] = "excluded_market"
    out.loc[out["strategy_market_ok"] & ~out["strategy_ev_ok"], "strategy_reason"] = "ev_outside_range"
    out.loc[
        out["strategy_market_ok"] & out["strategy_ev_ok"] & ~out["strategy_odds_ok"],
        "strategy_reason",
    ] = "odds_outside_range"
    out["profit_strategy"] = 0.0
    out.loc[out["strategy_bet_allowed"], "profit_strategy"] = out.loc[
        out["strategy_bet_allowed"], "profit"
    ]
    return out


def _wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _profit_ci(group: pd.DataFrame, z: float = 1.96) -> tuple[float, float]:
    n = len(group)
    if n <= 1:
        return 0.0, 0.0
    mean = float(group["profit_strategy"].mean())
    std = float(group["profit_strategy"].std(ddof=1))
    half = z * std / math.sqrt(n)
    return mean - half, mean + half


def _max_drawdown(profits: Iterable[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for profit in profits:
        equity += float(profit)
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return max_dd


def _summarize_group(
    model_variant: str,
    segment: str,
    group: pd.DataFrame,
    *,
    total_candidate_rows: int,
) -> dict:
    n = len(group)
    wins = int(group["win"].astype(bool).sum()) if n else 0
    profit = float(group["profit_strategy"].sum()) if n else 0.0
    hit_lo, hit_hi = _wilson_interval(wins, n)
    roi_lo, roi_hi = _profit_ci(group) if n else (0.0, 0.0)
    ordered = group.copy()
    if "match_key" in ordered.columns:
        ordered = ordered.sort_values(["match_key", "mercado", "cuota"])
    return {
        "model_variant": model_variant,
        "segment": segment,
        "candidate_rows": int(total_candidate_rows),
        "bets": n,
        "wins": wins,
        "hit_rate_pct": round(wins / n * 100.0, 2) if n else 0.0,
        "hit_rate_ci95_low": round(hit_lo * 100.0, 2),
        "hit_rate_ci95_high": round(hit_hi * 100.0, 2),
        "profit_units": round(profit, 3),
        "roi_pct": round(profit / n * 100.0, 2) if n else 0.0,
        "roi_ci95_low_pct": round(roi_lo * 100.0, 2),
        "roi_ci95_high_pct": round(roi_hi * 100.0, 2),
        "max_drawdown_units": round(_max_drawdown(ordered["profit_strategy"]) if n else 0.0, 3),
        "avg_odds": round(float(group["cuota"].mean()), 3) if n else 0.0,
        "min_odds": round(float(group["cuota"].min()), 3) if n else 0.0,
        "max_odds": round(float(group["cuota"].max()), 3) if n else 0.0,
        "avg_ev_pct": round(float(group["ev_pct"].mean()), 2) if n else 0.0,
        "unique_matches": int(group["match_key"].nunique()) if n and "match_key" in group.columns else 0,
    }


def summarize(filtered: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_variant, model_group in filtered.groupby("model_variant"):
        placed = model_group[model_group["strategy_bet_allowed"].astype(bool)].copy()
        rows.append(
            _summarize_group(
                str(model_variant),
                "overall",
                placed,
                total_candidate_rows=len(model_group),
            )
        )
        for market, group in placed.groupby("mercado"):
            rows.append(
                _summarize_group(
                    str(model_variant),
                    str(market),
                    group,
                    total_candidate_rows=len(model_group[model_group["mercado"] == market]),
                )
            )
    return pd.DataFrame(rows).sort_values(["segment", "model_variant"]).reset_index(drop=True)


def build_grid(df: pd.DataFrame, include_draws: bool = False) -> pd.DataFrame:
    rows = []
    for ev_min, ev_max in EV_BANDS:
        for odds_label, odds_min, odds_max in ODDS_BANDS:
            filtered = apply_strategy(
                df,
                Strategy(ev_min, ev_max, odds_min, odds_max, include_draws=include_draws),
            )
            for model_variant, model_group in filtered.groupby("model_variant"):
                placed = model_group[model_group["strategy_bet_allowed"].astype(bool)].copy()
                row = _summarize_group(
                    str(model_variant),
                    "overall",
                    placed,
                    total_candidate_rows=len(model_group),
                )
                row.update({
                    "ev_band": f"{ev_min:g}-{ev_max:g}",
                    "odds_band": odds_label,
                    "min_ev_pct": ev_min,
                    "max_ev_pct": ev_max,
                    "min_odds": odds_min,
                    "max_odds": odds_max,
                    "include_draws": include_draws,
                })
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["bets", "roi_pct", "profit_units"], ascending=[False, False, False]
    ).reset_index(drop=True)


def run(
    input_path: str = DEFAULT_INPUT,
    output_path: str = DEFAULT_OUTPUT,
    summary_path: str = DEFAULT_SUMMARY,
    grid_path: str = DEFAULT_GRID,
    min_ev_pct: float = DEFAULT_MIN_EV,
    max_ev_pct: float = DEFAULT_MAX_EV,
    min_odds: float = DEFAULT_MIN_ODDS,
    max_odds: float = DEFAULT_MAX_ODDS,
    include_draws: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(input_path)
    strategy = Strategy(
        min_ev_pct=min_ev_pct,
        max_ev_pct=max_ev_pct,
        min_odds=min_odds,
        max_odds=max_odds,
        include_draws=include_draws,
    )
    filtered = apply_strategy(df, strategy)
    summary = summarize(filtered)
    grid = build_grid(df, include_draws=include_draws)

    for path, frame in [
        (output_path, filtered),
        (summary_path, summary),
        (grid_path, grid),
    ]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        frame.to_csv(path, index=False)
    return filtered, summary, grid


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analiza estrategias de backtest sin llamar APIs.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY)
    parser.add_argument("--grid-output", default=DEFAULT_GRID)
    parser.add_argument("--min-ev-pct", type=float, default=DEFAULT_MIN_EV)
    parser.add_argument("--max-ev-pct", type=float, default=DEFAULT_MAX_EV)
    parser.add_argument("--min-odds", type=float, default=DEFAULT_MIN_ODDS)
    parser.add_argument("--max-odds", type=float, default=DEFAULT_MAX_ODDS)
    parser.add_argument("--include-draws", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    _, summary, grid = run(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary_output,
        grid_path=args.grid_output,
        min_ev_pct=args.min_ev_pct,
        max_ev_pct=args.max_ev_pct,
        min_odds=args.min_odds,
        max_odds=args.max_odds,
        include_draws=args.include_draws,
    )

    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print("\nTOP GRID ROWS (>=20 bets first)")
    top = grid[grid["bets"] >= 20].head(20)
    if top.empty:
        top = grid.head(20)
    print(top.to_string(index=False))
    print(f"\nGuardado: {args.output}")
    print(f"Guardado: {args.summary_output}")
    print(f"Guardado: {args.grid_output}")


if __name__ == "__main__":
    main()
