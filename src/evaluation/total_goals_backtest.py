"""
src/evaluation/total_goals_backtest.py
======================================
Dedicated backtest for Over/Under 2.5 goals.

This module does not call any API. It reuses outputs/backtest2022_bets.csv,
which already contains closing odds and model probabilities for WC2022 totals.

The goal is not to promote the market automatically. It answers:
  - does the current Over/Under 2.5 signal deserve real staking?
  - or should it remain paper-only?

Usage:
    python -m src.evaluation.total_goals_backtest
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


DEFAULT_INPUT = "outputs/backtest2022_bets.csv"
DEFAULT_OUTPUT = "outputs/backtest2022_total_goals_v2.csv"
DEFAULT_SUMMARY = "outputs/backtest2022_total_goals_v2_summary.csv"
DEFAULT_GRID = "outputs/backtest2022_total_goals_v2_grid.csv"

TOTAL_MARKETS = {"Over2.5", "Under2.5"}


def _profit(row: pd.Series, stake_units: float = 1.0) -> float:
    if bool(row["win"]):
        return round((float(row["cuota"]) - 1.0) * stake_units, 4)
    return -stake_units


def _roi_summary(df: pd.DataFrame, segment: str) -> dict[str, object]:
    if df.empty:
        return {
            "segment": segment,
            "bets": 0,
            "wins": 0,
            "hit_rate_pct": 0.0,
            "profit_units": 0.0,
            "roi_pct": 0.0,
            "avg_ev_pct": 0.0,
            "avg_odds": 0.0,
        }

    profit = float(df["strategy_profit_units"].sum())
    bets = len(df)
    wins = int(df["win"].sum())
    return {
        "segment": segment,
        "bets": bets,
        "wins": wins,
        "hit_rate_pct": round(wins / bets * 100.0, 2),
        "profit_units": round(profit, 3),
        "roi_pct": round(profit / bets * 100.0, 2),
        "avg_ev_pct": round(float(df["ev_pct"].mean()), 2),
        "avg_odds": round(float(df["cuota"].mean()), 3),
    }


def _decision(summary: pd.DataFrame) -> str:
    default = summary[summary["segment"].eq("default_strategy")]
    if default.empty:
        return "keep_paper_only_no_bets"
    row = default.iloc[0]
    if int(row["bets"]) < 20:
        return "keep_paper_only_sample_too_small"
    if float(row["roi_pct"]) <= 0:
        return "keep_paper_only_negative_roi"
    return "candidate_for_v2_review"


def build_total_goals_backtest(
    input_path: str = DEFAULT_INPUT,
    output_path: str = DEFAULT_OUTPUT,
    summary_path: str = DEFAULT_SUMMARY,
    grid_path: str = DEFAULT_GRID,
    min_ev_pct: float = 5.0,
    max_ev_pct: float = 25.0,
    min_odds: float = 1.50,
    max_odds: float = 2.50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} no existe. Ejecuta primero backtest_2022.")

    raw = pd.read_csv(input_path)
    df = raw[raw["mercado"].isin(TOTAL_MARKETS)].copy()
    if df.empty:
        raise ValueError(f"{input_path} no contiene mercados {sorted(TOTAL_MARKETS)}")

    df["market_family"] = "total_goals"
    df["market_key"] = "total_goals_2_5"
    df["line"] = 2.5
    df["strategy_ev_ok"] = df["ev_pct"].between(min_ev_pct, max_ev_pct, inclusive="both")
    df["strategy_odds_ok"] = df["cuota"].between(min_odds, max_odds, inclusive="both")
    df["strategy_bet_allowed"] = df["strategy_ev_ok"] & df["strategy_odds_ok"]
    df["strategy_reason"] = "allowed"
    df.loc[~df["strategy_ev_ok"], "strategy_reason"] = "ev_outside_range"
    df.loc[df["strategy_ev_ok"] & ~df["strategy_odds_ok"], "strategy_reason"] = "odds_outside_range"
    df["strategy_profit_units"] = 0.0
    df.loc[df["strategy_bet_allowed"], "strategy_profit_units"] = df.loc[
        df["strategy_bet_allowed"]
    ].apply(_profit, axis=1)

    placed = df[df["strategy_bet_allowed"]].copy()
    rows = [
        _roi_summary(placed, "default_strategy"),
    ]
    for market, group in placed.groupby("mercado"):
        rows.append(_roi_summary(group, str(market)))

    all_positive = df[(df["ev_pct"] > 0) & df["strategy_odds_ok"]].copy()
    all_positive["strategy_profit_units"] = all_positive.apply(_profit, axis=1)
    rows.append(_roi_summary(all_positive, "all_positive_ev"))

    all_rows = df.copy()
    all_rows["strategy_profit_units"] = all_rows.apply(_profit, axis=1)
    rows.append(_roi_summary(all_rows, "all_total_goals_rows"))

    summary = pd.DataFrame(rows)
    decision = _decision(summary)
    summary["decision"] = decision

    grid_rows = []
    for lo, hi in [(0, 25), (5, 25), (5, 40), (10, 25), (10, 40), (15, 40)]:
        candidate = df[
            df["ev_pct"].between(lo, hi, inclusive="both")
            & df["cuota"].between(min_odds, max_odds, inclusive="both")
        ].copy()
        candidate["strategy_profit_units"] = candidate.apply(_profit, axis=1)
        row = _roi_summary(candidate, f"ev_{lo}_{hi}")
        row.update({"min_ev_pct": lo, "max_ev_pct": hi, "min_odds": min_odds, "max_odds": max_odds})
        grid_rows.append(row)
    grid = pd.DataFrame(grid_rows)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    summary.to_csv(summary_path, index=False)
    grid.to_csv(grid_path, index=False)
    return df, summary, grid


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest Over/Under 2.5 goals from cached WC2022 bets.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY)
    parser.add_argument("--grid-output", default=DEFAULT_GRID)
    parser.add_argument("--min-ev-pct", type=float, default=5.0)
    parser.add_argument("--max-ev-pct", type=float, default=25.0)
    parser.add_argument("--min-odds", type=float, default=1.50)
    parser.add_argument("--max-odds", type=float, default=2.50)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    _, summary, grid = build_total_goals_backtest(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary_output,
        grid_path=args.grid_output,
        min_ev_pct=args.min_ev_pct,
        max_ev_pct=args.max_ev_pct,
        min_odds=args.min_odds,
        max_odds=args.max_odds,
    )
    print("TOTAL GOALS 2.5 BACKTEST")
    print(summary.to_string(index=False))
    print()
    print("THRESHOLD GRID")
    print(grid.to_string(index=False))
    print(f"\nGuardado: {args.output}")
    print(f"Resumen:  {args.summary_output}")
    print(f"Grid:     {args.grid_output}")


if __name__ == "__main__":
    main()
