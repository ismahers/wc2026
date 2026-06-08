"""
Reusable filters for turning model EV into actionable bets.

The first WC2022 backtest showed two systematic failure modes:
- 1X2 draws are overestimated by the current model.
- Very high EV values usually indicate model/market mismatch, not value.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


DEFAULT_MIN_EV_PCT = 10.0
DEFAULT_MAX_EV_PCT = 40.0
DEFAULT_BACKTEST_INPUT = "outputs/backtest2022_bets.csv"
DEFAULT_BACKTEST_OUTPUT = "outputs/backtest2022_filtered_bets.csv"
DEFAULT_BACKTEST_SUMMARY = "outputs/backtest2022_filtered_summary.csv"


def is_allowed_market_label(market_label: object) -> bool:
    """Only allow 1X2 home/away selections; draw is excluded."""
    label = str(market_label or "").strip()
    return label in {"1X2-H", "1X2-A", "1X2 Local (H)", "1X2 Visitante (A)"}


def apply_value_filters(
    df: pd.DataFrame,
    min_ev_pct: float = DEFAULT_MIN_EV_PCT,
    max_ev_pct: float = DEFAULT_MAX_EV_PCT,
    market_col: str = "mercado",
    ev_col: str = "ev_pct",
) -> pd.DataFrame:
    """Add strategy filter columns without dropping rows."""
    out = df.copy()
    out["strategy_market_ok"] = out[market_col].map(is_allowed_market_label)
    out["strategy_ev_ok"] = out[ev_col].between(min_ev_pct, max_ev_pct, inclusive="both")
    out["strategy_bet_allowed"] = out["strategy_market_ok"] & out["strategy_ev_ok"]
    out["strategy_reason"] = "allowed"
    out.loc[~out["strategy_market_ok"], "strategy_reason"] = "excluded_market_or_draw"
    out.loc[out["strategy_market_ok"] & ~out["strategy_ev_ok"], "strategy_reason"] = "ev_outside_range"
    return out


def summarize_backtest(filtered: pd.DataFrame) -> pd.DataFrame:
    """Return overall and per-market ROI summary for rows allowed by the strategy."""
    placed = filtered[filtered["strategy_bet_allowed"]].copy()
    rows = []

    def add_row(label: str, group: pd.DataFrame) -> None:
        n = len(group)
        wins = int(group["win"].sum()) if n else 0
        profit = float(group["profit_strategy"].sum()) if n else 0.0
        rows.append({
            "segment": label,
            "bets": n,
            "wins": wins,
            "hit_rate_pct": round(wins / n * 100, 2) if n else 0.0,
            "profit_units": round(profit, 3),
            "roi_pct": round(profit / n * 100, 2) if n else 0.0,
            "avg_ev_pct": round(float(group["ev_pct"].mean()), 2) if n else 0.0,
        })

    add_row("overall", placed)
    for market, group in placed.groupby("mercado"):
        add_row(str(market), group)
    return pd.DataFrame(rows)


def run_backtest_filter(
    input_path: str = DEFAULT_BACKTEST_INPUT,
    output_path: str = DEFAULT_BACKTEST_OUTPUT,
    summary_path: str = DEFAULT_BACKTEST_SUMMARY,
    min_ev_pct: float = DEFAULT_MIN_EV_PCT,
    max_ev_pct: float = DEFAULT_MAX_EV_PCT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} no existe. Ejecuta primero el backtest WC2022.")

    df = pd.read_csv(input_path)
    filtered = apply_value_filters(df, min_ev_pct=min_ev_pct, max_ev_pct=max_ev_pct)
    filtered["profit_strategy"] = 0.0
    filtered.loc[filtered["strategy_bet_allowed"], "profit_strategy"] = filtered.loc[
        filtered["strategy_bet_allowed"], "profit"
    ]
    summary = summarize_backtest(filtered)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    filtered.to_csv(output_path, index=False)
    summary.to_csv(summary_path, index=False)
    return filtered, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aplica la estrategia depurada al backtest WC2022.")
    parser.add_argument("--input", default=DEFAULT_BACKTEST_INPUT)
    parser.add_argument("--output", default=DEFAULT_BACKTEST_OUTPUT)
    parser.add_argument("--summary-output", default=DEFAULT_BACKTEST_SUMMARY)
    parser.add_argument("--min-ev-pct", type=float, default=DEFAULT_MIN_EV_PCT)
    parser.add_argument("--max-ev-pct", type=float, default=DEFAULT_MAX_EV_PCT)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    _, summary = run_backtest_filter(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary_output,
        min_ev_pct=args.min_ev_pct,
        max_ev_pct=args.max_ev_pct,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
