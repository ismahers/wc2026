"""
src/evaluation/multi_market_shortlist.py
========================================
Build a paper-trading shortlist from multi-market EV output.

This is intentionally separate from the operative 1X2 v1 shortlist. Markets
that are not backtested yet should be tracked on paper first, not staked.

Default paper markets:
  - total_goals Over/Under 2.5
  - btts Yes/No, when odds are available

Usage:
    python -m src.evaluation.multi_market_shortlist
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


DEFAULT_INPUT = "outputs/wc2026_multi_market_ev.csv"
DEFAULT_OUTPUT = "outputs/wc2026_multi_market_paper_shortlist.csv"
DEFAULT_SUMMARY = "outputs/wc2026_multi_market_paper_summary.csv"
DEFAULT_REVIEW_OUTPUT = "outputs/wc2026_multi_market_review.csv"
DEFAULT_REVIEW_SUMMARY = "outputs/wc2026_multi_market_review_summary.csv"

DEFAULT_MIN_EV_PCT = 5.0
DEFAULT_MAX_EV_PCT = 25.0
DEFAULT_MIN_ODDS = 1.50
DEFAULT_MAX_ODDS = 2.50
DEFAULT_PAPER_MARKETS = ("total_goals", "btts")

RELIABILITY_ORDER = {
    "N/A": -1,
    "MUY BAJA": 0,
    "BAJA": 1,
    "MEDIA": 2,
    "ALTA": 3,
    "MUY ALTA": 4,
}

CONFIDENCE_ORDER = {
    "N/A": -1,
    "derived": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}


def reliability_rank(value: object) -> int:
    return RELIABILITY_ORDER.get(str(value or "").strip().upper(), -1)


def confidence_rank(value: object) -> int:
    return CONFIDENCE_ORDER.get(str(value or "").strip().lower(), -1)


def _is_total_goals_line_ok(df: pd.DataFrame) -> pd.Series:
    line = pd.to_numeric(df.get("line"), errors="coerce")
    return (df["market"].ne("total_goals")) | line.eq(2.5)


def _market_reason(row: pd.Series) -> str:
    reasons = []
    if not row["_paper_market_ok"]:
        reasons.append("unsupported_market")
    if not row["_line_ok"]:
        reasons.append("unsupported_line")
    if not row["_ev_ok"]:
        reasons.append("ev_outside_paper_range")
    if not row["_odds_ok"]:
        reasons.append("odds_outside_core_range")
    if not row["_reliability_ok"]:
        reasons.append("low_reliability")
    if not row["_confidence_ok"]:
        reasons.append("low_model_confidence")
    if not reasons:
        return "paper_track"
    return "+".join(reasons)


def _add_filter_columns(
    df: pd.DataFrame,
    *,
    paper_markets: tuple[str, ...] = DEFAULT_PAPER_MARKETS,
    min_ev_pct: float = DEFAULT_MIN_EV_PCT,
    max_ev_pct: float = DEFAULT_MAX_EV_PCT,
    min_odds: float = DEFAULT_MIN_ODDS,
    max_odds: float = DEFAULT_MAX_ODDS,
    min_reliability: str = "MEDIA",
    min_confidence: str = "medium",
) -> pd.DataFrame:
    out = df.copy()
    out["_reliability_rank"] = out["fiabilidad_nivel"].map(reliability_rank)
    out["_confidence_rank"] = out["model_confidence"].map(confidence_rank)
    out["_paper_market_ok"] = out["market"].isin(paper_markets)
    out["_line_ok"] = _is_total_goals_line_ok(out)
    out["_ev_ok"] = out["ev_pct"].between(min_ev_pct, max_ev_pct, inclusive="both")
    out["_odds_ok"] = out["odds_decimal"].between(min_odds, max_odds, inclusive="both")
    out["_reliability_ok"] = out["_reliability_rank"] >= reliability_rank(min_reliability)
    out["_confidence_ok"] = out["_confidence_rank"] >= confidence_rank(min_confidence)
    out["paper_track_allowed"] = (
        out["_paper_market_ok"]
        & out["_line_ok"]
        & out["_ev_ok"]
        & out["_odds_ok"]
        & out["_reliability_ok"]
        & out["_confidence_ok"]
    )
    out["paper_reason"] = out.apply(_market_reason, axis=1)
    return out


def build_paper_shortlist(
    df: pd.DataFrame,
    *,
    allow_conflicts: bool = False,
    **filter_kwargs,
) -> pd.DataFrame:
    out = _add_filter_columns(df, **filter_kwargs)
    out = out[out["paper_track_allowed"]].copy()
    out = out.sort_values(["fiabilidad_pct", "ev_pct"], ascending=[False, False])

    if not allow_conflicts and not out.empty:
        out = out.drop_duplicates(["home_team", "away_team", "market"], keep="first")

    out["tracking_action"] = "paper_track"
    return _drop_internal_columns(out).reset_index(drop=True)


def build_review_list(
    df: pd.DataFrame,
    paper_shortlist: pd.DataFrame,
    *,
    min_ev_pct: float = 0.0,
    allow_conflicts: bool = False,
    **filter_kwargs,
) -> pd.DataFrame:
    out = _add_filter_columns(df, **filter_kwargs)
    out = out[out["market"].isin(filter_kwargs.get("paper_markets", DEFAULT_PAPER_MARKETS))].copy()
    out = out[out["ev_pct"] >= min_ev_pct].copy()
    out = out[~out["paper_track_allowed"]].copy()

    shortlist_keys = set()
    if not paper_shortlist.empty:
        shortlist_keys = set(
            paper_shortlist[["home_team", "away_team", "market", "selection", "bookmaker"]]
            .astype(str)
            .agg("|".join, axis=1)
        )
    out["_bet_key"] = out[["home_team", "away_team", "market", "selection", "bookmaker"]].astype(str).agg("|".join, axis=1)
    out = out[~out["_bet_key"].isin(shortlist_keys)].copy()

    if not allow_conflicts and not out.empty:
        out = out.sort_values(["fiabilidad_pct", "ev_pct"], ascending=[False, False])
        out = out.drop_duplicates(["home_team", "away_team", "market"], keep="first")

    out["tracking_action"] = "review_only"
    out = out.sort_values(["fiabilidad_pct", "ev_pct"], ascending=[False, False])
    return _drop_internal_columns(out).reset_index(drop=True)


def _drop_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    internal = [col for col in df.columns if col.startswith("_")]
    if "_bet_key" in df.columns and "_bet_key" not in internal:
        internal.append("_bet_key")
    return df.drop(columns=internal, errors="ignore")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([{
            "rows": 0,
            "markets": 0,
            "avg_ev_pct": 0.0,
            "avg_odds": 0.0,
            "avg_fiabilidad_pct": 0.0,
        }])
    return (
        df.groupby("market", dropna=False)
        .agg(
            rows=("market", "size"),
            avg_ev_pct=("ev_pct", "mean"),
            avg_odds=("odds_decimal", "mean"),
            avg_fiabilidad_pct=("fiabilidad_pct", "mean"),
        )
        .round(2)
        .reset_index()
    )


def run(
    input_path: str = DEFAULT_INPUT,
    output_path: str = DEFAULT_OUTPUT,
    summary_path: str = DEFAULT_SUMMARY,
    review_output_path: str = DEFAULT_REVIEW_OUTPUT,
    review_summary_path: str = DEFAULT_REVIEW_SUMMARY,
    min_ev_pct: float = DEFAULT_MIN_EV_PCT,
    max_ev_pct: float = DEFAULT_MAX_EV_PCT,
    min_odds: float = DEFAULT_MIN_ODDS,
    max_odds: float = DEFAULT_MAX_ODDS,
    min_reliability: str = "MEDIA",
    min_confidence: str = "medium",
    markets: tuple[str, ...] = DEFAULT_PAPER_MARKETS,
    allow_conflicts: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"{input_path} no existe. Ejecuta antes: python -m src.evaluation.multi_market_ev"
        )

    df = pd.read_csv(input_path)
    filter_kwargs = {
        "paper_markets": markets,
        "min_ev_pct": min_ev_pct,
        "max_ev_pct": max_ev_pct,
        "min_odds": min_odds,
        "max_odds": max_odds,
        "min_reliability": min_reliability,
        "min_confidence": min_confidence,
    }
    paper = build_paper_shortlist(df, allow_conflicts=allow_conflicts, **filter_kwargs)
    review = build_review_list(df, paper, allow_conflicts=allow_conflicts, **filter_kwargs)
    paper_summary = summarize(paper)
    review_summary = summarize(review)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    paper.to_csv(output_path, index=False)
    paper_summary.to_csv(summary_path, index=False)
    review.to_csv(review_output_path, index=False)
    review_summary.to_csv(review_summary_path, index=False)
    return paper, paper_summary, review, review_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build paper shortlist for non-v1 markets.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY)
    parser.add_argument("--review-output", default=DEFAULT_REVIEW_OUTPUT)
    parser.add_argument("--review-summary-output", default=DEFAULT_REVIEW_SUMMARY)
    parser.add_argument("--min-ev-pct", type=float, default=DEFAULT_MIN_EV_PCT)
    parser.add_argument("--max-ev-pct", type=float, default=DEFAULT_MAX_EV_PCT)
    parser.add_argument("--min-odds", type=float, default=DEFAULT_MIN_ODDS)
    parser.add_argument("--max-odds", type=float, default=DEFAULT_MAX_ODDS)
    parser.add_argument("--min-reliability", default="MEDIA", choices=["MUY BAJA", "BAJA", "MEDIA", "ALTA", "MUY ALTA"])
    parser.add_argument("--min-confidence", default="medium", choices=["derived", "low", "medium", "high"])
    parser.add_argument("--markets", default="total_goals,btts")
    parser.add_argument("--allow-conflicts", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    markets = tuple(part.strip() for part in args.markets.split(",") if part.strip())
    paper, paper_summary, review, review_summary = run(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary_output,
        review_output_path=args.review_output,
        review_summary_path=args.review_summary_output,
        min_ev_pct=args.min_ev_pct,
        max_ev_pct=args.max_ev_pct,
        min_odds=args.min_odds,
        max_odds=args.max_odds,
        min_reliability=args.min_reliability,
        min_confidence=args.min_confidence,
        markets=markets,
        allow_conflicts=args.allow_conflicts,
    )

    print("PAPER SHORTLIST")
    print(paper_summary.to_string(index=False))
    if not paper.empty:
        cols = [
            "home_team", "away_team", "market", "selection", "line",
            "bookmaker", "odds_decimal", "model_probability", "ev_pct",
            "fiabilidad_nivel", "model_confidence", "tracking_action",
        ]
        print()
        print(paper[cols].to_string(index=False))

    print()
    print("REVIEW ONLY")
    print(review_summary.to_string(index=False))
    if not review.empty:
        cols = [
            "home_team", "away_team", "market", "selection", "line",
            "bookmaker", "odds_decimal", "model_probability", "ev_pct",
            "fiabilidad_nivel", "model_confidence", "paper_reason",
        ]
        print()
        print(review[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
