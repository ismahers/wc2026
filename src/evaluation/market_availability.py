"""
src/evaluation/market_availability.py
=====================================
Local market availability report.

This module does not call any odds API. It only inspects files already present
in the repo/workspace and answers:
  - which markets have model probabilities;
  - which markets have odds in the latest saved snapshot;
  - which markets can currently produce EV rows;
  - which markets are waiting for bookmakers to unlock lines.

Usage:
    python -m src.evaluation.market_availability
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


DEFAULT_REGISTRY = "data/processed/market_registry.csv"
DEFAULT_PROBABILITIES = "outputs/wc2026_market_probabilities.csv"
DEFAULT_ODDS = "data/processed/odds_current_worldcup_flat.csv"
DEFAULT_EV = "outputs/wc2026_multi_market_ev.csv"
DEFAULT_PAPER_TRACKER = "data/tracking/wc2026_paper_tracker.csv"
DEFAULT_OUTPUT = "outputs/wc2026_market_availability.csv"
DEFAULT_SUMMARY = "outputs/wc2026_market_availability_summary.csv"

ODDS_MARKET_MAP = {
    "h2h": "h2h",
    "totals": "total_goals",
    "btts": "btts",
    "draw_no_bet": "draw_no_bet",
    "h2h_dnb": "draw_no_bet",
    "double_chance": "double_chance",
}


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _collapse(values: pd.Series) -> str:
    clean = sorted({str(value) for value in values.dropna() if str(value).strip()})
    return ";".join(clean)


def _any_true(values: pd.Series) -> bool:
    if values.empty:
        return False
    return values.astype(str).str.casefold().isin({"true", "1", "yes"}).any()


def _unique_matches(df: pd.DataFrame) -> int:
    if df.empty or not {"home_team", "away_team"}.issubset(df.columns):
        return 0
    return int(df[["home_team", "away_team"]].drop_duplicates().shape[0])


def _count_by_market(df: pd.DataFrame, market_col: str = "market") -> pd.DataFrame:
    if df.empty or market_col not in df.columns:
        return pd.DataFrame(columns=["market", "rows", "matches"])
    out = (
        df.assign(market=df[market_col].astype(str))
        .groupby("market", dropna=False)
        .agg(rows=("market", "size"))
        .reset_index()
    )
    match_counts = []
    for market, group in df.assign(market=df[market_col].astype(str)).groupby("market", dropna=False):
        match_counts.append({"market": market, "matches": _unique_matches(group)})
    return out.merge(pd.DataFrame(match_counts), on="market", how="left")


def _registry_by_market(registry: pd.DataFrame) -> pd.DataFrame:
    if registry.empty:
        return pd.DataFrame(columns=[
            "market", "registry_entries", "registry_keys", "model_statuses",
            "validation_statuses", "betting_statuses", "stake_allowed_any",
            "default_tracking_actions", "backtest_statuses", "odds_statuses",
            "next_steps",
        ])
    return (
        registry.groupby("market", dropna=False)
        .agg(
            registry_entries=("market_key", "size"),
            registry_keys=("market_key", _collapse),
            model_statuses=("model_status", _collapse),
            validation_statuses=("validation_status", _collapse),
            betting_statuses=("betting_status", _collapse),
            stake_allowed_any=("stake_allowed", _any_true),
            default_tracking_actions=("default_tracking_action", _collapse),
            backtest_statuses=("backtest_status", _collapse),
            odds_statuses=("odds_status", _collapse),
            next_steps=("next_step", _collapse),
        )
        .reset_index()
    )


def _mapped_odds_counts(odds: pd.DataFrame) -> pd.DataFrame:
    if odds.empty or "market" not in odds.columns:
        return pd.DataFrame(columns=["market", "odds_rows", "odds_matches", "odds_source_markets"])

    mapped = odds.copy()
    mapped["market"] = mapped["market"].astype(str).str.strip().str.lower()
    mapped["internal_market"] = mapped["market"].map(ODDS_MARKET_MAP)
    mapped = mapped[mapped["internal_market"].notna()].copy()
    if mapped.empty:
        return pd.DataFrame(columns=["market", "odds_rows", "odds_matches", "odds_source_markets"])

    rows = []
    for market, group in mapped.groupby("internal_market", dropna=False):
        rows.append({
            "market": market,
            "odds_rows": int(len(group)),
            "odds_matches": _unique_matches(group),
            "odds_source_markets": _collapse(group["market"]),
        })
    return pd.DataFrame(rows)


def _merge_counts(base: pd.DataFrame, counts: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if counts.empty:
        base[f"{prefix}_rows"] = 0
        base[f"{prefix}_matches"] = 0
        return base

    rename = {"rows": f"{prefix}_rows", "matches": f"{prefix}_matches"}
    out = base.merge(counts.rename(columns=rename), on="market", how="left")
    out[f"{prefix}_rows"] = out[f"{prefix}_rows"].fillna(0).astype(int)
    out[f"{prefix}_matches"] = out[f"{prefix}_matches"].fillna(0).astype(int)
    return out


def _availability_status(row: pd.Series) -> str:
    probability_rows = int(row.get("probability_rows", 0))
    odds_rows = int(row.get("odds_rows", 0))
    ev_rows = int(row.get("ev_rows", 0))
    paper_tracker_rows = int(row.get("paper_tracker_rows", 0))
    betting_statuses = str(row.get("betting_statuses") or "")
    model_statuses = str(row.get("model_statuses") or "")
    stake_allowed = bool(row.get("stake_allowed_any", False))

    if stake_allowed and ev_rows > 0:
        return "live_ev_ready"
    if "paper_only" in betting_statuses and ev_rows > 0:
        return "paper_ev_ready"
    if paper_tracker_rows > 0:
        return "paper_tracker_active"
    if probability_rows > 0 and odds_rows > 0 and ev_rows == 0:
        return "odds_mapping_gap"
    if probability_rows > 0 and odds_rows == 0:
        return "modeled_waiting_for_odds"
    if probability_rows == 0 and odds_rows > 0:
        return "odds_without_model"
    if "blocked" in str(row.get("validation_statuses") or "") or "not_ready" in model_statuses:
        return "blocked_not_modeled"
    return "registry_only"


def _recommended_action(row: pd.Series) -> str:
    status = row["availability_status"]
    market = str(row.get("market") or "")

    if status == "live_ev_ready":
        if market == "h2h":
            return "Run 1X2 v1 filters and update bet tracker; stake only core home/away selections."
        return "Review live EV because stake_allowed is true."
    if status == "paper_ev_ready":
        return "Update paper tracker and measure CLV/results; no real stake."
    if status == "paper_tracker_active":
        return "Keep tracking closing line and settlement."
    if status == "odds_mapping_gap":
        return "Odds exist locally but did not map to probabilities; inspect market/selection/line labels."
    if status == "modeled_waiting_for_odds":
        return "Wait for bookmaker market unlock; then refresh saved odds and rerun multi_market_ev."
    if status == "odds_without_model":
        return "Odds exist but no model probabilities; do not use until registry/model support exists."
    if status == "blocked_not_modeled":
        return str(row.get("next_steps") or "Finish data/model layer before using this market.")
    return str(row.get("next_steps") or "No immediate action.")


def build_market_availability(
    registry_path: str = DEFAULT_REGISTRY,
    probabilities_path: str = DEFAULT_PROBABILITIES,
    odds_path: str = DEFAULT_ODDS,
    ev_path: str = DEFAULT_EV,
    paper_tracker_path: str = DEFAULT_PAPER_TRACKER,
    output_path: str = DEFAULT_OUTPUT,
    summary_path: str = DEFAULT_SUMMARY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = _read_csv(registry_path)
    probabilities = _read_csv(probabilities_path)
    odds = _read_csv(odds_path)
    ev = _read_csv(ev_path)
    paper_tracker = _read_csv(paper_tracker_path)

    registry_summary = _registry_by_market(registry)
    probability_counts = _count_by_market(probabilities)
    ev_counts = _count_by_market(ev)
    paper_counts = _count_by_market(paper_tracker)
    odds_counts = _mapped_odds_counts(odds)

    all_markets = sorted(
        set(registry_summary.get("market", pd.Series(dtype=str)).astype(str))
        | set(probability_counts.get("market", pd.Series(dtype=str)).astype(str))
        | set(odds_counts.get("market", pd.Series(dtype=str)).astype(str))
        | set(ev_counts.get("market", pd.Series(dtype=str)).astype(str))
        | set(paper_counts.get("market", pd.Series(dtype=str)).astype(str))
    )
    report = pd.DataFrame({"market": all_markets})
    report = report.merge(registry_summary, on="market", how="left")
    report = _merge_counts(report, probability_counts, "probability")
    report = report.merge(odds_counts, on="market", how="left")
    report["odds_rows"] = report["odds_rows"].fillna(0).astype(int)
    report["odds_matches"] = report["odds_matches"].fillna(0).astype(int)
    report["odds_source_markets"] = report["odds_source_markets"].fillna("")
    report = _merge_counts(report, ev_counts, "ev")
    report = _merge_counts(report, paper_counts, "paper_tracker")

    for col in [
        "registry_entries", "registry_keys", "model_statuses", "validation_statuses",
        "betting_statuses", "default_tracking_actions", "backtest_statuses",
        "odds_statuses", "next_steps",
    ]:
        if col not in report.columns:
            report[col] = "" if col != "registry_entries" else 0
    report["registry_entries"] = report["registry_entries"].fillna(0).astype(int)
    report["stake_allowed_any"] = report["stake_allowed_any"].fillna(False).astype(bool)

    text_cols = [
        "registry_keys", "model_statuses", "validation_statuses", "betting_statuses",
        "default_tracking_actions", "backtest_statuses", "odds_statuses", "next_steps",
    ]
    report[text_cols] = report[text_cols].fillna("")

    report["availability_status"] = report.apply(_availability_status, axis=1)
    report["recommended_action"] = report.apply(_recommended_action, axis=1)
    report = report.sort_values(
        ["availability_status", "market"],
        ascending=[True, True],
    ).reset_index(drop=True)

    summary = (
        report.groupby("availability_status", dropna=False)
        .agg(
            markets=("market", "size"),
            probability_rows=("probability_rows", "sum"),
            odds_rows=("odds_rows", "sum"),
            ev_rows=("ev_rows", "sum"),
            paper_tracker_rows=("paper_tracker_rows", "sum"),
        )
        .reset_index()
        .sort_values("availability_status")
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    report.to_csv(output_path, index=False)
    summary.to_csv(summary_path, index=False)
    return report, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build local market availability report without API calls.")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--probabilities", default=DEFAULT_PROBABILITIES)
    parser.add_argument("--odds", default=DEFAULT_ODDS)
    parser.add_argument("--ev", default=DEFAULT_EV)
    parser.add_argument("--paper-tracker", default=DEFAULT_PAPER_TRACKER)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report, summary = build_market_availability(
        registry_path=args.registry,
        probabilities_path=args.probabilities,
        odds_path=args.odds,
        ev_path=args.ev,
        paper_tracker_path=args.paper_tracker,
        output_path=args.output,
        summary_path=args.summary_output,
    )
    print(f"Guardado en {args.output} ({len(report)} mercados)")
    print(summary.to_string(index=False))
    print()
    cols = [
        "market", "availability_status", "probability_rows", "odds_rows",
        "ev_rows", "paper_tracker_rows", "recommended_action",
    ]
    print(report[cols].to_string(index=False))


if __name__ == "__main__":
    main()
