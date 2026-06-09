"""
src/evaluation/btts_readiness.py
================================
Readiness report for BTTS.

BTTS has a trained model, but it should not move to staking until we have odds
coverage and a dedicated backtest/paper sample. This module makes that status
explicit without calling any API.

Usage:
    python -m src.evaluation.btts_readiness
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from src.evaluation.multi_market_shortlist import confidence_rank, reliability_rank


DEFAULT_METRICS = "outputs/xgb_baseline_metrics.json"
DEFAULT_CALIBRATION = "outputs/calibration_metrics.json"
DEFAULT_RADAR = "outputs/wc2026_market_probabilities.csv"
DEFAULT_ODDS = "data/processed/odds_current_worldcup_flat.csv"
DEFAULT_BACKTEST = "outputs/backtest2022_bets.csv"
DEFAULT_SUMMARY = "outputs/btts_readiness_summary.csv"
DEFAULT_CANDIDATES = "outputs/wc2026_btts_radar_candidates.csv"


def _read_json_records(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def _find_record(records: list[dict], *, market_contains: str = "", target_col: str = "") -> dict:
    needle = market_contains.casefold()
    for record in records:
        market = str(record.get("market") or "").casefold()
        if target_col and record.get("target_col") == target_col:
            return record
        if needle and needle in market:
            return record
    return {}


def _count_market_rows(path: str, market_col: str, market_values: set[str]) -> int:
    if not os.path.exists(path):
        return 0
    df = pd.read_csv(path)
    if market_col not in df.columns:
        return 0
    return int(df[market_col].astype(str).str.casefold().isin(market_values).sum())


def build_btts_readiness(
    metrics_path: str = DEFAULT_METRICS,
    calibration_path: str = DEFAULT_CALIBRATION,
    radar_path: str = DEFAULT_RADAR,
    odds_path: str = DEFAULT_ODDS,
    backtest_path: str = DEFAULT_BACKTEST,
    summary_path: str = DEFAULT_SUMMARY,
    candidates_path: str = DEFAULT_CANDIDATES,
    min_reliability: str = "MEDIA",
    min_confidence: str = "medium",
    max_min_odds_ev10: float = 2.50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = _find_record(_read_json_records(metrics_path), target_col="target_btts")
    calibration = _find_record(_read_json_records(calibration_path), market_contains="BTTS")

    radar = pd.read_csv(radar_path) if os.path.exists(radar_path) else pd.DataFrame()
    btts_radar = radar[radar.get("market", pd.Series(dtype=str)).eq("btts")].copy() if not radar.empty else pd.DataFrame()

    odds_btts_rows = _count_market_rows(odds_path, "market", {"btts"})
    backtest_btts_rows = _count_market_rows(backtest_path, "mercado", {"btts", "btts yes", "btts no"})

    if btts_radar.empty:
        candidates = pd.DataFrame()
    else:
        candidates = btts_radar.copy()
        candidates["_reliability_rank"] = candidates["fiabilidad_nivel"].map(reliability_rank)
        candidates["_confidence_rank"] = candidates["model_confidence"].map(confidence_rank)
        candidates = candidates[
            (candidates["_reliability_rank"] >= reliability_rank(min_reliability))
            & (candidates["_confidence_rank"] >= confidence_rank(min_confidence))
            & (pd.to_numeric(candidates["min_odds_ev10"], errors="coerce") <= max_min_odds_ev10)
        ].copy()
        candidates = candidates.sort_values(
            ["fiabilidad_pct", "model_probability"],
            ascending=[False, False],
        )
        candidates = candidates.drop(columns=["_reliability_rank", "_confidence_rank"])

    if odds_btts_rows > 0 and backtest_btts_rows > 0:
        decision = "ready_for_btts_backtest"
    elif odds_btts_rows > 0:
        decision = "ready_for_paper_tracking_needs_backtest"
    else:
        decision = "blocked_waiting_for_btts_odds"

    summary = pd.DataFrame([{
        "market_key": "btts",
        "model_status": "trained_direct",
        "validation_status": "paper_tracking",
        "betting_status": "paper_only",
        "stake_allowed": False,
        "xgb_auc": metrics.get("auc"),
        "xgb_brier": metrics.get("brier"),
        "xgb_log_loss": metrics.get("log_loss"),
        "xgb_pred_mean": metrics.get("pred_mean"),
        "actual_mean": metrics.get("actual_mean"),
        "calibration_ece": calibration.get("ece"),
        "calibration_brier": calibration.get("brier"),
        "radar_rows": len(btts_radar),
        "current_odds_rows": odds_btts_rows,
        "backtest_rows": backtest_btts_rows,
        "candidate_rows_if_odds_available": len(candidates),
        "decision": decision,
        "next_step": (
            "Fetch/collect BTTS odds, rerun multi_market_ev, then paper track; "
            "do not stake until a dedicated backtest is positive."
        ),
    }])

    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    summary.to_csv(summary_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    return summary, candidates


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build BTTS readiness report without calling any API.")
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    parser.add_argument("--radar", default=DEFAULT_RADAR)
    parser.add_argument("--odds", default=DEFAULT_ODDS)
    parser.add_argument("--backtest", default=DEFAULT_BACKTEST)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY)
    parser.add_argument("--candidates-output", default=DEFAULT_CANDIDATES)
    parser.add_argument("--min-reliability", default="MEDIA", choices=["MUY BAJA", "BAJA", "MEDIA", "ALTA", "MUY ALTA"])
    parser.add_argument("--min-confidence", default="medium", choices=["derived", "low", "medium", "high"])
    parser.add_argument("--max-min-odds-ev10", type=float, default=2.50)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary, candidates = build_btts_readiness(
        metrics_path=args.metrics,
        calibration_path=args.calibration,
        radar_path=args.radar,
        odds_path=args.odds,
        backtest_path=args.backtest,
        summary_path=args.summary_output,
        candidates_path=args.candidates_output,
        min_reliability=args.min_reliability,
        min_confidence=args.min_confidence,
        max_min_odds_ev10=args.max_min_odds_ev10,
    )
    print("BTTS READINESS")
    print(summary.to_string(index=False))
    print()
    print(f"Candidatos radar si aparece cuota BTTS: {len(candidates)}")
    if not candidates.empty:
        cols = [
            "date", "home_team", "away_team", "selection", "model_probability",
            "fair_odds", "min_odds_ev10", "fiabilidad_nivel", "model_confidence",
        ]
        print(candidates[cols].head(20).to_string(index=False))
    print(f"\nResumen:    {args.summary_output}")
    print(f"Candidatos: {args.candidates_output}")


if __name__ == "__main__":
    main()
