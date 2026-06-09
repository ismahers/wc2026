"""
src/evaluation/run_market_radar.py
==================================
One-command runner for the phase 2 market radar.

This runner never calls The Odds API. It:
  1. Builds model probabilities / fair odds for multiple markets.
  2. Optionally crosses an existing flat odds CSV.
  3. Builds a paper shortlist for non-v1 markets.
  4. Writes a local market availability report.

Usage:
    python -m src.evaluation.run_market_radar
"""

from __future__ import annotations

import argparse
import os

from src.evaluation import market_probabilities
from src.evaluation import market_registry
from src.evaluation import market_availability
from src.evaluation import multi_market_ev
from src.evaluation import multi_market_shortlist
from src.evaluation import paper_tracker


def run(args: argparse.Namespace) -> None:
    registry = market_registry.write_registry(args.registry_output)
    registry_summary = market_registry.summarize_registry(registry)

    print("\nMARKET REGISTRY")
    print("===============")
    print(registry_summary.to_string(index=False))

    radar = market_probabilities.build_market_probabilities(
        ensemble_path=args.ensemble,
        reliability_path=args.reliability,
        output_path=args.probabilities_output,
        markets=args.markets,
    )
    radar_summary = market_probabilities.summarize_market_probabilities(radar)

    print("\nMARKET RADAR")
    print("============")
    print(radar_summary.to_string(index=False))

    if args.skip_ev:
        print("\nSaltando EV local por --skip-ev.")
        return

    if not os.path.exists(args.odds):
        print(f"\nNo existe {args.odds}; radar generado, EV local saltado.")
        return

    ev = multi_market_ev.calculate_multi_market_ev(
        probabilities_path=args.probabilities_output,
        odds_path=args.odds,
        output_path=args.ev_output,
        min_ev_pct=args.min_ev_pct,
    )
    ev_summary = multi_market_ev.summarize_ev(ev)

    print("\nLOCAL ODDS EV")
    print("=============")
    print(ev_summary.to_string(index=False))

    paper, paper_summary, review, review_summary = multi_market_shortlist.run(
        input_path=args.ev_output,
        output_path=args.paper_output,
        summary_path=args.paper_summary_output,
        review_output_path=args.review_output,
        review_summary_path=args.review_summary_output,
        min_ev_pct=args.paper_min_ev_pct,
        max_ev_pct=args.paper_max_ev_pct,
        min_odds=args.paper_min_odds,
        max_odds=args.paper_max_odds,
        min_reliability=args.paper_min_reliability,
        min_confidence=args.paper_min_confidence,
        allow_conflicts=args.allow_conflicts,
    )

    print("\nPAPER SHORTLIST")
    print("===============")
    print(paper_summary.to_string(index=False))
    if not paper.empty:
        cols = [
            "home_team", "away_team", "market", "selection", "line",
            "bookmaker", "odds_decimal", "model_probability", "ev_pct",
            "fiabilidad_nivel", "model_confidence",
        ]
        print()
        print(paper[cols].to_string(index=False))

    print("\nREVIEW ONLY")
    print("===========")
    print(review_summary.to_string(index=False))

    if args.skip_paper_tracker:
        print("\nSaltando paper tracker por --skip-paper-tracker.")
    else:
        signals = paper_tracker.load_current_paper_signals(args.paper_output, args.review_output)
        tracker = paper_tracker.update_paper_tracker(
            signals,
            tracker_path=args.paper_tracker_output,
            paper_stake_units=args.paper_stake_units,
        )
        tracker_summary = paper_tracker.summarize_paper_tracker(tracker)
        print("\nPAPER TRACKER")
        print("=============")
        print(tracker_summary.to_string(index=False))

    availability, availability_summary = market_availability.build_market_availability(
        registry_path=args.registry_output,
        probabilities_path=args.probabilities_output,
        odds_path=args.odds,
        ev_path=args.ev_output,
        paper_tracker_path=args.paper_tracker_output,
        output_path=args.availability_output,
        summary_path=args.availability_summary_output,
    )
    print("\nMARKET AVAILABILITY")
    print("===================")
    print(availability_summary.to_string(index=False))

    print("\nOutputs")
    print(f"  Radar:  {args.probabilities_output}")
    print(f"  EV:     {args.ev_output}")
    print(f"  Paper:  {args.paper_output}")
    print(f"  Review: {args.review_output}")
    print(f"  Avail:  {args.availability_output}")
    if not args.skip_paper_tracker:
        print(f"  Tracker:{args.paper_tracker_output}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run phase 2 market radar without calling any API.")
    parser.add_argument("--ensemble", default=market_probabilities.DEFAULT_ENSEMBLE_INPUT)
    parser.add_argument("--reliability", default=market_probabilities.DEFAULT_RELIABILITY_INPUT)
    parser.add_argument("--registry-output", default=market_registry.DEFAULT_OUTPUT)
    parser.add_argument("--probabilities-output", default=market_probabilities.DEFAULT_OUTPUT)
    parser.add_argument("--markets", default="all")

    parser.add_argument("--skip-ev", action="store_true")
    parser.add_argument("--odds", default=multi_market_ev.DEFAULT_ODDS_INPUT)
    parser.add_argument("--ev-output", default=multi_market_ev.DEFAULT_OUTPUT)
    parser.add_argument("--min-ev-pct", type=float, default=None)

    parser.add_argument("--paper-output", default=multi_market_shortlist.DEFAULT_OUTPUT)
    parser.add_argument("--paper-summary-output", default=multi_market_shortlist.DEFAULT_SUMMARY)
    parser.add_argument("--review-output", default=multi_market_shortlist.DEFAULT_REVIEW_OUTPUT)
    parser.add_argument("--review-summary-output", default=multi_market_shortlist.DEFAULT_REVIEW_SUMMARY)
    parser.add_argument("--paper-min-ev-pct", type=float, default=multi_market_shortlist.DEFAULT_MIN_EV_PCT)
    parser.add_argument("--paper-max-ev-pct", type=float, default=multi_market_shortlist.DEFAULT_MAX_EV_PCT)
    parser.add_argument("--paper-min-odds", type=float, default=multi_market_shortlist.DEFAULT_MIN_ODDS)
    parser.add_argument("--paper-max-odds", type=float, default=multi_market_shortlist.DEFAULT_MAX_ODDS)
    parser.add_argument("--paper-min-reliability", default="MEDIA", choices=["MUY BAJA", "BAJA", "MEDIA", "ALTA", "MUY ALTA"])
    parser.add_argument("--paper-min-confidence", default="medium", choices=["derived", "low", "medium", "high"])
    parser.add_argument("--allow-conflicts", action="store_true")
    parser.add_argument("--skip-paper-tracker", action="store_true")
    parser.add_argument("--paper-tracker-output", default=paper_tracker.DEFAULT_TRACKER_OUTPUT)
    parser.add_argument("--paper-stake-units", type=float, default=paper_tracker.DEFAULT_PAPER_STAKE_UNITS)
    parser.add_argument("--availability-output", default=market_availability.DEFAULT_OUTPUT)
    parser.add_argument("--availability-summary-output", default=market_availability.DEFAULT_SUMMARY)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
