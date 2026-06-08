"""
Run the complete WC2026 1X2 v1 workflow.

Steps:
1. Fetch current h2h odds and calculate EV.
2. Build core shortlist plus manual-review/paper buckets.
3. Upsert all current signals into the local tracker.

Use --skip-ev to avoid spending an API credit when the EV CSV already exists.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from src.evaluation import bet_tracker, ev_calculator, strategy_shortlist


DEFAULT_EV_OUTPUT = "outputs/wc2026_ev_h2h_strategy.csv"
DEFAULT_SHORTLIST_OUTPUT = "outputs/wc2026_ev_h2h_shortlist.csv"
DEFAULT_SHORTLIST_SUMMARY = "outputs/wc2026_ev_h2h_shortlist_summary.csv"
DEFAULT_REVIEW_OUTPUT = "outputs/wc2026_ev_h2h_manual_review.csv"
DEFAULT_REVIEW_SUMMARY = "outputs/wc2026_ev_h2h_manual_review_summary.csv"
DEFAULT_TRACKER_OUTPUT = "data/tracking/wc2026_bet_tracker.csv"


def _print_frame(title: str, frame: pd.DataFrame) -> None:
    print()
    print(title)
    print("=" * len(title))
    print(frame.to_string(index=False))


def run(args: argparse.Namespace) -> None:
    if not args.skip_ev:
        ev_calculator.run(
            ensemble_path=args.ensemble,
            reliability_path=args.reliability,
            min_ev=args.min_ev,
            max_ev=args.max_ev,
            markets="h2h",
            bookmakers=args.bookmakers,
            regions=args.regions,
            sport=args.sport,
            output_path=args.ev_output,
            env_path=args.env_path,
            data_dir=args.data_dir,
        )
    elif not os.path.exists(args.ev_output):
        raise FileNotFoundError(f"{args.ev_output} no existe; no puedes usar --skip-ev.")
    else:
        print(f"Saltando EV/API. Usando snapshot existente: {args.ev_output}")

    shortlist, shortlist_summary, review, review_summary = strategy_shortlist.run(
        input_path=args.ev_output,
        output_path=args.shortlist_output,
        summary_path=args.shortlist_summary,
        review_output_path=args.review_output,
        review_summary_path=args.review_summary,
        min_reliability=args.min_reliability,
        min_odds=args.min_odds,
        max_odds=args.max_odds,
        allow_conflicts=args.allow_conflicts,
    )

    signals = bet_tracker.load_current_signals(args.shortlist_output, args.review_output)
    tracker = bet_tracker.update_tracker(
        signals,
        tracker_path=args.tracker_output,
        seen_at_utc=args.seen_at_utc,
        bankroll_units=args.bankroll_units,
        core_stake_units=args.core_stake_units,
        refresh_candidate_stakes=not args.no_refresh_candidate_stakes,
    )
    tracker_summary = bet_tracker.summarize_tracker(tracker)

    _print_frame("SHORTLIST CORE", shortlist_summary)
    if not shortlist.empty:
        cols = [
            "home_team", "away_team", "mercado", "seleccion", "bookmaker",
            "cuota", "prob_modelo", "ev_pct", "fiabilidad_nivel",
        ]
        _print_frame("CORE BETS", shortlist[cols])

    _print_frame("MANUAL REVIEW / PAPER", review_summary)
    if not review.empty:
        cols = [
            "home_team", "away_team", "mercado", "seleccion", "bookmaker",
            "cuota", "prob_modelo", "ev_pct", "fiabilidad_nivel",
            "review_reason", "review_action",
        ]
        _print_frame("REVIEW BETS", review[cols])

    _print_frame("TRACKER", tracker_summary)
    print()
    print(f"EV:        {args.ev_output}")
    print(f"Core:      {args.shortlist_output}")
    print(f"Review:    {args.review_output}")
    print(f"Tracker:   {args.tracker_output}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run WC2026 1X2 value betting v1 workflow.")
    parser.add_argument("--skip-ev", action="store_true", help="No llama a The Odds API; usa --ev-output existente.")
    parser.add_argument("--ensemble", default="outputs/wc2026_ensemble_predictions.csv")
    parser.add_argument("--reliability", default="outputs/wc2026_reliability.csv")
    parser.add_argument("--min-ev", type=float, default=0.10)
    parser.add_argument("--max-ev", type=float, default=0.40)
    parser.add_argument("--bookmakers", default="pinnacle")
    parser.add_argument("--regions", default="eu")
    parser.add_argument("--sport", default="soccer_fifa_world_cup")
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--min-reliability", default="MEDIA", choices=["MUY BAJA", "BAJA", "MEDIA", "ALTA", "MUY ALTA"])
    parser.add_argument("--min-odds", type=float, default=1.50)
    parser.add_argument("--max-odds", type=float, default=2.50)
    parser.add_argument("--allow-conflicts", action="store_true")
    parser.add_argument("--ev-output", default=DEFAULT_EV_OUTPUT)
    parser.add_argument("--shortlist-output", default=DEFAULT_SHORTLIST_OUTPUT)
    parser.add_argument("--shortlist-summary", default=DEFAULT_SHORTLIST_SUMMARY)
    parser.add_argument("--review-output", default=DEFAULT_REVIEW_OUTPUT)
    parser.add_argument("--review-summary", default=DEFAULT_REVIEW_SUMMARY)
    parser.add_argument("--tracker-output", default=DEFAULT_TRACKER_OUTPUT)
    parser.add_argument("--seen-at-utc", default=None)
    parser.add_argument("--bankroll-units", type=float, default=100.0)
    parser.add_argument("--core-stake-units", type=float, default=0.5)
    parser.add_argument("--no-refresh-candidate-stakes", action="store_true")
    return parser


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
