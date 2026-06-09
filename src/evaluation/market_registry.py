"""
src/evaluation/market_registry.py
=================================
Canonical registry for market status.

The registry separates three different ideas that are easy to mix up:
  - whether a market is trained directly;
  - whether it is derived from another trained model;
  - whether it is allowed for real staking.

Usage:
    python -m src.evaluation.market_registry
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass

import pandas as pd


DEFAULT_OUTPUT = "data/processed/market_registry.csv"


@dataclass(frozen=True)
class MarketRegistryEntry:
    market_key: str
    market: str
    display_name: str
    line: str
    selection_scope: str
    market_family: str
    model_status: str
    trained_directly: bool
    derived_from: str
    probability_source: str
    validation_status: str
    betting_status: str
    stake_allowed: bool
    backtest_status: str
    odds_status: str
    default_tracking_action: str
    notes: str
    next_step: str


REGISTRY: tuple[MarketRegistryEntry, ...] = (
    MarketRegistryEntry(
        market_key="h2h_home_away",
        market="h2h",
        display_name="1X2 home/away",
        line="",
        selection_scope="home_or_away",
        market_family="result",
        model_status="trained_direct",
        trained_directly=True,
        derived_from="",
        probability_source="xgb_result_1x2 + poisson_goals ensemble",
        validation_status="validated_v1",
        betting_status="live_core_restricted",
        stake_allowed=True,
        backtest_status="wc2022_pinnacle_filtered_positive",
        odds_status="available_current_and_historical",
        default_tracking_action="core_candidate",
        notes="Only home/away selections; draw is excluded.",
        next_step="Keep monitoring CLV and results during group stage.",
    ),
    MarketRegistryEntry(
        market_key="h2h_draw",
        market="h2h",
        display_name="1X2 draw",
        line="",
        selection_scope="draw",
        market_family="result",
        model_status="trained_direct",
        trained_directly=True,
        derived_from="",
        probability_source="xgb_result_1x2 + poisson_goals ensemble",
        validation_status="known_failure",
        betting_status="excluded",
        stake_allowed=False,
        backtest_status="wc2022_negative_draw_bias",
        odds_status="available_current_and_historical",
        default_tracking_action="no_track",
        notes="Current model overestimates draws; do not stake.",
        next_step="Recalibrate draw class or keep permanently excluded.",
    ),
    MarketRegistryEntry(
        market_key="double_chance",
        market="double_chance",
        display_name="Double chance",
        line="",
        selection_scope="1x_x2_12",
        market_family="result",
        model_status="derived",
        trained_directly=False,
        derived_from="h2h",
        probability_source="derived_from_ensemble_1x2",
        validation_status="radar_only",
        betting_status="no_stake",
        stake_allowed=False,
        backtest_status="not_backtested",
        odds_status="needs_market_odds",
        default_tracking_action="radar_only",
        notes="Derived from 1X2 probabilities; draw bias can leak into 1X/X2.",
        next_step="Backtest separately before paper shortlist.",
    ),
    MarketRegistryEntry(
        market_key="draw_no_bet",
        market="draw_no_bet",
        display_name="Draw no bet",
        line="",
        selection_scope="home_or_away_void_draw",
        market_family="result",
        model_status="derived",
        trained_directly=False,
        derived_from="h2h",
        probability_source="derived_from_ensemble_1x2_conditioned_no_draw",
        validation_status="radar_only",
        betting_status="no_stake",
        stake_allowed=False,
        backtest_status="not_backtested",
        odds_status="needs_market_odds",
        default_tracking_action="radar_only",
        notes="Uses conditional home/away probability once draw is void.",
        next_step="Collect odds and run DNB-specific backtest.",
    ),
    MarketRegistryEntry(
        market_key="total_goals_1_5",
        market="total_goals",
        display_name="Total goals 1.5",
        line="1.5",
        selection_scope="over_under",
        market_family="goals",
        model_status="derived",
        trained_directly=False,
        derived_from="poisson_goals",
        probability_source="poisson_total_goals",
        validation_status="radar_only",
        betting_status="no_stake",
        stake_allowed=False,
        backtest_status="not_backtested",
        odds_status="needs_market_odds",
        default_tracking_action="radar_only",
        notes="Line derived from Poisson score distribution; not independently validated.",
        next_step="Backtest 1.5 line if odds coverage is good.",
    ),
    MarketRegistryEntry(
        market_key="total_goals_2_5",
        market="total_goals",
        display_name="Total goals 2.5",
        line="2.5",
        selection_scope="over_under",
        market_family="goals",
        model_status="trained_direct",
        trained_directly=True,
        derived_from="",
        probability_source="xgb_over25 + poisson_goals ensemble",
        validation_status="paper_tracking",
        betting_status="paper_only",
        stake_allowed=False,
        backtest_status="wc2022_negative_small_sample",
        odds_status="available_current_partial_historical",
        default_tracking_action="paper_track",
        notes="Dedicated WC2022 test: 10 default bets, ROI -23.5%; keep paper only.",
        next_step="Track CLV/results in paper and collect more historical totals odds before promotion.",
    ),
    MarketRegistryEntry(
        market_key="total_goals_3_5",
        market="total_goals",
        display_name="Total goals 3.5",
        line="3.5",
        selection_scope="over_under",
        market_family="goals",
        model_status="derived",
        trained_directly=False,
        derived_from="poisson_goals",
        probability_source="poisson_total_goals",
        validation_status="radar_only",
        betting_status="no_stake",
        stake_allowed=False,
        backtest_status="not_backtested",
        odds_status="needs_market_odds",
        default_tracking_action="radar_only",
        notes="Line derived from Poisson score distribution; not independently validated.",
        next_step="Backtest 3.5 line if odds coverage is good.",
    ),
    MarketRegistryEntry(
        market_key="btts",
        market="btts",
        display_name="Both teams to score",
        line="",
        selection_scope="yes_no",
        market_family="goals",
        model_status="trained_direct",
        trained_directly=True,
        derived_from="",
        probability_source="xgb_btts + poisson_goals ensemble",
        validation_status="paper_tracking",
        betting_status="paper_only",
        stake_allowed=False,
        backtest_status="blocked_no_btts_odds",
        odds_status="no_current_or_historical_btts_odds_in_repo",
        default_tracking_action="paper_track",
        notes="Direct model exists and ECE is acceptable, but repo has no BTTS odds rows for EV/backtest yet.",
        next_step="Collect BTTS odds, rerun multi_market_ev, then run a dedicated BTTS backtest before staking.",
    ),
    MarketRegistryEntry(
        market_key="team_goals_0_5",
        market="team_goals",
        display_name="Team goals 0.5",
        line="0.5",
        selection_scope="team_over_under",
        market_family="team_goals",
        model_status="derived",
        trained_directly=False,
        derived_from="poisson_goals",
        probability_source="poisson_team_lambda",
        validation_status="radar_only",
        betting_status="no_stake",
        stake_allowed=False,
        backtest_status="not_backtested",
        odds_status="needs_market_odds",
        default_tracking_action="radar_only",
        notes="Derived from team lambda; useful for manual comparison only.",
        next_step="Collect team-goals odds before any shortlist.",
    ),
    MarketRegistryEntry(
        market_key="team_goals_1_5",
        market="team_goals",
        display_name="Team goals 1.5",
        line="1.5",
        selection_scope="team_over_under",
        market_family="team_goals",
        model_status="derived",
        trained_directly=False,
        derived_from="poisson_goals",
        probability_source="poisson_team_lambda",
        validation_status="radar_only",
        betting_status="no_stake",
        stake_allowed=False,
        backtest_status="not_backtested",
        odds_status="needs_market_odds",
        default_tracking_action="radar_only",
        notes="Derived from team lambda; useful for manual comparison only.",
        next_step="Collect team-goals odds before any shortlist.",
    ),
    MarketRegistryEntry(
        market_key="team_goals_2_5",
        market="team_goals",
        display_name="Team goals 2.5",
        line="2.5",
        selection_scope="team_over_under",
        market_family="team_goals",
        model_status="derived",
        trained_directly=False,
        derived_from="poisson_goals",
        probability_source="poisson_team_lambda",
        validation_status="radar_only",
        betting_status="no_stake",
        stake_allowed=False,
        backtest_status="not_backtested",
        odds_status="needs_market_odds",
        default_tracking_action="radar_only",
        notes="Derived from team lambda; useful for manual comparison only.",
        next_step="Collect team-goals odds before any shortlist.",
    ),
    MarketRegistryEntry(
        market_key="clean_sheet",
        market="clean_sheet",
        display_name="Clean sheet",
        line="",
        selection_scope="team_yes_no",
        market_family="team_goals",
        model_status="derived",
        trained_directly=False,
        derived_from="poisson_goals",
        probability_source="poisson_opponent_lambda_zero",
        validation_status="radar_only",
        betting_status="no_stake",
        stake_allowed=False,
        backtest_status="not_backtested",
        odds_status="needs_market_odds",
        default_tracking_action="radar_only",
        notes="Derived from opponent lambda; sensitive to Poisson independence assumptions.",
        next_step="Collect clean-sheet odds before any shortlist.",
    ),
    MarketRegistryEntry(
        market_key="total_corners_8_5",
        market="total_corners",
        display_name="Total corners 8.5",
        line="8.5",
        selection_scope="over_under",
        market_family="corners",
        model_status="derived_from_trained_mean",
        trained_directly=False,
        derived_from="corners_total_mean",
        probability_source="poisson_approx_from_xgb_pred_corners",
        validation_status="radar_only",
        betting_status="no_stake",
        stake_allowed=False,
        backtest_status="sample_small_not_backtested",
        odds_status="needs_market_odds",
        default_tracking_action="radar_only",
        notes="XGBoost predicts total-corners mean; line probability uses rough Poisson approximation.",
        next_step="Calibrate corners distribution/overdispersion before paper shortlist.",
    ),
    MarketRegistryEntry(
        market_key="total_corners_9_5",
        market="total_corners",
        display_name="Total corners 9.5",
        line="9.5",
        selection_scope="over_under",
        market_family="corners",
        model_status="derived_from_trained_mean",
        trained_directly=False,
        derived_from="corners_total_mean",
        probability_source="poisson_approx_from_xgb_pred_corners",
        validation_status="radar_only",
        betting_status="no_stake",
        stake_allowed=False,
        backtest_status="sample_small_not_backtested",
        odds_status="needs_market_odds",
        default_tracking_action="radar_only",
        notes="XGBoost predicts total-corners mean; line probability uses rough Poisson approximation.",
        next_step="Calibrate corners distribution/overdispersion before paper shortlist.",
    ),
    MarketRegistryEntry(
        market_key="total_corners_10_5",
        market="total_corners",
        display_name="Total corners 10.5",
        line="10.5",
        selection_scope="over_under",
        market_family="corners",
        model_status="derived_from_trained_mean",
        trained_directly=False,
        derived_from="corners_total_mean",
        probability_source="poisson_approx_from_xgb_pred_corners",
        validation_status="radar_only",
        betting_status="no_stake",
        stake_allowed=False,
        backtest_status="sample_small_not_backtested",
        odds_status="needs_market_odds",
        default_tracking_action="radar_only",
        notes="XGBoost predicts total-corners mean; line probability uses rough Poisson approximation.",
        next_step="Calibrate corners distribution/overdispersion before paper shortlist.",
    ),
    MarketRegistryEntry(
        market_key="total_yellow_cards",
        market="total_yellow_cards",
        display_name="Total yellow cards",
        line="",
        selection_scope="over_under",
        market_family="cards",
        model_status="incomplete",
        trained_directly=False,
        derived_from="",
        probability_source="not_ready",
        validation_status="blocked",
        betting_status="no_stake",
        stake_allowed=False,
        backtest_status="not_backtested",
        odds_status="needs_referee_assignments_and_odds",
        default_tracking_action="blocked",
        notes="Yellow-card config exists, but current WC2026 outputs do not contain pred_yellows.",
        next_step="Fix/generate yellows predictions once referees and card data are reliable.",
    ),
    MarketRegistryEntry(
        market_key="player_props",
        market="player_props",
        display_name="Player props",
        line="",
        selection_scope="player_market",
        market_family="player_props",
        model_status="not_started",
        trained_directly=False,
        derived_from="",
        probability_source="not_ready",
        validation_status="blocked",
        betting_status="no_stake",
        stake_allowed=False,
        backtest_status="not_backtested",
        odds_status="needs_player_prop_odds",
        default_tracking_action="blocked",
        notes="Requires expected minutes and player-level matching/stats before modelling.",
        next_step="Finish player master enrichment and expected-minutes layer.",
    ),
)


def normalize_line(value: object) -> str:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return ""
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value).strip()


def market_key_for(market: object, line: object = "", selection: object = "") -> str:
    market_text = str(market or "").strip()
    line_text = normalize_line(line)
    selection_text = str(selection or "").strip().casefold()

    if market_text == "h2h":
        return "h2h_draw" if selection_text == "draw" else "h2h_home_away"
    if market_text == "double_chance":
        return "double_chance"
    if market_text == "draw_no_bet":
        return "draw_no_bet"
    if market_text == "total_goals":
        return f"total_goals_{line_text.replace('.', '_')}"
    if market_text == "btts":
        return "btts"
    if market_text == "team_goals":
        return f"team_goals_{line_text.replace('.', '_')}"
    if market_text == "clean_sheet":
        return "clean_sheet"
    if market_text == "total_corners":
        return f"total_corners_{line_text.replace('.', '_')}"
    if market_text in {"total_yellow_cards", "yellows"}:
        return "total_yellow_cards"
    if market_text.startswith("player_"):
        return "player_props"
    return market_text


def registry_df() -> pd.DataFrame:
    return pd.DataFrame([asdict(entry) for entry in REGISTRY])


def registry_lookup() -> dict[str, dict[str, object]]:
    return registry_df().set_index("market_key").to_dict(orient="index")


def metadata_for(market: object, line: object = "", selection: object = "") -> dict[str, object]:
    key = market_key_for(market, line, selection)
    lookup = registry_lookup()
    if key in lookup:
        return {"registry_market_key": key, **lookup[key]}
    return {
        "registry_market_key": key,
        "market": market,
        "display_name": str(market or ""),
        "line": normalize_line(line),
        "selection_scope": "",
        "market_family": "unknown",
        "model_status": "unknown",
        "trained_directly": False,
        "derived_from": "",
        "probability_source": "unknown",
        "validation_status": "unknown",
        "betting_status": "no_stake",
        "stake_allowed": False,
        "backtest_status": "unknown",
        "odds_status": "unknown",
        "default_tracking_action": "review_required",
        "notes": "Market is not in registry.",
        "next_step": "Add explicit registry entry before using this market.",
    }


def write_registry(output_path: str = DEFAULT_OUTPUT) -> pd.DataFrame:
    df = registry_df()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def summarize_registry(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["validation_status", "betting_status"], dropna=False)
        .agg(markets=("market_key", "count"))
        .reset_index()
        .sort_values(["validation_status", "betting_status"])
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write the canonical market registry CSV.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    df = write_registry(args.output)
    print(f"Market registry guardado en {args.output} ({len(df)} mercados)")
    print(summarize_registry(df).to_string(index=False))


if __name__ == "__main__":
    main()
