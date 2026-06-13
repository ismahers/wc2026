"""
Build a WC2026 player-prop radar from free data sources.

This is a no-odds, no-stake layer. It converts FBref/Transfermarkt player rates
into fair probabilities for markets that are useful to review manually once
bookmaker player-prop odds appear.

Usage:
  python -m src.evaluation.player_prop_radar
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FBREF_FEATURES = Path("data/processed/fbref_player_features_wc2026.csv")
DEFAULT_TM_FEATURES = Path("data/processed/player_prop_features_wc2026.csv")
DEFAULT_FIXTURES = Path("data/raw/group_stage_wc2026.csv")
DEFAULT_MATCH_FEATURES = Path("data/processed/features_wc2026.csv")
DEFAULT_MATCH_PROJECTIONS = Path("outputs/wc2026_ensemble_predictions.csv")
DEFAULT_MINUTES_PROJECTION = Path("data/processed/player_expected_minutes_wc2026.csv")
DEFAULT_OUTPUT = Path("outputs/wc2026_player_prop_radar.csv")
DEFAULT_SUMMARY = Path("outputs/wc2026_player_prop_radar_summary.csv")

MIN_PROBABILITY_TO_OUTPUT = 0.08
MAX_FAIR_ODDS_TO_HIGHLIGHT = 4.50
MIN_FAIR_ODDS_TO_REVIEW = 1.45
STARTER_MINUTES_DISCOUNT = 0.94
STARTER_MINUTES_CAP = 85.0

BIG5_COMPETITIONS = {
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
}
STRONG_SECONDARY_COMPETITIONS = {
    "POR-Primeira Liga",
    "NED-Eredivisie",
    "USA-Major League Soccer",
    "BEL-Belgian Pro League",
    "TUR-Super Lig",
    "KSA-Saudi Pro League",
}
ATTACK_LAMBDA_BASELINE = 1.45
ATTACK_ADJUSTMENT_ELASTICITY = 0.50
SHOT_MARKET_POSITION_FACTORS = {
    "shots": {
        "Forward": 0.86,
        "Midfielder": 0.68,
        "Defender": 0.50,
        "Goalkeeper": 0.0,
    },
    "shots_on_target": {
        "Forward": 1.00,
        "Midfielder": 0.54,
        "Defender": 0.36,
        "Goalkeeper": 0.0,
    },
}
SHOT_MARKETS = {"shots", "shots_on_target"}
PROP_PROBABILITY_CAPS = {
    "shots": {
        0: 0.82,
        1: 0.72,
        2: 0.58,
    },
    "shots_on_target": {
        0: 0.82,
        1: 0.48,
    },
    "fouls_committed": {
        0: 0.82,
    },
    "goalkeeper_saves": {
        1: 0.82,
    },
}


@dataclass(frozen=True)
class PropSpec:
    market: str
    line: float
    rate_col: str
    readiness_col: str
    adjustment_kind: str
    positions: tuple[str, ...] | None = None
    source: str = "fbref"


PROP_SPECS = (
    PropSpec("shots", 0.5, "fbref_shots_per90", "fbref_shots_ready", "attack"),
    PropSpec("shots", 1.5, "fbref_shots_per90", "fbref_shots_ready", "attack"),
    PropSpec("shots", 2.5, "fbref_shots_per90", "fbref_shots_ready", "attack"),
    PropSpec("shots_on_target", 0.5, "fbref_shots_on_target_per90", "fbref_shots_ready", "attack"),
    PropSpec("shots_on_target", 1.5, "fbref_shots_on_target_per90", "fbref_shots_ready", "attack"),
    PropSpec("tackles", 1.5, "fbref_tackles_per90", "fbref_defense_ready", "defensive_work"),
    PropSpec("tackles", 2.5, "fbref_tackles_per90", "fbref_defense_ready", "defensive_work"),
    PropSpec("fouls_committed", 0.5, "fbref_fouls_committed_per90", "fbref_defense_ready", "defensive_work"),
    PropSpec("fouls_committed", 1.5, "fbref_fouls_committed_per90", "fbref_defense_ready", "defensive_work"),
    PropSpec("yellow_card", 0.5, "fbref_yellow_cards_per90", "fbref_defense_ready", "defensive_work"),
    PropSpec("goalkeeper_saves", 1.5, "fbref_goalkeeper_saves_per90", "fbref_keeper_ready", "defensive_work", ("Goalkeeper",)),
    PropSpec("goalkeeper_saves", 2.5, "fbref_goalkeeper_saves_per90", "fbref_keeper_ready", "defensive_work", ("Goalkeeper",)),
    PropSpec("goalkeeper_saves", 3.5, "fbref_goalkeeper_saves_per90", "fbref_keeper_ready", "defensive_work", ("Goalkeeper",)),
)


def poisson_over_probability(lam: float, line: float) -> float:
    """Return P(X > line) for a Poisson count variable."""
    if pd.isna(lam) or lam <= 0:
        return 0.0
    threshold = int(math.floor(line))
    p_le = 0.0
    term = math.exp(-lam)
    p_le += term
    for k in range(1, threshold + 1):
        term *= lam / k
        p_le += term
    return float(max(0.0, min(1.0, 1.0 - p_le)))


def player_prop_over_probability(lam: float, line: float, market: str) -> float:
    """Return over probability with conservative caps for shot props.

    Season per90 rates plus Poisson can make low player-prop lines look nearly
    certain. In real props, role changes and game state create extra zeros, so
    we cap fragile low-line probabilities rather than letting extreme lambdas
    imply untradeable fair odds.
    """
    probability = poisson_over_probability(lam, line)
    caps = PROP_PROBABILITY_CAPS.get(market)
    if not caps:
        return probability
    threshold = int(math.floor(line))
    cap = caps.get(threshold)
    if cap is None:
        return probability
    return float(np.clip(min(probability, cap), 0.0, 1.0))


def fair_odds(probability: float) -> float | None:
    if pd.isna(probability) or probability <= 0:
        return None
    return round(float(1.0 / probability), 3)


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def load_player_features(fbref_path: Path, tm_path: Path) -> pd.DataFrame:
    fbref = pd.read_csv(fbref_path)
    tm = pd.read_csv(tm_path)
    keep_tm = [
        "player_key",
        "transfermarkt_player_id",
        "age",
        "market_value_eur",
        "recent_minutes",
        "expected_minutes_baseline",
        "yellow_cards_per90",
        "goals_per90",
        "assists_per90",
        "international_caps",
        "has_minimum_prop_minutes",
        "can_model_cards_v0",
        "source_needs_manual_review",
    ]
    keep_tm = [col for col in keep_tm if col in tm.columns]
    players = fbref.merge(tm[keep_tm], on="player_key", how="left", suffixes=("", "_tm"))

    for col in [
        "has_fbref_season_stats",
        "fbref_props_ready",
        "fbref_shots_ready",
        "fbref_defense_ready",
        "fbref_keeper_ready",
        "fbref_xg_available",
        "has_minimum_prop_minutes",
        "can_model_cards_v0",
        "source_needs_manual_review",
    ]:
        if col in players.columns:
            players[col] = _to_bool(players[col])
        else:
            players[col] = False

    return players


def load_group_stage(
    fixtures_path: Path,
    match_features_path: Path,
    match_projections_path: Path = DEFAULT_MATCH_PROJECTIONS,
) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_path, parse_dates=["match_date"])
    features = pd.read_csv(match_features_path, parse_dates=["date"])
    context_cols = [
        "date",
        "home_team",
        "away_team",
        "rating_diff",
        "attack_diff",
        "defense_diff",
        "home_rating",
        "away_rating",
        "home_attack_norm",
        "away_attack_norm",
        "home_defense_norm",
        "away_defense_norm",
        "altitude_m",
        "is_indoor",
        "home_rest_days",
        "away_rest_days",
        "home_travel_km",
        "away_travel_km",
    ]
    context_cols = [col for col in context_cols if col in features.columns]
    merged = fixtures.merge(
        features[context_cols],
        left_on=["match_date", "home_team", "away_team"],
        right_on=["date", "home_team", "away_team"],
        how="left",
    )
    merged = merged.drop(columns=["date"], errors="ignore")

    if match_projections_path.exists():
        projections = pd.read_csv(match_projections_path, parse_dates=["date"])
        projection_cols = [
            "date",
            "home_team",
            "away_team",
            "lambda_home",
            "lambda_away",
            "final_expected_goals",
        ]
        projection_cols = [col for col in projection_cols if col in projections.columns]
        merged = merged.merge(
            projections[projection_cols],
            left_on=["match_date", "home_team", "away_team"],
            right_on=["date", "home_team", "away_team"],
            how="left",
            suffixes=("", "_projection"),
        ).drop(columns=["date"], errors="ignore")

    return merged


def estimate_expected_minutes(row: pd.Series) -> tuple[float, str]:
    position = str(row.get("position_broad", ""))
    fbref_minutes = pd.to_numeric(row.get("fbref_minutes"), errors="coerce")
    tm_expected = pd.to_numeric(row.get("expected_minutes_baseline"), errors="coerce")

    if position == "Goalkeeper" and bool(row.get("fbref_keeper_ready", False)):
        return 90.0, "goalkeeper_if_starter"

    if pd.notna(fbref_minutes):
        if fbref_minutes >= 2500:
            return 82.0, "fbref_minutes_elite_role"
        if fbref_minutes >= 1800:
            return 72.0, "fbref_minutes_regular_role"
        if fbref_minutes >= 1000:
            return 58.0, "fbref_minutes_rotation_role"
        if fbref_minutes >= 450:
            return 38.0, "fbref_minutes_squad_role"

    if pd.notna(tm_expected) and tm_expected > 0:
        return float(min(75.0, max(20.0, tm_expected))), "transfermarkt_recent_minutes"

    return 0.0, "no_minutes_signal"


def data_quality(row: pd.Series, *, source: str) -> tuple[str, float]:
    fbref_minutes = pd.to_numeric(row.get("fbref_minutes"), errors="coerce")
    competitions = str(row.get("fbref_competitions", ""))
    tm_minutes = pd.to_numeric(row.get("recent_minutes"), errors="coerce")

    if source == "transfermarkt_cards":
        if pd.notna(tm_minutes) and tm_minutes >= 1800:
            return "C", 0.55
        if pd.notna(tm_minutes) and tm_minutes >= 450:
            return "D", 0.35
        return "D", 0.20

    has_big5 = any(comp in competitions for comp in BIG5_COMPETITIONS)
    has_secondary = any(comp in competitions for comp in STRONG_SECONDARY_COMPETITIONS)
    minutes_score = min(float(fbref_minutes or 0) / 3000.0, 1.0) if pd.notna(fbref_minutes) else 0.0

    if has_big5 and minutes_score >= 0.30:
        return "A", round(0.70 + 0.25 * minutes_score, 3)
    if (has_big5 or has_secondary) and minutes_score >= 0.30:
        return "B", round(0.55 + 0.25 * minutes_score, 3)
    if pd.notna(fbref_minutes) and fbref_minutes >= 450:
        return "C", round(0.40 + 0.20 * minutes_score, 3)
    return "D", round(0.20 + 0.20 * minutes_score, 3)


def signed_match_strength(match: pd.Series, team_side: str) -> float:
    rating_diff = pd.to_numeric(match.get("rating_diff"), errors="coerce")
    if pd.isna(rating_diff):
        return 0.0
    return float(rating_diff if team_side == "home" else -rating_diff)


def match_adjustment(match: pd.Series, team_side: str, kind: str) -> float:
    strength = signed_match_strength(match, team_side)
    if kind == "attack":
        lambda_col = "lambda_home" if team_side == "home" else "lambda_away"
        team_lambda = pd.to_numeric(match.get(lambda_col), errors="coerce")
        if pd.notna(team_lambda) and team_lambda > 0:
            ratio = float(team_lambda) / ATTACK_LAMBDA_BASELINE
            return float(np.clip(ratio ** ATTACK_ADJUSTMENT_ELASTICITY, 0.65, 1.18))
        return float(np.clip(1.0 + 0.35 * strength, 0.75, 1.25))
    if kind == "defensive_work":
        return float(np.clip(1.0 - 0.35 * strength, 0.75, 1.30))
    return 1.0


def prop_calibration_factor(market: str, position: object) -> float:
    position = str(position or "")
    return SHOT_MARKET_POSITION_FACTORS.get(market, {}).get(position, 1.0)


def tracking_action(probability: float, tier: str, market: str, odds: float | None) -> str:
    if odds is not None and odds < MIN_FAIR_ODDS_TO_REVIEW:
        return "price_too_short_watch_only"
    if odds is not None and odds > 8.00:
        return "longshot_manual_only"
    if tier in {"A", "B"} and probability >= 0.22:
        return "manual_review_if_odds_available"
    if tier in {"A", "B"} and probability >= 0.14:
        return "radar_watch"
    if tier == "C" and probability >= 0.22:
        return "low_confidence_review"
    if market == "yellow_card" and probability >= 0.16 and tier in {"A", "B", "C"}:
        return "card_radar_watch"
    return "no_action"


def load_minutes_projection(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    projection = pd.read_csv(path)
    keep = [
        "match_number",
        "team",
        "player_key",
        "base_expected_minutes",
        "manual_expected_minutes",
        "final_expected_minutes",
        "final_start_probability",
        "lineup_status",
        "minutes_source",
        "confidence",
        "manual_override_applied",
    ]
    keep = [col for col in keep if col in projection.columns]
    projection = projection[keep].copy()
    projection["match_number"] = pd.to_numeric(projection["match_number"], errors="coerce").astype("Int64")
    projection = projection.dropna(subset=["match_number", "team", "player_key"])
    projection = projection.drop_duplicates(["match_number", "team", "player_key"], keep="last")
    return projection.set_index(["match_number", "team", "player_key"], drop=False)


def projected_minutes_for(
    player: pd.Series,
    match_number: object,
    team: str,
    minutes_projection: pd.DataFrame,
) -> dict:
    fallback_minutes, fallback_source = estimate_expected_minutes(player)
    result = {
        "expected_minutes": fallback_minutes,
        "expected_minutes_source": fallback_source,
        "base_expected_minutes": fallback_minutes,
        "manual_expected_minutes": np.nan,
        "final_start_probability": np.nan,
        "lineup_status": "legacy_auto_estimate",
        "minutes_confidence": 0.35,
        "manual_minutes_override_applied": False,
    }

    if minutes_projection.empty:
        return result

    key = (pd.to_numeric(match_number, errors="coerce"), team, player.get("player_key"))
    if pd.isna(key[0]) or key not in minutes_projection.index:
        return result

    row = minutes_projection.loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]

    result.update({
        "expected_minutes": _safe_float(row.get("final_expected_minutes"), fallback_minutes),
        "expected_minutes_source": row.get("minutes_source", fallback_source),
        "base_expected_minutes": _safe_float(row.get("base_expected_minutes"), fallback_minutes),
        "manual_expected_minutes": row.get("manual_expected_minutes", np.nan),
        "final_start_probability": row.get("final_start_probability", np.nan),
        "lineup_status": row.get("lineup_status", "auto_projection"),
        "minutes_confidence": _safe_float(row.get("confidence"), 0.35),
        "manual_minutes_override_applied": bool(row.get("manual_override_applied", False)),
    })
    return result


def _safe_float(value: object, default: float = 0.0) -> float:
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value):
        return default
    return float(value)


def starter_pricing_minutes(player: pd.Series, position: object) -> tuple[float, str]:
    """Estimate minutes conditional on the player starting.

    Player props are reviewed only if the player starts. Club minutes per start
    are therefore a better pricing input than unconditional pre-lineup minutes.
    """
    starts = _safe_float(player.get("fbref_starts"), np.nan)
    minutes = _safe_float(player.get("fbref_minutes"), np.nan)
    if pd.notna(starts) and starts >= 3 and pd.notna(minutes) and minutes > 0:
        raw = (minutes / starts) * STARTER_MINUTES_DISCOUNT
        return float(np.clip(raw, 55.0, STARTER_MINUTES_CAP)), "fbref_minutes_per_start_discounted"

    position = str(position or "")
    if position == "Goalkeeper":
        return 85.0, "starter_position_default"
    if position == "Defender":
        return 80.0, "starter_position_default"
    if position == "Midfielder":
        return 74.0, "starter_position_default"
    if position == "Forward":
        return 74.0, "starter_position_default"
    return 72.0, "starter_position_default"


def pricing_minutes_for_prop(player: pd.Series, minutes: dict) -> tuple[float, str]:
    """Return minutes used to price props.

    The minutes projection is intentionally conservative before lineups. For
    props we compare prices only when the player starts, so probable/locked
    starters should be priced with starter-conditional minutes.
    """
    expected_minutes = _safe_float(minutes.get("expected_minutes"), 0.0)
    lineup_status = str(minutes.get("lineup_status") or "")
    manual_minutes = pd.to_numeric(minutes.get("manual_expected_minutes"), errors="coerce")
    manual_override = bool(minutes.get("manual_minutes_override_applied", False))

    if manual_override and pd.notna(manual_minutes):
        return expected_minutes, "manual_expected_minutes"

    if lineup_status in {"locked_starter", "probable_starter", "confirmed_starter"}:
        starter_minutes, source = starter_pricing_minutes(player, player.get("position_broad"))
        return max(expected_minutes, starter_minutes), source

    return expected_minutes, minutes.get("expected_minutes_source", "expected_minutes")


def paper_tracking_allowed(
    *,
    tier: str,
    position: object,
    expected_minutes: float,
    minutes_confidence: float,
    lineup_status: object,
) -> bool:
    blocked_statuses = {"out", "not_in_squad", "suspended", "injury_doubt"}
    starter_statuses = {"probable_starter", "confirmed_starter", "locked_starter"}
    status = str(lineup_status or "")
    position = str(position or "")

    if status in blocked_statuses:
        return False

    if position == "Goalkeeper":
        return (
            tier in {"A", "B"}
            and status in starter_statuses
            and expected_minutes >= 75
            and minutes_confidence >= 0.50
        )

    return tier in {"A", "B"} and expected_minutes >= 45 and minutes_confidence >= 0.50


def build_player_prop_radar(
    fbref_features_path: Path = DEFAULT_FBREF_FEATURES,
    tm_features_path: Path = DEFAULT_TM_FEATURES,
    fixtures_path: Path = DEFAULT_FIXTURES,
    match_features_path: Path = DEFAULT_MATCH_FEATURES,
    match_projections_path: Path = DEFAULT_MATCH_PROJECTIONS,
    minutes_projection_path: Path = DEFAULT_MINUTES_PROJECTION,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    min_probability: float = MIN_PROBABILITY_TO_OUTPUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    players = load_player_features(fbref_features_path, tm_features_path)
    fixtures = load_group_stage(fixtures_path, match_features_path, match_projections_path)
    minutes_projection = load_minutes_projection(minutes_projection_path)
    rows: list[dict] = []

    for _, match in fixtures.iterrows():
        for team_side, team_col, opp_col in [
            ("home", "home_team", "away_team"),
            ("away", "away_team", "home_team"),
        ]:
            team = match[team_col]
            opponent = match[opp_col]
            team_players = players[players["team_canonical"] == team]
            adjustment_cache = {
                "attack": match_adjustment(match, team_side, "attack"),
                "defensive_work": match_adjustment(match, team_side, "defensive_work"),
            }

            for _, player in team_players.iterrows():
                minutes = projected_minutes_for(player, match.get("match_number"), team, minutes_projection)
                expected_minutes = minutes["expected_minutes"]
                if expected_minutes <= 0:
                    continue
                pricing_minutes, pricing_minutes_source = pricing_minutes_for_prop(player, minutes)

                for spec in PROP_SPECS:
                    if spec.positions and str(player.get("position_broad")) not in spec.positions:
                        continue
                    if not bool(player.get(spec.readiness_col, False)):
                        continue

                    rate = pd.to_numeric(player.get(spec.rate_col), errors="coerce")
                    source_used = spec.source
                    if spec.market == "yellow_card" and (pd.isna(rate) or rate <= 0):
                        rate = pd.to_numeric(player.get("yellow_cards_per90"), errors="coerce")
                        source_used = "transfermarkt_cards"
                        if not bool(player.get("can_model_cards_v0", False)):
                            continue
                    if pd.isna(rate) or rate < 0:
                        continue

                    adjustment = adjustment_cache.get(spec.adjustment_kind, 1.0)
                    calibration = prop_calibration_factor(spec.market, player.get("position_broad"))
                    lam = float(rate) * pricing_minutes / 90.0 * adjustment * calibration
                    probability = player_prop_over_probability(lam, spec.line, spec.market)
                    if probability < min_probability:
                        continue

                    tier, quality_score = data_quality(player, source=source_used)
                    odds = fair_odds(probability)
                    paper_allowed = paper_tracking_allowed(
                        tier=tier,
                        position=player.get("position_broad"),
                        expected_minutes=expected_minutes,
                        minutes_confidence=minutes["minutes_confidence"],
                        lineup_status=minutes["lineup_status"],
                    )
                    rows.append({
                        "match_number": match.get("match_number"),
                        "match_date": match.get("match_date"),
                        "stage": match.get("stage"),
                        "group": match.get("group"),
                        "team": team,
                        "opponent": opponent,
                        "team_side": team_side,
                        "player_key": player.get("player_key"),
                        "player_name": player.get("player_name"),
                        "position_broad": player.get("position_broad"),
                        "club_from_squad": player.get("club_from_squad"),
                        "market": spec.market,
                        "line": spec.line,
                        "selection": f"Over {spec.line}",
                        "model_probability": round(probability, 4),
                        "fair_odds": odds,
                        "event_lambda": round(lam, 4),
                        "event_rate_per90": round(float(rate), 4),
                        "expected_minutes": round(expected_minutes, 1),
                        "pricing_minutes": round(pricing_minutes, 1),
                        "pricing_minutes_source": pricing_minutes_source,
                        "expected_minutes_source": minutes["expected_minutes_source"],
                        "base_expected_minutes": round(minutes["base_expected_minutes"], 1),
                        "manual_expected_minutes": minutes["manual_expected_minutes"],
                        "final_start_probability": minutes["final_start_probability"],
                        "lineup_status": minutes["lineup_status"],
                        "minutes_confidence": minutes["minutes_confidence"],
                        "manual_minutes_override_applied": minutes["manual_minutes_override_applied"],
                        "match_adjustment": round(adjustment, 4),
                        "prop_calibration_factor": round(calibration, 4),
                        "signed_rating_diff": round(signed_match_strength(match, team_side), 4),
                        "data_quality_tier": tier,
                        "data_quality_score": quality_score,
                        "source_used": source_used,
                        "fbref_minutes": player.get("fbref_minutes"),
                        "fbref_competitions": player.get("fbref_competitions"),
                        "recent_minutes": player.get("recent_minutes"),
                        "market_value_eur": player.get("market_value_eur"),
                        "international_caps": player.get("international_caps"),
                        "tracking_action": tracking_action(probability, tier, spec.market, odds),
                        "stake_allowed": False,
                        "paper_tracking_allowed": paper_allowed,
                        "notes": "No odds/no lineups. Manual or paper review only.",
                    })

    radar = pd.DataFrame(rows)
    if radar.empty:
        summary = pd.DataFrame()
    else:
        radar = radar.sort_values(
            ["tracking_action", "data_quality_tier", "model_probability"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
        radar["highlight_fair_price"] = radar["fair_odds"].le(MAX_FAIR_ODDS_TO_HIGHLIGHT)
        summary = (
            radar.groupby(["market", "line", "data_quality_tier"], as_index=False)
            .agg(
                rows=("player_key", "count"),
                players=("player_key", "nunique"),
                avg_probability=("model_probability", "mean"),
                min_fair_odds=("fair_odds", "min"),
                max_fair_odds=("fair_odds", "max"),
                paper_tracking_rows=("paper_tracking_allowed", "sum"),
                manual_review_rows=("tracking_action", lambda s: int(s.isin({"manual_review_if_odds_available", "low_confidence_review"}).sum())),
            )
            .sort_values(["market", "line", "data_quality_tier"])
        )
        summary["avg_probability"] = summary["avg_probability"].round(4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    radar.to_csv(output_path, index=False)
    summary.to_csv(summary_path, index=False)
    return radar, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build WC2026 player-prop radar with fair probabilities.")
    parser.add_argument("--fbref-features", default=str(DEFAULT_FBREF_FEATURES))
    parser.add_argument("--tm-features", default=str(DEFAULT_TM_FEATURES))
    parser.add_argument("--fixtures", default=str(DEFAULT_FIXTURES))
    parser.add_argument("--match-features", default=str(DEFAULT_MATCH_FEATURES))
    parser.add_argument("--match-projections", default=str(DEFAULT_MATCH_PROJECTIONS))
    parser.add_argument("--minutes-projection", default=str(DEFAULT_MINUTES_PROJECTION))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--min-probability", type=float, default=MIN_PROBABILITY_TO_OUTPUT)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    radar, summary = build_player_prop_radar(
        fbref_features_path=Path(args.fbref_features),
        tm_features_path=Path(args.tm_features),
        fixtures_path=Path(args.fixtures),
        match_features_path=Path(args.match_features),
        match_projections_path=Path(args.match_projections),
        minutes_projection_path=Path(args.minutes_projection),
        output_path=Path(args.output),
        summary_path=Path(args.summary),
        min_probability=args.min_probability,
    )
    print("\nPLAYER PROP RADAR")
    print("-" * 72)
    print(f"Rows: {len(radar)}")
    if not radar.empty:
        print(f"Players: {radar['player_key'].nunique()}")
        print(f"Matches: {radar['match_number'].nunique()}")
        print("\nTop manual review candidates:")
        cols = [
            "match_number", "team", "opponent", "player_name", "market", "line",
            "model_probability", "fair_odds", "data_quality_tier", "expected_minutes",
        ]
        print(
            radar[radar["tracking_action"].eq("manual_review_if_odds_available")]
            .sort_values(["model_probability", "data_quality_score"], ascending=False)
            [cols]
            .head(20)
            .to_string(index=False)
        )
    print(f"\nSaved: {args.output}")
    print(f"Saved summary: {args.summary}")


if __name__ == "__main__":
    main()
