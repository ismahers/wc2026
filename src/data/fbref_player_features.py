"""
Build WC2026 FBref player features from normalized player-season stats.

This script is intentionally conservative with identity matching:
  1. exact normalized player name + normalized club/team
  2. exact normalized player name only when the squad name is unique

Anything else stays unmatched for manual review.

Usage:
  python -m src.data.fbref_player_features
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.player_master import normalize_identity_text

log = logging.getLogger(__name__)

DEFAULT_PLAYER_SEASON = Path("data/processed/player_season_stats.parquet")
DEFAULT_PLAYERS_MASTER = Path("data/processed/players_master.csv")
DEFAULT_OUTPUT = Path("data/processed/fbref_player_features_wc2026.csv")
DEFAULT_REVIEW_OUTPUT = Path("data/processed/fbref_player_match_review.csv")
DEFAULT_SUMMARY_OUTPUT = Path("data/processed/fbref_player_feature_summary.csv")
MIN_FBREF_MINUTES = 450

RATE_STATS = [
    "goals",
    "assists",
    "non_penalty_goals",
    "shots",
    "shots_on_target",
    "xg",
    "npxg",
    "xa",
    "key_passes",
    "tackles",
    "tackles_won",
    "interceptions",
    "fouls_committed",
    "fouls_drawn",
    "yellow_cards",
    "red_cards",
    "goalkeeper_saves",
    "goals_conceded",
    "psxg",
]


def _club_key(value: object) -> str:
    text = normalize_identity_text(value)
    text = re.sub(r"\b[a-z]{3}\b$", "", text).strip()
    text = re.sub(r"\b(fc|cf|sc|afc|ac|cd|ca|club|de|the)\b", " ", text)
    text = re.sub(r"^\d+\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    variants = {
        "bayer 04 leverkusen": "bayer leverkusen",
        "olympique marseille": "olympique marseille",
        "olympique de marseille": "olympique marseille",
        "paris saint germain": "psg",
        "paris st germain": "psg",
        "internazionale": "inter",
        "inter milan": "inter",
        "manchester utd": "manchester united",
        "man utd": "manchester united",
        "manchester city": "manchester city",
        "borussia monchengladbach": "borussia monchengladbach",
        "borussia m gladbach": "borussia monchengladbach",
    }
    return variants.get(text, text)


def _read_inputs(player_season_path: Path, players_master_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not player_season_path.exists():
        raise FileNotFoundError(
            f"No existe {player_season_path}. Ejecuta primero `python -m src.data.normalize_fbref`."
        )
    if not players_master_path.exists():
        raise FileNotFoundError(f"No existe {players_master_path}.")

    stats = pd.read_parquet(player_season_path)
    master = pd.read_csv(players_master_path)
    return stats, master


def attach_player_keys(stats: pd.DataFrame, master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = stats.copy()
    master = master.copy()

    if "player_key" in stats.columns:
        stats = stats.drop(columns=["player_key"])

    stats["fbref_row_id"] = np.arange(len(stats))
    stats["normalized_player_name"] = stats["normalized_player_name"].fillna(stats["player_name"].map(normalize_identity_text))
    stats["fbref_club_key"] = stats["team"].map(_club_key)

    master["master_name_key"] = master["player_name_normalized"].fillna(master["player_name"].map(normalize_identity_text))
    master["master_club_key"] = master["club_from_squad"].map(_club_key)

    exact = stats.merge(
        master[["player_key", "team_canonical", "position_broad", "master_name_key", "master_club_key"]],
        left_on=["normalized_player_name", "fbref_club_key"],
        right_on=["master_name_key", "master_club_key"],
        how="left",
    )
    exact["match_method"] = np.where(exact["player_key"].notna(), "exact_name_club", pd.NA)

    unmatched = exact[exact["player_key"].isna()].drop(
        columns=["player_key", "team_canonical", "position_broad", "master_name_key", "master_club_key", "match_method"]
    )
    matched = exact[exact["player_key"].notna()].copy()

    name_counts = master.groupby("master_name_key")["player_key"].nunique().reset_index(name="n_master_matches")
    unique_master = master.merge(name_counts[name_counts["n_master_matches"] == 1], on="master_name_key", how="inner")
    name_only = unmatched.merge(
        unique_master[["player_key", "team_canonical", "position_broad", "master_name_key"]],
        left_on="normalized_player_name",
        right_on="master_name_key",
        how="left",
    )
    name_only["match_method"] = np.where(name_only["player_key"].notna(), "exact_unique_name", pd.NA)

    combined = pd.concat([matched, name_only], ignore_index=True, sort=False)
    combined["needs_manual_review"] = combined["player_key"].isna()

    review = combined[combined["needs_manual_review"]].copy()
    review = review[[
        "fbref_row_id",
        "player_name",
        "team",
        "competition",
        "season",
        "normalized_player_name",
        "fbref_club_key",
        "minutes",
    ]].sort_values(["player_name", "season", "team"])

    return combined, review


def _numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    return df


def _sum_available(series: pd.Series) -> float:
    return pd.to_numeric(series, errors="coerce").sum(min_count=1)


def build_features(matched_stats: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    stats = matched_stats[matched_stats["player_key"].notna()].copy()
    stats = _numeric(stats, ["minutes", "starts", "clean_sheets", *RATE_STATS])

    grouped = stats.groupby("player_key", as_index=False).agg(
        fbref_rows=("fbref_row_id", "count"),
        fbref_competitions=("competition", lambda s: ", ".join(sorted(set(str(x) for x in s.dropna()))[:8])),
        fbref_seasons=("season", lambda s: ", ".join(sorted(set(str(x) for x in s.dropna())))),
        fbref_teams=("team", lambda s: ", ".join(sorted(set(str(x) for x in s.dropna()))[:8])),
        fbref_minutes=("minutes", _sum_available),
        fbref_starts=("starts", _sum_available),
        fbref_clean_sheets=("clean_sheets", _sum_available),
        **{stat: (stat, _sum_available) for stat in RATE_STATS},
    )

    nineties = grouped["fbref_minutes"].replace(0, np.nan) / 90
    for stat in RATE_STATS:
        grouped[f"fbref_{stat}_per90"] = grouped[stat] / nineties

    grouped["fbref_starts_rate"] = grouped["fbref_starts"] / nineties
    grouped["fbref_clean_sheets_per90"] = grouped["fbref_clean_sheets"] / nineties
    grouped["fbref_has_min_minutes"] = grouped["fbref_minutes"] >= MIN_FBREF_MINUTES

    output = master.merge(grouped, on="player_key", how="left")
    output["has_fbref_season_stats"] = output["fbref_minutes"].fillna(0) > 0
    output["fbref_props_ready"] = output["fbref_has_min_minutes"].fillna(False)
    is_goalkeeper = output["position_broad"].astype(str).str.casefold().eq("goalkeeper")
    output["fbref_shots_ready"] = output["fbref_props_ready"] & ~is_goalkeeper & output["fbref_shots_per90"].notna()
    output["fbref_defense_ready"] = output["fbref_props_ready"] & ~is_goalkeeper & output["fbref_tackles_per90"].notna()
    output["fbref_keeper_ready"] = output["fbref_props_ready"] & is_goalkeeper & output["fbref_goalkeeper_saves_per90"].notna()
    output["fbref_xg_available"] = output["fbref_xg_per90"].notna()
    return output.sort_values(["team_canonical", "position_broad", "player_name"]).reset_index(drop=True)


def build_summary(features: pd.DataFrame) -> pd.DataFrame:
    summary = features.groupby("team_canonical", as_index=False).agg(
        players=("player_key", "count"),
        matched_fbref=("has_fbref_season_stats", "sum"),
        props_ready=("fbref_props_ready", "sum"),
        shots_ready=("fbref_shots_ready", "sum"),
        defense_ready=("fbref_defense_ready", "sum"),
        keeper_ready=("fbref_keeper_ready", "sum"),
        total_fbref_minutes=("fbref_minutes", "sum"),
    )
    return summary.sort_values(["props_ready", "matched_fbref"], ascending=False)


def build_fbref_player_features(
    player_season_path: Path = DEFAULT_PLAYER_SEASON,
    players_master_path: Path = DEFAULT_PLAYERS_MASTER,
    output_path: Path = DEFAULT_OUTPUT,
    review_output_path: Path = DEFAULT_REVIEW_OUTPUT,
    summary_output_path: Path = DEFAULT_SUMMARY_OUTPUT,
) -> pd.DataFrame:
    stats, master = _read_inputs(player_season_path, players_master_path)
    matched_stats, review = attach_player_keys(stats, master)
    features = build_features(matched_stats, master)
    summary = build_summary(features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    review.to_csv(review_output_path, index=False)
    summary.to_csv(summary_output_path, index=False)

    print("\nFBREF PLAYER FEATURES WC2026")
    print("-" * 72)
    print(f"Jugadores:             {len(features)}")
    print(f"Con FBref stats:       {int(features['has_fbref_season_stats'].sum())}")
    print(f"Con >= {MIN_FBREF_MINUTES} min FBref: {int(features['fbref_props_ready'].sum())}")
    print(f"Pendientes review raw: {len(review)}")
    print("\nPeor cobertura por seleccion:")
    print(summary.sort_values(["props_ready", "matched_fbref"]).head(10).to_string(index=False))
    log.info("Guardado %s", output_path)
    log.info("Guardado %s", review_output_path)
    log.info("Guardado %s", summary_output_path)
    return features


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build FBref WC2026 player prop features.")
    parser.add_argument("--player-season", default=str(DEFAULT_PLAYER_SEASON))
    parser.add_argument("--players-master", default=str(DEFAULT_PLAYERS_MASTER))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--review-output", default=str(DEFAULT_REVIEW_OUTPUT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_OUTPUT))
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args()
    build_fbref_player_features(
        player_season_path=Path(args.player_season),
        players_master_path=Path(args.players_master),
        output_path=Path(args.output),
        review_output_path=Path(args.review_output),
        summary_output_path=Path(args.summary_output),
    )


if __name__ == "__main__":
    main()
