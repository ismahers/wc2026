"""
Build the WC2026 player master tables from the official squad CSV.

This layer keeps the official squad list immutable and creates the stable keys
that later collectors can use to attach FBref, Transfermarkt and StatsBomb data.
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.data.normalize_squads import (
    DEFAULT_INPUT,
    assert_valid_official_squads,
    load_official_squads,
    make_team_id,
)
from src.data.team_names import canonicalize


log = logging.getLogger(__name__)

DEFAULT_MASTER_OUTPUT = "data/processed/players_master.csv"
DEFAULT_SOURCE_MATCHES_OUTPUT = "data/processed/player_source_matches.csv"
EXPECTED_TEAMS = 48
EXPECTED_PLAYERS = 1248
EXPECTED_PLAYERS_PER_TEAM = 26

PLAYER_MASTER_COLUMNS = [
    "player_key",
    "team_id",
    "team_canonical",
    "player_name",
    "player_name_normalized",
    "club_from_squad",
    "club_normalized",
    "position_broad",
    "fbref_url",
    "transfermarkt_url",
    "statsbomb_player_id",
    "needs_manual_review",
    "source",
]

PLAYER_SOURCE_MATCH_COLUMNS = [
    "player_key",
    "team_id",
    "team_canonical",
    "player_name",
    "player_name_normalized",
    "club_from_squad",
    "club_normalized",
    "position_broad",
    "fbref_name",
    "fbref_url",
    "transfermarkt_name",
    "transfermarkt_url",
    "statsbomb_player_id",
    "match_confidence",
    "match_method",
    "needs_manual_review",
    "review_notes",
]


def normalize_identity_text(value: str) -> str:
    """Normalize identity text for matching: ascii, lowercase, compact spaces."""
    if value is None or pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = text.replace("'", " ").replace("`", " ").replace("’", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def slugify_identity(value: str) -> str:
    normalized = normalize_identity_text(value)
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def make_player_key(team_name: str, player_name: str) -> str:
    """
    Create a stable, readable player key scoped to the national team.

    It intentionally includes the team so common names in different countries
    do not collide. Source-specific IDs live in player_source_matches.csv.
    """
    team_slug = slugify_identity(canonicalize(team_name))
    player_slug = slugify_identity(player_name)
    return f"{team_slug}_{player_slug}"


def _ensure_unique_player_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    duplicates = df.duplicated("player_key", keep=False)
    if not duplicates.any():
        return df

    dupes = df[duplicates].copy()
    for idx, row in dupes.iterrows():
        club_slug = slugify_identity(row["club_from_squad"])
        position_slug = slugify_identity(row["position_broad"])
        suffix = "_".join(part for part in [club_slug, position_slug] if part)
        new_key = f"{row['player_key']}_{suffix}" if suffix else row["player_key"]
        df.at[idx, "player_key"] = f"{new_key}_{idx}"

    if df["player_key"].duplicated().any():
        raise ValueError("Could not create unique player_key values")
    return df


def build_players_master(squads: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in squads.iterrows():
        team = canonicalize(row["team_canonical"])
        player_name = row["player_name"]
        club = row["club"]
        rows.append({
            "player_key": make_player_key(team, player_name),
            "team_id": make_team_id(team),
            "team_canonical": team,
            "player_name": player_name,
            "player_name_normalized": normalize_identity_text(player_name),
            "club_from_squad": club,
            "club_normalized": normalize_identity_text(club),
            "position_broad": row["position_broad"],
            "fbref_url": pd.NA,
            "transfermarkt_url": pd.NA,
            "statsbomb_player_id": pd.NA,
            "needs_manual_review": False,
            "source": "wc2026_official_squad_corrected",
        })

    master = pd.DataFrame(rows)
    master = _ensure_unique_player_keys(master)
    return master[PLAYER_MASTER_COLUMNS].sort_values(["team_canonical", "player_name"]).reset_index(drop=True)


def build_player_source_matches(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in master.iterrows():
        rows.append({
            "player_key": row["player_key"],
            "team_id": row["team_id"],
            "team_canonical": row["team_canonical"],
            "player_name": row["player_name"],
            "player_name_normalized": row["player_name_normalized"],
            "club_from_squad": row["club_from_squad"],
            "club_normalized": row["club_normalized"],
            "position_broad": row["position_broad"],
            "fbref_name": pd.NA,
            "fbref_url": pd.NA,
            "transfermarkt_name": pd.NA,
            "transfermarkt_url": pd.NA,
            "statsbomb_player_id": pd.NA,
            "match_confidence": 0.0,
            "match_method": "unmatched",
            "needs_manual_review": True,
            "review_notes": pd.NA,
        })
    return pd.DataFrame(rows, columns=PLAYER_SOURCE_MATCH_COLUMNS)


def validate_players_master(master: pd.DataFrame) -> dict[str, int]:
    team_counts = master.groupby("team_canonical")["player_key"].count()

    return {
        "rows": len(master),
        "teams": int(master["team_canonical"].nunique()),
        "duplicate_player_keys": int(master["player_key"].duplicated().sum()),
        "empty_player_keys": int(master["player_key"].eq("").sum()),
        "empty_normalized_names": int(master["player_name_normalized"].eq("").sum()),
        "teams_not_26_players": int((team_counts != EXPECTED_PLAYERS_PER_TEAM).sum()),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build WC2026 player master identity tables.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--master-output", default=DEFAULT_MASTER_OUTPUT)
    parser.add_argument("--source-matches-output", default=DEFAULT_SOURCE_MATCHES_OUTPUT)
    parser.add_argument("--no-validate-squads", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args()

    squads = load_official_squads(args.input)
    if not args.no_validate_squads:
        assert_valid_official_squads(squads)

    master = build_players_master(squads)
    report = validate_players_master(master)
    if (
        report["rows"] != EXPECTED_PLAYERS
        or report["teams"] != EXPECTED_TEAMS
        or report["duplicate_player_keys"]
        or report["empty_player_keys"]
        or report["empty_normalized_names"]
        or report["teams_not_26_players"]
    ):
        raise ValueError(f"Player master validation failed: {report}")

    source_matches = build_player_source_matches(master)

    outputs = [
        (master, Path(args.master_output), "players master"),
        (source_matches, Path(args.source_matches_output), "player source matches"),
    ]
    for frame, output, label in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        log.info("Saved %d %s rows to %s", len(frame), label, output)
    log.info("Validation report: %s", report)


if __name__ == "__main__":
    main()
