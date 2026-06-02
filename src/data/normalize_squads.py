"""
Normalize official World Cup 2026 squad CSV into the project players table.

Input contract:
    data/raw/squads_wc2026_final_official_corrected.csv

Output contract:
    data/processed/players_wc2026.csv
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
from pathlib import Path

import pandas as pd

from src.data.schemas import table_columns
from src.data.team_names import CANONICAL_TEAMS, canonicalize, unknown_names


log = logging.getLogger(__name__)

VALID_POSITIONS = {"Goalkeeper", "Defender", "Midfielder", "Forward"}
SQUAD_COLUMNS = ["team_canonical", "player_name", "club", "position_broad"]
DEFAULT_INPUT = "data/raw/squads_wc2026_final_official_corrected.csv"
DEFAULT_OUTPUT = "data/processed/players_wc2026.csv"


def make_team_id(team_name: str) -> str:
    canonical = canonicalize(team_name)
    slug = re.sub(r"[^a-z0-9]+", "_", str(canonical).lower()).strip("_")
    return f"team_{slug}"


def make_player_id(team_name: str, player_name: str) -> str:
    key = f"{canonicalize(team_name)}|{player_name}".lower().encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()[:12]
    return f"wc2026_{digest}"


def load_official_squads(path: str | Path = DEFAULT_INPUT) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    expected = set(SQUAD_COLUMNS)
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in squad CSV: {sorted(missing)}")

    df = df[SQUAD_COLUMNS].copy()
    df["team_canonical"] = df["team_canonical"].map(canonicalize)
    df["player_name"] = df["player_name"].str.strip()
    df["club"] = df["club"].str.strip()
    df["position_broad"] = df["position_broad"].str.strip()
    return df


def validate_official_squads(df: pd.DataFrame) -> dict[str, object]:
    counts = df.groupby("team_canonical").size()
    duplicate_mask = df.duplicated(["team_canonical", "player_name"], keep=False)
    invalid_positions = sorted(set(df["position_broad"]) - VALID_POSITIONS)
    missing_teams = sorted(set(CANONICAL_TEAMS) - set(counts.index))
    extra_teams = sorted(set(counts.index) - set(CANONICAL_TEAMS))
    teams_not_26 = counts[counts.ne(26)].to_dict()

    report = {
        "rows": len(df),
        "teams": int(df["team_canonical"].nunique()),
        "teams_not_26": teams_not_26,
        "missing_teams": missing_teams,
        "extra_teams": extra_teams,
        "unknown_teams": unknown_names(df["team_canonical"].unique()),
        "invalid_positions": invalid_positions,
        "duplicate_team_player_rows": int(duplicate_mask.sum()),
        "empty_clubs": int(df["club"].eq("").sum()),
    }
    return report


def assert_valid_official_squads(df: pd.DataFrame) -> None:
    report = validate_official_squads(df)
    failures = {
        key: value
        for key, value in report.items()
        if key in {
            "teams_not_26",
            "missing_teams",
            "extra_teams",
            "unknown_teams",
            "invalid_positions",
            "duplicate_team_player_rows",
            "empty_clubs",
        }
        and value
    }
    if failures:
        raise ValueError(f"Official squad validation failed: {failures}")


def normalize_squads_to_players(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        team_name = row["team_canonical"]
        player_name = row["player_name"]
        rows.append({
            "player_id": make_player_id(team_name, player_name),
            "player_name": player_name,
            "team_id": make_team_id(team_name),
            "club": row["club"],
            "position": row["position_broad"],
            "birth_date": pd.NA,
            "preferred_foot": pd.NA,
            "height_cm": pd.NA,
            "market_value_eur": pd.NA,
        })

    players = pd.DataFrame(rows)
    return players.reindex(columns=table_columns("players")).reset_index(drop=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize official WC2026 squads into players table.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--no-validate", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args()

    squads = load_official_squads(args.input)
    if not args.no_validate:
        assert_valid_official_squads(squads)

    players = normalize_squads_to_players(squads)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    players.to_csv(output, index=False)
    log.info("Saved %d players to %s", len(players), output)


if __name__ == "__main__":
    main()
