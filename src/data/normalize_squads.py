"""
Normalize official World Cup 2026 squad CSV into the project players table.

Input contract:
    data/raw/squads_wc2026_final_official_corrected.csv

Output contract:
    data/processed/players_wc2026.csv
    data/processed/teams_wc2026.csv
    data/processed/squad_summary_wc2026.csv
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
DEFAULT_TEAMS_OUTPUT = "data/processed/teams_wc2026.csv"
DEFAULT_SQUAD_SUMMARY_OUTPUT = "data/processed/squad_summary_wc2026.csv"
DEFAULT_RATINGS_PATH = "data/processed/team_ratings.csv"

CONFEDERATION_MAP = {
    "Spain": "UEFA", "Germany": "UEFA", "France": "UEFA", "England": "UEFA",
    "Portugal": "UEFA", "Netherlands": "UEFA", "Belgium": "UEFA", "Croatia": "UEFA",
    "Poland": "UEFA", "Switzerland": "UEFA", "Austria": "UEFA", "Sweden": "UEFA",
    "Norway": "UEFA", "Scotland": "UEFA", "Turkey": "UEFA", "Czech Republic": "UEFA",
    "Bosnia and Herzegovina": "UEFA",
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL",
    "United States": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Panama": "CONCACAF", "Haiti": "CONCACAF", "Curacao": "CONCACAF",
    "Japan": "AFC", "South Korea": "AFC", "Iran": "AFC", "Australia": "AFC",
    "Saudi Arabia": "AFC", "Qatar": "AFC", "Jordan": "AFC", "Iraq": "AFC",
    "Uzbekistan": "AFC",
    "Morocco": "CAF", "Senegal": "CAF", "Ivory Coast": "CAF", "Ghana": "CAF",
    "Egypt": "CAF", "Tunisia": "CAF", "Algeria": "CAF", "South Africa": "CAF",
    "DR Congo": "CAF", "Cape Verde": "CAF",
    "New Zealand": "OFC",
}


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


def _latest_elo_by_team(ratings_path: str | Path = DEFAULT_RATINGS_PATH) -> dict[str, float]:
    path = Path(ratings_path)
    if not path.exists():
        return {}
    ratings = pd.read_csv(path, parse_dates=["rating_date"])
    required = {"team_name", "rating_date", "elo_rating"}
    if not required.issubset(ratings.columns):
        return {}
    ratings["team_name"] = ratings["team_name"].map(canonicalize)
    ratings = ratings.dropna(subset=["team_name", "rating_date", "elo_rating"])
    latest = ratings.sort_values("rating_date").groupby("team_name").tail(1)
    return dict(zip(latest["team_name"], latest["elo_rating"].astype(float)))


def normalize_squads_to_teams(
    df: pd.DataFrame,
    *,
    ratings_path: str | Path = DEFAULT_RATINGS_PATH,
) -> pd.DataFrame:
    latest_elo = _latest_elo_by_team(ratings_path)
    rows = []
    for team_name in sorted(df["team_canonical"].unique()):
        rows.append({
            "team_id": make_team_id(team_name),
            "team_name": team_name,
            "country_code": pd.NA,
            "confederation": CONFEDERATION_MAP.get(team_name, pd.NA),
            "fifa_rank": pd.NA,
            "elo_rating": latest_elo.get(team_name, pd.NA),
        })
    teams = pd.DataFrame(rows)
    return teams.reindex(columns=table_columns("teams")).reset_index(drop=True)


def build_squad_summary(df: pd.DataFrame) -> pd.DataFrame:
    position_counts = (
        df.pivot_table(
            index="team_canonical",
            columns="position_broad",
            values="player_name",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename(columns={
            "Goalkeeper": "goalkeepers",
            "Defender": "defenders",
            "Midfielder": "midfielders",
            "Forward": "forwards",
        })
    )
    for col in ["goalkeepers", "defenders", "midfielders", "forwards"]:
        if col not in position_counts.columns:
            position_counts[col] = 0

    club_counts = (
        df.assign(club=df["club"].replace("", pd.NA))
        .groupby("team_canonical")["club"]
        .nunique(dropna=True)
        .rename("unique_clubs")
        .reset_index()
    )
    summary = position_counts.merge(club_counts, on="team_canonical", how="left")
    summary["team_id"] = summary["team_canonical"].map(make_team_id)
    summary["team_name"] = summary["team_canonical"]
    summary["squad_size"] = (
        summary["goalkeepers"] + summary["defenders"] + summary["midfielders"] + summary["forwards"]
    )
    keep = [
        "team_id", "team_name", "squad_size", "goalkeepers",
        "defenders", "midfielders", "forwards", "unique_clubs",
    ]
    return summary[keep].sort_values("team_name").reset_index(drop=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize official WC2026 squads into players table.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--teams-output", default=DEFAULT_TEAMS_OUTPUT)
    parser.add_argument("--squad-summary-output", default=DEFAULT_SQUAD_SUMMARY_OUTPUT)
    parser.add_argument("--ratings-path", default=DEFAULT_RATINGS_PATH)
    parser.add_argument("--no-validate", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args()

    squads = load_official_squads(args.input)
    if not args.no_validate:
        assert_valid_official_squads(squads)

    players = normalize_squads_to_players(squads)
    teams = normalize_squads_to_teams(squads, ratings_path=args.ratings_path)
    squad_summary = build_squad_summary(squads)

    outputs = [
        (players, Path(args.output), "players"),
        (teams, Path(args.teams_output), "teams"),
        (squad_summary, Path(args.squad_summary_output), "squad summary"),
    ]
    for frame, output, label in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        log.info("Saved %d %s rows to %s", len(frame), label, output)


if __name__ == "__main__":
    main()
