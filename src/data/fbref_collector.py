"""
Collect FBref player season tables with soccerdata.

This collector intentionally starts with season-level tables. Match logs belong
in player_match_stats once we decide to scrape them; season aggregates are stored
separately under data/raw/fbref/player_season.

Usage:
  python -m src.data.fbref_collector
  python -m src.data.fbref_collector --seasons 2024-25 2025-26 --stat-types standard shooting
  python -m src.data.fbref_collector --leagues "NED-Eredivisie" --seasons 2025-26 --stat-types standard --no-overwrite
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]
DEFAULT_SEASONS = ["2023-24", "2024-25", "2025-26"]
DEFAULT_STAT_TYPES = ["standard", "shooting", "playing_time", "keeper", "misc"]
DEFAULT_OUTPUT_DIR = Path("data/raw/fbref/player_season")


def safe_path_part(value: str) -> str:
    """Make a readable path segment for league/season values."""
    value = str(value).strip()
    value = re.sub(r"[\\/]+", "-", value)
    value = re.sub(r"\s+", "_", value)
    return value


def output_path_for(
    output_dir: Path,
    *,
    league: str,
    season: str,
    stat_type: str,
    flat_output: bool,
) -> Path:
    if flat_output:
        return output_dir / f"fbref_player_season_{stat_type}.parquet"
    return (
        output_dir
        / f"league={safe_path_part(league)}"
        / f"season={safe_path_part(season)}"
        / f"stat_type={safe_path_part(stat_type)}.parquet"
    )


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten soccerdata/FBref MultiIndex output into stable snake-ish names."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(item) for item in col if str(item) and str(item) != "nan")
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            for col in df.columns
        ]
    else:
        df.columns = [
            str(col).strip().lower().replace(" ", "_").replace("-", "_")
            for col in df.columns
        ]
    return df.reset_index()


def collect_fbref_player_season_stats(
    *,
    leagues: list[str] = DEFAULT_LEAGUES,
    seasons: list[str] = DEFAULT_SEASONS,
    stat_types: list[str] = DEFAULT_STAT_TYPES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    no_cache: bool = False,
    no_overwrite: bool = False,
    flat_output: bool = False,
) -> None:
    os.environ.setdefault("SOCCERDATA_DIR", str(Path(".soccerdata").resolve()))
    try:
        import soccerdata as sd
    except ImportError as exc:
        raise ImportError(
            "Falta soccerdata. Instala dependencias con `pip install -r requirements.txt` "
            "antes de ejecutar el collector FBref."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    if flat_output:
        fbref = sd.FBref(leagues=leagues, seasons=seasons, no_cache=no_cache)
        for stat_type in stat_types:
            output_path = output_path_for(
                output_dir,
                league="combined",
                season="combined",
                stat_type=stat_type,
                flat_output=True,
            )
            if no_overwrite and output_path.exists():
                log.info("Skipping existing %s", output_path)
                continue
            log.info("Collecting FBref player season stats: %s", stat_type)
            df = fbref.read_player_season_stats(stat_type=stat_type)
            _save_table(df, output_path, stat_type)
        return

    for league in leagues:
        for season in seasons:
            fbref = sd.FBref(leagues=[league], seasons=[season], no_cache=no_cache)
            for stat_type in stat_types:
                output_path = output_path_for(
                    output_dir,
                    league=league,
                    season=season,
                    stat_type=stat_type,
                    flat_output=False,
                )
                if no_overwrite and output_path.exists():
                    log.info("Skipping existing %s", output_path)
                    continue
                log.info("Collecting FBref player season stats: league=%s season=%s stat=%s", league, season, stat_type)
                df = fbref.read_player_season_stats(stat_type=stat_type)
                _save_table(df, output_path, stat_type)


def _save_table(df: pd.DataFrame, output_path: Path, stat_type: str) -> None:
    df = flatten_columns(df)
    df["source"] = "fbref"
    df["raw_source_table"] = stat_type
    df["collected_at"] = datetime.now(timezone.utc).isoformat()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    log.info("Saved %d rows to %s", len(df), output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect FBref player season stats with soccerdata.")
    parser.add_argument("--leagues", nargs="+", default=DEFAULT_LEAGUES)
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--stat-types", nargs="+", default=DEFAULT_STAT_TYPES)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-cache", action="store_true", help="Bypass soccerdata cache.")
    parser.add_argument("--no-overwrite", action="store_true", help="Skip existing output parquet files.")
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Write legacy fbref_player_season_{stat_type}.parquet files instead of league/season partitions.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args()
    collect_fbref_player_season_stats(
        leagues=args.leagues,
        seasons=args.seasons,
        stat_types=args.stat_types,
        output_dir=Path(args.output_dir),
        no_cache=args.no_cache,
        no_overwrite=args.no_overwrite,
        flat_output=args.flat_output,
    )


if __name__ == "__main__":
    main()
