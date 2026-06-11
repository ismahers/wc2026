"""
Normalize raw FBref player-season tables into a common season-level schema.

Input:
  data/raw/fbref/player_season/fbref_player_season_{stat_type}.parquet
  data/raw/fbref/player_season/league={league}/season={season}/stat_type={stat_type}.parquet

Output:
  data/processed/player_season_stats.parquet
  data/processed/player_season_stats.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.player_master import normalize_identity_text

log = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/raw/fbref/player_season")
DEFAULT_OUTPUT = Path("data/processed/player_season_stats.parquet")
DEFAULT_CSV_OUTPUT = Path("data/processed/player_season_stats.csv")
DEFAULT_STAT_TYPES = ["standard", "shooting", "playing_time", "keeper", "misc"]

KEY_CANDIDATES = {
    "league": ["league", "competition"],
    "season": ["season"],
    "team": ["team", "squad"],
    "player": ["player", "player_name"],
    "nation": ["nation", "nationality"],
    "position": ["pos", "position"],
    "age": ["age"],
}

COMMON_COLUMNS = [
    "source",
    "source_player_id",
    "player_key",
    "player_name",
    "normalized_player_name",
    "team",
    "normalized_team",
    "competition",
    "season",
    "nation",
    "position",
    "age",
    "minutes",
    "starts",
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
    "clean_sheets",
    "psxg",
    "raw_source_table",
    "collected_at",
]


def _load_raw_table(raw_dir: Path, stat_type: str) -> pd.DataFrame:
    paths = []
    flat_path = raw_dir / f"fbref_player_season_{stat_type}.parquet"
    if flat_path.exists():
        paths.append(flat_path)
    paths.extend(sorted(raw_dir.glob(f"league=*/season=*/stat_type={stat_type}.parquet")))

    if not paths:
        log.warning("No hay parquet FBref para stat_type=%s en %s", stat_type, raw_dir)
        return pd.DataFrame()

    frames = []
    for path in paths:
        df_part = pd.read_parquet(path)
        df_part["raw_file"] = str(path)
        frames.append(df_part)

    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    df.columns = [str(col).strip().lower().replace(" ", "_").replace("-", "_") for col in df.columns]
    df["raw_source_table"] = stat_type
    return df


def _first_existing_name(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _canonical_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for canonical, candidates in KEY_CANDIDATES.items():
        col = _first_existing_name(out, candidates)
        if col and col != canonical:
            out[canonical] = out[col]
        elif not col and canonical not in out.columns:
            out[canonical] = pd.NA
    return out


def _prefix_stat_columns(df: pd.DataFrame, stat_type: str) -> pd.DataFrame:
    key_cols = set(KEY_CANDIDATES.keys()) | {"source", "raw_source_table", "collected_at"}
    renamed = {}
    for col in df.columns:
        if col in key_cols or col.startswith(f"{stat_type}_"):
            continue
        renamed[col] = f"{stat_type}_{col}"
    return df.rename(columns=renamed)


def _merge_raw_tables(raw_dir: Path, stat_types: list[str]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    merge_keys = list(KEY_CANDIDATES.keys())

    for stat_type in stat_types:
        table = _load_raw_table(raw_dir, stat_type)
        if table.empty:
            continue
        table = _canonical_key_columns(table)
        table = _prefix_stat_columns(table, stat_type)
        keep_cols = [col for col in table.columns if col in merge_keys or col.startswith(f"{stat_type}_") or col in {"source", "collected_at"}]
        table = table[keep_cols].drop_duplicates()

        if merged is None:
            merged = table
        else:
            merged = merged.merge(table, on=merge_keys, how="outer", suffixes=("", f"_{stat_type}"))

    if merged is None:
        raise FileNotFoundError(f"No hay tablas FBref player-season en {raw_dir}")
    return merged


def _series(df: pd.DataFrame, candidates: list[str], default=np.nan) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series(default, index=df.index)


def _numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def _age_years(series: pd.Series) -> pd.Series:
    """Convert FBref ages like 23 or 20-159 into numeric years."""
    text = series.astype("string")
    years = text.str.extract(r"^(\d+)", expand=False)
    return pd.to_numeric(years, errors="coerce")


def normalize_fbref_player_season(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_path: Path = DEFAULT_OUTPUT,
    csv_output_path: Path | None = DEFAULT_CSV_OUTPUT,
    stat_types: list[str] = DEFAULT_STAT_TYPES,
) -> pd.DataFrame:
    raw = _merge_raw_tables(raw_dir, stat_types)

    out = pd.DataFrame(index=raw.index)
    out["source"] = "fbref"
    out["source_player_id"] = pd.NA
    out["player_key"] = pd.NA
    out["player_name"] = _series(raw, ["player", "standard_player", "shooting_player"])
    out["normalized_player_name"] = out["player_name"].map(normalize_identity_text)
    out["team"] = _series(raw, ["team", "standard_team", "squad", "standard_squad"])
    out["normalized_team"] = out["team"].map(normalize_identity_text)
    out["competition"] = _series(raw, ["league", "competition"])
    out["season"] = _series(raw, ["season"])
    out["nation"] = _series(raw, ["nation", "standard_nation"])
    out["position"] = _series(raw, ["position", "standard_pos", "pos"])
    out["age"] = _age_years(_series(raw, ["age", "standard_age"]))

    column_map = {
        "minutes": ["playing_time_min", "standard_playing_time_min", "keeper_playing_time_min", "standard_min", "min"],
        "starts": ["playing_time_starts_starts", "standard_playing_time_starts", "keeper_playing_time_starts", "standard_starts", "starts"],
        "goals": ["standard_performance_gls", "shooting_standard_gls", "standard_gls", "gls"],
        "assists": ["standard_performance_ast", "standard_ast", "ast"],
        "non_penalty_goals": ["standard_performance_g_pk", "standard_npkg", "standard_npk", "shooting_npkg", "npg"],
        "shots": ["shooting_standard_sh", "shooting_sh", "standard_sh", "sh"],
        "shots_on_target": ["shooting_standard_sot", "shooting_sot", "standard_sot", "sot"],
        "xg": ["standard_xg", "shooting_xg", "xg"],
        "npxg": ["standard_npxg", "shooting_npxg", "npxg"],
        "xa": ["standard_xag", "standard_xa", "xag", "xa"],
        "key_passes": ["misc_kp", "standard_kp", "kp"],
        "tackles": ["misc_performance_tklw", "misc_tkl", "misc_tklw", "tkl"],
        "tackles_won": ["misc_performance_tklw", "misc_tklw", "tklw"],
        "interceptions": ["misc_performance_int", "misc_int", "int"],
        "fouls_committed": ["misc_performance_fls", "misc_fls", "fls"],
        "fouls_drawn": ["misc_performance_fld", "misc_fld", "fld"],
        "yellow_cards": ["misc_performance_crdy", "standard_performance_crdy", "misc_crdy", "standard_crdy", "crdy"],
        "red_cards": ["misc_performance_crdr", "standard_performance_crdr", "misc_crdr", "standard_crdr", "crdr"],
        "goalkeeper_saves": ["keeper_performance_saves", "keeper_saves", "saves"],
        "goals_conceded": ["keeper_performance_ga", "keeper_ga", "ga"],
        "clean_sheets": ["keeper_performance_cs", "keeper_cs", "cs"],
        "psxg": ["keeper_psxg", "psxg"],
    }
    for target_col, candidates in column_map.items():
        out[target_col] = _numeric(_series(raw, candidates))

    out["raw_source_table"] = "merged_fbref_player_season"
    out["collected_at"] = _series(raw, ["collected_at", "standard_collected_at"])
    out = out[COMMON_COLUMNS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    if csv_output_path:
        csv_output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(csv_output_path, index=False)

    log.info("Guardado %s (%d filas)", output_path, len(out))
    if csv_output_path:
        log.info("Guardado %s", csv_output_path)
    print(out[["player_name", "team", "competition", "season", "minutes", "shots", "shots_on_target", "xg"]].head().to_string(index=False))
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize raw FBref player season tables.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--csv-output", default=str(DEFAULT_CSV_OUTPUT))
    parser.add_argument("--stat-types", nargs="+", default=DEFAULT_STAT_TYPES)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args()
    normalize_fbref_player_season(
        raw_dir=Path(args.raw_dir),
        output_path=Path(args.output),
        csv_output_path=Path(args.csv_output) if args.csv_output else None,
        stat_types=args.stat_types,
    )


if __name__ == "__main__":
    main()
