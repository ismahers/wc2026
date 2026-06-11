"""
Build WC2026 player-level features for prop markets.

This module turns the official squad master plus Transfermarkt matches into a
player table that can feed future player-prop models. Transfermarkt is useful
for minutes, goals, assists, cards and international experience; it does not
contain shots, shots on target, tackles or saves, so those markets are flagged
as requiring another source.

Usage:
  python -m src.data.player_prop_features
  python -m src.data.player_prop_features --seasons 24/25 25/26
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_PLAYERS_MASTER = Path("data/processed/players_master.csv")
DEFAULT_TM_PROFILES = Path("data/processed/transfermarkt_player_profiles.csv")
DEFAULT_SOURCE_MATCHES = Path("data/processed/player_source_matches.csv")
DEFAULT_PERFORMANCES = Path("data/transfermarkt/player_performances.csv")
DEFAULT_NATIONAL = Path("data/transfermarkt/player_national_performances.csv")
DEFAULT_OUTPUT = Path("data/processed/player_prop_features_wc2026.csv")
DEFAULT_SUMMARY_OUTPUT = Path("data/processed/player_prop_readiness_summary.csv")
DEFAULT_SEASONS = ["23/24", "24/25", "25/26"]
MIN_PROP_MINUTES = 450


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_per90(stat: pd.Series, minutes: pd.Series) -> pd.Series:
    minutes = _to_numeric(minutes)
    stat = _to_numeric(stat).fillna(0)
    return np.where(minutes > 0, stat / minutes * 90, np.nan)


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}")
    return pd.read_csv(path, **kwargs)


def load_players_master(path: Path) -> pd.DataFrame:
    master = _read_csv(path)
    required = {
        "player_key",
        "team_id",
        "team_canonical",
        "player_name",
        "club_from_squad",
        "position_broad",
    }
    missing = required.difference(master.columns)
    if missing:
        raise ValueError(f"{path} no contiene columnas requeridas: {sorted(missing)}")
    log.info("Players master: %d jugadores, %d selecciones", len(master), master["team_canonical"].nunique())
    return master


def load_transfermarkt_profiles(path: Path) -> pd.DataFrame:
    profiles = _read_csv(path)
    cols = [
        "player_key",
        "transfermarkt_player_id",
        "transfermarkt_name",
        "transfermarkt_url",
        "date_of_birth",
        "age",
        "height_cm",
        "foot",
        "position_detail",
        "market_value_eur",
        "current_club_transfermarkt",
    ]
    available = [col for col in cols if col in profiles.columns]
    profiles = profiles[available].copy()
    profiles["transfermarkt_player_id"] = _to_numeric(profiles["transfermarkt_player_id"]).astype("Int64")
    profiles = profiles.drop_duplicates("player_key", keep="first")
    log.info("Transfermarkt profiles matched: %d jugadores", len(profiles))
    return profiles


def load_source_match_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.warning("%s no encontrado; se omite metadata de matching", path)
        return pd.DataFrame(columns=["player_key"])

    matches = pd.read_csv(path, usecols=lambda c: c in {
        "player_key",
        "match_confidence",
        "match_method",
        "needs_manual_review",
        "review_notes",
    })
    matches = matches.rename(columns={
        "needs_manual_review": "source_needs_manual_review",
        "review_notes": "source_review_notes",
    })
    return matches.drop_duplicates("player_key", keep="first")


def aggregate_recent_performances(path: Path, seasons: list[str]) -> pd.DataFrame:
    usecols = [
        "player_id",
        "season_name",
        "team_id",
        "team_name",
        "nb_in_group",
        "nb_on_pitch",
        "goals",
        "assists",
        "subed_in",
        "subed_out",
        "yellow_cards",
        "second_yellow_cards",
        "direct_red_cards",
        "penalty_goals",
        "minutes_played",
        "goals_conceded",
        "clean_sheets",
    ]
    log.info("Cargando %s para temporadas %s", path, seasons)
    perf = _read_csv(path, low_memory=False, usecols=usecols)
    perf = perf[perf["season_name"].isin(seasons)].copy()

    numeric_cols = [
        "player_id",
        "team_id",
        "nb_in_group",
        "nb_on_pitch",
        "goals",
        "assists",
        "subed_in",
        "subed_out",
        "yellow_cards",
        "second_yellow_cards",
        "direct_red_cards",
        "penalty_goals",
        "minutes_played",
        "goals_conceded",
        "clean_sheets",
    ]
    for col in numeric_cols:
        perf[col] = _to_numeric(perf[col]).fillna(0)

    perf["estimated_starts"] = (perf["nb_on_pitch"] - perf["subed_in"]).clip(lower=0)
    perf["red_cards"] = perf["second_yellow_cards"] + perf["direct_red_cards"]

    grouped = perf.groupby("player_id", as_index=False).agg(
        performance_rows=("player_id", "size"),
        seasons_with_performance=("season_name", "nunique"),
        recent_teams=("team_id", "nunique"),
        recent_minutes=("minutes_played", "sum"),
        recent_squad_apps=("nb_in_group", "sum"),
        recent_appearances=("nb_on_pitch", "sum"),
        recent_starts=("estimated_starts", "sum"),
        recent_sub_ins=("subed_in", "sum"),
        recent_goals=("goals", "sum"),
        recent_assists=("assists", "sum"),
        recent_yellow_cards=("yellow_cards", "sum"),
        recent_red_cards=("red_cards", "sum"),
        recent_penalty_goals=("penalty_goals", "sum"),
        recent_goals_conceded=("goals_conceded", "sum"),
        recent_clean_sheets=("clean_sheets", "sum"),
    )
    grouped["transfermarkt_player_id"] = _to_numeric(grouped["player_id"]).astype("Int64")
    grouped = grouped.drop(columns=["player_id"])

    grouped["minutes_per_appearance"] = np.where(
        grouped["recent_appearances"] > 0,
        grouped["recent_minutes"] / grouped["recent_appearances"],
        np.nan,
    )
    grouped["start_rate_recent"] = np.where(
        grouped["recent_squad_apps"] > 0,
        grouped["recent_starts"] / grouped["recent_squad_apps"],
        np.nan,
    )
    grouped["appearance_rate_recent"] = np.where(
        grouped["recent_squad_apps"] > 0,
        grouped["recent_appearances"] / grouped["recent_squad_apps"],
        np.nan,
    )
    grouped["expected_minutes_baseline"] = np.where(
        grouped["recent_squad_apps"] > 0,
        grouped["recent_minutes"] / grouped["recent_squad_apps"],
        np.nan,
    )
    grouped["expected_minutes_baseline"] = grouped["expected_minutes_baseline"].clip(lower=0, upper=90)

    for stat in [
        "recent_goals",
        "recent_assists",
        "recent_yellow_cards",
        "recent_red_cards",
        "recent_penalty_goals",
        "recent_goals_conceded",
        "recent_clean_sheets",
    ]:
        grouped[stat.replace("recent_", "") + "_per90"] = _safe_per90(grouped[stat], grouped["recent_minutes"])

    log.info("Recent performances agregadas: %d jugadores", len(grouped))
    return grouped


def aggregate_national_performances(path: Path) -> pd.DataFrame:
    national = _read_csv(path, low_memory=False, usecols=["player_id", "matches", "goals"])
    for col in ["player_id", "matches", "goals"]:
        national[col] = _to_numeric(national[col]).fillna(0)

    grouped = national.groupby("player_id", as_index=False).agg(
        international_caps=("matches", "sum"),
        international_goals=("goals", "sum"),
    )
    grouped["transfermarkt_player_id"] = _to_numeric(grouped["player_id"]).astype("Int64")
    grouped = grouped.drop(columns=["player_id"])
    grouped["international_goals_per_cap"] = np.where(
        grouped["international_caps"] > 0,
        grouped["international_goals"] / grouped["international_caps"],
        np.nan,
    )
    log.info("National performances agregadas: %d jugadores", len(grouped))
    return grouped


def build_player_prop_features(
    players_master_path: Path = DEFAULT_PLAYERS_MASTER,
    tm_profiles_path: Path = DEFAULT_TM_PROFILES,
    source_matches_path: Path = DEFAULT_SOURCE_MATCHES,
    performances_path: Path = DEFAULT_PERFORMANCES,
    national_path: Path = DEFAULT_NATIONAL,
    output_path: Path = DEFAULT_OUTPUT,
    summary_output_path: Path = DEFAULT_SUMMARY_OUTPUT,
    seasons: list[str] = DEFAULT_SEASONS,
) -> pd.DataFrame:
    master = load_players_master(players_master_path)
    profiles = load_transfermarkt_profiles(tm_profiles_path)
    source_matches = load_source_match_metadata(source_matches_path)
    performances = aggregate_recent_performances(performances_path, seasons)
    national = aggregate_national_performances(national_path)

    df = master.merge(profiles, on="player_key", how="left")
    df = df.merge(source_matches, on="player_key", how="left")
    df = df.merge(performances, on="transfermarkt_player_id", how="left")
    df = df.merge(national, on="transfermarkt_player_id", how="left")

    if "transfermarkt_url_x" in df.columns or "transfermarkt_url_y" in df.columns:
        url_master = df["transfermarkt_url_x"] if "transfermarkt_url_x" in df.columns else pd.Series(index=df.index, dtype=object)
        url_profile = df["transfermarkt_url_y"] if "transfermarkt_url_y" in df.columns else pd.Series(index=df.index, dtype=object)
        df["transfermarkt_url"] = url_profile.combine_first(url_master)
        df = df.drop(columns=[col for col in ["transfermarkt_url_x", "transfermarkt_url_y"] if col in df.columns])

    df["has_transfermarkt_profile"] = df["transfermarkt_player_id"].notna()
    df["has_recent_performance"] = df["recent_minutes"].fillna(0) > 0
    df["has_minimum_prop_minutes"] = df["recent_minutes"].fillna(0) >= MIN_PROP_MINUTES
    df["is_goalkeeper"] = df["position_broad"].astype(str).str.casefold().eq("goalkeeper")

    df["can_model_cards_v0"] = df["has_minimum_prop_minutes"]
    df["can_model_goal_or_assist_v0"] = df["has_minimum_prop_minutes"] & ~df["is_goalkeeper"]
    df["can_model_goalkeeper_conceded_proxy_v0"] = df["has_minimum_prop_minutes"] & df["is_goalkeeper"]

    # Transfermarkt does not expose these event-level prop stats.
    df["has_shots_data"] = False
    df["has_shots_on_target_data"] = False
    df["has_tackles_data"] = False
    df["has_saves_data"] = False
    df["needs_fbref_for_shots"] = ~df["is_goalkeeper"]
    df["needs_fbref_for_tackles"] = True
    df["needs_external_for_goalkeeper_saves"] = df["is_goalkeeper"]

    if "source_needs_manual_review" not in df.columns:
        df["source_needs_manual_review"] = False
    df["source_needs_manual_review"] = df["source_needs_manual_review"].fillna(False).astype(bool)

    first_cols = [
        "player_key",
        "team_id",
        "team_canonical",
        "player_name",
        "club_from_squad",
        "position_broad",
        "transfermarkt_player_id",
        "transfermarkt_name",
        "transfermarkt_url",
        "match_confidence",
        "match_method",
        "source_needs_manual_review",
        "age",
        "height_cm",
        "foot",
        "position_detail",
        "market_value_eur",
        "current_club_transfermarkt",
    ]
    other_cols = [col for col in df.columns if col not in first_cols]
    df = df[first_cols + other_cols]
    df = df.sort_values(["team_canonical", "position_broad", "player_name"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    summary = build_readiness_summary(df)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output_path, index=False)

    validate_output(df)
    log.info("Guardado %s (%d filas)", output_path, len(df))
    log.info("Guardado %s", summary_output_path)
    print_summary(df, summary)
    return df


def build_readiness_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("team_canonical", as_index=False).agg(
        players=("player_key", "count"),
        tm_profiles=("has_transfermarkt_profile", "sum"),
        recent_performance=("has_recent_performance", "sum"),
        prop_minutes=("has_minimum_prop_minutes", "sum"),
        cards_ready=("can_model_cards_v0", "sum"),
        goal_assist_ready=("can_model_goal_or_assist_v0", "sum"),
        gk_proxy_ready=("can_model_goalkeeper_conceded_proxy_v0", "sum"),
        manual_reviews=("source_needs_manual_review", "sum"),
        avg_expected_minutes=("expected_minutes_baseline", "mean"),
        total_recent_minutes=("recent_minutes", "sum"),
        avg_international_caps=("international_caps", "mean"),
    )
    numeric_cols = grouped.select_dtypes(include=["float"]).columns
    grouped[numeric_cols] = grouped[numeric_cols].round(3)
    return grouped.sort_values(["prop_minutes", "recent_performance", "tm_profiles"], ascending=False)


def validate_output(df: pd.DataFrame) -> None:
    report = {
        "rows": int(len(df)),
        "teams": int(df["team_canonical"].nunique()),
        "duplicate_player_keys": int(df["player_key"].duplicated().sum()),
        "empty_player_keys": int(df["player_key"].fillna("").eq("").sum()),
    }
    if report["rows"] != 1248 or report["teams"] != 48 or report["duplicate_player_keys"] or report["empty_player_keys"]:
        raise ValueError(f"Player prop feature validation failed: {report}")


def print_summary(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    print("\nPLAYER PROP FEATURES WC2026")
    print("-" * 72)
    print(f"Jugadores:                  {len(df)}")
    print(f"Selecciones:                {df['team_canonical'].nunique()}")
    print(f"Con perfil Transfermarkt:   {int(df['has_transfermarkt_profile'].sum())}")
    print(f"Con minutos recientes:      {int(df['has_recent_performance'].sum())}")
    print(f"Con >= {MIN_PROP_MINUTES} minutos:       {int(df['has_minimum_prop_minutes'].sum())}")
    print(f"Cards v0 ready:             {int(df['can_model_cards_v0'].sum())}")
    print(f"Goal/assist v0 ready:       {int(df['can_model_goal_or_assist_v0'].sum())}")
    print(f"Porteros proxy ready:       {int(df['can_model_goalkeeper_conceded_proxy_v0'].sum())}")
    print("\nPeor cobertura por seleccion:")
    cols = ["team_canonical", "players", "tm_profiles", "recent_performance", "prop_minutes", "manual_reviews"]
    print(summary.sort_values(["prop_minutes", "recent_performance", "tm_profiles"], ascending=True)[cols].head(10).to_string(index=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build WC2026 player-level prop feature table.")
    parser.add_argument("--players-master", default=str(DEFAULT_PLAYERS_MASTER))
    parser.add_argument("--tm-profiles", default=str(DEFAULT_TM_PROFILES))
    parser.add_argument("--source-matches", default=str(DEFAULT_SOURCE_MATCHES))
    parser.add_argument("--performances", default=str(DEFAULT_PERFORMANCES))
    parser.add_argument("--national", default=str(DEFAULT_NATIONAL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_OUTPUT))
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args()
    build_player_prop_features(
        players_master_path=Path(args.players_master),
        tm_profiles_path=Path(args.tm_profiles),
        source_matches_path=Path(args.source_matches),
        performances_path=Path(args.performances),
        national_path=Path(args.national),
        output_path=Path(args.output),
        summary_output_path=Path(args.summary_output),
        seasons=args.seasons,
    )


if __name__ == "__main__":
    main()
