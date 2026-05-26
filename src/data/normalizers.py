"""
Provider normalizers for the normalized event-market tables.

The functions in this module accept already-fetched provider payloads. Network
access, caching, API keys, and scraping politeness should live in thin loaders
above this layer.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import re
import unicodedata
from typing import Any

import pandas as pd

from .schemas import create_empty_table, table_columns


SHOT_ON_TARGET_OUTCOMES = {"Goal", "Saved", "Saved to Post"}
SHARP_BOOKMAKERS = {"pinnacle", "betfair_ex_uk", "betfair_ex_eu", "matchbook"}
SELECTION_LINE_RE = re.compile(r"^(?P<name>.+?)\s*\((?P<side>Over|Under)\s+(?P<line>\d+(?:\.\d+)?)\)$")


def _as_source_id(source: str, entity: str, raw_id: Any, fallback: str = "") -> str:
    if raw_id not in (None, ""):
        return f"{source}:{entity}:{raw_id}"
    clean = unicodedata.normalize("NFKD", str(fallback))
    clean = clean.encode("ascii", "ignore").decode("ascii")
    clean = re.sub(r"[^a-zA-Z0-9]+", "-", clean.strip().lower()).strip("-")
    return f"{source}:{entity}:{clean}"


def _stable_id(*parts: Any) -> str:
    raw = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _list_get(values: Any, idx: int) -> Any:
    if values is None:
        return None
    try:
        return values[idx]
    except (IndexError, TypeError):
        return None


def _event_minute(event: dict[str, Any]) -> float | None:
    minute = event.get("minute")
    second = event.get("second") or 0
    if minute is None:
        return None
    return float(minute) + float(second) / 60.0


def _team_id(team: dict[str, Any], source: str = "statsbomb") -> str:
    return _as_source_id(source, "team", team.get("id"), team.get("name", "unknown"))


def _player_id(player: dict[str, Any], source: str = "statsbomb") -> str:
    return _as_source_id(source, "player", player.get("id"), player.get("name", "unknown"))


def _parse_selection_and_line(selection: Any, line: Any = None) -> tuple[Any, float | None]:
    if line not in (None, ""):
        return selection, float(line)
    if not isinstance(selection, str):
        return selection, None
    match = SELECTION_LINE_RE.match(selection.strip())
    if not match:
        return selection, None
    return f"{match.group('name')} {match.group('side')}", float(match.group("line"))


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "starter", "starting"}
    return bool(value)


def _rate_or_value(value: Any, numerator: Any, denominator: Any) -> Any:
    if value not in (None, "") and not pd.isna(value):
        return value
    if numerator in (None, "") or denominator in (None, ""):
        return None
    if pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0:
        return None
    return float(numerator) / float(denominator)


def _add_opening_closing_flags(odds: pd.DataFrame) -> pd.DataFrame:
    if odds.empty:
        return odds

    odds = odds.copy()
    odds["quote_time"] = pd.to_datetime(odds["quote_time"], errors="coerce", utc=True)
    group_cols = ["match_id", "bookmaker", "market", "selection", "line"]
    odds["is_opening"] = False
    odds["is_closing"] = False

    for _, group in odds.dropna(subset=["quote_time"]).groupby(group_cols, dropna=False):
        odds.loc[group["quote_time"].idxmin(), "is_opening"] = True
        odds.loc[group["quote_time"].idxmax(), "is_closing"] = True

    return odds


def normalize_statsbomb_events(
    events: list[dict[str, Any]],
    match_id: str | int,
    *,
    source: str = "statsbomb",
) -> dict[str, pd.DataFrame]:
    """
    Normalize StatsBomb event JSON into players, lineups, and player stats.

    Parameters
    ----------
    events:
        Raw list from `data/events/{match_id}.json`.
    match_id:
        Provider match id.

    Returns
    -------
    dict[str, DataFrame]
        DataFrames keyed by normalized table name.
    """
    normalized_match_id = _as_source_id(source, "match", match_id)
    players: dict[str, dict[str, Any]] = {}
    lineups: dict[tuple[str, str, str], dict[str, Any]] = {}
    stats = defaultdict(lambda: {
        "match_id": normalized_match_id,
        "team_id": None,
        "player_id": None,
        "minutes_played": None,
        "goals": 0.0,
        "assists": 0.0,
        "shots": 0.0,
        "shots_on_target": 0.0,
        "xg": 0.0,
        "xa": None,
        "key_passes": 0.0,
        "corners_taken": 0.0,
        "fouls_committed": 0.0,
        "fouls_won": 0.0,
        "yellow_cards": 0.0,
        "red_cards": 0.0,
    })

    def ensure_player(player: dict[str, Any], team: dict[str, Any]) -> tuple[str, str]:
        player_id = _player_id(player, source)
        team_id = _team_id(team, source)
        players[player_id] = {
            "player_id": player_id,
            "player_name": player.get("name"),
            "team_id": team_id,
            "club": None,
            "position": players.get(player_id, {}).get("position"),
            "birth_date": None,
            "preferred_foot": None,
            "height_cm": None,
            "market_value_eur": None,
        }
        key = (normalized_match_id, team_id, player_id)
        row = stats[key]
        row["team_id"] = team_id
        row["player_id"] = player_id
        return player_id, team_id

    for event in events:
        event_type = event.get("type", {}).get("name")
        team = event.get("team") or {}
        player = event.get("player") or {}

        if event_type == "Starting XI":
            team_id = _team_id(team, source)
            for lineup_player in event.get("tactics", {}).get("lineup", []):
                raw_player = lineup_player.get("player") or {}
                player_id = _player_id(raw_player, source)
                position = (lineup_player.get("position") or {}).get("name")
                players[player_id] = {
                    "player_id": player_id,
                    "player_name": raw_player.get("name"),
                    "team_id": team_id,
                    "club": None,
                    "position": position,
                    "birth_date": None,
                    "preferred_foot": None,
                    "height_cm": None,
                    "market_value_eur": None,
                }
                key = (normalized_match_id, team_id, player_id)
                lineups[key] = {
                    "match_id": normalized_match_id,
                    "team_id": team_id,
                    "player_id": player_id,
                    "is_starter": True,
                    "position": position,
                    "shirt_number": lineup_player.get("jersey_number"),
                    "minute_on": 0.0,
                    "minute_off": None,
                    "minutes_played": None,
                }
                stats[key]["team_id"] = team_id
                stats[key]["player_id"] = player_id
            continue

        if not player or not team:
            continue

        player_id, team_id = ensure_player(player, team)
        key = (normalized_match_id, team_id, player_id)
        stat_row = stats[key]

        if event_type == "Substitution":
            minute = _event_minute(event)
            if key not in lineups:
                lineups[key] = {
                    "match_id": normalized_match_id,
                    "team_id": team_id,
                    "player_id": player_id,
                    "is_starter": True,
                    "position": None,
                    "shirt_number": None,
                    "minute_on": 0.0,
                    "minute_off": minute,
                    "minutes_played": minute,
                }
            else:
                lineups[key]["minute_off"] = minute
                lineups[key]["minutes_played"] = minute

            replacement = event.get("substitution", {}).get("replacement") or {}
            if replacement:
                repl_id = _player_id(replacement, source)
                players[repl_id] = {
                    "player_id": repl_id,
                    "player_name": replacement.get("name"),
                    "team_id": team_id,
                    "club": None,
                    "position": None,
                    "birth_date": None,
                    "preferred_foot": None,
                    "height_cm": None,
                    "market_value_eur": None,
                }
                repl_key = (normalized_match_id, team_id, repl_id)
                lineups[repl_key] = {
                    "match_id": normalized_match_id,
                    "team_id": team_id,
                    "player_id": repl_id,
                    "is_starter": False,
                    "position": None,
                    "shirt_number": None,
                    "minute_on": minute,
                    "minute_off": None,
                    "minutes_played": None,
                }
                stats[repl_key]["team_id"] = team_id
                stats[repl_key]["player_id"] = repl_id
            continue

        if event_type == "Shot":
            shot = event.get("shot", {})
            outcome = (shot.get("outcome") or {}).get("name")
            stat_row["shots"] += 1
            stat_row["xg"] += float(shot.get("statsbomb_xg") or 0.0)
            if outcome in SHOT_ON_TARGET_OUTCOMES:
                stat_row["shots_on_target"] += 1
            if outcome == "Goal":
                stat_row["goals"] += 1
            continue

        if event_type == "Pass":
            pass_data = event.get("pass", {})
            if (pass_data.get("type") or {}).get("name") == "Corner":
                stat_row["corners_taken"] += 1
            if pass_data.get("shot_assist"):
                stat_row["key_passes"] += 1
            if pass_data.get("goal_assist"):
                stat_row["assists"] += 1
            continue

        if event_type == "Foul Committed":
            stat_row["fouls_committed"] += 1
            card = (event.get("foul_committed", {}).get("card") or {}).get("name")
            if card == "Yellow Card":
                stat_row["yellow_cards"] += 1
            elif card in {"Red Card", "Second Yellow"}:
                stat_row["red_cards"] += 1
            continue

        if event_type == "Foul Won":
            stat_row["fouls_won"] += 1
            continue

        if event_type == "Bad Behaviour":
            card = (event.get("bad_behaviour", {}).get("card") or {}).get("name")
            if card == "Yellow Card":
                stat_row["yellow_cards"] += 1
            elif card in {"Red Card", "Second Yellow"}:
                stat_row["red_cards"] += 1

    players_df = pd.DataFrame(players.values(), columns=table_columns("players"))
    lineups_df = pd.DataFrame(lineups.values(), columns=table_columns("lineups"))
    stats_df = pd.DataFrame(stats.values(), columns=table_columns("player_match_stats"))

    return {
        "players": players_df if not players_df.empty else create_empty_table("players"),
        "lineups": lineups_df if not lineups_df.empty else create_empty_table("lineups"),
        "player_match_stats": stats_df if not stats_df.empty else create_empty_table("player_match_stats"),
    }


FBREF_PLAYER_STAT_MAP = {
    "Player": "player_name",
    "Pos": "position",
    "Min": "minutes_played",
    "Gls": "goals",
    "Ast": "assists",
    "Sh": "shots",
    "SoT": "shots_on_target",
    "xG": "xg",
    "xA": "xa",
    "KP": "key_passes",
    "Fls": "fouls_committed",
    "Fld": "fouls_won",
    "CrdY": "yellow_cards",
    "CrdR": "red_cards",
}


def normalize_fbref_player_match_log(
    df: pd.DataFrame,
    *,
    match_id: str,
    team_id: str,
    source: str = "fbref",
) -> pd.DataFrame:
    """
    Normalize one FBref player match-log table into player_match_stats.

    FBref tables vary by competition and stat group. Unknown columns are ignored;
    missing target columns are left null except identifiers.
    """
    rows = []
    for _, raw in df.iterrows():
        player_name = raw.get("Player")
        if pd.isna(player_name):
            continue
        player_id = _as_source_id(source, "player", raw.get("Player ID"), str(player_name))
        row = {column: None for column in table_columns("player_match_stats")}
        row.update({
            "match_id": match_id,
            "team_id": team_id,
            "player_id": player_id,
        })
        for provider_col, target_col in FBREF_PLAYER_STAT_MAP.items():
            if provider_col in raw.index and target_col in row:
                row[target_col] = raw.get(provider_col)
        rows.append(row)

    if not rows:
        return create_empty_table("player_match_stats")
    return pd.DataFrame(rows, columns=table_columns("player_match_stats"))


def normalize_api_football_player_stats(
    payload: list[dict[str, Any]],
    *,
    fixture_id: str | int,
    source: str = "api_football",
) -> pd.DataFrame:
    """
    Normalize API-Football `/fixtures/players` response into player_match_stats.

    Expected input is the decoded `response` list, grouped by team.
    """
    match_id = _as_source_id(source, "match", fixture_id)
    rows = []

    for team_block in payload:
        team = team_block.get("team") or {}
        team_id = _as_source_id(source, "team", team.get("id"), team.get("name", "unknown"))
        for player_block in team_block.get("players", []):
            player = player_block.get("player") or {}
            player_id = _as_source_id(source, "player", player.get("id"), player.get("name", "unknown"))
            for stat in player_block.get("statistics", []):
                games = stat.get("games") or {}
                shots = stat.get("shots") or {}
                goals = stat.get("goals") or {}
                passes = stat.get("passes") or {}
                fouls = stat.get("fouls") or {}
                cards = stat.get("cards") or {}
                rows.append({
                    "match_id": match_id,
                    "team_id": team_id,
                    "player_id": player_id,
                    "minutes_played": games.get("minutes"),
                    "goals": goals.get("total"),
                    "assists": goals.get("assists"),
                    "shots": shots.get("total"),
                    "shots_on_target": shots.get("on"),
                    "xg": None,
                    "xa": None,
                    "key_passes": passes.get("key"),
                    "corners_taken": None,
                    "fouls_committed": fouls.get("committed"),
                    "fouls_won": fouls.get("drawn"),
                    "yellow_cards": cards.get("yellow"),
                    "red_cards": cards.get("red"),
                })

    if not rows:
        return create_empty_table("player_match_stats")
    return pd.DataFrame(rows, columns=table_columns("player_match_stats"))


def normalize_transfermarkt_lineups(
    df: pd.DataFrame,
    *,
    source: str = "transfermarkt",
) -> dict[str, pd.DataFrame]:
    """
    Normalize Transfermarkt dataset lineup rows into players and lineups.

    Expected columns include `game_id`, `player_id`, `player_name`, `position`,
    `starting_lineup`, and `minutes_played`. The normalizer also accepts common
    variants such as `club_id`, `team_id`, `market_value_in_eur`, and `number`.
    """
    players = {}
    lineups = []

    for _, raw in df.iterrows():
        raw_match_id = raw.get("game_id", raw.get("match_id"))
        raw_player_id = raw.get("player_id")
        player_name = raw.get("player_name", raw.get("name"))
        if pd.isna(raw_match_id) or pd.isna(raw_player_id):
            continue

        raw_team_id = raw.get("team_id", raw.get("club_id"))
        match_id = _as_source_id(source, "match", raw_match_id)
        team_id = _as_source_id(source, "team", raw_team_id) if not pd.isna(raw_team_id) else None
        player_id = _as_source_id(source, "player", raw_player_id, player_name)
        position = raw.get("position")

        players[player_id] = {
            "player_id": player_id,
            "player_name": player_name,
            "team_id": team_id,
            "club": raw.get("club_name", raw.get("current_club_name")),
            "position": position,
            "birth_date": raw.get("date_of_birth", raw.get("birth_date")),
            "preferred_foot": raw.get("foot", raw.get("preferred_foot")),
            "height_cm": raw.get("height_in_cm", raw.get("height_cm")),
            "market_value_eur": raw.get("market_value_in_eur", raw.get("market_value_eur")),
        }

        is_starter = raw.get("starting_lineup", raw.get("is_starter"))
        minute_on = raw.get("minute_on")
        minutes_played = raw.get("minutes_played")
        is_starter_bool = _to_bool(is_starter)
        if minute_on is None and is_starter_bool:
            minute_on = 0.0

        lineups.append({
            "match_id": match_id,
            "team_id": team_id,
            "player_id": player_id,
            "is_starter": is_starter_bool,
            "position": position,
            "shirt_number": raw.get("shirt_number", raw.get("number")),
            "minute_on": minute_on,
            "minute_off": raw.get("minute_off"),
            "minutes_played": minutes_played,
        })

    players_df = pd.DataFrame(players.values(), columns=table_columns("players"))
    lineups_df = pd.DataFrame(lineups, columns=table_columns("lineups"))

    return {
        "players": players_df if not players_df.empty else create_empty_table("players"),
        "lineups": lineups_df if not lineups_df.empty else create_empty_table("lineups"),
    }


def normalize_manual_venues(
    df: pd.DataFrame,
    *,
    source: str = "wc2026",
) -> pd.DataFrame:
    """
    Normalize the manual WC 2026 venues CSV into the `venues` table.

    Accepts raw columns such as `stadium_name`, `lat`, `lon`, and `roof_type`.
    Timezones should preferably be IANA names, e.g. `America/Mexico_City`.
    """
    rows = []
    for _, raw in df.iterrows():
        venue_id = raw.get("venue_id")
        stadium_name = raw.get("stadium_name", raw.get("venue_name"))
        roof_type = raw.get("roof_type")
        indoor = raw.get("indoor")
        if indoor is None or pd.isna(indoor):
            indoor = str(roof_type).strip().lower() in {"fixed", "retractable", "closed"}

        rows.append({
            "venue_id": _as_source_id(source, "venue", venue_id, stadium_name),
            "venue_name": stadium_name,
            "city": raw.get("city"),
            "country": raw.get("country"),
            "latitude": raw.get("lat", raw.get("latitude")),
            "longitude": raw.get("lon", raw.get("longitude")),
            "altitude_m": raw.get("altitude_m"),
            "indoor": _to_bool(indoor),
            "roof_type": roof_type,
            "surface": raw.get("surface"),
            "timezone": raw.get("timezone"),
        })

    if not rows:
        return create_empty_table("venues")
    return pd.DataFrame(rows, columns=table_columns("venues"))


def normalize_open_meteo_hourly(
    payload: dict[str, Any],
    *,
    venue_id: str,
    source: str = "open_meteo",
    is_forecast: bool = False,
) -> pd.DataFrame:
    """
    Normalize an Open-Meteo hourly JSON response into `weather_hourly`.

    Expected shape:
    `{ "latitude": ..., "longitude": ..., "hourly": { "time": [...],
    "temperature_2m": [...], ... } }`.
    """
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or hourly.get("datetime") or []
    rows = []

    for idx, dt in enumerate(times):
        rows.append({
            "venue_id": venue_id,
            "datetime": dt,
            "latitude": payload.get("latitude"),
            "longitude": payload.get("longitude"),
            "temperature_2m": _list_get(hourly.get("temperature_2m"), idx),
            "relative_humidity_2m": _list_get(hourly.get("relative_humidity_2m"), idx),
            "apparent_temperature": _list_get(hourly.get("apparent_temperature"), idx),
            "precipitation": _list_get(hourly.get("precipitation"), idx),
            "wind_speed_10m": _list_get(hourly.get("wind_speed_10m"), idx),
            "source": source,
            "is_forecast": is_forecast,
        })

    if not rows:
        return create_empty_table("weather_hourly")
    return pd.DataFrame(rows, columns=table_columns("weather_hourly"))


def normalize_open_meteo_flat(
    df: pd.DataFrame,
    *,
    venue_id: str | None = None,
    source: str = "open_meteo",
    is_forecast: bool = False,
) -> pd.DataFrame:
    """Normalize an Open-Meteo CSV/DataFrame export into `weather_hourly`."""
    rows = []
    for _, raw in df.iterrows():
        raw_venue_id = raw.get("venue_id", venue_id)
        rows.append({
            "venue_id": raw_venue_id,
            "datetime": raw.get("datetime", raw.get("time")),
            "latitude": raw.get("lat", raw.get("latitude")),
            "longitude": raw.get("lon", raw.get("longitude")),
            "temperature_2m": raw.get("temperature_2m"),
            "relative_humidity_2m": raw.get("relative_humidity_2m"),
            "apparent_temperature": raw.get("apparent_temperature"),
            "precipitation": raw.get("precipitation"),
            "wind_speed_10m": raw.get("wind_speed_10m"),
            "source": raw.get("source", source),
            "is_forecast": raw.get("is_forecast", is_forecast),
        })

    if not rows:
        return create_empty_table("weather_hourly")
    return pd.DataFrame(rows, columns=table_columns("weather_hourly"))


def normalize_referee_aggregates(
    df: pd.DataFrame,
    *,
    source: str = "worldreferee",
) -> pd.DataFrame:
    """
    Normalize referee aggregate rows into the `referees` table.

    Accepts WorldReferee/Kaggle-style columns such as `referee_name`,
    `tournament`, `matches`, `yellow_cards`, `red_cards`,
    `penalties_awarded`, and per-match rates.
    """
    rows = []
    for _, raw in df.iterrows():
        referee_name = raw.get("referee_name", raw.get("name"))
        if pd.isna(referee_name):
            continue

        tournament = raw.get("tournament", raw.get("competition", "All"))
        matches = raw.get("matches", raw.get("matches_officiated"))
        yellow_cards = raw.get("yellow_cards")
        red_cards = raw.get("red_cards")
        penalties_awarded = raw.get("penalties_awarded", raw.get("penalties"))

        rows.append({
            "referee_id": _as_source_id(source, "referee", raw.get("referee_id"), referee_name),
            "referee_name": referee_name,
            "nationality": raw.get("nationality"),
            "confederation": raw.get("confederation"),
            "tournament": tournament,
            "matches": matches,
            "yellow_cards": yellow_cards,
            "red_cards": red_cards,
            "penalties_awarded": penalties_awarded,
            "yellow_per_match": _rate_or_value(raw.get("yellow_per_match"), yellow_cards, matches),
            "red_per_match": _rate_or_value(raw.get("red_per_match"), red_cards, matches),
            "fouls_per_match": raw.get("fouls_per_match"),
            "penalties_per_match": _rate_or_value(raw.get("penalties_per_match"), penalties_awarded, matches),
        })

    if not rows:
        return create_empty_table("referees")
    return pd.DataFrame(rows, columns=table_columns("referees"))


def normalize_manual_referees(
    df: pd.DataFrame,
    *,
    source: str = "wc2026",
    tournament: str = "World Cup 2026",
) -> pd.DataFrame:
    """Normalize a manual referee list into the `referees` table."""
    rows = []
    for _, raw in df.iterrows():
        referee_name = raw.get("referee_name", raw.get("name"))
        if pd.isna(referee_name):
            continue

        rows.append({
            "referee_id": _as_source_id(source, "referee", raw.get("referee_id"), referee_name),
            "referee_name": str(referee_name).strip(),
            "nationality": raw.get("country", raw.get("nationality")),
            "confederation": raw.get("confederation"),
            "tournament": raw.get("tournament", tournament),
            "matches": raw.get("matches"),
            "yellow_cards": raw.get("yellow_cards"),
            "red_cards": raw.get("red_cards"),
            "penalties_awarded": raw.get("penalties_awarded"),
            "yellow_per_match": raw.get("yellow_per_match"),
            "red_per_match": raw.get("red_per_match"),
            "fouls_per_match": raw.get("fouls_per_match"),
            "penalties_per_match": raw.get("penalties_per_match"),
        })

    if not rows:
        return create_empty_table("referees")
    return pd.DataFrame(rows, columns=table_columns("referees"))


def normalize_manual_base_camps(
    df: pd.DataFrame,
    *,
    source: str = "wc2026",
    tournament: str = "World Cup 2026",
) -> pd.DataFrame:
    """Normalize a manual WC 2026 base-camps CSV into `base_camps`."""
    rows = []
    for _, raw in df.iterrows():
        team_name = raw.get("team_name", raw.get("team"))
        if pd.isna(team_name):
            continue

        rows.append({
            "team_id": _as_source_id(source, "team", raw.get("team_id"), team_name),
            "team_name": str(team_name).strip(),
            "tournament": raw.get("tournament", tournament),
            "city": raw.get("city"),
            "country": raw.get("country"),
            "accommodation": raw.get("accommodation", raw.get("hotel")),
            "training_site": raw.get("training_site"),
            "latitude": raw.get("lat", raw.get("latitude")),
            "longitude": raw.get("lon", raw.get("longitude")),
            "timezone": raw.get("timezone"),
        })

    if not rows:
        return create_empty_table("base_camps")
    return pd.DataFrame(rows, columns=table_columns("base_camps"))


def normalize_manual_group_stage_matches(
    df: pd.DataFrame,
    *,
    source: str = "wc2026",
    competition: str = "FIFA World Cup",
    season: str = "2026",
) -> pd.DataFrame:
    """Normalize the manual WC 2026 group-stage fixtures into `matches`."""
    rows = []
    for _, raw in df.iterrows():
        match_number = raw.get("match_number")
        home_team = raw.get("home_team")
        away_team = raw.get("away_team")
        if pd.isna(match_number) or pd.isna(home_team) or pd.isna(away_team):
            continue

        rows.append({
            "match_id": _as_source_id(source, "match", match_number),
            "competition": raw.get("competition", competition),
            "season": raw.get("season", season),
            "stage": raw.get("stage", "Group Stage"),
            "group": raw.get("group"),
            "match_date": raw.get("match_date"),
            "kickoff_local": raw.get("kickoff_local"),
            "home_team_id": _as_source_id(source, "team", raw.get("home_team_id"), home_team),
            "away_team_id": _as_source_id(source, "team", raw.get("away_team_id"), away_team),
            "venue_id": _as_source_id(source, "venue", raw.get("venue_id")),
            "referee_id": raw.get("referee_id"),
            "home_score": raw.get("home_score"),
            "away_score": raw.get("away_score"),
            "neutral": raw.get("neutral", True),
            "home_rest_days": raw.get("home_rest_days"),
            "away_rest_days": raw.get("away_rest_days"),
            "home_travel_km": raw.get("home_travel_km"),
            "away_travel_km": raw.get("away_travel_km"),
            "temperature_c": raw.get("temperature_c"),
            "humidity_pct": raw.get("humidity_pct"),
        })

    if not rows:
        return create_empty_table("matches")
    return pd.DataFrame(rows, columns=table_columns("matches"))


def normalize_manual_knockout_matches(
    df: pd.DataFrame,
    *,
    source: str = "wc2026",
    competition: str = "FIFA World Cup",
    season: str = "2026",
) -> pd.DataFrame:
    """Normalize the manual WC 2026 knockout fixtures into `matches`."""
    rows = []
    for _, raw in df.iterrows():
        match_number = raw.get("match_number")
        home_slot = raw.get("home_slot")
        away_slot = raw.get("away_slot")
        if pd.isna(match_number) or pd.isna(home_slot) or pd.isna(away_slot):
            continue

        rows.append({
            "match_id": _as_source_id(source, "match", match_number),
            "competition": raw.get("competition", competition),
            "season": raw.get("season", season),
            "stage": raw.get("stage"),
            "group": None,
            "match_date": raw.get("match_date"),
            "kickoff_local": raw.get("kickoff_local"),
            "home_team_id": _as_source_id(source, "slot", home_slot),
            "away_team_id": _as_source_id(source, "slot", away_slot),
            "venue_id": _as_source_id(source, "venue", raw.get("venue_id")),
            "referee_id": raw.get("referee_id"),
            "home_score": raw.get("home_score"),
            "away_score": raw.get("away_score"),
            "neutral": raw.get("neutral", True),
            "home_rest_days": raw.get("home_rest_days"),
            "away_rest_days": raw.get("away_rest_days"),
            "home_travel_km": raw.get("home_travel_km"),
            "away_travel_km": raw.get("away_travel_km"),
            "temperature_c": raw.get("temperature_c"),
            "humidity_pct": raw.get("humidity_pct"),
        })

    if not rows:
        return create_empty_table("matches")
    return pd.DataFrame(rows, columns=table_columns("matches"))


def normalize_team_ratings(
    df: pd.DataFrame,
    *,
    source: str = "world_football_elo",
) -> pd.DataFrame:
    """
    Normalize team-strength rating snapshots into `team_ratings`.

    Accepts columns such as `date`, `team`, `elo_rating`, `rank`,
    `1_year_change`, and `confederation`.
    """
    rows = []
    for _, raw in df.iterrows():
        team_name = raw.get("team", raw.get("team_name"))
        rating_date = raw.get("date", raw.get("rating_date"))
        if pd.isna(team_name) or pd.isna(rating_date):
            continue

        rows.append({
            "team_id": _as_source_id("wc2026", "team", raw.get("team_id"), team_name),
            "team_name": str(team_name).strip(),
            "rating_date": rating_date,
            "source": source,
            "elo_rating": raw.get("elo_rating", raw.get("rating")),
            "rank": raw.get("rank"),
            "one_year_change": raw.get("1_year_change", raw.get("one_year_change")),
            "confederation": raw.get("confederation"),
        })

    if not rows:
        return create_empty_table("team_ratings")
    return pd.DataFrame(rows, columns=table_columns("team_ratings"))


def normalize_the_odds_api_history(
    payload: list[dict[str, Any]],
    *,
    source: str = "the_odds_api",
    mark_open_close: bool = True,
) -> pd.DataFrame:
    """
    Normalize The Odds API historical snapshots into the `odds` table.

    The normalizer accepts a list of event snapshots. Each event can contain
    `bookmakers -> markets -> outcomes`, matching The Odds API's common JSON
    shape. For already-flattened rows, use `normalize_flat_odds_history`.
    """
    rows = []
    for event in payload:
        match_id = _as_source_id(source, "match", event.get("id"), event.get("match_id", "unknown"))
        quote_time = event.get("timestamp") or event.get("commence_time")

        for bookmaker in event.get("bookmakers", []):
            bookmaker_key = bookmaker.get("key") or bookmaker.get("title")
            bookmaker_time = bookmaker.get("last_update") or quote_time
            is_sharp = str(bookmaker_key).lower() in SHARP_BOOKMAKERS

            for market in bookmaker.get("markets", []):
                market_key = market.get("key")
                market_time = market.get("last_update") or bookmaker_time

                for outcome in market.get("outcomes", []):
                    selection = outcome.get("description") or outcome.get("name")
                    line = outcome.get("point")
                    selection, line = _parse_selection_and_line(selection, line)
                    odds_decimal = outcome.get("price")
                    if odds_decimal in (None, ""):
                        continue
                    odds_id = _stable_id(match_id, bookmaker_key, market_key, selection, line, market_time)
                    odds_decimal = float(odds_decimal)
                    rows.append({
                        "odds_id": _as_source_id(source, "odds", odds_id),
                        "match_id": match_id,
                        "bookmaker": bookmaker_key,
                        "is_sharp": is_sharp,
                        "market": market_key,
                        "selection": selection,
                        "line": line,
                        "odds_decimal": odds_decimal,
                        "implied_probability": 1 / odds_decimal if odds_decimal > 0 else None,
                        "no_vig_probability": None,
                        "quote_time": market_time,
                        "is_opening": None,
                        "is_closing": None,
                    })

    if not rows:
        return create_empty_table("odds")

    odds = pd.DataFrame(rows, columns=table_columns("odds"))
    if mark_open_close:
        odds = _add_opening_closing_flags(odds)
    return odds


def normalize_flat_odds_history(
    df: pd.DataFrame,
    *,
    source: str = "the_odds_api",
    mark_open_close: bool = True,
) -> pd.DataFrame:
    """Normalize flattened historical odds rows into the `odds` table."""
    rows = []
    for _, raw in df.iterrows():
        bookmaker = raw.get("bookmaker")
        selection, line = _parse_selection_and_line(raw.get("selection"), raw.get("line"))
        odds_decimal = raw.get("price", raw.get("odds_decimal"))
        if odds_decimal in (None, "") or pd.isna(odds_decimal):
            continue
        match_id = _as_source_id(source, "match", raw.get("match_id"))
        quote_time = raw.get("timestamp", raw.get("quote_time"))
        odds_id = _stable_id(match_id, bookmaker, raw.get("market"), selection, line, quote_time)
        odds_decimal = float(odds_decimal)
        rows.append({
            "odds_id": _as_source_id(source, "odds", odds_id),
            "match_id": match_id,
            "bookmaker": bookmaker,
            "is_sharp": str(bookmaker).lower() in SHARP_BOOKMAKERS,
            "market": raw.get("market"),
            "selection": selection,
            "line": line,
            "odds_decimal": odds_decimal,
            "implied_probability": 1 / odds_decimal if odds_decimal > 0 else None,
            "no_vig_probability": raw.get("no_vig_probability"),
            "quote_time": quote_time,
            "is_opening": raw.get("is_opening"),
            "is_closing": raw.get("is_closing"),
        })

    if not rows:
        return create_empty_table("odds")

    odds = pd.DataFrame(rows, columns=table_columns("odds"))
    if mark_open_close:
        odds = _add_opening_closing_flags(odds)
    return odds
