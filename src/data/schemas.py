"""
Normalized data contracts for market and event modeling.

The current project started from match-level outcomes. These schemas establish
the entities needed for event markets and player props: matches, teams,
players, lineups, player stats, venues, base camps, team ratings, weather,
referees, and odds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ColumnSpec:
    """Column contract for a normalized table."""

    name: str
    dtype: str
    nullable: bool = True
    description: str = ""


@dataclass(frozen=True)
class TableSchema:
    """Schema definition for one normalized table."""

    name: str
    primary_key: tuple[str, ...]
    columns: tuple[ColumnSpec, ...]
    description: str = ""

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(col.name for col in self.columns if not col.nullable)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(col.name for col in self.columns)


TABLE_SCHEMAS: dict[str, TableSchema] = {
    "matches": TableSchema(
        name="matches",
        primary_key=("match_id",),
        description="One row per match, including fixture context and final score.",
        columns=(
            ColumnSpec("match_id", "string", False, "Stable source match id."),
            ColumnSpec("competition", "string", False, "Competition name."),
            ColumnSpec("season", "string", True, "Season or tournament edition."),
            ColumnSpec("stage", "string", True, "Tournament stage or round."),
            ColumnSpec("group", "string", True, "Group-stage group label."),
            ColumnSpec("match_date", "datetime64[ns]", False, "Kickoff date."),
            ColumnSpec("kickoff_local", "datetime64[ns]", True, "Local kickoff timestamp."),
            ColumnSpec("home_team_id", "string", False, "Home team id."),
            ColumnSpec("away_team_id", "string", False, "Away team id."),
            ColumnSpec("venue_id", "string", True, "Venue id."),
            ColumnSpec("referee_id", "string", True, "Assigned referee id."),
            ColumnSpec("home_score", "float64", True, "Final home goals."),
            ColumnSpec("away_score", "float64", True, "Final away goals."),
            ColumnSpec("neutral", "bool", True, "Whether venue is neutral."),
            ColumnSpec("home_rest_days", "float64", True, "Days since home team's previous match."),
            ColumnSpec("away_rest_days", "float64", True, "Days since away team's previous match."),
            ColumnSpec("home_travel_km", "float64", True, "Estimated travel distance before match."),
            ColumnSpec("away_travel_km", "float64", True, "Estimated travel distance before match."),
            ColumnSpec("temperature_c", "float64", True, "Forecast or observed temperature."),
            ColumnSpec("humidity_pct", "float64", True, "Forecast or observed humidity."),
        ),
    ),
    "teams": TableSchema(
        name="teams",
        primary_key=("team_id",),
        description="National teams and slowly changing team attributes.",
        columns=(
            ColumnSpec("team_id", "string", False, "Stable team id."),
            ColumnSpec("team_name", "string", False, "Display team name."),
            ColumnSpec("country_code", "string", True, "ISO or FIFA country code."),
            ColumnSpec("confederation", "string", True, "FIFA confederation."),
            ColumnSpec("fifa_rank", "float64", True, "Latest FIFA ranking before match."),
            ColumnSpec("elo_rating", "float64", True, "Elo rating before match."),
        ),
    ),
    "team_ratings": TableSchema(
        name="team_ratings",
        primary_key=("team_id", "rating_date", "source"),
        description="Time-series team strength ratings for as-of feature joins.",
        columns=(
            ColumnSpec("team_id", "string", False, "Stable team id."),
            ColumnSpec("team_name", "string", False, "Display team name."),
            ColumnSpec("rating_date", "datetime64[ns]", False, "Date the rating applies to."),
            ColumnSpec("source", "string", False, "Rating provider."),
            ColumnSpec("elo_rating", "float64", True, "World Football Elo rating."),
            ColumnSpec("rank", "float64", True, "Ranking position at rating date."),
            ColumnSpec("one_year_change", "float64", True, "Rating or rank change over one year."),
            ColumnSpec("confederation", "string", True, "FIFA confederation."),
        ),
    ),
    "players": TableSchema(
        name="players",
        primary_key=("player_id",),
        description="Player identity, team membership, and player profile fields.",
        columns=(
            ColumnSpec("player_id", "string", False, "Stable player id."),
            ColumnSpec("player_name", "string", False, "Display player name."),
            ColumnSpec("team_id", "string", False, "National team id."),
            ColumnSpec("club", "string", True, "Club at time of tournament."),
            ColumnSpec("position", "string", True, "Primary position."),
            ColumnSpec("birth_date", "datetime64[ns]", True, "Date of birth."),
            ColumnSpec("preferred_foot", "string", True, "Preferred foot."),
            ColumnSpec("height_cm", "float64", True, "Player height."),
            ColumnSpec("market_value_eur", "float64", True, "Transfermarkt market value in EUR."),
        ),
    ),
    "lineups": TableSchema(
        name="lineups",
        primary_key=("match_id", "team_id", "player_id"),
        description="Match squad, starting XI, substitutions, and minutes played.",
        columns=(
            ColumnSpec("match_id", "string", False, "Match id."),
            ColumnSpec("team_id", "string", False, "Team id."),
            ColumnSpec("player_id", "string", False, "Player id."),
            ColumnSpec("is_starter", "bool", False, "Whether player started."),
            ColumnSpec("position", "string", True, "Listed match position."),
            ColumnSpec("shirt_number", "float64", True, "Shirt number."),
            ColumnSpec("minute_on", "float64", True, "Minute entered."),
            ColumnSpec("minute_off", "float64", True, "Minute left."),
            ColumnSpec("minutes_played", "float64", True, "Total minutes played."),
        ),
    ),
    "player_match_stats": TableSchema(
        name="player_match_stats",
        primary_key=("match_id", "team_id", "player_id"),
        description="Observed player event stats for goals, assists, shots, cards, and props.",
        columns=(
            ColumnSpec("match_id", "string", False, "Match id."),
            ColumnSpec("team_id", "string", False, "Team id."),
            ColumnSpec("player_id", "string", False, "Player id."),
            ColumnSpec("minutes_played", "float64", True, "Minutes played."),
            ColumnSpec("goals", "float64", True, "Goals."),
            ColumnSpec("assists", "float64", True, "Assists."),
            ColumnSpec("shots", "float64", True, "Total shots."),
            ColumnSpec("shots_on_target", "float64", True, "Shots on target."),
            ColumnSpec("xg", "float64", True, "Expected goals."),
            ColumnSpec("xa", "float64", True, "Expected assists."),
            ColumnSpec("key_passes", "float64", True, "Key passes or chances created."),
            ColumnSpec("corners_taken", "float64", True, "Corners taken."),
            ColumnSpec("fouls_committed", "float64", True, "Fouls committed."),
            ColumnSpec("fouls_won", "float64", True, "Fouls won."),
            ColumnSpec("yellow_cards", "float64", True, "Yellow cards."),
            ColumnSpec("red_cards", "float64", True, "Red cards."),
        ),
    ),
    "venues": TableSchema(
        name="venues",
        primary_key=("venue_id",),
        description="Venue location and environmental context for WC 2026.",
        columns=(
            ColumnSpec("venue_id", "string", False, "Stable venue id."),
            ColumnSpec("venue_name", "string", False, "Display venue name."),
            ColumnSpec("city", "string", False, "Host city."),
            ColumnSpec("country", "string", False, "Host country."),
            ColumnSpec("latitude", "float64", True, "Latitude."),
            ColumnSpec("longitude", "float64", True, "Longitude."),
            ColumnSpec("altitude_m", "float64", True, "Altitude above sea level."),
            ColumnSpec("indoor", "bool", True, "Whether the stadium is indoor or roofed."),
            ColumnSpec("roof_type", "string", True, "Open, fixed, or retractable roof."),
            ColumnSpec("surface", "string", True, "Pitch surface."),
            ColumnSpec("timezone", "string", True, "Local timezone."),
        ),
    ),
    "weather_hourly": TableSchema(
        name="weather_hourly",
        primary_key=("venue_id", "datetime"),
        description="Hourly weather observations or forecasts for each venue.",
        columns=(
            ColumnSpec("venue_id", "string", False, "Venue id."),
            ColumnSpec("datetime", "datetime64[ns]", False, "Local or UTC hourly timestamp."),
            ColumnSpec("latitude", "float64", True, "Latitude used for weather lookup."),
            ColumnSpec("longitude", "float64", True, "Longitude used for weather lookup."),
            ColumnSpec("temperature_2m", "float64", True, "Air temperature in Celsius."),
            ColumnSpec("relative_humidity_2m", "float64", True, "Relative humidity percentage."),
            ColumnSpec("apparent_temperature", "float64", True, "Feels-like temperature in Celsius."),
            ColumnSpec("precipitation", "float64", True, "Precipitation amount in millimeters."),
            ColumnSpec("wind_speed_10m", "float64", True, "Wind speed at 10m, usually km/h."),
            ColumnSpec("source", "string", True, "Weather provider or dataset."),
            ColumnSpec("is_forecast", "bool", True, "Whether data came from forecast endpoint."),
        ),
    ),
    "base_camps": TableSchema(
        name="base_camps",
        primary_key=("team_id", "tournament"),
        description="Team base camp lodging and training site for a tournament.",
        columns=(
            ColumnSpec("team_id", "string", False, "Stable team id."),
            ColumnSpec("team_name", "string", False, "Display team name."),
            ColumnSpec("tournament", "string", False, "Tournament name."),
            ColumnSpec("city", "string", False, "Base camp city."),
            ColumnSpec("country", "string", True, "Base camp country when known."),
            ColumnSpec("accommodation", "string", True, "Team lodging site."),
            ColumnSpec("training_site", "string", True, "Team training site."),
            ColumnSpec("latitude", "float64", True, "Base camp city or training-site latitude."),
            ColumnSpec("longitude", "float64", True, "Base camp city or training-site longitude."),
            ColumnSpec("altitude_m", "float64", True, "Base camp city altitude above sea level."),
            ColumnSpec("timezone", "string", True, "Base camp timezone."),
        ),
    ),
    "referees": TableSchema(
        name="referees",
        primary_key=("referee_id", "tournament"),
        description="Referee identity and disciplinary tendencies by context/tournament.",
        columns=(
            ColumnSpec("referee_id", "string", False, "Stable referee id."),
            ColumnSpec("referee_name", "string", False, "Display referee name."),
            ColumnSpec("nationality", "string", True, "Nationality."),
            ColumnSpec("confederation", "string", True, "Confederation."),
            ColumnSpec("tournament", "string", False, "Competition or context for the aggregate."),
            ColumnSpec("matches", "float64", True, "Career matches in source."),
            ColumnSpec("yellow_cards", "float64", True, "Total yellow cards in context."),
            ColumnSpec("red_cards", "float64", True, "Total red cards in context."),
            ColumnSpec("penalties_awarded", "float64", True, "Total penalties awarded in context."),
            ColumnSpec("yellow_per_match", "float64", True, "Average yellow cards per match."),
            ColumnSpec("red_per_match", "float64", True, "Average red cards per match."),
            ColumnSpec("fouls_per_match", "float64", True, "Average fouls called per match."),
            ColumnSpec("penalties_per_match", "float64", True, "Average penalties awarded per match."),
        ),
    ),
    "odds": TableSchema(
        name="odds",
        primary_key=("odds_id",),
        description="Bookmaker and sharp odds by match, market, selection, and timestamp.",
        columns=(
            ColumnSpec("odds_id", "string", False, "Stable quote id."),
            ColumnSpec("match_id", "string", False, "Match id."),
            ColumnSpec("bookmaker", "string", False, "Bookmaker or exchange."),
            ColumnSpec("is_sharp", "bool", True, "Whether source is treated as sharp baseline."),
            ColumnSpec("market", "string", False, "Market name, e.g. player_shots_on_target."),
            ColumnSpec("selection", "string", False, "Outcome or player selection."),
            ColumnSpec("line", "float64", True, "Market line for totals/handicaps."),
            ColumnSpec("odds_decimal", "float64", False, "Decimal odds."),
            ColumnSpec("implied_probability", "float64", True, "Raw implied probability."),
            ColumnSpec("no_vig_probability", "float64", True, "Margin-adjusted probability."),
            ColumnSpec("quote_time", "datetime64[ns]", False, "Timestamp of odds quote."),
            ColumnSpec("is_opening", "bool", True, "Opening quote marker."),
            ColumnSpec("is_closing", "bool", True, "Closing quote marker."),
        ),
    ),
}


def get_schema(table_name: str) -> TableSchema:
    """Return the schema for a known table."""
    try:
        return TABLE_SCHEMAS[table_name]
    except KeyError as exc:
        known = ", ".join(sorted(TABLE_SCHEMAS))
        raise KeyError(f"Unknown table '{table_name}'. Known tables: {known}") from exc


def table_columns(table_name: str) -> list[str]:
    """Return ordered column names for a schema."""
    return list(get_schema(table_name).column_names)


def create_empty_table(table_name: str) -> pd.DataFrame:
    """Create an empty DataFrame with the schema's ordered columns."""
    schema = get_schema(table_name)
    return pd.DataFrame(columns=schema.column_names)


def validate_columns(
    df: pd.DataFrame,
    table_name: str,
    *,
    require_primary_key: bool = True,
) -> dict[str, Any]:
    """
    Validate that a DataFrame satisfies the minimum column contract.

    This intentionally avoids strict dtype coercion because raw provider feeds
    often arrive as strings. Type normalization should happen in ingestion
    adapters before model training.
    """
    schema = get_schema(table_name)
    required = set(schema.required_columns)
    if require_primary_key:
        required.update(schema.primary_key)

    present = set(df.columns)
    missing_required = sorted(required - present)
    extra_columns = sorted(present - set(schema.column_names))

    return {
        "table": table_name,
        "ok": not missing_required,
        "missing_required": missing_required,
        "extra_columns": extra_columns,
    }
