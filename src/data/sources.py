"""
Source catalog for player props and event-market data.

These definitions keep provider metadata next to the ingestion layer: coverage,
granularity, access method, rate limits, and the normalized tables each source
can populate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataSource:
    """Metadata for one external data source."""

    name: str
    url: str
    access_method: str
    granularity: tuple[str, ...]
    coverage: str
    normalized_tables: tuple[str, ...]
    rate_limit_note: str = ""
    license_note: str = ""


SOURCE_CATALOG: dict[str, DataSource] = {
    "fbref": DataSource(
        name="FBref",
        url="https://fbref.com/",
        access_method="HTML scraping with pandas.read_html or soccerdata",
        granularity=("match", "player"),
        coverage=(
            "Top European leagues from roughly 2017 with xG, plus complete "
            "international competitions such as World Cups 2018 and 2022."
        ),
        normalized_tables=("players", "lineups", "player_match_stats"),
        rate_limit_note="Respect a strict delay of at least 3.5 seconds between requests.",
        license_note="Use for personal or academic workflows; avoid bulk redistribution.",
    ),
    "statsbomb_open_data": DataSource(
        name="StatsBomb Open Data",
        url="https://github.com/statsbomb/open-data",
        access_method="JSON files or statsbombpy DataFrames",
        granularity=("event", "player", "match"),
        coverage=(
            "Men's World Cups 2018 and 2022, Euro 2020 and 2024, and selected "
            "club seasons."
        ),
        normalized_tables=("matches", "teams", "players", "lineups", "player_match_stats"),
        rate_limit_note="Public GitHub data; still cache locally and avoid repeated full pulls.",
        license_note="Free for non-commercial use with StatsBomb attribution.",
    ),
    "api_football": DataSource(
        name="API-Football",
        url="https://www.api-football.com/",
        access_method="REST API returning JSON",
        granularity=("match", "event", "player", "venue"),
        coverage="More than 10 years of fixtures across leagues and international competitions.",
        normalized_tables=(
            "matches",
            "teams",
            "players",
            "lineups",
            "player_match_stats",
            "venues",
            "referees",
        ),
        rate_limit_note="Free plan is useful for tests; production usage needs a paid tier.",
        license_note="Commercial provider; check plan terms before storage or redistribution.",
    ),
    "the_odds_api": DataSource(
        name="The Odds API",
        url="https://the-odds-api.com/",
        access_method="REST API historical odds endpoint returning JSON snapshots",
        granularity=("odds", "market", "bookmaker", "timestamp"),
        coverage=(
            "Historical availability depends on plan, commonly around 12-24 months. "
            "Useful for recent qualifiers, market movement, opening lines, and CLV."
        ),
        normalized_tables=("odds",),
        rate_limit_note="Historical endpoints are paid and quota-limited; cache every response.",
        license_note="Commercial provider; check plan terms before storing or redistributing history.",
    ),
    "transfermarkt_datasets": DataSource(
        name="Transfermarkt Datasets",
        url="https://github.com/dcaribou/transfermarkt-datasets",
        access_method="CSV or Parquet snapshots maintained on GitHub/Kaggle",
        granularity=("match", "player"),
        coverage="Transfermarkt-derived football data from 2012 to present, updated weekly.",
        normalized_tables=("players", "lineups"),
        rate_limit_note="Prefer downloaded snapshots over scraping Transfermarkt directly.",
        license_note="Community dataset published as CC0; verify current repository terms before use.",
    ),
    "wc2026_manual_venues": DataSource(
        name="WC 2026 Manual Venues",
        url="data/raw/venues.csv",
        access_method="Manual CSV maintained in the repository workspace",
        granularity=("venue",),
        coverage="The 16 official FIFA World Cup 2026 host venues.",
        normalized_tables=("venues",),
        rate_limit_note="Static file; review manually if FIFA venue names or surfaces change.",
        license_note="Curated factual venue metadata for local modeling.",
    ),
    "wc2026_manual_referees": DataSource(
        name="WC 2026 Manual Referees",
        url="data/raw/referees_wc2026.csv",
        access_method="Manual CSV maintained in the repository workspace",
        granularity=("referee",),
        coverage="Listed FIFA World Cup 2026 referees provided for local modeling.",
        normalized_tables=("referees",),
        rate_limit_note="Static file; reconcile against official FIFA updates before tournament use.",
        license_note="Curated factual referee metadata for local modeling.",
    ),
    "wc2026_manual_base_camps": DataSource(
        name="WC 2026 Manual Base Camps",
        url="data/raw/base_camps_wc2026.csv",
        access_method="Manual CSV maintained in the repository workspace",
        granularity=("team", "base_camp"),
        coverage="Listed FIFA World Cup 2026 team base camps for all 48 teams.",
        normalized_tables=("base_camps",),
        rate_limit_note="Static file; enrich coordinates/timezones before travel-distance modeling.",
        license_note="Curated factual base-camp metadata for local modeling.",
    ),
    "wc2026_manual_group_stage": DataSource(
        name="WC 2026 Manual Group Stage",
        url="data/raw/group_stage_wc2026.csv",
        access_method="Manual CSV maintained in the repository workspace",
        granularity=("match",),
        coverage="FIFA World Cup 2026 group-stage fixtures, groups A-L.",
        normalized_tables=("matches",),
        rate_limit_note="Static file; add kickoff times once confirmed/available.",
        license_note="Curated factual fixture metadata for local modeling.",
    ),
    "wc2026_manual_knockout": DataSource(
        name="WC 2026 Manual Knockout",
        url="data/raw/knockout_wc2026.csv",
        access_method="Manual CSV maintained in the repository workspace",
        granularity=("match", "bracket_slot"),
        coverage="FIFA World Cup 2026 knockout bracket fixtures, matches 73-104.",
        normalized_tables=("matches",),
        rate_limit_note="Static file; slots resolve after group-stage/knockout results.",
        license_note="Curated factual fixture metadata for local modeling.",
    ),
    "open_meteo": DataSource(
        name="Open-Meteo API",
        url="https://open-meteo.com/",
        access_method="REST API JSON or CSV via openmeteo-requests",
        granularity=("weather", "venue", "hour"),
        coverage="ERA5-based historical hourly weather from 1940 and forecasts up to 14 days.",
        normalized_tables=("weather_hourly",),
        rate_limit_note="Free non-commercial tier without API key; cache venue-hour responses.",
        license_note="Free for non-commercial use; check terms for commercial deployment.",
    ),
    "world_football_elo": DataSource(
        name="World Football Elo Ratings",
        url="https://www.eloratings.net/",
        access_method="HTML scraping or GitHub CSV mirror",
        granularity=("team", "date"),
        coverage="International football Elo ratings from 1872 to present.",
        normalized_tables=("team_ratings",),
        rate_limit_note="Prefer CSV mirrors for history; cache scraped snapshots.",
        license_note="Public rating data; verify terms of any chosen mirror.",
    ),
    "worldreferee": DataSource(
        name="WorldReferee",
        url="https://worldreferee.com/",
        access_method="HTML scraping into pandas DataFrames",
        granularity=("referee", "tournament"),
        coverage="Historical international referee profiles and disciplinary aggregates.",
        normalized_tables=("referees",),
        rate_limit_note="Use polite scraping with delays and cache parsed pages.",
        license_note="Public website data; review terms before automated collection.",
    ),
    "football_referees_stats": DataSource(
        name="Football Referees Stats",
        url="Kaggle search: Football Referees Stats",
        access_method="CSV dataset",
        granularity=("referee", "tournament"),
        coverage="Depends on selected Kaggle dataset; useful fallback to avoid scraping.",
        normalized_tables=("referees",),
        rate_limit_note="Prefer local CSV snapshots once downloaded.",
        license_note="Check the selected Kaggle dataset license.",
    ),
}


def get_source(source_key: str) -> DataSource:
    """Return source metadata by key."""
    try:
        return SOURCE_CATALOG[source_key]
    except KeyError as exc:
        known = ", ".join(sorted(SOURCE_CATALOG))
        raise KeyError(f"Unknown source '{source_key}'. Known sources: {known}") from exc
