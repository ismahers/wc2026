"""
StatsBomb Open Data collector for international event-market targets.

The collector aggregates event-level open data into match-level stats used by
corners, cards, shots, and xG models. It intentionally downloads JSON directly
from the public StatsBomb repository, avoiding an extra dependency for this
small, reproducible ingestion step.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any

import pandas as pd
import requests

from src.data.team_names import canonicalize


log = logging.getLogger(__name__)

SB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
REQUEST_DELAY_SECONDS = 0.25


@dataclass(frozen=True)
class StatsBombCompetition:
    """StatsBomb competition/season pair available in open data."""

    competition_id: int
    season_id: int
    competition: str
    edition: str
    host_team: str | None = None


INTERNATIONAL_COMPETITIONS: tuple[StatsBombCompetition, ...] = (
    StatsBombCompetition(43, 3, "FIFA World Cup", "2018", "Russia"),
    StatsBombCompetition(43, 106, "FIFA World Cup", "2022", "Qatar"),
    StatsBombCompetition(55, 43, "UEFA Euro", "2020", None),
    StatsBombCompetition(55, 282, "UEFA Euro", "2024", "Germany"),
    # ── Nuevas (verificadas jun 2026) ──
    StatsBombCompetition(223, 282, "Copa America", "2024", "United States"),
    StatsBombCompetition(1267, 107, "African Cup of Nations", "2023", "Ivory Coast"),
    # ── WC históricos sueltos (19 partidos). Descomenta los que quieras ──
    # StatsBombCompetition(43, 272, "FIFA World Cup", "1970", None),
    # StatsBombCompetition(43, 51, "FIFA World Cup", "1974", None),
    # StatsBombCompetition(43, 54, "FIFA World Cup", "1986", None),
)


class StatsBombInternationalCollector:
    """Collect match-level international stats from StatsBomb Open Data."""

    def __init__(
        self,
        *,
        base_url: str = SB_BASE,
        request_delay_seconds: float = REQUEST_DELAY_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_delay_seconds = request_delay_seconds
        self.session = session or requests.Session()

    def collect(self, competitions: tuple[StatsBombCompetition, ...] = INTERNATIONAL_COMPETITIONS) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        for competition in competitions:
            matches = self._fetch_matches(competition)
            log.info(
                "%s %s: %d matches",
                competition.competition,
                competition.edition,
                len(matches),
            )

            for idx, match in enumerate(matches, start=1):
                match_id = match["match_id"]
                home_team = canonicalize(match.get("home_team", {}).get("home_team_name", ""))
                away_team = canonicalize(match.get("away_team", {}).get("away_team_name", ""))
                log.info(
                    "  %s %s match %d/%d: %s vs %s",
                    competition.competition,
                    competition.edition,
                    idx,
                    len(matches),
                    home_team,
                    away_team,
                )
                try:
                    events = self._fetch_events(match_id)
                    agg = aggregate_match_events(events, home_team, away_team)
                except requests.RequestException as exc:
                    log.warning("  Could not fetch events for match_id=%s: %s", match_id, exc)
                    agg = empty_event_stats()

                home_score = match.get("home_score")
                away_score = match.get("away_score")
                rows.append({
                    "statsbomb_match_id": match_id,
                    "date": match.get("match_date"),
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": home_score,
                    "away_score": away_score,
                    "result": _result(home_score, away_score),
                    "competition": competition.competition,
                    "season": competition.edition,
                    "tournament": competition.competition,
                    "stage": match.get("competition_stage", {}).get("name"),
                    "stadium": match.get("stadium", {}).get("name"),
                    "referee": (match.get("referee") or {}).get("name"),
                    "neutral": _is_neutral(home_team, away_team, competition.host_team),
                    **agg,
                })

        result = pd.DataFrame(rows)
        if result.empty:
            return result
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        return result.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)

    def save(
        self,
        df: pd.DataFrame,
        *,
        raw_output: str | Path = "data/raw/international_match_stats.csv",
        statsbomb_output: str | Path = "data/statsbomb_matches.csv",
    ) -> tuple[Path, Path]:
        """Save both a readable raw stats table and the legacy pipeline table."""
        raw_path = Path(raw_output)
        statsbomb_path = Path(statsbomb_output)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        statsbomb_path.parent.mkdir(parents=True, exist_ok=True)

        raw_df = to_international_match_stats(df)
        raw_df.to_csv(raw_path, index=False)
        df.to_csv(statsbomb_path, index=False)
        log.info("Saved %d rows to %s", len(raw_df), raw_path)
        log.info("Saved %d rows to %s", len(df), statsbomb_path)
        return raw_path, statsbomb_path

    def _fetch_matches(self, competition: StatsBombCompetition) -> list[dict[str, Any]]:
        time.sleep(self.request_delay_seconds)
        url = f"{self.base_url}/matches/{competition.competition_id}/{competition.season_id}.json"
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return response.json()

    def _fetch_events(self, match_id: int) -> list[dict[str, Any]]:
        time.sleep(self.request_delay_seconds)
        url = f"{self.base_url}/events/{match_id}.json"
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return response.json()


def empty_event_stats() -> dict[str, float]:
    return {
        "corners_home": 0,
        "corners_away": 0,
        "corners_total": 0,
        "yellow_home": 0,
        "yellow_away": 0,
        "yellow_total": 0,
        "red_home": 0,
        "red_away": 0,
        "red_total": 0,
        "shots_home": 0,
        "shots_away": 0,
        "shots_total": 0,
        "shots_on_target_home": 0,
        "shots_on_target_away": 0,
        "shots_on_target_total": 0,
        "xg_home": 0.0,
        "xg_away": 0.0,
        "xg_total": 0.0,
    }


def aggregate_match_events(
    events: list[dict[str, Any]],
    home_team: str,
    away_team: str,
) -> dict[str, float]:
    """Aggregate StatsBomb events into home/away match stats."""
    stats = empty_event_stats()

    for event in events:
        team = canonicalize((event.get("team") or {}).get("name", ""))
        if team == home_team:
            side = "home"
        elif team == away_team:
            side = "away"
        else:
            continue

        event_type = (event.get("type") or {}).get("name")
        if event_type == "Pass":
            pass_type = (event.get("pass") or {}).get("type", {}).get("name")
            if pass_type == "Corner":
                stats[f"corners_{side}"] += 1

        elif event_type == "Foul Committed":
            card = (event.get("foul_committed") or {}).get("card", {}).get("name")
            _add_card(stats, side, card)

        elif event_type == "Bad Behaviour":
            card = (event.get("bad_behaviour") or {}).get("card", {}).get("name")
            _add_card(stats, side, card)

        elif event_type == "Shot":
            stats[f"shots_{side}"] += 1
            shot = event.get("shot") or {}
            stats[f"xg_{side}"] += float(shot.get("statsbomb_xg") or 0.0)
            outcome = (shot.get("outcome") or {}).get("name")
            if outcome in {"Goal", "Saved", "Saved to Post"}:
                stats[f"shots_on_target_{side}"] += 1

    stats["corners_total"] = stats["corners_home"] + stats["corners_away"]
    stats["yellow_total"] = stats["yellow_home"] + stats["yellow_away"]
    stats["red_total"] = stats["red_home"] + stats["red_away"]
    stats["shots_total"] = stats["shots_home"] + stats["shots_away"]
    stats["shots_on_target_total"] = stats["shots_on_target_home"] + stats["shots_on_target_away"]
    stats["xg_home"] = round(stats["xg_home"], 3)
    stats["xg_away"] = round(stats["xg_away"], 3)
    stats["xg_total"] = round(stats["xg_home"] + stats["xg_away"], 3)
    return stats


def to_international_match_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return the compact CSV shape requested for event-market modeling."""
    if df.empty:
        return df.copy()

    result = pd.DataFrame({
        "date": df["date"],
        "home_team": df["home_team"],
        "away_team": df["away_team"],
        "home_corners": df["corners_home"],
        "away_corners": df["corners_away"],
        "home_yellow_cards": df["yellow_home"],
        "away_yellow_cards": df["yellow_away"],
        "home_red_cards": df["red_home"],
        "away_red_cards": df["red_away"],
        "referee": df.get("referee"),
        "competition": df["competition"],
        "season": df["season"],
        "neutral": df["neutral"],
    })
    return result.sort_values("date").reset_index(drop=True)


def _add_card(stats: dict[str, float], side: str, card: str | None) -> None:
    if card == "Yellow Card":
        stats[f"yellow_{side}"] += 1
    elif card in {"Red Card", "Second Yellow"}:
        stats[f"red_{side}"] += 1


def _result(home_score: Any, away_score: Any) -> str | None:
    if home_score is None or away_score is None:
        return None
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def _is_neutral(home_team: str, away_team: str, host_team: str | None) -> bool:
    if not host_team:
        return True
    host = canonicalize(host_team)
    return home_team != host and away_team != host


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect international match stats from StatsBomb Open Data.")
    parser.add_argument("--raw-output", default="data/raw/international_match_stats.csv")
    parser.add_argument("--statsbomb-output", default="data/statsbomb_matches.csv")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY_SECONDS)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args()
    collector = StatsBombInternationalCollector(request_delay_seconds=args.delay)
    df = collector.collect()
    collector.save(df, raw_output=args.raw_output, statsbomb_output=args.statsbomb_output)
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
