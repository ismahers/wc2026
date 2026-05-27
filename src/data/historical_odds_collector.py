"""
The Odds API historical odds collector.

This loader fetches historical market snapshots and stores both the cached raw
JSON and a normalized `odds` table compatible with src.data.schemas.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Iterable

import pandas as pd
import requests

from src.data.normalizers import normalize_flat_odds_history


log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.the-odds-api.com/v4/historical/sports"
CURRENT_BASE_URL = "https://api.the-odds-api.com/v4/sports"
DEFAULT_SPORT = "soccer_fifa_world_cup"
DEFAULT_MARKETS = ("h2h", "totals")
DEFAULT_BOOKMAKERS = ("pinnacle", "bet365")


def _read_dotenv_value(key: str, env_path: Path) -> str | None:
    if not env_path.exists():
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip('"').strip("'")
    return None


def _api_key(explicit_api_key: str | None, env_path: Path) -> str:
    key = explicit_api_key or os.getenv("ODDS_API_KEY") or _read_dotenv_value("ODDS_API_KEY", env_path)
    if not key:
        raise ValueError("ODDS_API_KEY no encontrado en el entorno ni en .env")
    return key


def _parse_utc(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _csv_arg(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part).strip() for part in value if str(part).strip())


def _safe_snapshot_name(snapshot_date: str) -> str:
    return snapshot_date.replace(":", "").replace("-", "").replace("T", "_").replace("Z", "Z")


def _response_error_message(response: requests.Response) -> str:
    text = response.text.strip()
    if len(text) > 500:
        text = text[:500] + "..."
    detail = f": {text}" if text else ""

    if response.status_code == 401:
        return (
            "The Odds API rechazó la petición (401 Unauthorized). "
            "Revisa que ODDS_API_KEY sea correcta y que tu plan tenga acceso al endpoint solicitado."
            f"{detail}"
        )
    if response.status_code == 403:
        return (
            "The Odds API rechazó la petición (403 Forbidden). "
            "Normalmente significa que tu plan no incluye este endpoint/mercado, "
            "por ejemplo histórico o player props."
            f"{detail}"
        )
    if response.status_code == 429:
        return f"The Odds API devolvió 429 Rate Limit. Espera antes de repetir la llamada.{detail}"

    return f"The Odds API devolvió HTTP {response.status_code}.{detail}"


def _raise_for_api_error(response: requests.Response) -> None:
    if response.ok:
        return
    raise RuntimeError(_response_error_message(response))


@dataclass(frozen=True)
class SnapshotResult:
    """Raw API wrapper and metadata for one historical snapshot."""

    snapshot_type: str
    requested_date: str
    api_timestamp: str | None
    events: list[dict[str, Any]]
    raw: dict[str, Any]


class HistoricalOddsCollector:
    """
    Fetch opening and closing snapshots from The Odds API.

    For broad match markets (`h2h`, `totals`, `spreads`) the historical odds
    endpoint is enough. For player props and other extended markets, pass the
    concrete market keys supported by the sport/event and consider fetching by
    historical event id once those markets are available for your plan.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        data_dir: str | Path = "data",
        env_path: str | Path = ".env",
        base_url: str = DEFAULT_BASE_URL,
        session: requests.Session | None = None,
        request_timeout: int = 30,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.raw_cache_dir = self.data_dir / "raw" / "odds" / "the_odds_api"
        self.processed_dir = self.data_dir / "processed"
        self.env_path = Path(env_path)
        self.api_key = _api_key(api_key, self.env_path)
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.request_timeout = request_timeout

    def get_historical_snapshot(
        self,
        snapshot_date: str,
        *,
        sport: str = DEFAULT_SPORT,
        markets: str | Iterable[str] = DEFAULT_MARKETS,
        bookmakers: str | Iterable[str] = DEFAULT_BOOKMAKERS,
        regions: str | Iterable[str] = ("eu", "us"),
        odds_format: str = "decimal",
        use_cache: bool = True,
    ) -> SnapshotResult:
        """
        Fetch a market snapshot for a UTC timestamp.

        The Odds API returns the closest snapshot equal to or earlier than the
        requested date. The raw response is cached under data/raw/odds.
        """
        markets_arg = ",".join(_csv_arg(markets))
        bookmakers_arg = ",".join(_csv_arg(bookmakers))
        regions_arg = ",".join(_csv_arg(regions))
        snapshot_date = _iso_z(_parse_utc(snapshot_date))

        cache_path = self._cache_path(sport, snapshot_date, markets_arg, bookmakers_arg)
        if use_cache and cache_path.exists():
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            return self._snapshot_result("cached", snapshot_date, raw)

        endpoint = f"{self.base_url}/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions_arg,
            "markets": markets_arg,
            "bookmakers": bookmakers_arg,
            "date": snapshot_date,
            "oddsFormat": odds_format,
        }
        log.info("Fetching The Odds API snapshot %s markets=%s", snapshot_date, markets_arg)
        response = self.session.get(endpoint, params=params, timeout=self.request_timeout)

        if response.status_code in {404, 422}:
            log.warning("No historical odds returned for %s (%s)", snapshot_date, response.status_code)
            raw = {"timestamp": snapshot_date, "data": [], "error": response.text}
        else:
            _raise_for_api_error(response)
            raw = response.json()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
        return self._snapshot_result("api", snapshot_date, raw)

    def get_current_odds(
        self,
        *,
        sport: str = DEFAULT_SPORT,
        markets: str | Iterable[str] = DEFAULT_MARKETS,
        bookmakers: str | Iterable[str] = DEFAULT_BOOKMAKERS,
        regions: str | Iterable[str] = ("eu", "us"),
        odds_format: str = "decimal",
    ) -> pd.DataFrame:
        """
        Fetch current/pre-match odds.

        This endpoint is useful for validating a free API key. It does not give
        opening/closing history, so it cannot be used for CLV backtests.
        """
        markets_arg = ",".join(_csv_arg(markets))
        bookmakers_arg = ",".join(_csv_arg(bookmakers))
        regions_arg = ",".join(_csv_arg(regions))
        endpoint = f"{CURRENT_BASE_URL}/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions_arg,
            "markets": markets_arg,
            "bookmakers": bookmakers_arg,
            "oddsFormat": odds_format,
        }
        log.info("Fetching The Odds API current odds sport=%s markets=%s", sport, markets_arg)
        response = self.session.get(endpoint, params=params, timeout=self.request_timeout)

        if response.status_code in {404, 422}:
            log.warning("No current odds returned for sport=%s (%s)", sport, response.status_code)
            return normalize_flat_odds_history(pd.DataFrame(), mark_open_close=False)

        _raise_for_api_error(response)
        quote_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        result = SnapshotResult(
            snapshot_type="current",
            requested_date=quote_time,
            api_timestamp=quote_time,
            events=response.json(),
            raw={"timestamp": quote_time},
        )
        flat_rows = self._flatten_snapshot(result, "current")
        return normalize_flat_odds_history(pd.DataFrame(flat_rows), mark_open_close=False)

    def get_current_odds_flat(
        self,
        *,
        sport: str = DEFAULT_SPORT,
        markets: str | Iterable[str] = DEFAULT_MARKETS,
        bookmakers: str | Iterable[str] = DEFAULT_BOOKMAKERS,
        regions: str | Iterable[str] = ("eu", "us"),
        odds_format: str = "decimal",
    ) -> pd.DataFrame:
        """
        Fetch current odds and keep provider fixture fields.

        Use this file for ad-hoc model joins because it retains home_team,
        away_team, and match_date in addition to the normalized odds columns.
        """
        markets_arg = ",".join(_csv_arg(markets))
        bookmakers_arg = ",".join(_csv_arg(bookmakers))
        regions_arg = ",".join(_csv_arg(regions))
        endpoint = f"{CURRENT_BASE_URL}/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions_arg,
            "markets": markets_arg,
            "bookmakers": bookmakers_arg,
            "oddsFormat": odds_format,
        }
        log.info("Fetching The Odds API current odds flat sport=%s markets=%s", sport, markets_arg)
        response = self.session.get(endpoint, params=params, timeout=self.request_timeout)

        if response.status_code in {404, 422}:
            log.warning("No current odds returned for sport=%s (%s)", sport, response.status_code)
            return pd.DataFrame()

        _raise_for_api_error(response)
        quote_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        result = SnapshotResult(
            snapshot_type="current",
            requested_date=quote_time,
            api_timestamp=quote_time,
            events=response.json(),
            raw={"timestamp": quote_time},
        )
        return pd.DataFrame(self._flatten_snapshot(result, "current"))

    def get_historical_events(
        self,
        snapshot_date: str,
        *,
        sport: str = DEFAULT_SPORT,
        use_cache: bool = True,
    ) -> SnapshotResult:
        """
        Fetch historical event ids available at a snapshot.

        Use this before player props or alternate markets; the event id returned
        here is required by The Odds API historical event odds endpoint.
        """
        snapshot_date = _iso_z(_parse_utc(snapshot_date))
        cache_path = self.raw_cache_dir / sport / f"{_safe_snapshot_name(snapshot_date)}__events.json"
        if use_cache and cache_path.exists():
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            return self._snapshot_result("cached", snapshot_date, raw)

        endpoint = f"{self.base_url}/{sport}/events"
        params = {"apiKey": self.api_key, "date": snapshot_date}
        log.info("Fetching The Odds API historical events %s", snapshot_date)
        response = self.session.get(endpoint, params=params, timeout=self.request_timeout)

        if response.status_code in {404, 422}:
            log.warning("No historical events returned for %s (%s)", snapshot_date, response.status_code)
            raw = {"timestamp": snapshot_date, "data": [], "error": response.text}
        else:
            _raise_for_api_error(response)
            raw = response.json()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
        return self._snapshot_result("api", snapshot_date, raw)

    def get_historical_event_snapshot(
        self,
        event_id: str,
        snapshot_date: str,
        *,
        sport: str = DEFAULT_SPORT,
        markets: str | Iterable[str] = DEFAULT_MARKETS,
        bookmakers: str | Iterable[str] = DEFAULT_BOOKMAKERS,
        regions: str | Iterable[str] = ("eu", "us"),
        odds_format: str = "decimal",
        use_cache: bool = True,
    ) -> SnapshotResult:
        """
        Fetch historical odds for one event id.

        This is the endpoint to use for player props, alternate lines, and other
        non-featured markets when your plan and sport support them.
        """
        markets_arg = ",".join(_csv_arg(markets))
        bookmakers_arg = ",".join(_csv_arg(bookmakers))
        regions_arg = ",".join(_csv_arg(regions))
        snapshot_date = _iso_z(_parse_utc(snapshot_date))

        cache_path = self._event_cache_path(sport, event_id, snapshot_date, markets_arg, bookmakers_arg)
        if use_cache and cache_path.exists():
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            return self._snapshot_result("cached", snapshot_date, raw)

        endpoint = f"{self.base_url}/{sport}/events/{event_id}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions_arg,
            "markets": markets_arg,
            "bookmakers": bookmakers_arg,
            "date": snapshot_date,
            "oddsFormat": odds_format,
        }
        log.info("Fetching The Odds API event snapshot %s event=%s markets=%s", snapshot_date, event_id, markets_arg)
        response = self.session.get(endpoint, params=params, timeout=self.request_timeout)

        if response.status_code in {404, 422}:
            log.warning("No historical event odds returned for %s (%s)", snapshot_date, response.status_code)
            raw = {"timestamp": snapshot_date, "data": [], "error": response.text}
        else:
            _raise_for_api_error(response)
            raw = response.json()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
        return self._snapshot_result("api", snapshot_date, raw)

    def fetch_match_lifecycle(
        self,
        target_match_date: str,
        *,
        sport: str = DEFAULT_SPORT,
        markets: str | Iterable[str] = DEFAULT_MARKETS,
        bookmakers: str | Iterable[str] = DEFAULT_BOOKMAKERS,
        regions: str | Iterable[str] = ("eu", "us"),
        opening_hours: int = 48,
        closing_hours: int = 1,
        pause_seconds: float = 1.0,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch opening and closing snapshots for one match kickoff time."""
        match_dt = _parse_utc(target_match_date)
        snapshots = [
            ("opening", match_dt - timedelta(hours=opening_hours)),
            ("closing", match_dt - timedelta(hours=closing_hours)),
        ]

        flat_rows: list[dict[str, Any]] = []
        for idx, (snapshot_type, snapshot_dt) in enumerate(snapshots):
            result = self.get_historical_snapshot(
                _iso_z(snapshot_dt),
                sport=sport,
                markets=markets,
                bookmakers=bookmakers,
                regions=regions,
                use_cache=use_cache,
            )
            flat_rows.extend(self._flatten_snapshot(result, snapshot_type))
            if idx < len(snapshots) - 1 and pause_seconds > 0:
                time.sleep(pause_seconds)

        odds = normalize_flat_odds_history(pd.DataFrame(flat_rows), mark_open_close=True)
        if odds.empty:
            return odds
        return odds

    def fetch_event_lifecycle(
        self,
        event_id: str,
        target_match_date: str,
        *,
        sport: str = DEFAULT_SPORT,
        markets: str | Iterable[str] = DEFAULT_MARKETS,
        bookmakers: str | Iterable[str] = DEFAULT_BOOKMAKERS,
        regions: str | Iterable[str] = ("eu", "us"),
        opening_hours: int = 48,
        closing_hours: int = 1,
        pause_seconds: float = 1.0,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch opening and closing snapshots for one historical event id."""
        match_dt = _parse_utc(target_match_date)
        snapshots = [
            ("opening", match_dt - timedelta(hours=opening_hours)),
            ("closing", match_dt - timedelta(hours=closing_hours)),
        ]

        flat_rows: list[dict[str, Any]] = []
        for idx, (snapshot_type, snapshot_dt) in enumerate(snapshots):
            result = self.get_historical_event_snapshot(
                event_id,
                _iso_z(snapshot_dt),
                sport=sport,
                markets=markets,
                bookmakers=bookmakers,
                regions=regions,
                use_cache=use_cache,
            )
            flat_rows.extend(self._flatten_snapshot(result, snapshot_type))
            if idx < len(snapshots) - 1 and pause_seconds > 0:
                time.sleep(pause_seconds)

        return normalize_flat_odds_history(pd.DataFrame(flat_rows), mark_open_close=True)

    def save_lifecycle(
        self,
        odds: pd.DataFrame,
        *,
        output_path: str | Path | None = None,
    ) -> Path:
        """Persist normalized lifecycle odds to CSV."""
        if output_path is None:
            output_path = self.processed_dir / "odds_history.csv"
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        odds.to_csv(output, index=False)
        log.info("Saved %d normalized odds rows to %s", len(odds), output)
        return output

    def save_current_odds(
        self,
        odds: pd.DataFrame,
        *,
        output_path: str | Path | None = None,
    ) -> Path:
        """Persist current odds to CSV."""
        if output_path is None:
            output_path = self.processed_dir / "odds_current.csv"
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        odds.to_csv(output, index=False)
        log.info("Saved %d current odds rows to %s", len(odds), output)
        return output

    def save_current_odds_flat(
        self,
        odds: pd.DataFrame,
        *,
        output_path: str | Path | None = None,
    ) -> Path:
        """Persist current odds with provider fixture fields to CSV."""
        if output_path is None:
            output_path = self.processed_dir / "odds_current_flat.csv"
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        odds.to_csv(output, index=False)
        log.info("Saved %d current flat odds rows to %s", len(odds), output)
        return output

    def save_open_close_pivot(
        self,
        odds: pd.DataFrame,
        *,
        output_path: str | Path | None = None,
    ) -> Path:
        """Persist one row per bet with opening and closing odds side by side."""
        if output_path is None:
            output_path = self.processed_dir / "odds_open_close.csv"
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        pivot = self.open_close_pivot(odds)
        pivot.to_csv(output, index=False)
        log.info("Saved %d open/close odds rows to %s", len(pivot), output)
        return output

    def save_events(self, snapshot: SnapshotResult, *, output_path: str | Path | None = None) -> Path:
        """Persist historical event ids to CSV so they can be used for event odds."""
        if output_path is None:
            output_path = self.processed_dir / "odds_events.csv"
        rows = [
            {
                "event_id": event.get("id"),
                "commence_time": event.get("commence_time"),
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
                "sport_key": event.get("sport_key"),
                "sport_title": event.get("sport_title"),
            }
            for event in snapshot.events
        ]
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)
        log.info("Saved %d historical events to %s", len(rows), output)
        return output

    @staticmethod
    def open_close_pivot(odds: pd.DataFrame) -> pd.DataFrame:
        """Return opening_odds and closing_odds columns for CLV backtests."""
        if odds.empty:
            return pd.DataFrame()

        flagged = odds[odds["is_opening"].fillna(False) | odds["is_closing"].fillna(False)].copy()
        if flagged.empty:
            return pd.DataFrame()

        flagged["quote_label"] = "opening"
        flagged.loc[flagged["is_closing"].fillna(False), "quote_label"] = "closing"
        flagged["_line_key"] = flagged["line"].where(flagged["line"].notna(), "__NO_LINE__")
        index_cols = ["match_id", "bookmaker", "market", "selection", "_line_key"]
        line_values = flagged[index_cols + ["line"]].drop_duplicates(index_cols)
        odds_pivot = flagged.pivot_table(
            index=index_cols,
            columns="quote_label",
            values="odds_decimal",
            aggfunc="first",
        ).reset_index()
        time_pivot = flagged.pivot_table(
            index=index_cols,
            columns="quote_label",
            values="quote_time",
            aggfunc="first",
        ).reset_index()

        odds_pivot = odds_pivot.rename(columns={"opening": "opening_odds", "closing": "closing_odds"})
        time_pivot = time_pivot.rename(columns={"opening": "opening_quote_time", "closing": "closing_quote_time"})
        pivot = odds_pivot.merge(time_pivot, on=index_cols, how="left")
        pivot = pivot.merge(line_values, on=index_cols, how="left")
        pivot = pivot.drop(columns=["_line_key"])
        ordered_cols = ["match_id", "bookmaker", "market", "selection", "line"]
        extra_cols = [col for col in pivot.columns if col not in ordered_cols]
        return pivot[ordered_cols + extra_cols]

    def _cache_path(self, sport: str, snapshot_date: str, markets: str, bookmakers: str) -> Path:
        market_part = markets.replace(",", "-") or "all"
        bookmaker_part = bookmakers.replace(",", "-") or "all"
        filename = f"{_safe_snapshot_name(snapshot_date)}__{market_part}__{bookmaker_part}.json"
        return self.raw_cache_dir / sport / filename

    def _event_cache_path(
        self,
        sport: str,
        event_id: str,
        snapshot_date: str,
        markets: str,
        bookmakers: str,
    ) -> Path:
        market_part = markets.replace(",", "-") or "all"
        bookmaker_part = bookmakers.replace(",", "-") or "all"
        filename = f"{_safe_snapshot_name(snapshot_date)}__{market_part}__{bookmaker_part}.json"
        return self.raw_cache_dir / sport / "events" / event_id / filename

    @staticmethod
    def _snapshot_result(snapshot_type: str, requested_date: str, raw: dict[str, Any]) -> SnapshotResult:
        api_timestamp = raw.get("timestamp")
        data = raw.get("data", [])
        if isinstance(data, dict):
            events = [data]
        elif isinstance(data, list):
            events = data
        else:
            events = []
        return SnapshotResult(
            snapshot_type=snapshot_type,
            requested_date=requested_date,
            api_timestamp=api_timestamp,
            events=events,
            raw=raw,
        )

    @staticmethod
    def _flatten_snapshot(result: SnapshotResult, snapshot_type: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        quote_time = result.api_timestamp or result.requested_date

        for event in result.events:
            match_id = event.get("id")
            for bookmaker in event.get("bookmakers", []):
                bookmaker_key = bookmaker.get("key") or bookmaker.get("title")
                for market in bookmaker.get("markets", []):
                    market_key = market.get("key")
                    for outcome in market.get("outcomes", []):
                        odds_decimal = outcome.get("price")
                        if odds_decimal in (None, ""):
                            continue
                        rows.append({
                            "match_id": match_id,
                            "match_date": event.get("commence_time"),
                            "home_team": event.get("home_team"),
                            "away_team": event.get("away_team"),
                            "bookmaker": bookmaker_key,
                            "market": market_key,
                            "selection": outcome.get("description") or outcome.get("name"),
                            "line": outcome.get("point"),
                            "odds_decimal": odds_decimal,
                            "timestamp": quote_time,
                            "snapshot_type": snapshot_type,
                        })

        return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch historical odds snapshots from The Odds API.")
    parser.add_argument("--target-date", default=None, help="Kickoff UTC timestamp, e.g. 2022-12-18T15:00:00Z.")
    parser.add_argument("--list-events-date", default=None, help="Only fetch historical event ids at this UTC date.")
    parser.add_argument("--current", action="store_true", help="Fetch current odds instead of historical snapshots.")
    parser.add_argument("--current-flat", action="store_true", help="Fetch current odds retaining teams and match dates.")
    parser.add_argument("--sport", default=DEFAULT_SPORT, help=f"Sport key. Default: {DEFAULT_SPORT}.")
    parser.add_argument("--event-id", default=None, help="Historical event id for props/alternate markets.")
    parser.add_argument("--markets", default=",".join(DEFAULT_MARKETS), help="Comma-separated market keys.")
    parser.add_argument("--bookmakers", default=",".join(DEFAULT_BOOKMAKERS), help="Comma-separated bookmakers.")
    parser.add_argument("--regions", default="eu,us", help="Comma-separated regions.")
    parser.add_argument("--opening-hours", type=int, default=48, help="Hours before kickoff for opening snapshot.")
    parser.add_argument("--closing-hours", type=int, default=1, help="Hours before kickoff for closing snapshot.")
    parser.add_argument("--pause-seconds", type=float, default=1.0, help="Delay between API calls.")
    parser.add_argument("--data-dir", default="data", help="Project data directory.")
    parser.add_argument("--env-path", default=".env", help="Path to .env containing ODDS_API_KEY.")
    parser.add_argument("--output", default=None, help="Output CSV path. Defaults to data/processed/odds_history.csv.")
    parser.add_argument(
        "--pivot-output",
        default=None,
        help="Open/close pivot CSV path. Defaults to data/processed/odds_open_close.csv.",
    )
    parser.add_argument("--events-output", default=None, help="Events CSV path when using --list-events-date.")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached raw snapshots.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args()
    collector = HistoricalOddsCollector(data_dir=args.data_dir, env_path=args.env_path)

    if args.current:
        odds = collector.get_current_odds(
            sport=args.sport,
            markets=args.markets,
            bookmakers=args.bookmakers,
            regions=args.regions,
        )
        output = collector.save_current_odds(odds, output_path=args.output)
        print(f"Saved {len(odds)} current odds rows to {output}")
        return

    if args.current_flat:
        odds = collector.get_current_odds_flat(
            sport=args.sport,
            markets=args.markets,
            bookmakers=args.bookmakers,
            regions=args.regions,
        )
        output = collector.save_current_odds_flat(odds, output_path=args.output)
        print(f"Saved {len(odds)} current flat odds rows to {output}")
        return

    if args.list_events_date:
        snapshot = collector.get_historical_events(args.list_events_date, sport=args.sport, use_cache=not args.no_cache)
        events_output = collector.save_events(snapshot, output_path=args.events_output)
        print(f"Saved {len(snapshot.events)} historical events to {events_output}")
        return

    if not args.target_date:
        raise SystemExit("--target-date es obligatorio salvo que uses --list-events-date")

    lifecycle_kwargs = {
        "sport": args.sport,
        "markets": args.markets,
        "bookmakers": args.bookmakers,
        "regions": args.regions,
        "opening_hours": args.opening_hours,
        "closing_hours": args.closing_hours,
        "pause_seconds": args.pause_seconds,
        "use_cache": not args.no_cache,
    }
    if args.event_id:
        odds = collector.fetch_event_lifecycle(args.event_id, args.target_date, **lifecycle_kwargs)
    else:
        odds = collector.fetch_match_lifecycle(args.target_date, **lifecycle_kwargs)
    output = collector.save_lifecycle(odds, output_path=args.output)
    pivot_output = collector.save_open_close_pivot(odds, output_path=args.pivot_output)
    print(f"Saved {len(odds)} odds rows to {output}")
    print(f"Saved open/close pivot to {pivot_output}")


if __name__ == "__main__":
    main()
