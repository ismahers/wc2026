"""
Push the WC2026 player-prop mobile radar to Supabase.

This uploads a compact, phone-friendly subset of outputs/wc2026_player_prop_radar.csv.
It does not call odds APIs and it never enables stake automatically.

Expected Supabase table: player_prop_radar

Usage:
  python -m src.evaluation.player_prop_radar_push --dry-run
  python -m src.evaluation.player_prop_radar_push
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

try:
    from phase_helper import get_phase
except ImportError:
    from src.evaluation.phase_helper import get_phase


DEFAULT_INPUT = "outputs/wc2026_player_prop_radar.csv"
TABLE = "player_prop_radar"
LOCAL_TZ = ZoneInfo("Europe/Madrid")
ACTIONABLE = {
    "manual_review_if_odds_available",
    "radar_watch",
    "card_radar_watch",
    "low_confidence_review",
    "longshot_manual_only",
}
PREFERRED_ACTION_ORDER = {
    "manual_review_if_odds_available": 1,
    "radar_watch": 2,
    "card_radar_watch": 3,
    "longshot_manual_only": 4,
    "low_confidence_review": 5,
}


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


def _env(key: str, env_path: Path, default: str | None = None) -> str:
    value = os.getenv(key) or _read_dotenv_value(key, env_path) or default
    if not value:
        raise ValueError(f"{key} no encontrado en el entorno ni en {env_path}")
    return value


def _supabase_rest_url(url: str, table: str) -> str:
    base = url.rstrip("/")
    if base.endswith("/rest/v1"):
        return f"{base}/{table}"
    return f"{base}/rest/v1/{table}"


def _num(value, digits: int = 4):
    out = pd.to_numeric(value, errors="coerce")
    if pd.isna(out):
        return None
    return round(float(out), digits)


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "si", "sí"}


def _signal_id(row: pd.Series) -> str:
    raw = "|".join(str(row.get(col, "")) for col in [
        "match_number", "team", "player_key", "market", "line", "selection"
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _home_away(row: pd.Series) -> tuple[str, str]:
    team = str(row.get("team") or "")
    opponent = str(row.get("opponent") or "")
    if str(row.get("team_side") or "") == "away":
        return opponent, team
    return team, opponent


def _market_label(row: pd.Series) -> str:
    market = str(row.get("market") or "")
    line = _num(row.get("line"), 1)
    selection = str(row.get("selection") or "Over")
    labels = {
        "shots": "Tiros",
        "shots_on_target": "Tiros a puerta",
        "tackles": "Entradas",
        "fouls_committed": "Faltas cometidas",
        "yellow_card": "Tarjeta amarilla",
        "goalkeeper_saves": "Paradas",
    }
    if market == "yellow_card":
        return labels.get(market, market)
    if selection:
        return f"{labels.get(market, market)} {selection}"
    return f"{labels.get(market, market)} Over {line:g}" if line is not None else labels.get(market, market)


def _lineup_rule(row: pd.Series) -> str:
    market = str(row.get("market") or "")
    expected_minutes = _num(row.get("pricing_minutes"), 1) or _num(row.get("expected_minutes"), 1) or 0.0
    lineup_status = str(row.get("lineup_status") or "")

    if market == "goalkeeper_saves":
        return "solo si portero titular"
    if lineup_status in {"locked_starter", "probable_starter"} and expected_minutes >= 45:
        return "revisar si mantiene titularidad"
    if expected_minutes < 45:
        return "solo si confirma titular"
    return "revisar alineacion"


def _mobile_action(row: pd.Series) -> str:
    if not _bool(row.get("paper_tracking_allowed", False)):
        return "baja confianza - solo paper"

    tracking = str(row.get("tracking_action") or "")
    fair_odds = _num(row.get("fair_odds"), 3)
    min_odds = None if fair_odds is None else round(fair_odds * 1.10, 2)
    suffix = "" if min_odds is None else f" y cuota >= {min_odds:.2f}"

    if tracking == "manual_review_if_odds_available":
        return f"revisar si titular{suffix}"
    if tracking == "radar_watch":
        return f"watchlist{suffix}"
    if tracking == "card_radar_watch":
        return f"amarilla: solo revision{suffix}"
    if tracking == "longshot_manual_only":
        return "cuota alta: revisar manual"
    if tracking == "low_confidence_review":
        return "baja confianza: paper"
    return "no usar"


def _today_local() -> str:
    return datetime.now(LOCAL_TZ).date().isoformat()


def _normalize_date(value: str | None) -> str | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Fecha no valida: {value}")
    return parsed.date().isoformat()


def _filter_scope(
    df: pd.DataFrame,
    *,
    target_date: str | None,
    match_numbers: list[int] | None,
    all_matches: bool,
) -> tuple[pd.DataFrame, str]:
    if match_numbers:
        return df[df["match_number"].isin(match_numbers)].copy(), "matches=" + ",".join(map(str, match_numbers))
    if all_matches:
        return df.copy(), "all"

    target = _normalize_date(target_date) or _today_local()
    dates = pd.to_datetime(df["match_date"], errors="coerce").dt.date.astype(str)
    return df[dates.eq(target)].copy(), f"date={target}"


def load_player_prop_rows(
    path: str = DEFAULT_INPUT,
    *,
    max_rows: int = 600,
    target_date: str | None = None,
    match_numbers: list[int] | None = None,
    all_matches: bool = False,
) -> tuple[list[dict], str]:
    if not os.path.exists(path):
        return [], "missing_input"

    df = pd.read_csv(path)
    df, scope = _filter_scope(
        df,
        target_date=target_date,
        match_numbers=match_numbers,
        all_matches=all_matches,
    )
    df = df[df["tracking_action"].isin(ACTIONABLE)].copy()
    if "fair_odds" in df.columns:
        df = df[df["fair_odds"].between(1.35, 8.0, inclusive="both")]

    df["mobile_priority"] = df["tracking_action"].map(PREFERRED_ACTION_ORDER).fillna(99)
    df["min_odds_review"] = pd.to_numeric(df["fair_odds"], errors="coerce") * 1.10
    df = df.sort_values(
        ["match_number", "mobile_priority", "paper_tracking_allowed", "model_probability"],
        ascending=[True, True, False, False],
    ).head(max_rows)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[dict] = []
    for _, r in df.iterrows():
        home, away = _home_away(r)
        display_minutes = _num(r.get("pricing_minutes"), 1) or _num(r.get("expected_minutes"), 1)
        rows.append({
            "signal_id": _signal_id(r),
            "match_number": int(r["match_number"]),
            "match_date": str(r.get("match_date") or ""),
            "phase": get_phase(home, away),
            "home_team": home,
            "away_team": away,
            "team": str(r.get("team") or ""),
            "opponent": str(r.get("opponent") or ""),
            "player_key": str(r.get("player_key") or ""),
            "player_name": str(r.get("player_name") or ""),
            "position_broad": str(r.get("position_broad") or ""),
            "market": str(r.get("market") or ""),
            "market_label": _market_label(r),
            "line": _num(r.get("line"), 1),
            "selection": str(r.get("selection") or ""),
            "model_probability": _num(r.get("model_probability")),
            "fair_odds": _num(r.get("fair_odds"), 3),
            "min_odds_review": _num(r.get("min_odds_review"), 3),
            "expected_minutes": display_minutes,
            "lineup_status": str(r.get("lineup_status") or ""),
            "lineup_rule": _lineup_rule(r),
            "mobile_action": _mobile_action(r),
            "action_if_starter": "revisar cuota real",
            "action_if_bench": "descartar salvo linea muy baja",
            "tracking_action": str(r.get("tracking_action") or ""),
            "data_quality_tier": str(r.get("data_quality_tier") or ""),
            "paper_tracking_allowed": _bool(r.get("paper_tracking_allowed", False)),
            "updated_at": now,
        })
    return rows, scope


def push(rows: list[dict], *, url: str, service_key: str, timeout: int = 30) -> None:
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    base = _supabase_rest_url(url, TABLE)

    r = requests.delete(f"{base}?signal_id=neq.__never__", headers=headers, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"DELETE {TABLE} fallo ({r.status_code}): {r.text[:300]}")

    for i in range(0, len(rows), 200):
        chunk = rows[i:i + 200]
        r = requests.post(base, headers=headers, data=json.dumps(chunk), timeout=timeout)
        if not r.ok:
            raise RuntimeError(f"INSERT {TABLE} fallo ({r.status_code}): {r.text[:300]}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sube radar de player props a Supabase para vista movil.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--max-rows", type=int, default=600)
    parser.add_argument("--date", help="Fecha YYYY-MM-DD a subir. Por defecto, hoy en Europe/Madrid.")
    parser.add_argument("--match-number", type=int, action="append", help="Partido concreto. Puede repetirse.")
    parser.add_argument("--all", action="store_true", help="Subir todas las props accionables del torneo.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows, scope = load_player_prop_rows(
        args.input,
        max_rows=args.max_rows,
        target_date=args.date,
        match_numbers=args.match_number,
        all_matches=args.all,
    )
    print(f"Player prop mobile rows: {len(rows)} ({scope})")

    if args.dry_run:
        for row in rows[:30]:
            print(
                f"{row['match_number']:>2} {row['team']} | {row['player_name']} | "
                f"{row['market_label']} | justa {row['fair_odds']} | {row['mobile_action']}"
            )
        return

    env_path = Path(args.env_path)
    url = _env("SUPABASE_URL", env_path, default="https://bcqfyipszvktufenlcep.supabase.co")
    key = _env("SUPABASE_SERVICE_KEY", env_path)
    push(rows, url=url, service_key=key)
    print(f"Subidas {len(rows)} filas a {TABLE}.")


if __name__ == "__main__":
    main()
