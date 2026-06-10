"""
src/evaluation/push_signals_supabase.py
=======================================
Sube las senales del modelo a la tabla `model_signals` de Supabase para que
el tracker web (Netlify) las muestre antes de apostar.

Fuentes (todas locales, sin llamar a The Odds API):
  - outputs/wc2026_ev_h2h_shortlist.csv        -> bucket core   (apostar 0.5u)
  - outputs/wc2026_ev_h2h_manual_review.csv    -> bucket manual_review
  - outputs/wc2026_multi_market_paper_shortlist.csv -> bucket paper (O/U, BTTS)

Semantica de snapshot: cada run BORRA todas las senales y sube las actuales.
El tracker siempre muestra el ultimo estado del modelo, no un historico.

Requiere en .env (NUNCA subir a Git, ya esta en .gitignore via .env):
  SUPABASE_URL=https://bcqfyipszvktufenlcep.supabase.co
  SUPABASE_SERVICE_KEY=eyJ...   (service_role key, Settings > API en Supabase)

La service key se queda SOLO en tu .env local. El HTML usa la anon key y solo
puede LEER la tabla (RLS), nunca escribir.

Uso:
    python -m src.evaluation.push_signals_supabase
    python -m src.evaluation.push_signals_supabase --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


DEFAULT_CORE_INPUT = "outputs/wc2026_ev_h2h_shortlist.csv"
DEFAULT_REVIEW_INPUT = "outputs/wc2026_ev_h2h_manual_review.csv"
DEFAULT_PAPER_INPUT = "outputs/wc2026_multi_market_paper_shortlist.csv"
TABLE = "model_signals"

# Etiquetas de mercado que el HTML sabe mapear (MODEL_TO_TRACKER)
PAPER_MARKET_LABEL = {
    ("total_goals", "over"): "Over 2.5",
    ("total_goals", "under"): "Under 2.5",
    ("btts", "yes"): "BTTS Si",
    ("btts", "no"): "BTTS No",
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


def _signal_id(date, home, away, mercado, seleccion, bookmaker) -> str:
    raw = "|".join(str(v or "") for v in (date, home, away, mercado, seleccion, bookmaker))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _num(value):
    out = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(out) else float(out)


def _load_h2h(path: str, bucket: str, default_action: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        accion = str(r.get("review_action") or default_action)
        rows.append({
            "signal_id": _signal_id(r.get("date"), r.get("home_team"), r.get("away_team"),
                                    r.get("mercado"), r.get("seleccion"), r.get("bookmaker")),
            "match_date": str(r.get("date") or ""),
            "home_team": str(r.get("home_team") or ""),
            "away_team": str(r.get("away_team") or ""),
            "mercado": str(r.get("mercado") or ""),
            "seleccion": str(r.get("seleccion") or ""),
            "bookmaker": str(r.get("bookmaker") or ""),
            "cuota": _num(r.get("cuota")),
            "prob_modelo": _num(r.get("prob_modelo")),
            "cuota_justa": _num(r.get("cuota_justa")),
            "ev_pct": _num(r.get("ev_pct")),
            "fiabilidad_pct": _num(r.get("fiabilidad_pct")),
            "fiabilidad_nivel": str(r.get("fiabilidad_nivel") or ""),
            "bucket": bucket,
            "accion": accion,
        })
    return rows


def _load_paper(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        market = str(r.get("market") or "").strip().lower()
        selection = str(r.get("selection") or "").strip().lower()
        mercado = PAPER_MARKET_LABEL.get((market, selection))
        if mercado is None:
            # linea no 2.5 u otro mercado: etiqueta generica legible
            line = r.get("line")
            line_txt = "" if pd.isna(line) else f" {line}"
            mercado = f"{str(r.get('selection') or '')}{line_txt} ({market})"
        prob = _num(r.get("model_probability"))
        rows.append({
            "signal_id": _signal_id(r.get("date"), r.get("home_team"), r.get("away_team"),
                                    mercado, r.get("selection"), r.get("bookmaker")),
            "match_date": str(r.get("date") or ""),
            "home_team": str(r.get("home_team") or ""),
            "away_team": str(r.get("away_team") or ""),
            "mercado": mercado,
            "seleccion": str(r.get("selection") or ""),
            "bookmaker": str(r.get("bookmaker") or ""),
            "cuota": _num(r.get("odds_decimal")),
            "prob_modelo": prob,
            "cuota_justa": _num(r.get("fair_odds")) or (round(1.0 / prob, 3) if prob else None),
            "ev_pct": _num(r.get("ev_pct")),
            "fiabilidad_pct": _num(r.get("fiabilidad_pct")),
            "fiabilidad_nivel": str(r.get("fiabilidad_nivel") or ""),
            "bucket": "paper",
            "accion": str(r.get("tracking_action") or "paper_track"),
        })
    return rows


def collect_signals(core_input: str, review_input: str, paper_input: str) -> list[dict]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = (
        _load_h2h(core_input, bucket="core", default_action="core_candidate")
        + _load_h2h(review_input, bucket="manual_review", default_action="paper_only")
        + _load_paper(paper_input)
    )
    seen: set[str] = set()
    out = []
    for row in rows:
        if row["signal_id"] in seen:
            continue
        seen.add(row["signal_id"])
        row["updated_at"] = now
        out.append(row)
    return out


def push(rows: list[dict], *, url: str, service_key: str, timeout: int = 30) -> None:
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    base = f"{url.rstrip('/')}/rest/v1/{TABLE}"

    # Snapshot: borrar todo y subir lo actual
    r = requests.delete(f"{base}?signal_id=neq.__never__", headers=headers, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"DELETE fallo ({r.status_code}): {r.text[:300]}")

    if not rows:
        return
    for i in range(0, len(rows), 200):
        chunk = rows[i:i + 200]
        r = requests.post(base, headers=headers, data=json.dumps(chunk), timeout=timeout)
        if not r.ok:
            raise RuntimeError(f"INSERT fallo ({r.status_code}): {r.text[:300]}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sube senales del modelo a Supabase para el tracker web.")
    parser.add_argument("--core-input", default=DEFAULT_CORE_INPUT)
    parser.add_argument("--review-input", default=DEFAULT_REVIEW_INPUT)
    parser.add_argument("--paper-input", default=DEFAULT_PAPER_INPUT)
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--dry-run", action="store_true", help="Muestra las senales sin subirlas.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = collect_signals(args.core_input, args.review_input, args.paper_input)

    by_bucket: dict[str, int] = {}
    for row in rows:
        by_bucket[row["bucket"]] = by_bucket.get(row["bucket"], 0) + 1
    print(f"Senales recolectadas: {len(rows)}  {by_bucket}")

    if args.dry_run:
        for row in rows:
            print(f"  [{row['bucket']:<13}] {row['home_team']} vs {row['away_team']} | "
                  f"{row['mercado']:<18} @{row['cuota']} EV={row['ev_pct']}% -> {row['accion']}")
        return

    env_path = Path(args.env_path)
    url = _env("SUPABASE_URL", env_path, default="https://bcqfyipszvktufenlcep.supabase.co")
    key = _env("SUPABASE_SERVICE_KEY", env_path)
    push(rows, url=url, service_key=key)
    print(f"Subidas {len(rows)} senales a {TABLE}. El tracker web ya las ve.")


if __name__ == "__main__":
    main()
    