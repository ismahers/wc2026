"""
Extensión de push_signals_supabase para subir también el radar de mercados.
Añadir al final de push_signals_supabase.py, o ejecutar por separado.
"""
import json, os, hashlib, requests, pandas as pd
from pathlib import Path
from datetime import datetime, timezone

DEFAULT_RADAR_INPUT = "outputs/wc2026_market_probabilities.csv"
RADAR_TABLE = "market_radar"

RADAR_COLS = [
    "signal_id","match_date","home_team","away_team","market","selection","line",
    "model_probability","fair_odds","min_odds_ev5","min_odds_ev10","min_odds_ev15",
    "betting_status","default_tracking_action","fiabilidad_pct","fiabilidad_nivel",
    "model_confidence","source_model","updated_at"
]

def _num(v):
    x = pd.to_numeric(v, errors='coerce')
    return None if pd.isna(x) else round(float(x), 4)

try:
    from phase_helper import get_phase
except ImportError:
    from src.evaluation.phase_helper import get_phase

def _get_phase_radar(home, away):
    return get_phase(home, away)

def load_radar(path=DEFAULT_RADAR_INPUT):
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for _, r in df.iterrows():
        sid = hashlib.sha1("|".join(str(r.get(c,"")) for c in
            ["date","home_team","away_team","market","selection","line"]).encode()).hexdigest()[:16]
        rows.append({
            "signal_id": sid,
            "match_date": str(r.get("date") or ""),
            "home_team": str(r.get("home_team") or ""),
            "away_team": str(r.get("away_team") or ""),
            "market": str(r.get("market") or ""),
            "selection": str(r.get("selection") or ""),
            "line": _num(r.get("line")),
            "model_probability": _num(r.get("model_probability")),
            "fair_odds": _num(r.get("fair_odds")),
            "min_odds_ev5": _num(r.get("min_odds_ev5")),
            "min_odds_ev10": _num(r.get("min_odds_ev10")),
            "min_odds_ev15": _num(r.get("min_odds_ev15")),
            "betting_status": str(r.get("betting_status") or ""),
            "default_tracking_action": str(r.get("default_tracking_action") or ""),
            "fiabilidad_pct": _num(r.get("fiabilidad_pct")),
            "fiabilidad_nivel": str(r.get("fiabilidad_nivel") or ""),
            "model_confidence": str(r.get("model_confidence") or ""),
            "source_model": str(r.get("source_model") or ""),
            "phase": _get_phase_radar(str(r.get("home_team","")), str(r.get("away_team",""))),
            "updated_at": now,
        })
    return rows

def push_radar(rows, *, url, service_key, timeout=30):
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    base = f"{url.rstrip('/')}/rest/v1/{RADAR_TABLE}"
    r = requests.delete(f"{base}?signal_id=neq.__never__", headers=headers, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"DELETE radar fallo ({r.status_code}): {r.text[:300]}")
    for i in range(0, len(rows), 200):
        chunk = rows[i:i+200]
        r = requests.post(base, headers=headers, data=json.dumps(chunk), timeout=timeout)
        if not r.ok:
            raise RuntimeError(f"INSERT radar fallo ({r.status_code}): {r.text[:300]}")

if __name__ == "__main__":
    from pathlib import Path
    def _env(key):
        v = os.getenv(key)
        if v: return v
        env = Path(".env")
        if env.exists():
            for l in env.read_text().splitlines():
                if l.strip().startswith(key+"="):
                    return l.split("=",1)[1].strip().strip('"').strip("'")
        raise ValueError(f"{key} no encontrado")
    
    rows = load_radar()
    print(f"Radar: {len(rows)} filas")
    url = _env("SUPABASE_URL") if os.getenv("SUPABASE_URL") else "https://bcqfyipszvktufenlcep.supabase.co"
    key = _env("SUPABASE_SERVICE_KEY")
    push_radar(rows, url=url, service_key=key)
    print(f"Subidas {len(rows)} filas a {RADAR_TABLE}")
    