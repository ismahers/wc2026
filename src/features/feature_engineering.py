"""
src/features/feature_engineering.py
=====================================
Construcción de features para el Pronosticador Mundial 2026.

Para cada partido genera:
  - Features de forma reciente (últimos N partidos de cada equipo)
  - Features de contexto (fase, sede, confederación, neutral/host)
  - Features del árbitro
  - Rating compuesto WC2026 (valor mercado + Elo + racha)
  - Media de córners por selección (todos sus partidos anteriores con dato)
  - Variables objetivo para cada mercado

Optimización de rendimiento
-----------------------------
La versión original era O(n²): para cada partido filtraba todo el DataFrame.
Esta versión preconstruye un índice {equipo: [filas ordenadas por fecha]}
y usa búsqueda binaria para encontrar los N partidos anteriores en O(log n).
En un dataset de 32k partidos esto reduce el tiempo de ~20 min a ~30 seg.

Media de córners por selección (sin data leakage)
--------------------------------------------------
La feature de forma "corners_for_10" se queda vacía para el Mundial porque las
selecciones no tienen córners en su historial reciente (solo los 314 partidos de
StatsBomb los tienen). En su lugar usamos la MEDIA de córners de cada selección
sobre TODOS sus partidos anteriores con dato. Para selecciones sin historial de
córners, se imputa desde el rating de ataque (recta ataque→córners ajustada con
los equipos que sí tienen ambos), y como último recurso la media global.
Todo estrictamente con partidos ANTERIORES a la fecha → sin leakage.

Mundial = sede neutral
------------------------
En el Mundial todos los partidos son en sede neutral. Las features de rating
usan diferencia absoluta (no direccional) entre equipos. La única excepción
son USA/México/Canadá cuando juegan en su país sede.
"""

import os
import bisect
from typing import Optional, Callable

import numpy as np
import pandas as pd


# ── Configuración ─────────────────────────────────────────────────────────────

N_RECENT = 10


# ── Carga de ratings WC2026 ───────────────────────────────────────────────────

_RATINGS_CACHE: dict = {}
_PLAYER_STATS_CACHE: dict = {}

def _load_ratings(ratings_path: str = "./data/processed/team_ratings_wc2026.csv") -> dict:
    """Carga el rating compuesto de selecciones WC2026 en memoria (singleton)."""
    global _RATINGS_CACHE
    if _RATINGS_CACHE:
        return _RATINGS_CACHE
    try:
        df = pd.read_csv(ratings_path)
        for _, row in df.iterrows():
            _RATINGS_CACHE[row["team_canonical"]] = {
                "rating":        float(row.get("rating", 50.0)),
                "attack_norm":   float(row.get("attack_value_norm", 0.5)),
                "midfield_norm": float(row.get("midfield_value_norm", 0.5)),
                "defense_norm":  float(row.get("defense_value_norm", 0.5)),
            }
        print(f"  → Ratings WC2026 cargados: {len(_RATINGS_CACHE)} selecciones")
    except Exception as e:
        print(f"  ⚠ No se pudo cargar team_ratings_wc2026.csv: {e}")
    return _RATINGS_CACHE


def _load_player_stats(stats_path: str = "./data/processed/team_player_stats_wc2026.csv") -> dict:
    """Carga stats de jugadores por selección WC2026 en memoria (singleton)."""
    global _PLAYER_STATS_CACHE
    if _PLAYER_STATS_CACHE:
        return _PLAYER_STATS_CACHE
    try:
        df = pd.read_csv(stats_path)
        for _, row in df.iterrows():
            _PLAYER_STATS_CACHE[row["team_canonical"]] = {
                "yellow_per_90":  float(row["avg_yellow_per_90"])  if pd.notna(row["avg_yellow_per_90"])  else np.nan,
                "red_per_90":     float(row["avg_red_per_90"])     if pd.notna(row["avg_red_per_90"])     else np.nan,
                "goals_per_90":   float(row["avg_goals_per_90"])   if pd.notna(row["avg_goals_per_90"])   else np.nan,
                "assists_per_90": float(row["avg_assists_per_90"]) if pd.notna(row["avg_assists_per_90"]) else np.nan,
                "avg_caps":       float(row["avg_international_caps"]) if pd.notna(row["avg_international_caps"]) else np.nan,
            }
        print(f"  → Player stats WC2026 cargados: {len(_PLAYER_STATS_CACHE)} selecciones")
    except Exception as e:
        print(f"  ⚠ No se pudo cargar team_player_stats_wc2026.csv: {e}")
    return _PLAYER_STATS_CACHE


# ── Índice de historial por equipo (optimización O(n) → O(log n)) ─────────────

def build_team_index(df: pd.DataFrame) -> dict[str, list]:
    """
    Preconstruye un índice {equipo: [(fecha, idx_fila), ...]} ordenado por fecha.
    Permite encontrar los N partidos anteriores con búsqueda binaria en O(log n)
    en lugar de filtrar todo el DataFrame en cada llamada O(n).

    Con 32k partidos y ~600 equipos esto pasa de ~20 min a ~30 seg.
    """
    index: dict[str, list] = {}
    for i, row in df.iterrows():
        date = row["date"]
        for team_col in ["home_team", "away_team"]:
            team = row[team_col]
            if pd.isna(team):
                continue
            if team not in index:
                index[team] = []
            index[team].append((date, i))

    # Ordenar por fecha
    for team in index:
        index[team].sort(key=lambda x: x[0])

    return index


def _get_team_history_fast(
    df: pd.DataFrame,
    index: dict[str, list],
    team: str,
    before_date: pd.Timestamp,
    n: int,
) -> pd.DataFrame:
    """
    Devuelve los últimos N partidos de un equipo antes de before_date
    usando el índice preconstruido. O(log n + n) en lugar de O(total_partidos).
    """
    if "result" not in df.columns:
        return pd.DataFrame()

    entries = index.get(team, [])
    if not entries:
        return pd.DataFrame()

    # Búsqueda binaria: encontrar el punto de corte antes de before_date
    dates = [e[0] for e in entries]
    cut = bisect.bisect_left(dates, before_date)

    # Tomar los N anteriores al corte, de más reciente a más antiguo
    recent_entries = entries[max(0, cut - n * 3): cut]  # coger más para filtrar result
    recent_entries = list(reversed(recent_entries))

    rows = []
    for _, idx in recent_entries:
        row = df.iloc[idx]
        if pd.notna(row.get("result")):
            rows.append(row)
        if len(rows) >= n:
            break

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ── Media de córners por selección (prefix sums, sin leakage) ─────────────────

def build_corners_prefix(df: pd.DataFrame, index: dict[str, list]) -> dict[str, tuple]:
    """
    Para cada equipo, precomputa (dates, prefix_sum, prefix_cnt) de los córners
    A FAVOR alineados con su historial ordenado por fecha.
      - prefix_sum[k] = suma de córners-a-favor en los primeros k partidos
      - prefix_cnt[k] = nº de esos partidos que tenían dato de córners
    Permite la media de "todos los partidos antes de X" en O(log n) sin leakage.
    """
    prefix: dict[str, tuple] = {}
    for team, entries in index.items():
        dates = [e[0] for e in entries]
        sums = [0.0]
        cnts = [0]
        for _, idx in entries:
            row = df.iloc[idx]
            if row["home_team"] == team:
                c = row.get("corners_home", np.nan)
            else:
                c = row.get("corners_away", np.nan)
            if pd.notna(c):
                sums.append(sums[-1] + float(c))
                cnts.append(cnts[-1] + 1)
            else:
                sums.append(sums[-1])
                cnts.append(cnts[-1])
        prefix[team] = (dates, sums, cnts)
    return prefix


def _global_corners_for(prefix: dict[str, tuple]) -> float:
    """Media global de córners-a-favor por equipo-partido (fallback de último recurso)."""
    tot, n = 0.0, 0
    for _, (_, sums, cnts) in prefix.items():
        tot += sums[-1]
        n += cnts[-1]
    return tot / n if n else 5.0


def _build_attack_corners_model(prefix: dict[str, tuple], ratings: dict) -> Optional[tuple]:
    """
    Ajusta una recta córners-a-favor ~ attack_norm usando los equipos que tienen
    AMBOS (media de córners sobre ≥3 partidos + rating de ataque WC2026).
    Devuelve (slope, intercept) o None si no hay puntos suficientes.
    """
    xs, ys = [], []
    for team, (_, sums, cnts) in prefix.items():
        r = ratings.get(team)
        if not r:
            continue
        att = r.get("attack_norm", np.nan)
        if att is None or (isinstance(att, float) and np.isnan(att)):
            continue
        if cnts[-1] >= 3:  # mínimo de partidos para una media fiable
            xs.append(att)
            ys.append(sums[-1] / cnts[-1])
    if len(xs) >= 5:
        slope, intercept = np.polyfit(np.array(xs), np.array(ys), 1)
        return float(slope), float(intercept)
    return None


def _make_corners_fallback(
    ratings: dict,
    attack_model: Optional[tuple],
    global_val: float,
) -> Callable[[str], float]:
    """
    Cascada de fallback para selecciones sin historial de córners:
      1) si tiene rating de ataque → estimación por la recta ataque→córners
      2) si no → media global
    """
    def fallback(team: str) -> float:
        if attack_model is not None:
            r = ratings.get(team)
            if r:
                att = r.get("attack_norm", np.nan)
                if att is not None and not (isinstance(att, float) and np.isnan(att)):
                    val = attack_model[0] * att + attack_model[1]
                    return float(np.clip(val, 2.0, 8.0))  # rango sensato de córners/equipo
        return global_val
    return fallback


def compute_corners_avg(
    prefix: dict[str, tuple],
    team: str,
    before_date: pd.Timestamp,
    fallback_fn: Callable[[str], float],
) -> float:
    """
    Media de córners-a-favor del equipo en TODOS sus partidos anteriores a
    before_date que tengan dato. Sin leakage (estrictamente < before_date).
    Si no hay ninguno, usa la cascada de fallback.
    """
    p = prefix.get(team)
    if not p:
        return fallback_fn(team)
    dates, sums, cnts = p
    cut = bisect.bisect_left(dates, before_date)
    if cnts[cut] > 0:
        return sums[cut] / cnts[cut]
    return fallback_fn(team)


# ── Utilidades ────────────────────────────────────────────────────────────────

def _safe_nanmean(values: list) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if len(arr) > 0 else np.nan


def _safe_diff(a, b):
    if a is None or b is None:
        return np.nan
    try:
        fa, fb = float(a), float(b)
        return np.nan if (np.isnan(fa) or np.isnan(fb)) else fa - fb
    except (TypeError, ValueError):
        return np.nan


def _safe_bool(value) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    try:
        return bool(value)
    except Exception:
        return False


# ── Features de forma reciente ────────────────────────────────────────────────

def compute_recent_form(
    df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    n: int = N_RECENT,
    index: Optional[dict] = None,
) -> dict:
    """
    Forma reciente de un equipo en los últimos N partidos.
    Si se pasa index usa la versión rápida, si no usa filtro directo.
    """
    if index is not None:
        history = _get_team_history_fast(df, index, team, before_date, n)
    else:
        mask = (
            ((df["home_team"] == team) | (df["away_team"] == team)) &
            (df["date"] < before_date) &
            df["result"].notna()
        )
        history = df[mask].sort_values("date", ascending=False).head(n)

    empty = {
        f"form_wins_{n}":         np.nan,
        f"form_draws_{n}":        np.nan,
        f"form_losses_{n}":       np.nan,
        f"form_gf_{n}":           np.nan,
        f"form_gc_{n}":           np.nan,
        f"form_ppg_{n}":          np.nan,
        f"form_corners_for_{n}":  np.nan,
        f"form_yellows_{n}":      np.nan,
        f"form_streak":           0,
        f"form_neutral_wins_{n}": np.nan,
    }
    if history.empty:
        return empty

    results, gf_list, gc_list, corners_list, yellow_list, neutral_wins = [], [], [], [], [], []

    for _, row in history.iterrows():
        is_home = row["home_team"] == team
        if is_home:
            r  = row["result"]
            gf = row.get("home_score", np.nan)
            gc = row.get("away_score", np.nan)
            c  = row.get("corners_home", np.nan)
            y  = row.get("yellow_home",  np.nan)
        else:
            r  = {"H": "A", "A": "H", "D": "D"}.get(row["result"], "D")
            gf = row.get("away_score", np.nan)
            gc = row.get("home_score", np.nan)
            c  = row.get("corners_away", np.nan)
            y  = row.get("yellow_away",  np.nan)

        results.append(r)
        gf_list.append(gf)
        gc_list.append(gc)
        corners_list.append(c)
        yellow_list.append(y)

        is_neutral = _safe_bool(row.get("neutral", False)) or _safe_bool(row.get("is_neutral", False))
        if is_neutral:
            neutral_wins.append(1 if r == "H" else 0)

    n_played = len(results)
    wins   = results.count("H")
    draws  = results.count("D")
    losses = results.count("A")

    streak = 0
    for r in results:
        if r == "H":
            if streak >= 0: streak += 1
            else: break
        elif r == "A":
            if streak <= 0: streak -= 1
            else: break
        else:
            break

    return {
        f"form_wins_{n}":         wins / n_played,
        f"form_draws_{n}":        draws / n_played,
        f"form_losses_{n}":       losses / n_played,
        f"form_gf_{n}":           _safe_nanmean(gf_list),
        f"form_gc_{n}":           _safe_nanmean(gc_list),
        f"form_ppg_{n}":          (wins * 3 + draws) / n_played,
        f"form_corners_for_{n}":  _safe_nanmean(corners_list),
        f"form_yellows_{n}":      _safe_nanmean(yellow_list),
        f"form_streak":           streak,
        f"form_neutral_wins_{n}": np.mean(neutral_wins) if neutral_wins else np.nan,
    }


# ── Features de contexto ──────────────────────────────────────────────────────

CONFEDERATION_MAP = {
    "Spain": "UEFA", "Germany": "UEFA", "France": "UEFA", "England": "UEFA",
    "Portugal": "UEFA", "Netherlands": "UEFA", "Belgium": "UEFA", "Italy": "UEFA",
    "Croatia": "UEFA", "Poland": "UEFA", "Switzerland": "UEFA", "Denmark": "UEFA",
    "Austria": "UEFA", "Sweden": "UEFA", "Norway": "UEFA", "Serbia": "UEFA",
    "Slovakia": "UEFA", "Slovenia": "UEFA", "Scotland": "UEFA", "Turkey": "UEFA",
    "Czech Republic": "UEFA", "Bosnia and Herzegovina": "UEFA",
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Chile": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL", "Peru": "CONMEBOL", "Venezuela": "CONMEBOL",
    "Bolivia": "CONMEBOL",
    "United States": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Costa Rica": "CONCACAF", "Panama": "CONCACAF", "Jamaica": "CONCACAF",
    "Honduras": "CONCACAF", "El Salvador": "CONCACAF", "Haiti": "CONCACAF",
    "Curacao": "CONCACAF",
    "Japan": "AFC", "South Korea": "AFC", "Iran": "AFC", "Australia": "AFC",
    "Saudi Arabia": "AFC", "Qatar": "AFC", "Jordan": "AFC", "Iraq": "AFC",
    "Uzbekistan": "AFC", "China": "AFC",
    "Morocco": "CAF", "Senegal": "CAF", "Nigeria": "CAF", "Ivory Coast": "CAF",
    "Ghana": "CAF", "Cameroon": "CAF", "Egypt": "CAF", "Tunisia": "CAF",
    "Algeria": "CAF", "South Africa": "CAF", "DR Congo": "CAF", "Cape Verde": "CAF",
    "New Zealand": "OFC",
}

HOST_COUNTRY_BY_TEAM = {
    "United States": "United States",
    "Mexico": "Mexico",
    "Canada": "Canada",
}

STAGE_ORDER = {
    "Group Stage": 0, "Round of 32": 1, "Round of 16": 2,
    "Quarter-finals": 3, "Semi-finals": 4,
    "Third-place play-off": 5, "Final": 6,
}


def compute_context_features(row: pd.Series, ratings: dict) -> dict:
    """
    Features de contexto del partido, incluyendo rating WC2026.

    En el Mundial todo es sede neutral — las features de rating
    usan diferencia absoluta, no direccional.
    """
    home_team = str(row.get("home_team", ""))
    away_team = str(row.get("away_team", ""))
    home_conf = CONFEDERATION_MAP.get(home_team, "OTHER")
    away_conf = CONFEDERATION_MAP.get(away_team, "OTHER")

    is_neutral = _safe_bool(row.get("neutral", False)) or _safe_bool(row.get("is_neutral", False))
    venue_country = str(row.get("country", "") or "")
    home_is_host = int(HOST_COUNTRY_BY_TEAM.get(home_team) == venue_country)
    away_is_host = int(HOST_COUNTRY_BY_TEAM.get(away_team) == venue_country)

    if not is_neutral:
        effective_home_adv = 1.0
    elif home_is_host:
        effective_home_adv = 0.5
    elif away_is_host:
        effective_home_adv = -0.5
    else:
        effective_home_adv = 0.0

    # Convocatorias
    squad_features = {}
    for key in ["squad_size", "goalkeepers", "defenders", "midfielders", "forwards", "unique_clubs"]:
        home_key = f"home_{key}"
        away_key = f"away_{key}"
        squad_features[home_key]       = row.get(home_key, np.nan)
        squad_features[away_key]       = row.get(away_key, np.nan)
        squad_features[f"{key}_diff"]  = _safe_diff(row.get(home_key), row.get(away_key))

    # Rating WC2026 — diferencia absoluta (campo neutral, no hay home/away real)
    home_r = ratings.get(home_team, {})
    away_r = ratings.get(away_team, {})

    # Stats de jugadores WC2026
    player_stats = _load_player_stats()
    home_ps = player_stats.get(home_team, {})
    away_ps = player_stats.get(away_team, {})

    home_rating      = home_r.get("rating",        np.nan)
    away_rating      = away_r.get("rating",        np.nan)
    home_attack      = home_r.get("attack_norm",   np.nan)
    away_attack      = away_r.get("attack_norm",   np.nan)
    home_midfield    = home_r.get("midfield_norm", np.nan)
    away_midfield    = away_r.get("midfield_norm", np.nan)
    home_defense     = home_r.get("defense_norm",  np.nan)
    away_defense     = away_r.get("defense_norm",  np.nan)

    rd = _safe_diff(home_rating, away_rating)
    rating_diff = rd / 100.0 if not np.isnan(rd) else np.nan
    attack_diff   = _safe_diff(home_attack,   away_attack)         
    midfield_diff = _safe_diff(home_midfield, away_midfield)       
    defense_diff  = _safe_diff(home_defense,  away_defense)        

    return {
        # Ventaja local
        "is_neutral":           int(is_neutral),
        "home_is_host":         home_is_host,
        "away_is_host":         away_is_host,
        "effective_home_adv":   effective_home_adv,
        # Confederaciones
        "same_confederation":   int(home_conf == away_conf),
        "home_confederation":   home_conf,
        "away_confederation":   away_conf,
        # Fase
        "stage_ordinal":        STAGE_ORDER.get(str(row.get("stage", "")), 0),
        # Sede
        "altitude_m":           row.get("altitude_m", np.nan),
        "is_indoor":            int(str(row.get("roof_type", "")).lower() in {"fixed", "retractable"}),
        # Descanso
        "home_rest_days":       row.get("home_rest_days", np.nan),
        "away_rest_days":       row.get("away_rest_days", np.nan),
        "rest_days_diff":       _safe_diff(row.get("home_rest_days"), row.get("away_rest_days")),
        # Viaje
        "home_travel_km":       row.get("home_travel_km", np.nan),
        "away_travel_km":       row.get("away_travel_km", np.nan),
        "travel_km_diff":       _safe_diff(row.get("home_travel_km"), row.get("away_travel_km")),
        # Elo
        "elo_diff":             row.get("elo_diff", np.nan),
        # Rating WC2026 (NaN para histórico, solo WC2026 tiene valores)
        "home_rating":          home_rating,
        "away_rating":          away_rating,
        "rating_diff":          rating_diff,
        "home_attack_norm":     home_attack,
        "away_attack_norm":     away_attack,
        "attack_diff":          attack_diff,
        "home_midfield_norm":   home_midfield,
        "away_midfield_norm":   away_midfield,
        "midfield_diff":        midfield_diff,
        "home_defense_norm":    home_defense,
        "away_defense_norm":    away_defense,
        "defense_diff":         defense_diff,
        # Convocatorias
        **squad_features,
        # Stats de jugadores WC2026 (NaN para histórico)
        "home_yellow_per_90":   home_ps.get("yellow_per_90",  np.nan),
        "away_yellow_per_90":   away_ps.get("yellow_per_90",  np.nan),
        "yellow_per_90_diff":   abs(_safe_diff(home_ps.get("yellow_per_90"), away_ps.get("yellow_per_90"))),
        "home_red_per_90":      home_ps.get("red_per_90",     np.nan),
        "away_red_per_90":      away_ps.get("red_per_90",     np.nan),
        "home_goals_per_90":    home_ps.get("goals_per_90",   np.nan),
        "away_goals_per_90":    away_ps.get("goals_per_90",   np.nan),
        "goals_per_90_diff":    abs(_safe_diff(home_ps.get("goals_per_90"), away_ps.get("goals_per_90"))),
        "home_assists_per_90":  home_ps.get("assists_per_90", np.nan),
        "away_assists_per_90":  away_ps.get("assists_per_90", np.nan),
        "home_avg_caps":        home_ps.get("avg_caps",       np.nan),
        "away_avg_caps":        away_ps.get("avg_caps",       np.nan),
    }


# ── Features del árbitro ──────────────────────────────────────────────────────

def compute_referee_features(row: pd.Series) -> dict:
    def _pick(model_key, raw_key):
        v = row.get(model_key, np.nan)
        return v if pd.notna(v) else row.get(raw_key, np.nan)
    return {
        "ref_yellow_per_match": _pick("ref_yellow_per_match_model", "ref_yellow_per_match"),
        "ref_red_per_match":    _pick("ref_red_per_match_model",    "ref_red_per_match"),
    }


# ── Variables objetivo ────────────────────────────────────────────────────────

def compute_targets(row: pd.Series) -> dict:
    hs  = row.get("home_score", np.nan)
    as_ = row.get("away_score", np.nan)
    total_goals = (hs + as_) if pd.notna(hs) and pd.notna(as_) else np.nan

    ct        = row.get("corners_total", np.nan)
    yt        = row.get("yellow_total",  np.nan)
    rh        = row.get("red_home",      np.nan)
    ra        = row.get("red_away",      np.nan)
    red_total = ((rh or 0) + (ra or 0)) if pd.notna(rh) and pd.notna(ra) else np.nan

    return {
        "target_result":       row.get("result", np.nan),
        "target_over25":       int(total_goals > 2.5) if pd.notna(total_goals) else np.nan,
        "target_btts":         int((hs or 0) > 0 and (as_ or 0) > 0) if pd.notna(hs) and pd.notna(as_) else np.nan,
        "target_total_goals":  total_goals,
        "target_ht_result":    row.get("ht_result", np.nan),
        "target_corners":      ct,
        "target_over85c":      int(ct > 8.5) if pd.notna(ct) else np.nan,
        "target_yellows":      yt,
        "target_over35y":      int(yt > 3.5) if pd.notna(yt) else np.nan,
        "target_red":          int(red_total > 0) if pd.notna(red_total) else np.nan,
        "target_first_scorer": np.nan,
    }


# ── Pipeline principal ────────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    n_recent: int = N_RECENT,
    require_result: bool = False,
    ratings_path: str = "./data/processed/team_ratings_wc2026.csv",
) -> pd.DataFrame:
    """
    Construye la matriz de features completa sin data leakage.

    Optimización: preconstruye índice por equipo para búsqueda O(log n)
    en lugar de filtrar el DataFrame completo en cada iteración O(n²).

    Parámetros
    ----------
    df            : DataFrame de matches_enriched.csv
    n_recent      : ventana de partidos recientes para forma
    require_result: True para entrenamiento, False para incluir WC2026 futuro
    ratings_path  : ruta al CSV de ratings WC2026
    """
    df = df.copy()

    # Normalizar columna de fecha
    date_col = "match_date" if "match_date" in df.columns else "date"
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"]).reset_index(drop=True)
    df = df.sort_values("date").reset_index(drop=True)

    if require_result:
        df = df[df["result"].notna()].reset_index(drop=True)

    # Cargar ratings WC2026
    ratings = _load_ratings(ratings_path)

    # Preconstruir índice de historial por equipo
    print("  Construyendo índice de historial por equipo...")
    team_index = build_team_index(df)
    print(f"  → Índice listo: {len(team_index)} equipos")

    # Media de córners por selección (prefix sums) + cascada de fallback
    print("  Preparando medias de córners por selección...")
    corners_prefix = build_corners_prefix(df, team_index)
    global_corners_for = _global_corners_for(corners_prefix)
    attack_model = _build_attack_corners_model(corners_prefix, ratings)
    corners_fallback = _make_corners_fallback(ratings, attack_model, global_corners_for)
    if attack_model is not None:
        print(f"  → Recta ataque→córners: corners ≈ {attack_model[0]:.2f}·attack + {attack_model[1]:.2f}")
    print(f"  → Media global córners-a-favor (último fallback): {global_corners_for:.2f}")

    rows = []
    total = len(df)

    for i, row in df.iterrows():
        if i % 2000 == 0:
            print(f"  Procesando {i+1}/{total}...")

        date      = row["date"]
        home_team = row["home_team"]
        away_team = row["away_team"]

        home_form = compute_recent_form(df, home_team, date, n_recent, index=team_index)
        away_form = compute_recent_form(df, away_team, date, n_recent, index=team_index)

        home_feats = {f"home_{k}": v for k, v in home_form.items()}
        away_feats = {f"away_{k}": v for k, v in away_form.items()}

        diff_feats = {}
        for key in home_form:
            hv = home_feats.get(f"home_{key}")
            av = away_feats.get(f"away_{key}")
            diff_feats[f"diff_{key}"] = _safe_diff(hv, av)

        # Media de córners por selección (todos los partidos anteriores con dato)
        home_corners_avg = compute_corners_avg(corners_prefix, home_team, date, corners_fallback)
        away_corners_avg = compute_corners_avg(corners_prefix, away_team, date, corners_fallback)

        ctx_feats = compute_context_features(row, ratings)
        ref_feats = compute_referee_features(row)
        targets   = compute_targets(row)

        rows.append({
            "date":       date,
            "home_team":  home_team,
            "away_team":  away_team,
            "tournament": row.get("tournament") or row.get("competition"),
            "_source":    row.get("_source", "historical"),
            "home_score": row.get("home_score", np.nan),
            "away_score": row.get("away_score", np.nan),
            **home_feats,
            **away_feats,
            **diff_feats,
            # Media de córners por selección (cobertura ~100%, sin leakage)
            "home_corners_avg_all": home_corners_avg,
            "away_corners_avg_all": away_corners_avg,
            "corners_avg_all_sum":  home_corners_avg + away_corners_avg,
            **ctx_feats,
            **ref_feats,
            **targets,
        })

    result_df = pd.DataFrame(rows)
    print(f"\nFeature matrix: {len(result_df)} partidos × {len(result_df.columns)} columnas")
    return result_df


# ── Script standalone ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_DIR = "./data"
    enriched_path = os.path.join(DATA_DIR, "processed", "matches_enriched.csv")
    unified_path  = os.path.join(DATA_DIR, "unified.csv")

    if os.path.exists(enriched_path):
        print(f"Cargando {enriched_path}...")
        df = pd.read_csv(enriched_path, low_memory=False)
    elif os.path.exists(unified_path):
        print(f"Fallback a {unified_path}...")
        df = pd.read_csv(unified_path, low_memory=False)
    else:
        print("No se encuentra dataset. Ejecuta data_collector.py y builder.py primero.")
        exit(1)

    print(f"  → {len(df)} partidos, {len(df.columns)} columnas")
    print("\nConstruyendo feature matrix...")

    features = build_feature_matrix(df, require_result=False)

    os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)

    # Todos los partidos
    out = os.path.join(DATA_DIR, "processed", "features.csv")
    features.to_csv(out, index=False)
    print(f"Guardado en {out}")

    # Solo histórico con resultado → entrenamiento
    train = features[features["target_result"].notna()].copy()
    out_train = os.path.join(DATA_DIR, "processed", "features_train.csv")
    train.to_csv(out_train, index=False)
    print(f"Entrenamiento: {out_train}  ({len(train)} partidos)")

    # Solo WC2026 → predicción
    wc = features[features["_source"] == "wc2026"].copy()
    out_wc = os.path.join(DATA_DIR, "processed", "features_wc2026.csv")
    wc.to_csv(out_wc, index=False)
    print(f"WC2026: {out_wc}  ({len(wc)} partidos)")

    # Resumen de cobertura
    print("\nCobertura de features:")
    key_cols = [
        "effective_home_adv", "is_neutral", "home_is_host",
        "altitude_m", "home_rest_days", "home_travel_km",
        "home_form_ppg_10", "diff_form_ppg_10",
        "rating_diff", "attack_diff", "midfield_diff", "defense_diff",
        "home_corners_avg_all", "corners_avg_all_sum",
        "home_yellow_per_90", "home_goals_per_90", "home_avg_caps",
    ]
    for col in key_cols:
        if col in features.columns:
            pct = features[col].notna().mean() * 100
            print(f"  {col:<32} {pct:.0f}%")
            