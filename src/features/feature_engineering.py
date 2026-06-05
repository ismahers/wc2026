"""
src/features/feature_engineering.py
=====================================
Construcción de features para el Pronosticador Mundial 2026.

Para cada partido genera:
  - Features de forma reciente (últimos N partidos de cada equipo)
  - Features de contexto (fase, sede, confederación, neutral/host)
  - Features del árbitro
  - Variables objetivo para cada mercado

Ventaja de local en el Mundial
-------------------------------
En el Mundial todos los partidos son en sede neutral, así que la ventaja
de local tradicional no existe. El feature `effective_home_adv` lo captura:
  - Partido no neutral (clasificatorias, amistosos): 1.0
  - Partido neutral sin anfitrión (la mayoría del WC2026): 0.0
  - Partido neutral con home anfitrión en su país sede: 0.5
  - Partido neutral con away anfitrión en su país sede: -0.5

El modelo aprende solo del histórico que `is_neutral=1` anula la ventaja,
y al predecir el Mundial solo aplica ventaja si USA/México/Canadá juegan
en su propio país sede, aunque el fixture los liste como away.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ── Configuración ─────────────────────────────────────────────────────────────

N_RECENT = 10


# ── Utilidades ────────────────────────────────────────────────────────────────

def _get_team_history(
    df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    n: int,
) -> pd.DataFrame:
    """Últimos N partidos de un equipo antes de una fecha (sin leakage)."""
    if "result" not in df.columns:
        return pd.DataFrame()

    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["date"] < before_date) &
        df["result"].notna()
    )
    return df[mask].sort_values("date", ascending=False).head(n)


def _safe_nanmean(values: list[float]) -> float:
    """Mean that stays quiet when a market has no historical coverage."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan
    return float(arr.mean())


# ── Features de forma reciente ────────────────────────────────────────────────

def compute_recent_form(
    df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    n: int = N_RECENT,
) -> dict:
    """
    Forma reciente de un equipo en los últimos N partidos.
    Incluye win_rate en partidos neutrales para capturar rendimiento
    en condiciones similares al Mundial.
    """
    history = _get_team_history(df, team, before_date, n)

    empty = {
        f"form_wins_{n}":          np.nan,
        f"form_draws_{n}":         np.nan,
        f"form_losses_{n}":        np.nan,
        f"form_gf_{n}":            np.nan,
        f"form_gc_{n}":            np.nan,
        f"form_ppg_{n}":           np.nan,
        f"form_corners_for_{n}":   np.nan,
        f"form_yellows_{n}":       np.nan,
        f"form_streak":            0,
        f"form_neutral_wins_{n}":  np.nan,
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

    # Racha actual (+ victorias consecutivas, - derrotas)
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


def compute_context_features(row: pd.Series) -> dict:
    """
    Features de contexto del partido.

    effective_home_adv:
      1.0 → partido no neutral (ventaja local real)
      0.5 → partido neutral, home juega en su país anfitrión
      0.0 → partido neutral, sin anfitrión en su país sede
     -0.5 → partido neutral, away juega en su país anfitrión
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

    squad_features = {}
    for key in [
        "squad_size", "goalkeepers", "defenders",
        "midfielders", "forwards", "unique_clubs",
    ]:
        home_key = f"home_{key}"
        away_key = f"away_{key}"
        squad_features[home_key] = row.get(home_key, np.nan)
        squad_features[away_key] = row.get(away_key, np.nan)
        squad_features[f"{key}_diff"] = _safe_diff(row.get(home_key), row.get(away_key))

    return {
        # Ventaja de local
        "is_neutral":          int(is_neutral),
        "home_is_host":        home_is_host,
        "away_is_host":        away_is_host,
        "effective_home_adv":  effective_home_adv,
        # Confederaciones
        "same_confederation":  int(home_conf == away_conf),
        "home_confederation":  home_conf,
        "away_confederation":  away_conf,
        # Fase del torneo
        "stage_ordinal":       STAGE_ORDER.get(str(row.get("stage", "")), 0),
        # Sede (solo WC2026)
        "altitude_m":          row.get("altitude_m", np.nan),
        "is_indoor":           int(str(row.get("roof_type", "")).lower() in {"fixed", "retractable"}),
        # Descanso
        "home_rest_days":      row.get("home_rest_days", np.nan),
        "away_rest_days":      row.get("away_rest_days", np.nan),
        "rest_days_diff":      _safe_diff(row.get("home_rest_days"), row.get("away_rest_days")),
        # Viaje (solo WC2026)
        "home_travel_km":      row.get("home_travel_km", np.nan),
        "away_travel_km":      row.get("away_travel_km", np.nan),
        "travel_km_diff":      _safe_diff(row.get("home_travel_km"), row.get("away_travel_km")),
        # Fuerza relativa de equipo. No usar home_elo/away_elo separados
        # como features del modelo en sede neutral: el lado home/away es
        # administrativo salvo anfitriones.
        "elo_diff":            row.get("elo_diff", np.nan),
        # Convocatorias WC2026 (histórico queda NaN para evitar leakage)
        **squad_features,
    }


def _safe_diff(a, b):
    if a is None or b is None:
        return np.nan
    try:
        fa, fb = float(a), float(b)
        return np.nan if (np.isnan(fa) or np.isnan(fb)) else fa - fb
    except (TypeError, ValueError):
        return np.nan


def _safe_bool(value) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


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
) -> pd.DataFrame:
    """
    Construye la matriz de features completa sin data leakage.

    Parámetros
    ----------
    df : DataFrame de matches_enriched.csv (salida del builder)
    n_recent : ventana de partidos recientes para forma
    require_result : True para solo entrenamiento, False para incluir WC2026 futuro
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

    rows = []
    total = len(df)

    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(f"  Procesando {i+1}/{total}...")

        date      = row["date"]
        home_team = row["home_team"]
        away_team = row["away_team"]

        home_form = compute_recent_form(df, home_team, date, n_recent)
        away_form = compute_recent_form(df, away_team, date, n_recent)

        home_feats = {f"home_{k}": v for k, v in home_form.items()}
        away_feats = {f"away_{k}": v for k, v in away_form.items()}

        diff_feats = {}
        for key in home_form:
            hv = home_feats.get(f"home_{key}")
            av = away_feats.get(f"away_{key}")
            diff_feats[f"diff_{key}"] = _safe_diff(hv, av)

        ctx_feats = compute_context_features(row)
        ref_feats = compute_referee_features(row)
        targets   = compute_targets(row)

        rows.append({
            "date":       date,
            "home_team":  home_team,
            "away_team":  away_team,
            "tournament": row.get("tournament") or row.get("competition"),
            "_source":    row.get("_source", "historical"),
            **home_feats,
            **away_feats,
            **diff_feats,
            **ctx_feats,
            **ref_feats,
            **targets,
        })

    result_df = pd.DataFrame(rows)
    print(f"\nFeature matrix: {len(result_df)} partidos × {len(result_df.columns)} columnas")
    return result_df


# ── Script standalone ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

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

    # Resumen de features clave
    print("\nCobertura de features:")
    key = ["effective_home_adv", "is_neutral", "home_is_host",
           "altitude_m", "home_rest_days", "home_travel_km",
           "home_form_ppg_10", "diff_form_ppg_10"]
    for col in key:
        if col in features.columns:
            pct = features[col].notna().mean() * 100
            print(f"  {col:<30} {pct:.0f}%")
            
