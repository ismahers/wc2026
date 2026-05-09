"""
src/features/feature_engineering.py
=====================================
Construcción de features para el Pronosticador Mundial 2026.

Para cada partido genera:
  - Features de forma reciente (últimos N partidos de cada equipo)
  - Features de ranking FIFA
  - Features del árbitro
  - Features de contexto (fase, sede, confederación)
  - Variables objetivo para cada mercado
"""

import pandas as pd
import numpy as np
from typing import Optional


# ── Configuración ─────────────────────────────────────────────────────────────

N_RECENT = 10   # Ventana de partidos recientes para calcular forma


# ── Utilidades ────────────────────────────────────────────────────────────────

def _result_points(result: str, perspective: str) -> float:
    """Convierte resultado en puntos desde la perspectiva del equipo (H o A)."""
    if perspective == "H":
        return {"H": 3.0, "D": 1.0, "A": 0.0}.get(result, np.nan)
    else:
        return {"A": 3.0, "D": 1.0, "H": 0.0}.get(result, np.nan)


def _get_team_history(df: pd.DataFrame, team: str, before_date: pd.Timestamp, n: int) -> pd.DataFrame:
    """
    Devuelve los últimos N partidos de un equipo anteriores a una fecha dada.
    Incluye tanto partidos como local como visitante.
    """
    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["date"] < before_date)
    )
    return df[mask].sort_values("date", ascending=False).head(n)


# ── Features de forma reciente ────────────────────────────────────────────────

def compute_recent_form(df: pd.DataFrame, team: str, before_date: pd.Timestamp, n: int = N_RECENT) -> dict:
    """
    Calcula features de forma reciente para un equipo:
      - win_rate, draw_rate, loss_rate
      - avg_goals_scored, avg_goals_conceded
      - avg_corners_for, avg_corners_against   (si disponible)
      - avg_yellow_cards                        (si disponible)
      - points_per_game
      - form_streak: racha actual (+ victorias, - derrotas)
    """
    history = _get_team_history(df, team, before_date, n)
    feats = {}

    if history.empty:
        return {
            f"form_wins_{n}":        np.nan,
            f"form_draws_{n}":       np.nan,
            f"form_losses_{n}":      np.nan,
            f"form_gf_{n}":          np.nan,
            f"form_gc_{n}":          np.nan,
            f"form_ppg_{n}":         np.nan,
            f"form_corners_for_{n}": np.nan,
            f"form_yellows_{n}":     np.nan,
            f"form_streak":          0,
        }

    results, gf_list, gc_list, corners_list, yellow_list = [], [], [], [], []

    for _, row in history.iterrows():
        is_home = row["home_team"] == team
        if is_home:
            r  = row["result"]
            gf = row["home_score"]
            gc = row["away_score"]
            c  = row.get("corners_home", np.nan)
            y  = row.get("yellow_home",  np.nan)
        else:
            r  = {"H": "A", "A": "H", "D": "D"}.get(row["result"], "D")
            gf = row["away_score"]
            gc = row["home_score"]
            c  = row.get("corners_away", np.nan)
            y  = row.get("yellow_away",  np.nan)

        results.append(r)
        gf_list.append(gf)
        gc_list.append(gc)
        corners_list.append(c)
        yellow_list.append(y)

    n_played = len(results)
    wins     = results.count("H")
    draws    = results.count("D")
    losses   = results.count("A")

    # Racha actual
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

    feats[f"form_wins_{n}"]        = wins / n_played
    feats[f"form_draws_{n}"]       = draws / n_played
    feats[f"form_losses_{n}"]      = losses / n_played
    feats[f"form_gf_{n}"]          = np.nanmean(gf_list)
    feats[f"form_gc_{n}"]          = np.nanmean(gc_list)
    feats[f"form_ppg_{n}"]         = (wins * 3 + draws) / n_played
    feats[f"form_corners_for_{n}"] = np.nanmean(corners_list)
    feats[f"form_yellows_{n}"]     = np.nanmean(yellow_list)
    feats[f"form_streak"]          = streak

    return feats


# ── Features de contexto ──────────────────────────────────────────────────────

CONFEDERATION_MAP = {
    # UEFA
    "Spain": "UEFA", "Germany": "UEFA", "France": "UEFA", "England": "UEFA",
    "Portugal": "UEFA", "Netherlands": "UEFA", "Belgium": "UEFA", "Italy": "UEFA",
    "Croatia": "UEFA", "Poland": "UEFA", "Switzerland": "UEFA", "Denmark": "UEFA",
    "Austria": "UEFA", "Sweden": "UEFA", "Norway": "UEFA", "Serbia": "UEFA",
    "Slovakia": "UEFA", "Slovenia": "UEFA", "Scotland": "UEFA", "Turkey": "UEFA",
    # CONMEBOL
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Chile": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL", "Peru": "CONMEBOL", "Venezuela": "CONMEBOL",
    "Bolivia": "CONMEBOL",
    # CONCACAF
    "United States": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Costa Rica": "CONCACAF", "Panama": "CONCACAF", "Jamaica": "CONCACAF",
    "Honduras": "CONCACAF", "El Salvador": "CONCACAF", "Haiti": "CONCACAF",
    # AFC
    "Japan": "AFC", "South Korea": "AFC", "Iran": "AFC", "Australia": "AFC",
    "Saudi Arabia": "AFC", "Qatar": "AFC", "Jordan": "AFC", "Iraq": "AFC",
    "Uzbekistan": "AFC",
    # CAF
    "Morocco": "CAF", "Senegal": "CAF", "Nigeria": "CAF", "Ivory Coast": "CAF",
    "Ghana": "CAF", "Cameroon": "CAF", "Egypt": "CAF", "Tunisia": "CAF",
    "Algeria": "CAF", "South Africa": "CAF", "DR Congo": "CAF",
    # OFC
    "New Zealand": "OFC",
}

STAGE_ORDER = {
    "Group Stage": 0, "Round of 16": 1, "Quarter-finals": 2,
    "Semi-finals": 3, "3rd Place Final": 4, "Final": 5,
}


def compute_context_features(row: pd.Series) -> dict:
    """Features de contexto del partido."""
    home_conf = CONFEDERATION_MAP.get(row["home_team"], "OTHER")
    away_conf = CONFEDERATION_MAP.get(row["away_team"], "OTHER")

    return {
        "is_neutral":         int(row.get("neutral", False)),
        "same_confederation": int(home_conf == away_conf),
        "stage_ordinal":      STAGE_ORDER.get(row.get("stage", "Group Stage"), 0),
        "home_confederation": home_conf,
        "away_confederation": away_conf,
    }


# ── Features del árbitro ──────────────────────────────────────────────────────

def compute_referee_features(row: pd.Series) -> dict:
    """Features del árbitro asignado al partido."""
    return {
        "ref_yellow_per_match": row.get("yellow_per_match", np.nan),
        "ref_red_per_match":    row.get("red_per_match",    np.nan),
    }


# ── Variables objetivo ────────────────────────────────────────────────────────

def compute_targets(row: pd.Series) -> dict:
    """
    Calcula todas las variables objetivo para los distintos mercados.
    """
    hs = row.get("home_score", np.nan)
    as_ = row.get("away_score", np.nan)
    total_goals = (hs or 0) + (as_ or 0) if pd.notna(hs) and pd.notna(as_) else np.nan

    ct = row.get("corners_total", np.nan)
    yt = row.get("yellow_total",  np.nan)
    rh = row.get("red_home",      np.nan)
    ra = row.get("red_away",      np.nan)
    red_total = (rh or 0) + (ra or 0) if pd.notna(rh) and pd.notna(ra) else np.nan

    return {
        # Mercado principal
        "target_result":      row.get("result", np.nan),       # H / D / A
        # Goles
        "target_over25":      int(total_goals > 2.5) if pd.notna(total_goals) else np.nan,
        "target_btts":        int((hs or 0) > 0 and (as_ or 0) > 0) if pd.notna(hs) and pd.notna(as_) else np.nan,
        "target_total_goals": total_goals,
        # Descanso
        "target_ht_result":   row.get("ht_result", np.nan),    # H / D / A (si disponible)
        # Córners
        "target_corners":     ct,
        "target_over85c":     int(ct > 8.5) if pd.notna(ct) else np.nan,
        # Tarjetas
        "target_yellows":     yt,
        "target_over35y":     int(yt > 3.5) if pd.notna(yt) else np.nan,
        "target_red":         int(red_total > 0) if pd.notna(red_total) else np.nan,
        # Quién marca primero (requeriría eventos, dejamos NaN por ahora)
        "target_first_scorer": np.nan,
    }


# ── Pipeline principal ────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame, n_recent: int = N_RECENT) -> pd.DataFrame:
    """
    Construye la matriz de features completa a partir del dataset unificado.

    Para cada partido calcula features de forma de ambos equipos usando
    solo información disponible ANTES del partido (sin data leakage).

    Parámetros
    ----------
    df : DataFrame con el dataset unificado (unified.csv)
    n_recent : ventana de partidos recientes

    Retorna
    -------
    DataFrame con una fila por partido y todas las features + targets
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "result"]).reset_index(drop=True)
    df = df.sort_values("date").reset_index(drop=True)

    rows = []
    total = len(df)

    for i, row in df.iterrows():
        if i % 500 == 0:
            print(f"  Procesando partido {i+1}/{total}...")

        date      = row["date"]
        home_team = row["home_team"]
        away_team = row["away_team"]

        # Forma reciente de ambos equipos (sin data leakage)
        home_form = compute_recent_form(df, home_team, date, n_recent)
        away_form = compute_recent_form(df, away_team, date, n_recent)

        # Renombrar con prefijo
        home_feats = {f"home_{k}": v for k, v in home_form.items()}
        away_feats = {f"away_{k}": v for k, v in away_form.items()}

        # Diferencias entre equipos (features de diferencia)
        diff_feats = {}
        for key in home_form:
            hk, ak = f"home_{key}", f"away_{key}"
            hv, av = home_feats.get(hk), away_feats.get(ak)
            if isinstance(hv, float) and isinstance(av, float):
                diff_feats[f"diff_{key}"] = hv - av

        # Contexto y árbitro
        ctx_feats = compute_context_features(row)
        ref_feats = compute_referee_features(row)

        # Targets
        targets = compute_targets(row)

        # Fila final
        feat_row = {
            "date":      date,
            "home_team": home_team,
            "away_team": away_team,
            "tournament": row.get("tournament"),
            **home_feats,
            **away_feats,
            **diff_feats,
            **ctx_feats,
            **ref_feats,
            **targets,
        }
        rows.append(feat_row)

    result_df = pd.DataFrame(rows)
    print(f"\nFeature matrix: {len(result_df)} partidos × {len(result_df.columns)} columnas")
    return result_df


# ── Script standalone ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    DATA_DIR = "./data"
    unified_path = os.path.join(DATA_DIR, "unified.csv")

    if not os.path.exists(unified_path):
        print(f"No se encuentra {unified_path}. Ejecuta primero data_collector.py")
        exit(1)

    print("Cargando unified.csv...")
    df = pd.read_csv(unified_path, parse_dates=["date"])
    print(f"  → {len(df)} partidos cargados")

    print("\nConstruyendo feature matrix...")
    features = build_feature_matrix(df)

    out = os.path.join(DATA_DIR, "features.csv")
    features.to_csv(out, index=False)
    print(f"\nGuardado en {out}")
    print(features.describe().to_string())
