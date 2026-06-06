"""
src/data/elo_loader.py
=======================
Calcula ratings Elo históricos de selecciones nacionales a partir de
los resultados históricos (results.csv de Kaggle / unified.csv).

Metodología eloratings.net:
  - K variable según la importancia del partido
  - Factor de goles (Goal Difference Weight)
  - Sin ventaja local para WC2026 (sede neutral)

NOVEDAD — Anclaje inicial real
------------------------------
Antes todas las selecciones arrancaban en 1500 (con un puñado de potencias
en STRONG_TEAMS_INIT). Eso deprimía a confederaciones enteras (CONCACAF, AFC)
porque el Elo es suma cero DENTRO de cada confederación, que juega casi
siempre aislada. Resultado: USA, México, etc. salían infravalorados.

Ahora cada selección arranca en su Elo REAL de eloratings.net, cargado desde
data/raw/elo_initial_ratings.csv (las 244 selecciones). Esto fija el centro de
masa de cada confederación en un nivel realista y elimina la deriva.

Uso:
    python -m src.data.elo_loader
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parámetros del algoritmo Elo
# ---------------------------------------------------------------------------

ELO_START = 1500.0        # Solo fallback si una selección no está en el CSV de anclaje
HOME_ADVANTAGE = 100.0    # Puntos de ventaja local (no aplica en neutrales)

# Ruta del CSV con el Elo inicial real (eloratings.net)
INITIAL_ELO_PATH = "data/raw/elo_initial_ratings.csv"

K_VALUES = {
    "FIFA World Cup":                    60,
    "FIFA World Cup qualification":      40,
    "UEFA Euro":                         60,
    "UEFA Euro qualification":           40,
    "Copa América":                      60,
    "Africa Cup of Nations":             60,
    "AFC Asian Cup":                     60,
    "CONCACAF Gold Cup":                 60,
    "Friendly":                          20,
    "Confederations Cup":                50,
    "Nations League":                    40,
    "default":                           30,
}


def _load_initial_elo(path: str = INITIAL_ELO_PATH) -> dict[str, float]:
    """
    Carga el Elo inicial real de cada selección desde el CSV de anclaje.
    Devuelve {team_name_canonico: elo_inicial}. Si no existe, dict vacío
    (todas arrancarían en ELO_START — comportamiento antiguo).
    """
    from src.data.team_names import canonicalize

    if not os.path.exists(path):
        log.warning(
            "No se encontró %s — todas las selecciones arrancarán en %.0f. "
            "Sin anclaje real, las confederaciones se deprimen.",
            path, ELO_START,
        )
        return {}

    df = pd.read_csv(path)
    col_name = "team_name" if "team_name" in df.columns else df.columns[0]
    col_elo = "elo_initial" if "elo_initial" in df.columns else df.columns[1]

    anchor: dict[str, float] = {}
    for _, row in df.iterrows():
        name = canonicalize(row[col_name])
        if name is None:
            name = str(row[col_name]).strip()
        try:
            anchor[name] = float(row[col_elo])
        except (ValueError, TypeError):
            continue

    log.info("Elo inicial cargado: %d selecciones ancladas desde %s", len(anchor), path)
    return anchor


# ---------------------------------------------------------------------------
# Cálculo de Elo
# ---------------------------------------------------------------------------

def _k_factor(tournament: str) -> float:
    for key, k in K_VALUES.items():
        if key.lower() in str(tournament).lower():
            return float(k)
    return float(K_VALUES["default"])


def _goal_weight(home_goals: int, away_goals: int) -> float:
    diff = abs(home_goals - away_goals)
    if diff == 0 or diff == 1:
        return 1.0
    elif diff == 2:
        return 1.5
    else:
        return (11 + diff) / 8.0


def _expected_score(rating_home: float, rating_away: float, is_neutral: bool) -> float:
    advantage = 0.0 if is_neutral else HOME_ADVANTAGE
    dr = rating_home + advantage - rating_away
    return 1.0 / (10.0 ** (-dr / 400.0) + 1.0)


def _actual_score(home_goals: int, away_goals: int) -> float:
    if home_goals > away_goals:
        return 1.0
    elif home_goals == away_goals:
        return 0.5
    return 0.0


def compute_elo_history(
    df: pd.DataFrame,
    save_snapshots_every_n: int = 1,
    initial_elo: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Calcula el rating Elo de cada selección partido a partido.

    initial_elo : dict {team: elo_inicial}. Si None, se carga del CSV de anclaje.
                  Las selecciones no presentes arrancan en ELO_START.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team",
                            "home_score", "away_score"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    if initial_elo is None:
        initial_elo = _load_initial_elo()

    ratings: dict[str, float] = {}

    def get_rating(team: str) -> float:
        if team not in ratings:
            ratings[team] = float(initial_elo.get(team, ELO_START))
        return ratings[team]

    snapshots: list[tuple[str, pd.Timestamp, float]] = []

    all_teams = set(df["home_team"]).union(set(df["away_team"]))
    first_date = df["date"].min() - pd.Timedelta(days=1)
    for team in all_teams:
        snapshots.append((team, first_date, get_rating(team)))

    n_anchored = sum(1 for t in all_teams if t in initial_elo)
    log.info("Calculando Elo para %d equipos sobre %d partidos (%d con anclaje real)...",
             len(all_teams), len(df), n_anchored)

    for i, row in df.iterrows():
        if i % 5000 == 0 and i > 0:
            log.info("  Procesado %d/%d partidos...", i, len(df))

        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]
        tournament = row.get("tournament", "Friendly")
        is_neutral = bool(row.get("neutral", False))

        try:
            hg = int(row["home_score"])
            ag = int(row["away_score"])
        except (ValueError, TypeError):
            continue

        r_home = get_rating(home)
        r_away = get_rating(away)

        k = _k_factor(tournament)
        gw = _goal_weight(hg, ag)
        we = _expected_score(r_home, r_away, is_neutral)
        w = _actual_score(hg, ag)

        delta = k * gw * (w - we)

        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

        if i % save_snapshots_every_n == 0:
            snapshots.append((home, date, ratings[home]))
            snapshots.append((away, date, ratings[away]))

    log.info("  ✓ Elo calculado para %d equipos", len(ratings))

    result = pd.DataFrame(snapshots, columns=["team_name", "rating_date", "elo_rating"])
    result = result.sort_values(["team_name", "rating_date"]).reset_index(drop=True)
    result["source"] = "computed_elo"

    result["rank"] = result.groupby("rating_date")["elo_rating"].rank(
        ascending=False, method="min"
    )

    log.info("Snapshots totales: %d", len(result))
    return result


def get_latest_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    return (
        ratings_df.sort_values("rating_date")
        .groupby("team_name")
        .last()
        .reset_index()
        .sort_values("elo_rating", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Script standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    DATA_DIR = "./data"
    OUT_DIR = os.path.join(DATA_DIR, "processed")
    os.makedirs(OUT_DIR, exist_ok=True)

    unified_path = os.path.join(DATA_DIR, "unified.csv")
    raw_path = os.path.join(DATA_DIR, "results_raw.csv")

    if os.path.exists(unified_path):
        log.info("Cargando %s...", unified_path)
        df = pd.read_csv(unified_path, parse_dates=["date"])
    elif os.path.exists(raw_path):
        log.info("Cargando %s...", raw_path)
        df = pd.read_csv(raw_path, parse_dates=["date"])
    else:
        log.error("No se encuentra unified.csv ni results_raw.csv")
        exit(1)

    log.info("  → %d partidos cargados", len(df))

    from src.data.team_names import add_canonical_columns
    df = add_canonical_columns(df, ["home_team", "away_team"], suffix="")

    ratings = compute_elo_history(df, save_snapshots_every_n=1)

    out_path = os.path.join(OUT_DIR, "team_ratings.csv")
    ratings.to_csv(out_path, index=False)
    log.info("Historial Elo guardado en %s  (%d filas)", out_path, len(ratings))

    latest = get_latest_ratings(ratings)
    print("\n" + "=" * 55)
    print("TOP 30 SELECCIONES POR ELO (rating actual)")
    print("=" * 55)
    for i, row in latest.head(30).iterrows():
        print(f"  {i+1:>2}. {row['team_name']:<25} {row['elo_rating']:>7.1f}")
    print("=" * 55)

    wc_teams = [
        "Brazil", "Argentina", "France", "England", "Spain",
        "Germany", "Portugal", "Netherlands", "Belgium",
        "Morocco", "Japan", "South Korea", "Mexico",
        "United States", "Canada", "Paraguay",
    ]
    print("\nEquipos WC2026 relevantes:")
    wc_ratings = latest[latest["team_name"].isin(wc_teams)]
    for _, row in wc_ratings.iterrows():
        print(f"  {row['team_name']:<25} {row['elo_rating']:>7.1f}  (rank #{int(row['rank'])})")
        