"""
src/data/elo_loader.py
=======================
Calcula ratings Elo históricos de selecciones nacionales a partir de
los resultados históricos (results.csv de Kaggle / unified.csv).

El algoritmo sigue la metodología de eloratings.net:
  - K variable según la importancia del partido
  - Factor de goles (Goal Difference Weight)
  - Sin factor de ventaja local para partidos del WC2026
    (todos en sede neutral)

La salida es data/processed/team_ratings.csv, que el builder.py
ya sabe leer y hacer join as-of sin leakage.

Uso:
    python src/data/elo_loader.py
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

ELO_START = 1500.0        # Rating inicial para equipos sin historial
HOME_ADVANTAGE = 100.0    # Puntos de ventaja local (no aplica en neutrales)

# K base según importancia del partido (basado en metodología eloratings.net)
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

# Selecciones con rating inicial elevado (potencias históricas)
STRONG_TEAMS_INIT = {
    "Brazil": 1900, "Germany": 1880, "Argentina": 1870, "Spain": 1860,
    "France": 1850, "Italy": 1840, "Netherlands": 1830, "England": 1820,
    "Portugal": 1800, "Belgium": 1790, "Uruguay": 1780, "Croatia": 1770,
}


# ---------------------------------------------------------------------------
# Cálculo de Elo
# ---------------------------------------------------------------------------

def _k_factor(tournament: str) -> float:
    """K factor según el torneo."""
    for key, k in K_VALUES.items():
        if key.lower() in str(tournament).lower():
            return float(k)
    return float(K_VALUES["default"])


def _goal_weight(home_goals: int, away_goals: int) -> float:
    """
    Factor multiplicador por diferencia de goles (metodología eloratings.net).
    Victorias ajustadas: no es lo mismo ganar 1-0 que 5-0.
    """
    diff = abs(home_goals - away_goals)
    if diff == 0 or diff == 1:
        return 1.0
    elif diff == 2:
        return 1.5
    else:
        return (11 + diff) / 8.0


def _expected_score(rating_home: float, rating_away: float, is_neutral: bool) -> float:
    """
    Probabilidad esperada de victoria del equipo home según Elo.
    Incluye ventaja local si el partido no es neutral.
    """
    advantage = 0.0 if is_neutral else HOME_ADVANTAGE
    dr = rating_home + advantage - rating_away
    return 1.0 / (10.0 ** (-dr / 400.0) + 1.0)


def _actual_score(home_goals: int, away_goals: int) -> float:
    """Resultado real: 1.0 victoria local, 0.5 empate, 0.0 derrota local."""
    if home_goals > away_goals:
        return 1.0
    elif home_goals == away_goals:
        return 0.5
    return 0.0


def compute_elo_history(
    df: pd.DataFrame,
    save_snapshots_every_n: int = 1,
) -> pd.DataFrame:
    """
    Calcula el rating Elo de cada selección partido a partido.

    Parámetros
    ----------
    df : DataFrame con columnas: date, home_team, away_team,
         home_score, away_score, tournament, neutral
    save_snapshots_every_n : cada cuántos partidos guardar snapshot
         (1 = guardar después de cada partido, más preciso para joins)

    Retorna
    -------
    DataFrame con columnas: team_name, rating_date, elo_rating, source
    Ordenado por team_name, rating_date.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team",
                            "home_score", "away_score"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Inicializar ratings
    ratings: dict[str, float] = {}

    def get_rating(team: str) -> float:
        if team not in ratings:
            ratings[team] = float(STRONG_TEAMS_INIT.get(team, ELO_START))
        return ratings[team]

    # Guardar snapshots: (team, date, rating)
    snapshots: list[tuple[str, pd.Timestamp, float]] = []

    # Snapshot inicial para todos los equipos conocidos
    all_teams = set(df["home_team"]).union(set(df["away_team"]))
    first_date = df["date"].min() - pd.Timedelta(days=1)
    for team in all_teams:
        snapshots.append((team, first_date, get_rating(team)))

    log.info("Calculando Elo para %d equipos sobre %d partidos...", len(all_teams), len(df))

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

    # Construir DataFrame
    result = pd.DataFrame(snapshots, columns=["team_name", "rating_date", "elo_rating"])
    result = result.sort_values(["team_name", "rating_date"]).reset_index(drop=True)
    result["source"] = "computed_elo"

    # Añadir ranking por fecha (aproximado)
    result["rank"] = result.groupby("rating_date")["elo_rating"].rank(
        ascending=False, method="min"
    )

    log.info("Snapshots totales: %d", len(result))
    return result


def get_latest_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el rating más reciente de cada equipo."""
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

    # Cargar datos históricos
    # Preferir unified.csv (ya filtrado) sobre results_raw.csv
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

    # Calcular Elo
    ratings = compute_elo_history(df, save_snapshots_every_n=1)

    # Guardar historial completo (para joins as-of en builder.py)
    out_path = os.path.join(OUT_DIR, "team_ratings.csv")
    ratings.to_csv(out_path, index=False)
    log.info("Historial Elo guardado en %s  (%d filas)", out_path, len(ratings))

    # Mostrar top 30 actuales
    latest = get_latest_ratings(ratings)
    print("\n" + "=" * 55)
    print("TOP 30 SELECCIONES POR ELO (rating actual)")
    print("=" * 55)
    for i, row in latest.head(30).iterrows():
        print(f"  {i+1:>2}. {row['team_name']:<25} {row['elo_rating']:>7.1f}")
    print("=" * 55)

    # Verificar equipos del WC2026
    wc_teams = [
        "Brazil", "Argentina", "France", "England", "Spain",
        "Germany", "Portugal", "Netherlands", "Belgium",
        "Morocco", "Japan", "South Korea", "Mexico",
        "United States", "Canada",
    ]
    print("\nEquipos WC2026 relevantes:")
    wc_ratings = latest[latest["team_name"].isin(wc_teams)].head(20)
    for _, row in wc_ratings.iterrows():
        print(f"  {row['team_name']:<25} {row['elo_rating']:>7.1f}  (rank #{int(row['rank'])})")
        