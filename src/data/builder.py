"""
src/data/builder.py
====================
Builder principal del dataset del Mundial 2026.

Une las tablas normalizadas en un DataFrame por partido listo para
feature engineering. Diseñado para funcionar incrementalmente: cada
fuente es opcional y enriquece el dataset sin romper el pipeline si falta.

Uso básico
----------
    from src.data.builder import MatchDatasetBuilder

    builder = MatchDatasetBuilder(data_dir="./data")
    df = builder.build()
    df.to_csv("data/processed/matches_enriched.csv", index=False)

Fuentes que une
---------------
    matches       ← group_stage_wc2026.csv + knockout_wc2026.csv  (y cualquier histórico)
    venues        ← venues.csv
    base_camps    ← base_camps_wc2026.csv
    referees      ← referees_wc2026.csv  (+  referees.csv si existe con stats históricas)
    team_ratings  ← Elo / FIFA ranking (join as-of sin leakage)
    weather       ← Open-Meteo por sede/hora  (opcional, si existe weather_hourly.csv)

El builder NO toca features.csv ni los modelos. Su salida es
data/processed/matches_enriched.csv.
"""

from __future__ import annotations

import logging
import math
import os
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.data.normalize_squads import make_team_id

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM = 6371.0

# Venue IDs tal como aparecen en los CSVs de Dani
VENUE_ID_COL = "venue_id"

# Nombres de columnas de salida (contrato con feature_engineering.py)
OUTPUT_COLS_MATCHES = [
    "match_id", "competition", "season", "stage", "group",
    "match_date", "kickoff_local",
    "home_team", "away_team",
    "venue_id", "referee_name",
    "home_score", "away_score", "result",
    "neutral",
    # venue
    "venue_name", "city", "country", "altitude_m",
    "roof_type", "surface", "timezone",
    "venue_lat", "venue_lon",
    # rest
    "home_rest_days", "away_rest_days",
    # travel (base_camp → venue)
    "home_travel_km", "away_travel_km",
    "home_tz_change", "away_tz_change",
    # altitude delta entre campamento base y sede
    "home_altitude_delta_m", "away_altitude_delta_m",
    # referee stats históricos
    "ref_yellow_per_match", "ref_red_per_match",
    "ref_matches", "ref_confederation",
    # team ratings as-of (Elo)
    "home_elo", "away_elo", "elo_diff",
    # weather at kickoff
    "temperature_c", "humidity_pct", "precipitation_mm", "wind_speed_kmh",
    # host advantage flag
    "home_is_host", "away_is_host",
    # squad summary (WC2026 only)
    "home_team_id", "away_team_id",
    "home_squad_size", "away_squad_size",
    "home_goalkeepers", "away_goalkeepers",
    "home_defenders", "away_defenders",
    "home_midfielders", "away_midfielders",
    "home_forwards", "away_forwards",
    "home_unique_clubs", "away_unique_clubs",
]

HOST_NATIONS = {"United States", "Mexico", "Canada"}
HOST_COUNTRY_BY_TEAM = {
    "United States": "United States",
    "Mexico": "Mexico",
    "Canada": "Canada",
}


# ---------------------------------------------------------------------------
# Utilidades geográficas
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia en km entre dos puntos (lat/lon en grados decimales)."""
    if any(math.isnan(v) for v in (lat1, lon1, lat2, lon2)):
        return float("nan")
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _tz_offset_hours(tz_name: str) -> float:
    """
    Devuelve el offset UTC en horas de una zona horaria IANA.
    Usa pandas que ya tiene pytz/zoneinfo integrado.
    Retorna NaN si la zona no se reconoce.
    """
    try:
        ts = pd.Timestamp.now(tz=tz_name)
        return ts.utcoffset().total_seconds() / 3600.0
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Loaders de CSVs estáticos
# ---------------------------------------------------------------------------

def _load_csv(path: str, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        log.warning("Archivo no encontrado: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=parse_dates or [])
    log.info("Cargado %s  (%d filas)", os.path.basename(path), len(df))
    return df


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _compute_rest_days(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada partido calcula los días de descanso de cada equipo desde
    su partido anterior en el mismo torneo (sin leakage: usa fecha anterior).

    Añade columnas: home_rest_days, away_rest_days.
    """
    matches = matches.copy()
    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")
    matches = matches.sort_values("match_date").reset_index(drop=True)

    # Construir índice de último partido por equipo
    last_match: dict[str, pd.Timestamp] = {}
    home_rest, away_rest = [], []

    for _, row in matches.iterrows():
        date = row["match_date"]
        ht = row.get("home_team", row.get("home_team_id", ""))
        at = row.get("away_team", row.get("away_team_id", ""))

        hr = (date - last_match[ht]).days if ht in last_match else float("nan")
        ar = (date - last_match[at]).days if at in last_match else float("nan")
        home_rest.append(hr)
        away_rest.append(ar)

        last_match[ht] = date
        last_match[at] = date

    matches["home_rest_days"] = home_rest
    matches["away_rest_days"] = away_rest
    return matches

def _build_ratings_index(ratings: pd.DataFrame) -> dict:
    """Precalcula, por equipo, (fechas, elos) ordenados para búsqueda as-of rápida."""
    ratings = ratings.dropna(subset=["team_name", "rating_date", "elo_rating"]).copy()
    ratings["rating_date"] = pd.to_datetime(ratings["rating_date"], errors="coerce")
    ratings = ratings.dropna(subset=["rating_date"]).sort_values("rating_date")
    index = {}
    for team, sub in ratings.groupby("team_name"):
        index[team] = (sub["rating_date"].values, sub["elo_rating"].values)
    return index


def _as_of_rating_fast(index: dict, team_name: str, before_date) -> float:
    """Elo más reciente ESTRICTAMENTE antes de before_date, con índice precalculado."""
    data = index.get(team_name)
    if data is None:
        return float("nan")
    dates, elos = data
    pos = np.searchsorted(dates, np.datetime64(before_date), side="left")
    if pos == 0:
        return float(elos[0])
    return float(elos[pos - 1])


def _as_of_rating(
    ratings: pd.DataFrame,
    team_name: str,
    before_date: pd.Timestamp,
) -> Optional[float]:
    """
    Devuelve el Elo más reciente de un equipo estrictamente antes de before_date.
    Evita data leakage.
    """
    if ratings.empty:
        return float("nan")
    mask = (ratings["team_name"] == team_name) & (ratings["rating_date"] < before_date)
    subset = ratings[mask]
    if subset.empty:
        # Fallback: rating más antiguo disponible
        fallback = ratings[ratings["team_name"] == team_name]
        return float(fallback["elo_rating"].iloc[0]) if not fallback.empty else float("nan")
    return float(subset.sort_values("rating_date").iloc[-1]["elo_rating"])


# ---------------------------------------------------------------------------
# Builder principal
# ---------------------------------------------------------------------------

class MatchDatasetBuilder:
    """
    Une matches + venues + base_camps + referees + team_ratings + weather
    en un único DataFrame por partido.

    Parámetros
    ----------
    data_dir : str
        Carpeta raíz con subcarpetas raw/ y processed/.
    require_result : bool
        Si True (default) filtra partidos sin resultado (histórico).
        Si False incluye partidos futuros del Mundial 2026 sin score.
    """

    def __init__(self, data_dir: str = "./data", require_result: bool = True):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        self.require_result = require_result

        # DataFrames internos (se cargan en build())
        self._venues: pd.DataFrame = pd.DataFrame()
        self._base_camps: pd.DataFrame = pd.DataFrame()
        self._referees: pd.DataFrame = pd.DataFrame()
        self._ratings: pd.DataFrame = pd.DataFrame()
        self._weather: pd.DataFrame = pd.DataFrame()
        self._squad_summary: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # Carga de fuentes
    # ------------------------------------------------------------------

    def _load_all_sources(self) -> None:
        r = self.raw_dir
        p = self.processed_dir

        self._venues = _load_csv(os.path.join(r, "venues.csv"))
        self._base_camps = _load_csv(os.path.join(r, "base_camps_wc2026.csv"))

        # Árbitros: intentar versión con stats históricas primero
        ref_stats = _load_csv(os.path.join(p, "referees_with_stats.csv"))
        ref_raw = _load_csv(os.path.join(r, "referees_wc2026.csv"))
        if not ref_stats.empty:
            self._referees = ref_stats
            log.info("Usando referees_with_stats.csv (con datos históricos)")
        elif not ref_raw.empty:
            self._referees = ref_raw
            log.info("Usando referees_wc2026.csv (sin stats históricas aún)")

        # Ratings Elo (si existen)
        ratings_path = os.path.join(p, "team_ratings.csv")
        if os.path.exists(ratings_path):
            self._ratings = _load_csv(
                ratings_path, parse_dates=["rating_date"]
            )
        else:
            log.warning(
                "No se encontró team_ratings.csv — Elo no disponible. "
                "Ejecuta el loader de Elo o descarga el CSV de eloratings.net"
            )

        # Clima por sede/hora (opcional)
        weather_path = os.path.join(p, "weather_hourly.csv")
        if os.path.exists(weather_path):
            self._weather = _load_csv(
                weather_path, parse_dates=["datetime"]
            )
        else:
            log.info("weather_hourly.csv no encontrado — features de clima omitidas")

        squad_summary_path = os.path.join(p, "squad_summary_wc2026.csv")
        if os.path.exists(squad_summary_path):
            self._squad_summary = _load_csv(squad_summary_path)
        else:
            log.info(
                "squad_summary_wc2026.csv no encontrado — ejecuta "
                "python -m src.data.normalize_squads para conectar convocatorias"
            )

    # ------------------------------------------------------------------
    # Carga de partidos
    # ------------------------------------------------------------------

    def _load_matches(self) -> pd.DataFrame:
        """
        Carga partidos de dos fuentes con esquemas distintos y las une limpiamente.

        Fuente A — Histórico (unified.csv / Kaggle+StatsBomb):
            Columnas clave: date, home_team, away_team, home_score, away_score,
            tournament, neutral, result, corners_*, yellow_*, xg_*, etc.
            NO tiene venue_id, competition, season, match_number.

        Fuente B — WC2026 (group_stage + knockout CSVs):
            Columnas clave: match_date, home_team, away_team, venue_id,
            competition, season, stage, group, match_number.
            NO tiene scores (partidos futuros) ni stats avanzadas.

        Las features de sede/viaje/clima solo se aplican a partidos WC2026
        donde existe venue_id. El histórico solo recibe rest_days.
        """
        frames = []

        # ── Fuente A: histórico unificado ──────────────────────────────────
        unified_path = os.path.join(self.data_dir, "unified.csv")
        if os.path.exists(unified_path):
            hist = _load_csv(unified_path, parse_dates=["date"])
            if not hist.empty:
                if "date" in hist.columns and "match_date" not in hist.columns:
                    hist = hist.rename(columns={"date": "match_date"})
                hist["_source"] = "historical"
                frames.append(hist)
                log.info("Histórico: %d partidos desde unified.csv", len(hist))
        else:
            log.warning(
                "unified.csv no encontrado — ejecuta data_collector.py primero. "
                "El builder continuará solo con los partidos WC2026."
            )

        # ── Fuente B: partidos WC2026 (fase de grupos) ─────────────────────
        gs = _load_csv(
            os.path.join(self.raw_dir, "group_stage_wc2026.csv"),
            parse_dates=["match_date"],
        )
        if not gs.empty:
            gs["_source"] = "wc2026"
            gs["neutral"] = True
            frames.append(gs)
            log.info("WC2026 grupos: %d partidos", len(gs))

        # ── Fuente B: partidos WC2026 (eliminatorias) ──────────────────────
        ko = _load_csv(
            os.path.join(self.raw_dir, "knockout_wc2026.csv"),
            parse_dates=["match_date"],
        )
        if not ko.empty:
            if "home_slot" in ko.columns:
                ko = ko.rename(columns={"home_slot": "home_team", "away_slot": "away_team"})
            ko["_source"] = "wc2026"
            ko["neutral"] = True
            frames.append(ko)
            log.info("WC2026 eliminatorias: %d partidos", len(ko))

        if not frames:
            log.error("No se encontraron partidos. Ejecuta data_collector.py primero.")
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True, sort=False)
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
        df = df.sort_values("match_date").reset_index(drop=True)
        log.info("Total partidos: %d (%d histórico, %d WC2026)",
                 len(df),
                 (df["_source"] == "historical").sum(),
                 (df["_source"] == "wc2026").sum())
        from src.data.team_names import add_canonical_columns
        df = add_canonical_columns(df, ["home_team", "away_team"], suffix="")
        return df


    # ------------------------------------------------------------------
    # Joins de enriquecimiento
    # ------------------------------------------------------------------

    def _join_venues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade coordenadas, altitud, superficie y timezone de cada sede."""
        if self._venues.empty:
            log.warning("venues.csv vacío — features de sede omitidas")
            return df

        venues = self._venues.rename(columns={
            "lat": "venue_lat",
            "lon": "venue_lon",
            "stadium_name": "venue_name",
        }).copy()

        # Seleccionar columnas relevantes
        keep = ["venue_id", "venue_name", "city", "country",
                "venue_lat", "venue_lon", "altitude_m",
                "roof_type", "surface", "timezone"]
        venues = venues[[c for c in keep if c in venues.columns]]

        # Solo aplicar a partidos WC2026 que tienen venue_id
        if "venue_id" not in df.columns:
            log.info("Venues: ningún partido tiene venue_id — omitido")
            return df

        # Eliminar columnas que colisionarían con el merge (evita sufijos _x/_y)
        conflict_cols = [c for c in ["city", "country"] if c in df.columns and c in venues.columns]
        if conflict_cols:
            df = df.drop(columns=conflict_cols)

        has_venue = df["venue_id"].notna()
        wc_rows = df[has_venue].merge(venues, on=VENUE_ID_COL, how="left")
        hist_rows = df[~has_venue]

        df = pd.concat([wc_rows, hist_rows], ignore_index=True, sort=False)
        df = df.sort_values("match_date").reset_index(drop=True)

        n_matched = df["altitude_m"].notna().sum() if "altitude_m" in df.columns else 0
        log.info("Venues: %d/%d partidos con sede enriquecida", n_matched, len(df))
        return df

    def _join_referee_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cruza el árbitro del partido con su perfil histórico.
        Busca por nombre exacto o fuzzy si rapidfuzz está disponible.
        """
        if self._referees.empty:
            return df

        ref = self._referees.copy()

        # Columna de nombre en el CSV de árbitros
        name_col = next(
            (c for c in ["referee_name", "name", "referee"] if c in ref.columns),
            None,
        )
        if name_col is None:
            log.warning("referees CSV no tiene columna de nombre reconocible")
            return df

        ref = ref.rename(columns={name_col: "referee_name"})

        # Stats que nos interesan
        stat_cols = [c for c in [
            "yellow_per_match", "red_per_match", "matches",
            "yellow_cards", "red_cards", "confederation",
            "yellow_per_match_model", "red_per_match_model",   # <-- añadir
        ] if c in ref.columns]

        ref_stats = ref[["referee_name"] + stat_cols].drop_duplicates("referee_name")
        ref_stats.columns = ["referee_name"] + [f"ref_{c}" for c in stat_cols]

        # Merge por nombre exacto primero
        if "referee" not in df.columns and "referee_name" not in df.columns:
            log.info("Partidos históricos sin columna referee — stats de árbitro omitidas")
            return df

        ref_col = "referee" if "referee" in df.columns else "referee_name"
        df = df.rename(columns={ref_col: "referee_name"})
        df = df.merge(ref_stats, on="referee_name", how="left")

        n = df["ref_yellow_per_match"].notna().sum() if "ref_yellow_per_match" in df.columns else 0
        log.info("Árbitros: %d/%d partidos con stats históricas", n, len(df))
        return df

    def _join_base_camps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula distancia base_camp→venue y cambio de timezone para cada equipo.
        Requiere que venues ya estén unidas (venue_lat, venue_lon, timezone).
        """
        if self._base_camps.empty:
            log.info("base_camps_wc2026.csv no disponible — features de viaje omitidas")
            return df

        # Solo calcular viajes para partidos WC2026 con venue_id
        if "venue_id" not in df.columns or df["venue_id"].isna().all():
            log.info("Viajes: no hay partidos WC2026 con venue_id — omitido")
            return df

        bc = self._base_camps.copy()

        # Necesitamos lat/lon de los campamentos.
        # Si el CSV no tiene lat/lon aún (punto 2 de la lista de Dani),
        # intentamos geocodificar desde city+country usando un diccionario
        # básico de ciudades WC2026 como fallback.
        if "lat" not in bc.columns or bc["lat"].isna().all():
            bc = _add_base_camp_coords(bc)

        if "timezone" not in bc.columns:
            bc["timezone"] = float("nan")
        bc["bc_tz_offset"] = bc["timezone"].apply(
            lambda tz: _tz_offset_hours(str(tz)) if pd.notna(tz) else float("nan")
        )

        # Construir lookup: team_name → (lat, lon, tz_offset)
        bc_lookup = bc.set_index("team_name")[
            [c for c in ["lat", "lon", "bc_tz_offset"] if c in bc.columns]
        ].to_dict("index")

        home_km, away_km = [], []
        home_tz, away_tz = [], []

        venue_tz_cache: dict[str, float] = {}

        for _, row in df.iterrows():
            v_lat = row.get("venue_lat", float("nan"))
            v_lon = row.get("venue_lon", float("nan"))
            v_tz_name = row.get("timezone", "")
            if v_tz_name not in venue_tz_cache:
                venue_tz_cache[v_tz_name] = _tz_offset_hours(v_tz_name) if v_tz_name else float("nan")
            v_tz = venue_tz_cache[v_tz_name]

            for team_col, km_list, tz_list in [
                ("home_team", home_km, home_tz),
                ("away_team", away_km, away_tz),
            ]:
                team = row.get(team_col, "")
                bc_data = bc_lookup.get(team, {})
                b_lat = bc_data.get("lat", float("nan"))
                b_lon = bc_data.get("lon", float("nan"))
                b_tz = bc_data.get("bc_tz_offset", float("nan"))

                dist = haversine_km(b_lat, b_lon, v_lat, v_lon) if not (
                    math.isnan(b_lat) or math.isnan(v_lat)
                ) else float("nan")
                tz_delta = abs(b_tz - v_tz) if not (
                    math.isnan(b_tz) or math.isnan(v_tz)
                ) else float("nan")

                km_list.append(dist)
                tz_list.append(tz_delta)

        df["home_travel_km"] = home_km
        df["away_travel_km"] = away_km
        df["home_tz_change"] = home_tz
        df["away_tz_change"] = away_tz

        n_travel = df["home_travel_km"].notna().sum()
        log.info("Viajes: %d/%d partidos con distancia base_camp→sede", n_travel, len(df))
        return df

    def _join_altitude_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Altitud del campamento base vs altitud de la sede.
        Útil para partidos en Ciudad de México (2240m) o Guadalajara (1560m).
        """
        if self._base_camps.empty or "altitude_m" not in df.columns:
            return df

        bc = self._base_camps.copy()
        # altitude_m del campamento (si existe)
        if "altitude_m" not in bc.columns:
            return df

        bc_alt = bc.set_index("team_name")["altitude_m"].to_dict()

        df["home_altitude_delta_m"] = df.apply(
            lambda r: (bc_alt.get(r.get("home_team", ""), float("nan")) or float("nan"))
                      - (r.get("altitude_m") or float("nan")),
            axis=1,
        )
        df["away_altitude_delta_m"] = df.apply(
            lambda r: (bc_alt.get(r.get("away_team", ""), float("nan")) or float("nan"))
                      - (r.get("altitude_m") or float("nan")),
            axis=1,
        )
        return df

    def _join_team_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade Elo as-of para home y away sin data leakage."""
        if self._ratings.empty:
            df["home_elo"] = float("nan")
            df["away_elo"] = float("nan")
            df["elo_diff"] = float("nan")
            return df

        # ratings = self._ratings.copy()
        # ratings["rating_date"] = pd.to_datetime(ratings["rating_date"], errors="coerce")

        # home_elo_list, away_elo_list = [], []
        # for _, row in df.iterrows():
        #     date = row.get("match_date")
        #     ht = row.get("home_team", "")
        #     at = row.get("away_team", "")
        #     if pd.isna(date):
        #         home_elo_list.append(float("nan"))
        #         away_elo_list.append(float("nan"))
        #         continue
        #     home_elo_list.append(_as_of_rating(ratings, ht, date))
        #     away_elo_list.append(_as_of_rating(ratings, at, date))

        ratings = self._ratings.copy()
        ratings["rating_date"] = pd.to_datetime(ratings["rating_date"], errors="coerce")
        index = _build_ratings_index(ratings)

        home_elo_list, away_elo_list = [], []
        for _, row in df.iterrows():
            date = row.get("match_date")
            ht = row.get("home_team", "")
            at = row.get("away_team", "")
            if pd.isna(date):
                home_elo_list.append(float("nan"))
                away_elo_list.append(float("nan"))
                continue
            home_elo_list.append(_as_of_rating_fast(index, ht, date))
            away_elo_list.append(_as_of_rating_fast(index, at, date))

        df["home_elo"] = home_elo_list
        df["away_elo"] = away_elo_list
        df["elo_diff"] = df["home_elo"] - df["away_elo"]

        n = df["home_elo"].notna().sum()
        log.info("Elo: %d/%d partidos con rating enriquecido", n, len(df))
        return df

    def _join_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Une clima horario por sede/hora con cada partido.
        Usa kickoff_local si existe, si no usa match_date a las 20:00 local.
        """
        if self._weather.empty:
            return df

        weather = self._weather.copy()
        weather["datetime"] = pd.to_datetime(weather["datetime"], errors="coerce", utc=True)

        # Redondear a hora exacta para el join
        weather["hour"] = weather["datetime"].dt.floor("h")
        weather_idx = weather.set_index(["venue_id", "hour"])

        temp_list, hum_list, prec_list, wind_list = [], [], [], []

        for _, row in df.iterrows():
            vid = row.get("venue_id")
            kt = row.get("kickoff_local")
            if pd.isna(kt):
                # Fallback: medianoche de la fecha
                kt = pd.Timestamp(row.get("match_date"))
            else:
                kt = pd.Timestamp(kt)

            kt_hour = kt.floor("h")

            try:
                w = weather_idx.loc[(vid, kt_hour)]
                temp_list.append(w.get("temperature_2m") if hasattr(w, "get") else w["temperature_2m"])
                hum_list.append(w.get("relative_humidity_2m") if hasattr(w, "get") else w["relative_humidity_2m"])
                prec_list.append(w.get("precipitation") if hasattr(w, "get") else w["precipitation"])
                wind_list.append(w.get("wind_speed_10m") if hasattr(w, "get") else w["wind_speed_10m"])
            except KeyError:
                temp_list.append(float("nan"))
                hum_list.append(float("nan"))
                prec_list.append(float("nan"))
                wind_list.append(float("nan"))

        df["temperature_c"] = temp_list
        df["humidity_pct"] = hum_list
        df["precipitation_mm"] = prec_list
        df["wind_speed_kmh"] = wind_list

        n = df["temperature_c"].notna().sum()
        log.info("Clima: %d/%d partidos con datos meteorológicos", n, len(df))
        return df

    def _add_host_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Marca ventaja de anfitrión solo cuando el equipo juega en su país sede."""
        venue_country = df.get("country", pd.Series(index=df.index, dtype=object)).fillna("")
        df["home_is_host"] = [
            int(HOST_COUNTRY_BY_TEAM.get(team) == country)
            for team, country in zip(df.get("home_team", pd.Series(dtype=str)), venue_country)
        ]
        df["away_is_host"] = [
            int(HOST_COUNTRY_BY_TEAM.get(team) == country)
            for team, country in zip(df.get("away_team", pd.Series(dtype=str)), venue_country)
        ]
        return df

    def _add_team_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade IDs estables para enlazar partidos con tablas de equipos/jugadores."""
        df["home_team_id"] = df["home_team"].map(make_team_id)
        df["away_team_id"] = df["away_team"].map(make_team_id)
        return df

    def _join_squad_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade conteos de convocatoria solo a filas WC2026.

        El histórico no tiene una fuente equivalente de convocatorias, así que
        dejamos esas columnas a NaN para evitar entrenar con datos no comparables.
        """
        squad_cols = [
            "squad_size", "goalkeepers", "defenders",
            "midfielders", "forwards", "unique_clubs",
        ]
        for side in ["home", "away"]:
            for col in squad_cols:
                out_col = f"{side}_{col}"
                if out_col not in df.columns:
                    df[out_col] = float("nan")

        if self._squad_summary.empty:
            return df
        if "team_id" not in self._squad_summary.columns:
            log.warning("squad_summary_wc2026.csv no tiene team_id — convocatorias omitidas")
            return df
        if "_source" not in df.columns:
            return df

        summary = self._squad_summary[["team_id"] + [
            c for c in squad_cols if c in self._squad_summary.columns
        ]].drop_duplicates("team_id")

        wc_mask = df["_source"].eq("wc2026")
        wc = df[wc_mask].copy()
        rest = df[~wc_mask].copy()

        for side in ["home", "away"]:
            wc = wc.drop(
                columns=[f"{side}_{col}" for col in squad_cols],
                errors="ignore",
            )
            prefixed = summary.rename(columns={
                col: f"{side}_{col}" for col in squad_cols if col in summary.columns
            })
            wc = wc.merge(
                prefixed,
                left_on=f"{side}_team_id",
                right_on="team_id",
                how="left",
                suffixes=("", "_squad"),
            )
            wc = wc.drop(columns=["team_id"], errors="ignore")

        df = pd.concat([wc, rest], ignore_index=True, sort=False)
        for side in ["home", "away"]:
            for col in squad_cols:
                out_col = f"{side}_{col}"
                if out_col not in df.columns:
                    df[out_col] = float("nan")
        df = df.sort_values("match_date").reset_index(drop=True)

        n_home = df.loc[df["_source"].eq("wc2026"), "home_squad_size"].notna().sum()
        n_away = df.loc[df["_source"].eq("wc2026"), "away_squad_size"].notna().sum()
        log.info("Convocatorias: %d/%d home y %d/%d away WC2026 enlazadas",
                 n_home, wc_mask.sum(), n_away, wc_mask.sum())
        return df

    def _add_result_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade columna result (H/D/A) si los scores están disponibles."""
        if "result" in df.columns:
            return df
        if "home_score" not in df.columns or "away_score" not in df.columns:
            df["result"] = float("nan")
            return df

        def _result(row):
            hs = row.get("home_score")
            as_ = row.get("away_score")
            if pd.isna(hs) or pd.isna(as_):
                return float("nan")
            if hs > as_:
                return "H"
            if hs < as_:
                return "A"
            return "D"

        df["result"] = df.apply(_result, axis=1)
        return df

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------

    def build(self, save: bool = True) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo y devuelve el DataFrame enriquecido.

        Parámetros
        ----------
        save : bool
            Si True, guarda el resultado en data/processed/matches_enriched.csv
        """
        log.info("=" * 60)
        log.info("MatchDatasetBuilder — iniciando pipeline")
        log.info("=" * 60)

        self._load_all_sources()

        df = self._load_matches()
        if df.empty:
            return df

        # Filtrar por resultado si aplica — SOLO sobre el histórico.
        # Los partidos WC2026 se conservan siempre (aún no tienen score).
        if self.require_result and "_source" in df.columns:
            before = len(df)
            has_score = "home_score" in df.columns and "away_score" in df.columns
            mask_hist_with_result = (
                (df["_source"] == "historical") &
                (df["home_score"].notna() & df["away_score"].notna() if has_score else False)
            )
            mask_wc2026 = df["_source"] == "wc2026"
            df = df[mask_hist_with_result | mask_wc2026].copy()
            log.info(
                "require_result=True: %d → %d partidos "
                "(%d histórico con resultado + %d WC2026)",
                before, len(df),
                mask_hist_with_result.sum(),
                mask_wc2026.sum(),
            )

        # Pipeline de enriquecimiento (el orden importa)
        df = _compute_rest_days(df)
        df = self._join_venues(df)
        df = self._join_base_camps(df)
        df = self._join_altitude_delta(df)
        df = self._join_referee_stats(df)
        df = self._join_team_ratings(df)
        df = self._join_weather(df)
        df = self._add_host_flags(df)
        df = self._add_team_ids(df)
        df = self._join_squad_summary(df)
        df = self._add_result_column(df)

        log.info("-" * 60)
        log.info("Dataset final: %d partidos × %d columnas", len(df), len(df.columns))

        # Resumen de cobertura por columna clave
        key_cols = [
            "altitude_m", "home_travel_km", "home_elo",
            "ref_yellow_per_match", "temperature_c", "home_squad_size",
        ]
        for col in key_cols:
            if col in df.columns:
                pct = df[col].notna().mean() * 100
                log.info("  %-28s cobertura: %.1f%%", col, pct)

        if save:
            out = os.path.join(self.processed_dir, "matches_enriched.csv")
            df.to_csv(out, index=False)
            log.info("Guardado en %s", out)

        return df


# ---------------------------------------------------------------------------
# Geocodificación básica de campamentos base (fallback sin API)
# ---------------------------------------------------------------------------

# Coordenadas aproximadas de las ciudades de campamento base del WC2026.
# Extraídas de base_camps_wc2026.csv. Se usa solo si lat/lon no están en el CSV.
_BASE_CAMP_COORDS: dict[str, tuple[float, float]] = {
    "Lawrence": (38.9717, -95.2353),
    "Kansas City": (39.0997, -94.5786),
    "Berkeley": (37.8716, -122.2727),
    "Goleta": (34.4258, -119.8276),
    "Renton": (47.4829, -122.2171),
    "Salt Lake City": (40.7608, -111.8910),
    "Basking Ridge": (40.7057, -74.5488),
    "Vancouver": (49.2827, -123.1207),
    "Tampa": (27.9506, -82.4572),
    "Guadalajara": (20.6597, -103.3496),
    "Alexandria": (38.8048, -77.0469),
    "Boca Raton": (26.3683, -80.1289),
    "Arlington": (32.7357, -97.1081),
    "Houston": (29.7604, -95.3698),
    "Columbus": (39.9612, -82.9988),
    "Airway Heights": (47.6449, -117.5920),
    "Kansas City": (39.0997, -94.5786),
    "Boston": (42.3601, -71.0589),
    "Winston-Salem": (36.0999, -80.2442),
    "Providence": (41.8240, -71.4128),
    "Atlantic City": (39.3643, -74.4229),
    "Tijuana": (32.5149, -117.0382),
    "White Sulphur Springs": (37.7965, -80.2997),
    "Wilmington": (39.7447, -75.5484),
    "Nashville": (36.1627, -86.7816),
    "Portland": (45.5051, -122.6750),
    "Mexico City": (19.4326, -99.1332),
    "Warren": (40.7684, -74.4143),
    "New Brunswick": (40.4862, -74.4518),
    "San Jose": (37.3382, -121.8863),
    "Palm Beach": (26.7056, -80.0364),
    "Austin": (30.2672, -97.7431),
    "Charlotte": (35.2271, -80.8431),
    "Frisco": (33.1584, -96.8236),
    "San Diego": (32.7157, -117.1611),
    "San Pedro": (23.7464, -110.6318),
    "Mesa": (33.4152, -111.8315),
    "Irvine": (33.6846, -117.8265),
    "Playa del Carmen": (20.6296, -87.0739),
    "Atlanta": (33.7490, -84.3880),
    "Pachuca": (20.1011, -98.7591),
    "New Tecumseth": (44.0780, -79.8133),
    "Greensboro": (36.0726, -79.7920),
    "Chattanooga": (35.0456, -85.3097),
    "Seattle": (47.6062, -122.3321),  # fallback
}


def _add_base_camp_coords(bc: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas lat/lon a base_camps usando el diccionario interno de coords.
    Solo se usa si el CSV no tiene lat/lon.
    """
    if "lat" not in bc.columns:
        bc["lat"] = float("nan")
    if "lon" not in bc.columns:
        bc["lon"] = float("nan")

    for idx, row in bc.iterrows():
        city = row.get("city", "")
        coords = _BASE_CAMP_COORDS.get(city)
        if coords and (pd.isna(row.get("lat")) or row.get("lat") == ""):
            bc.at[idx, "lat"] = coords[0]
            bc.at[idx, "lon"] = coords[1]

    n = bc["lat"].notna().sum()
    log.info(
        "base_camp coords: %d/%d equipos con lat/lon (desde diccionario interno)",
        n, len(bc),
    )
    return bc


# ---------------------------------------------------------------------------
# CLI / uso standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Modo: "historical" (con resultado) o "wc2026" (con y sin resultado)
    mode = sys.argv[1] if len(sys.argv) > 1 else "historical"
    require_result = mode != "wc2026"

    builder = MatchDatasetBuilder(
        data_dir="./data",
        require_result=require_result,
    )
    df = builder.build(save=True)

    print(f"\n{'='*60}")
    print(f"RESUMEN — {len(df)} partidos × {len(df.columns)} columnas")
    print(f"{'='*60}")
    if not df.empty:
        print(f"Rango temporal:  {df['match_date'].min()} → {df['match_date'].max()}")
        for col in df.columns:
            pct_null = df[col].isna().mean() * 100
            if pct_null > 0:
                print(f"  {col:<35}  {pct_null:5.1f}% nulos")
