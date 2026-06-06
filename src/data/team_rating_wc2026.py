"""
src/data/team_rating_wc2026.py
================================
Rating compuesto de selecciones para el Mundial 2026.

Fórmula:
    rating = 0.30 * elo_norm + 0.50 * valor_mercado_norm + 0.20 * racha_norm

Racha ponderada (últimos 10 partidos vs rivales WC2026):
    factor_rival = 0.5 * elo_norm_rival + 0.5 * valor_mercado_norm_rival
    puntos_partido = resultado(0/1/3) × factor_rival
    racha_norm = suma(puntos_ponderados) / suma(3 × factor_rival)

Output: data/processed/team_ratings_wc2026.csv

Columnas:
    team_canonical, elo_norm, market_value_norm, racha_norm, rating,
    attack_value, midfield_value, defense_value,
    market_value_total, elo_raw, n_racha_matches

Fallbacks:
    - Si no existe team_ratings.csv → Elo calculado desde results.csv/unified.csv
    - Si no existe transfermarkt_player_profiles.csv → market_value = 0 para todos
      (el rating cae a 0.30*elo + 0.20*racha, renormalizado)
    - Si un equipo no tiene partidos vs rivales WC2026 → racha_norm = 0.5 (neutral)

Uso:
    python -m src.data.team_rating_wc2026
    python -m src.data.team_rating_wc2026 --results data/unified.csv
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.team_names import CANONICAL_TEAMS, canonicalize

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

WC2026_TEAMS = set(CANONICAL_TEAMS)

POSITION_TO_LINE = {
    "Goalkeeper": None,       # porteros no cuentan en ataque/medio/defensa
    "Defender":   "defense",
    "Midfielder": "midfield",
    "Forward":    "attack",
}

DEFAULT_DATA_DIR      = Path("data")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_OUTPUT        = Path("data/processed/team_ratings_wc2026.csv")


# ---------------------------------------------------------------------------
# 1. Carga y fallback de Elo
# ---------------------------------------------------------------------------

def _load_elo(processed_dir: Path, data_dir: Path) -> dict[str, float]:
    """
    Devuelve {team_canonical: elo_rating} con el Elo más reciente disponible.
    Fallback: calcula Elo desde unified.csv o results_raw.csv.
    """
    elo_path = processed_dir / "team_ratings.csv"

    if elo_path.exists():
        log.info("Cargando Elo desde %s", elo_path)
        ratings = pd.read_csv(elo_path, parse_dates=["rating_date"])
        ratings["team_name"] = ratings["team_name"].map(canonicalize)
        ratings = ratings.dropna(subset=["team_name", "elo_rating"])
        latest = (
            ratings.sort_values("rating_date")
            .groupby("team_name")["elo_rating"]
            .last()
        )
        return latest.to_dict()

    # Fallback: calcular desde results
    log.warning("team_ratings.csv no encontrado — calculando Elo desde resultados históricos")
    for candidate in ["unified.csv", "results.csv", "results_raw.csv"]:
        results_path = data_dir / candidate
        if results_path.exists():
            log.info("Usando %s para calcular Elo", results_path)
            df = pd.read_csv(results_path, parse_dates=["date"] if "date" in
                             pd.read_csv(results_path, nrows=0).columns else [])
            break
    else:
        log.error("No se encontró ningún archivo de resultados históricos")
        return {}

    try:
        from src.data.elo_loader import compute_elo_history, get_latest_ratings
        from src.data.team_names import add_canonical_columns
        df = add_canonical_columns(df, ["home_team", "away_team"], suffix="")
        elo_history = compute_elo_history(df)
        latest = get_latest_ratings(elo_history)
        return dict(zip(latest["team_name"].map(canonicalize), latest["elo_rating"]))
    except Exception as e:
        log.error("No se pudo calcular Elo: %s", e)
        return {}


# ---------------------------------------------------------------------------
# 2. Carga y fallback de valor de mercado
# ---------------------------------------------------------------------------

def _load_market_values(processed_dir: Path, squads_path: Path) -> pd.DataFrame:
    """
    Devuelve DataFrame con columnas:
        team_canonical, market_value_total, attack_value, midfield_value, defense_value

    Fuente: player_profiles.csv + player_latest_market_value.csv de Transfermarkt,
    cruzados con squads_wc2026 por nombre de jugador normalizado.
    Fallback: todo a 0 (el rating funcionará sin este componente).
    """
    raw_dir = squads_path.parent.parent  # data/raw
    tm_dir  = raw_dir / "transfermarkt"

    profiles_path = tm_dir / "player_profiles.csv"
    values_path   = tm_dir / "player_latest_market_value.csv"

    if not profiles_path.exists() or not values_path.exists():
        log.warning(
            "Archivos de Transfermarkt no encontrados en %s\n"
            "  Necesarios: player_profiles.csv, player_latest_market_value.csv\n"
            "  El rating usará solo Elo + racha (renormalizado).",
            tm_dir,
        )
        return _empty_market_values()

    if not squads_path.exists():
        log.warning("squads_wc2026_final_official_corrected.csv no encontrado")
        return _empty_market_values()

    log.info("Cargando perfiles de Transfermarkt...")
    profiles = pd.read_csv(profiles_path, low_memory=False)
    values   = pd.read_csv(values_path,   low_memory=False)

    # Quedarse con el valor más reciente por jugador
    values["date_unix"] = pd.to_numeric(values["date_unix"], errors="coerce")
    values["value"]     = pd.to_numeric(values["value"],     errors="coerce").fillna(0)
    latest_values = (
        values.sort_values("date_unix", ascending=False)
        .groupby("player_id")["value"]
        .first()
        .reset_index()
        .rename(columns={"value": "market_value_eur"})
    )

    # Cruzar perfiles con valores
    tm = profiles[["player_id", "player_name", "citizenship", "main_position"]].copy()
    tm = tm.merge(latest_values, on="player_id", how="left")
    tm["market_value_eur"] = tm["market_value_eur"].fillna(0)

    # Normalizar nombres para el cruce con convocatorias
    import unicodedata, re
    def _norm_name(s):
        if pd.isna(s):
            return ""
        s = str(s).strip()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = s.lower()
        s = re.sub(r"[^a-z\s]", "", s)
        return re.sub(r"\s+", " ", s).strip()

    tm["name_norm"] = tm["player_name"].map(_norm_name)

    # Generar también la variante con nombre invertido y añadirla como filas extra
    # Así "Min-Jae Kim" matchea con "Kim Min-Jae" del CSV de convocatorias
    def _invert_name(s):
        parts = s.split()
        return " ".join(parts[1:] + [parts[0]]) if len(parts) >= 2 else s

    tm_inv = tm.copy()
    tm_inv["name_norm"] = tm["name_norm"].map(_invert_name)

    # Unir original + invertido, quedarse con el de mayor valor si hay duplicado
    tm_both = pd.concat([tm, tm_inv], ignore_index=True)
    tm_both = (
        tm_both.sort_values("market_value_eur", ascending=False)
        .drop_duplicates("name_norm", keep="first")
    )

    # Cargar convocatorias WC2026
    squads = pd.read_csv(squads_path)[["team_canonical", "player_name", "position_broad"]]
    squads["team_canonical"] = squads["team_canonical"].map(canonicalize)
    squads["name_norm"]      = squads["player_name"].map(_norm_name)

    # Cruzar por nombre normalizado (ya incluye variante invertida)
    merged = squads.merge(
        tm_both[["name_norm", "market_value_eur", "main_position"]],
        on="name_norm",
        how="left",
    )
    merged["market_value_eur"] = merged["market_value_eur"].fillna(0)

    n_matched = (merged["market_value_eur"] > 0).sum()
    log.info("Transfermarkt: %d/%d jugadores con valor de mercado", n_matched, len(merged))

    # Agregar por selección y línea
    rows = []
    for team, group in merged.groupby("team_canonical"):
        total    = group["market_value_eur"].sum()
        attack   = group[group["position_broad"] == "Forward"]["market_value_eur"].sum()
        midfield = group[group["position_broad"] == "Midfielder"]["market_value_eur"].sum()
        defense  = group[group["position_broad"] == "Defender"]["market_value_eur"].sum()
        rows.append({
            "team_canonical":     team,
            "market_value_total": total,
            "attack_value":       attack,
            "midfield_value":     midfield,
            "defense_value":      defense,
        })

    result = pd.DataFrame(rows)
    n_with_data = (result["market_value_total"] > 0).sum()
    log.info("Valor de mercado agregado: %d/%d selecciones con datos", n_with_data, len(result))
    return result


def _empty_market_values() -> pd.DataFrame:
    """DataFrame vacío con las columnas correctas para el fallback."""
    return pd.DataFrame(columns=[
        "team_canonical", "market_value_total",
        "attack_value", "midfield_value", "defense_value",
    ])


# ---------------------------------------------------------------------------
# 3. Racha ponderada vs rivales WC2026
# ---------------------------------------------------------------------------

def _load_results(data_dir: Path) -> pd.DataFrame:
    """Carga resultados históricos para calcular la racha."""
    for candidate in ["unified.csv", "results.csv", "results_raw.csv"]:
        path = data_dir / candidate
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            date_col = "date" if "date" in df.columns else "match_date"
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            if date_col != "date":
                df = df.rename(columns={date_col: "date"})
            df["home_team"] = df["home_team"].map(canonicalize)
            df["away_team"] = df["away_team"].map(canonicalize)
            df = df.dropna(subset=["date", "home_team", "away_team",
                                   "home_score", "away_score"])
            log.info("Resultados cargados desde %s (%d partidos)", path, len(df))
            return df
    log.error("No se encontró ningún archivo de resultados históricos")
    return pd.DataFrame()


def _compute_result(home_score: float, away_score: float, is_home: bool) -> int:
    """Devuelve puntos (3/1/0) desde la perspectiva del equipo."""
    if home_score > away_score:
        return 3 if is_home else 0
    if home_score < away_score:
        return 0 if is_home else 3
    return 1


def compute_weighted_streak(
    results: pd.DataFrame,
    team: str,
    elo_norms: dict[str, float],
    market_norms: dict[str, float],
    n: int = 10,
    elo_median: float = 0.5,
) -> tuple[float, int]:
    """
    Calcula la racha ponderada de un equipo contra rivales WC2026
    con Elo por encima de la mediana del torneo.

    Devuelve (racha_norm, n_partidos_encontrados).
    Si no hay partidos, devuelve (0.5, 0) — valor neutral.
    """
    # Filtrar partidos donde jugó este equipo contra un rival del WC2026
    # con Elo por encima de la mediana (evita inflar racha con rivales débiles)
    mask = (
        (
            (results["home_team"] == team) |
            (results["away_team"] == team)
        ) &
        (
            results["home_team"].isin(WC2026_TEAMS) &
            results["away_team"].isin(WC2026_TEAMS)
        )
    )
    candidates = results[mask].copy()

    # Filtrar solo rivales con Elo >= mediana del torneo
    def _rival_elo(row):
        rival = row["away_team"] if row["home_team"] == team else row["home_team"]
        return elo_norms.get(rival, 0.0)

    candidates["rival_elo"] = candidates.apply(_rival_elo, axis=1)
    strong = candidates[candidates["rival_elo"] >= elo_median]

    # Si no hay suficientes rivales fuertes, usar todos (fallback)
    if len(strong) < 5:
        strong = candidates

    team_matches = strong.sort_values("date", ascending=False).head(n)

    if team_matches.empty:
        log.debug("Sin partidos WC2026 para %s → racha neutral 0.5", team)
        return 0.5, 0

    total_weighted_points = 0.0
    total_max_points      = 0.0

    for _, row in team_matches.iterrows():
        is_home = row["home_team"] == team
        rival   = row["away_team"] if is_home else row["home_team"]

        resultado = _compute_result(
            float(row["home_score"]),
            float(row["away_score"]),
            is_home,
        )

        # Factor del rival basado en Elo + valor de mercado (sin racha para evitar circularidad)
        elo_r    = elo_norms.get(rival, 0.5)
        market_r = market_norms.get(rival, 0.5)
        factor   = 0.5 * elo_r + 0.5 * market_r

        total_weighted_points += resultado * factor
        total_max_points      += 3.0 * factor

    if total_max_points == 0:
        return 0.5, len(team_matches)

    racha_norm = total_weighted_points / total_max_points
    return float(racha_norm), len(team_matches)


# ---------------------------------------------------------------------------
# 4. Normalización min-max
# ---------------------------------------------------------------------------

def _minmax_norm(series: pd.Series, fallback: float = 0.5) -> pd.Series:
    """Normaliza una serie a [0, 1]. Si todos los valores son iguales → fallback."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(fallback, index=series.index)
    return (series - mn) / (mx - mn)


# ---------------------------------------------------------------------------
# 5. Pipeline principal
# ---------------------------------------------------------------------------

def build_team_ratings(
    data_dir: Path = DEFAULT_DATA_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    output_path: Path = DEFAULT_OUTPUT,
    n_racha: int = 10,
) -> pd.DataFrame:
    """
    Construye el rating compuesto de las 48 selecciones WC2026.

    Pasos:
      1. Cargar Elo (con fallback a cálculo propio)
      2. Cargar valor de mercado por línea (con fallback a 0)
      3. Cargar resultados históricos y calcular racha ponderada
      4. Normalizar cada componente min-max sobre las 48 selecciones
      5. Calcular rating final y guardar CSV
    """
    log.info("=" * 60)
    log.info("Building WC2026 team ratings")
    log.info("=" * 60)

    squads_path = data_dir / "raw" / "squads_wc2026_final_official_corrected.csv"

    # ── 1. Elo ────────────────────────────────────────────────────────────────
    elo_raw = _load_elo(processed_dir, data_dir)
    log.info("Elo disponible para %d equipos", len(elo_raw))

    # ── 2. Valor de mercado ───────────────────────────────────────────────────
    market_df = _load_market_values(processed_dir, squads_path)
    has_market = not market_df.empty and (market_df["market_value_total"] > 0).any()

    # ── 3. Resultados para racha ──────────────────────────────────────────────
    results = _load_results(data_dir)

    # ── 4. Construir DataFrame base con las 48 selecciones ────────────────────
    df = pd.DataFrame({"team_canonical": sorted(WC2026_TEAMS)})

    # Elo raw
    df["elo_raw"] = df["team_canonical"].map(elo_raw)
    missing_elo = df["elo_raw"].isna().sum()
    if missing_elo > 0:
        global_mean_elo = df["elo_raw"].mean()
        df["elo_raw"] = df["elo_raw"].fillna(global_mean_elo)
        log.warning("%d equipos sin Elo → imputado con media global (%.0f)",
                    missing_elo, global_mean_elo)

    # Valor de mercado
    if has_market:
        df = df.merge(market_df, on="team_canonical", how="left")
        df["market_value_total"] = df["market_value_total"].fillna(0)
        df["attack_value"]       = df["attack_value"].fillna(0)
        df["midfield_value"]     = df["midfield_value"].fillna(0)
        df["defense_value"]      = df["defense_value"].fillna(0)
    else:
        df["market_value_total"] = 0.0
        df["attack_value"]       = 0.0
        df["midfield_value"]     = 0.0
        df["defense_value"]      = 0.0

    # ── 5. Normalizar Elo y valor de mercado ANTES de calcular racha ──────────
    # (la racha necesita estos norms para ponderar rivales)
    df["elo_norm"]    = _minmax_norm(df["elo_raw"])
    df["market_norm_pre"] = _minmax_norm(df["market_value_total"]) if has_market \
                            else pd.Series(0.5, index=df.index)

    elo_norms_dict    = dict(zip(df["team_canonical"], df["elo_norm"]))
    market_norms_dict = dict(zip(df["team_canonical"], df["market_norm_pre"]))

    # ── 6. Racha ponderada ────────────────────────────────────────────────────
    racha_vals, racha_counts = [], []
    if results.empty:
        log.warning("Sin resultados históricos → racha neutral 0.5 para todos")
        racha_vals  = [0.5] * len(df)
        racha_counts = [0] * len(df)
    else:
        for team in df["team_canonical"]:
            r, n = compute_weighted_streak(
                results, team, elo_norms_dict, market_norms_dict,
                n=n_racha, elo_median=float(pd.Series(list(elo_norms_dict.values())).median())
            )
            racha_vals.append(r)
            racha_counts.append(n)

    df["racha_raw"]         = racha_vals
    df["n_racha_matches"]   = racha_counts

    # ── 7. Normalización final de los tres componentes ────────────────────────
    df["elo_norm"]          = _minmax_norm(df["elo_raw"])
    # Log del valor de mercado para comprimir diferencias extremas entre selecciones
    if has_market:
        df["market_value_log"]  = np.log1p(df["market_value_total"])
        df["market_value_norm"] = _minmax_norm(df["market_value_log"])
    else:
        df["market_value_norm"] = pd.Series(0.5, index=df.index)
    df["racha_norm"]        = _minmax_norm(df["racha_raw"])

    # Normalizar valores por línea también
    df["attack_value_norm"]   = _minmax_norm(df["attack_value"])   if has_market else pd.Series(0.5, index=df.index)
    df["midfield_value_norm"] = _minmax_norm(df["midfield_value"]) if has_market else pd.Series(0.5, index=df.index)
    df["defense_value_norm"]  = _minmax_norm(df["defense_value"])  if has_market else pd.Series(0.5, index=df.index)

    # ── 8. Rating compuesto ───────────────────────────────────────────────────
    if has_market:
        # Pesos: 20% Elo, 60% mercado (log), 20% racha
        df["rating_raw"] = (
            0.20 * df["elo_norm"] +
            0.60 * df["market_value_norm"] +
            0.20 * df["racha_norm"]
        )
    else:
        # Sin valor de mercado: redistribuir pesos → 60% Elo, 40% racha
        log.warning("Sin valor de mercado → pesos ajustados: 60%% Elo + 40%% racha")
        df["rating_raw"] = (
            0.60 * df["elo_norm"] +
            0.40 * df["racha_norm"]
        )

    # Escalar a 0-100
    df["rating"] = (_minmax_norm(df["rating_raw"]) * 100).round(2)

    # ── 9. Seleccionar y ordenar columnas de salida ───────────────────────────
    out_cols = [
        "team_canonical",
        "rating",
        "elo_norm", "market_value_norm", "racha_norm",
        "attack_value_norm", "midfield_value_norm", "defense_value_norm",
        "elo_raw", "market_value_total",
        "attack_value", "midfield_value", "defense_value",
        "racha_raw", "n_racha_matches",
    ]
    df = df[out_cols].sort_values("rating", ascending=False).reset_index(drop=True)

    # ── 10. Guardar ───────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("Rating guardado en %s", output_path)

    # ── 11. Resumen ───────────────────────────────────────────────────────────
    _print_summary(df, has_market)

    return df


def _print_summary(df: pd.DataFrame, has_market: bool) -> None:
    print("\n" + "=" * 55)
    print("TOP 20 — RATING WC2026")
    print(f"{'Rank':<5} {'Selección':<25} {'Rating':>7} {'Elo':>7} {'Mercado':>9} {'Racha':>7} {'N':>4}")
    print("-" * 55)
    for i, row in df.head(20).iterrows():
        mercado_str = f"{row['market_value_norm']:.2f}" if has_market else "  N/A"
        print(
            f"  {i+1:<3} {row['team_canonical']:<25} "
            f"{row['rating']:>7.1f} "
            f"{row['elo_norm']:>7.2f} "
            f"{mercado_str:>9} "
            f"{row['racha_norm']:>7.2f} "
            f"{int(row['n_racha_matches']):>4}"
        )
    print("=" * 55)
    print(f"\nMedia del torneo: {df['rating'].mean():.1f}")
    print(f"Std:              {df['rating'].std():.1f}")
    print(f"Rango:            {df['rating'].min():.1f} – {df['rating'].max():.1f}")
    if not has_market:
        print("\n⚠ Valor de mercado no disponible. Para incluirlo:")
        print("  1. Descarga transfermarkt-datasets de Kaggle")
        print("  2. python -m src.data.normalize_squads")
        print("  3. python -m src.data.match_transfermarkt --transfermarkt-input data/raw/transfermarkt/players.csv.gz")
        print("  4. Vuelve a ejecutar este script")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calcula el rating compuesto de las 48 selecciones WC2026."
    )
    parser.add_argument("--data-dir",       default="data",                              help="Directorio raíz de datos.")
    parser.add_argument("--processed-dir",  default="data/processed",                   help="Directorio de datos procesados.")
    parser.add_argument("--output",         default="data/processed/team_ratings_wc2026.csv", help="Ruta del CSV de salida.")
    parser.add_argument("--n-racha",        type=int, default=10,                        help="Últimos N partidos para la racha.")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args()
    build_team_ratings(
        data_dir      = Path(args.data_dir),
        processed_dir = Path(args.processed_dir),
        output_path   = Path(args.output),
        n_racha       = args.n_racha,
    )


if __name__ == "__main__":
    main()
           