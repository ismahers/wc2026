"""
src/data/player_stats_aggregator.py
=====================================
Agrega estadísticas de jugadores de Transfermarkt por selección WC2026.

Fuentes:
  - data/transfermarkt/player_performances.csv  → stats por temporada/club
  - data/transfermarkt/player_national_performances.csv → stats en selección
  - data/transfermarkt/player_profiles.csv → player_id + nombre
  - data/raw/squads_wc2026_final_official_corrected.csv → convocados

Output: data/processed/team_player_stats_wc2026.csv

Columnas de salida por selección:
  team_canonical,
  # Disciplina (tarjetas)
  avg_yellow_per_90, avg_red_per_90,
  # Ataque
  avg_goals_per_90, avg_assists_per_90,
  # Experiencia internacional
  avg_international_caps,
  # Cobertura
  n_players_matched, n_players_total

Uso:
  python -m src.data.player_stats_aggregator
  python -m src.data.player_stats_aggregator --seasons 24/25 25/26
  python -m src.data.player_stats_aggregator --performances data/transfermarkt/player_performances.csv
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.team_names import canonicalize

log = logging.getLogger(__name__)

DEFAULT_DATA_DIR      = Path("data")
DEFAULT_OUTPUT        = Path("data/processed/team_player_stats_wc2026.csv")
DEFAULT_SEASONS       = ["23/24", "24/25", "25/26"]
MIN_MINUTES           = 90  # mínimo de minutos para incluir al jugador en el cálculo


# ---------------------------------------------------------------------------
# Normalización de nombres (igual que en team_rating_wc2026.py)
# ---------------------------------------------------------------------------

def _norm_name(s) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _invert_name(s: str) -> str:
    parts = s.split()
    return " ".join(parts[1:] + [parts[0]]) if len(parts) >= 2 else s


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def _load_squads(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "raw" / "squads_wc2026_final_official_corrected.csv"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}")
    df = pd.read_csv(path)
    df["team_canonical"] = df["team_canonical"].map(canonicalize)
    df["name_norm"]      = df["player_name"].map(_norm_name)
    df["name_norm_inv"]  = df["name_norm"].map(_invert_name)
    log.info("Convocados WC2026: %d jugadores de %d selecciones",
             len(df), df["team_canonical"].nunique())
    return df


def _load_profiles(tm_dir: Path) -> pd.DataFrame:
    """Carga perfiles de Transfermarkt para obtener player_id por nombre."""
    path = tm_dir / "player_profiles.csv"
    if not path.exists():
        log.warning("player_profiles.csv no encontrado en %s", tm_dir)
        return pd.DataFrame()
    profiles = pd.read_csv(path, low_memory=False,
                           usecols=["player_id", "player_name"])
    profiles["name_norm"]     = profiles["player_name"].map(_norm_name)
    profiles["name_norm_inv"] = profiles["name_norm"].map(_invert_name)
    log.info("Perfiles Transfermarkt: %d jugadores", len(profiles))
    return profiles


def _match_player_ids(squads: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Cruza convocados con perfiles para obtener player_id.
    Intenta nombre directo primero, luego nombre invertido (coreanos, etc).
    """
    if profiles.empty:
        squads["player_id"] = np.nan
        return squads

    # Generar variante invertida de perfiles y concatenar
    # Así "Min-Jae Kim" en Transfermarkt matchea con "Kim Min-Jae" en convocatorias
    profiles_inv = profiles.copy()
    profiles_inv["name_norm"] = profiles_inv["name_norm_inv"]
    profiles_both = pd.concat([profiles, profiles_inv], ignore_index=True)
    profiles_both = (
        profiles_both.sort_values("player_id")
        .drop_duplicates("name_norm", keep="first")
    )

    merged = squads.merge(
        profiles_both[["name_norm", "player_id"]],
        on="name_norm",
        how="left",
    )

    n_matched = merged["player_id"].notna().sum()
    log.info("Matching jugadores: %d/%d con player_id", n_matched, len(merged))
    return merged


def _load_performances(
    tm_dir: Path,
    seasons: list[str],
) -> pd.DataFrame:
    """Carga stats de club filtradas por temporadas recientes."""
    path = tm_dir / "player_performances.csv"
    if not path.exists():
        log.warning("player_performances.csv no encontrado en %s", tm_dir)
        return pd.DataFrame()

    log.info("Cargando player_performances.csv (puede tardar unos segundos)...")
    df = pd.read_csv(path, low_memory=False, usecols=[
        "player_id", "season_name",
        "yellow_cards", "second_yellow_cards", "direct_red_cards",
        "goals", "assists", "minutes_played",
    ])

    df = df[df["season_name"].isin(seasons)].copy()
    df["minutes_played"]   = pd.to_numeric(df["minutes_played"],   errors="coerce")
    df["yellow_cards"]     = pd.to_numeric(df["yellow_cards"],     errors="coerce").fillna(0)
    df["second_yellow_cards"] = pd.to_numeric(df["second_yellow_cards"], errors="coerce").fillna(0)
    df["direct_red_cards"] = pd.to_numeric(df["direct_red_cards"], errors="coerce").fillna(0)
    df["goals"]            = pd.to_numeric(df["goals"],            errors="coerce").fillna(0)
    df["assists"]          = pd.to_numeric(df["assists"],          errors="coerce").fillna(0)
    df["total_red_cards"]  = df["second_yellow_cards"] + df["direct_red_cards"]

    log.info("Performances filtradas (%s): %d filas", seasons, len(df))
    return df


def _load_national_performances(tm_dir: Path) -> pd.DataFrame:
    """Carga stats en selección (partidos internacionales totales)."""
    path = tm_dir / "player_national_performances.csv"
    if not path.exists():
        log.warning("player_national_performances.csv no encontrado en %s", tm_dir)
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False, usecols=["player_id", "matches", "goals"])
    df["matches"] = pd.to_numeric(df["matches"], errors="coerce").fillna(0)

    # Sumar todos los partidos internacionales del jugador (puede tener varias selecciones)
    caps = df.groupby("player_id")["matches"].sum().reset_index()
    caps.columns = ["player_id", "international_caps"]
    log.info("Caps internacionales: %d jugadores", len(caps))
    return caps


# ---------------------------------------------------------------------------
# Agregación por selección
# ---------------------------------------------------------------------------

def _per90(stat: pd.Series, minutes: pd.Series) -> float:
    """Calcula stat por 90 minutos ponderado por minutos jugados."""
    valid = minutes > 0
    if not valid.any():
        return np.nan
    total_stat    = stat[valid].sum()
    total_minutes = minutes[valid].sum()
    if total_minutes < MIN_MINUTES:
        return np.nan
    return float(total_stat / total_minutes * 90)


def aggregate_by_team(
    squads_with_ids: pd.DataFrame,
    performances: pd.DataFrame,
    national_caps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrega stats de jugadores por selección.
    Solo incluye jugadores con player_id matcheado.
    """
    rows = []

    for team, group in squads_with_ids.groupby("team_canonical"):
        n_total   = len(group)
        matched   = group[group["player_id"].notna()].copy()
        n_matched = len(matched)

        if matched.empty or performances.empty:
            rows.append({
                "team_canonical":       team,
                "avg_yellow_per_90":    np.nan,
                "avg_red_per_90":       np.nan,
                "avg_goals_per_90":     np.nan,
                "avg_assists_per_90":   np.nan,
                "avg_international_caps": np.nan,
                "n_players_matched":    n_matched,
                "n_players_total":      n_total,
            })
            continue

        player_ids = matched["player_id"].astype(int).tolist()

        # Stats de club (últimas temporadas)
        perf = performances[performances["player_id"].isin(player_ids)].copy()

        # Agregar por jugador primero (sumar todas las temporadas/competiciones)
        player_stats = perf.groupby("player_id").agg(
            yellow_cards    = ("yellow_cards",     "sum"),
            red_cards       = ("total_red_cards",  "sum"),
            goals           = ("goals",            "sum"),
            assists         = ("assists",          "sum"),
            minutes_played  = ("minutes_played",   "sum"),
        ).reset_index()

        # Filtrar jugadores con mínimo de minutos
        player_stats = player_stats[player_stats["minutes_played"] >= MIN_MINUTES]

        if player_stats.empty:
            avg_yellow  = np.nan
            avg_red     = np.nan
            avg_goals   = np.nan
            avg_assists = np.nan
        else:
            # Per-90 por jugador, luego media del equipo
            player_stats["yellow_per_90"]  = player_stats["yellow_cards"]  / player_stats["minutes_played"] * 90
            player_stats["red_per_90"]     = player_stats["red_cards"]     / player_stats["minutes_played"] * 90
            player_stats["goals_per_90"]   = player_stats["goals"]         / player_stats["minutes_played"] * 90
            player_stats["assists_per_90"] = player_stats["assists"]       / player_stats["minutes_played"] * 90

            avg_yellow  = float(player_stats["yellow_per_90"].mean())
            avg_red     = float(player_stats["red_per_90"].mean())
            avg_goals   = float(player_stats["goals_per_90"].mean())
            avg_assists = float(player_stats["assists_per_90"].mean())

        # Caps internacionales
        if not national_caps.empty:
            team_caps = national_caps[national_caps["player_id"].isin(player_ids)]
            avg_caps  = float(team_caps["international_caps"].mean()) if not team_caps.empty else np.nan
        else:
            avg_caps = np.nan

        rows.append({
            "team_canonical":         team,
            "avg_yellow_per_90":      round(avg_yellow,  4) if not np.isnan(avg_yellow)  else np.nan,
            "avg_red_per_90":         round(avg_red,     4) if not np.isnan(avg_red)     else np.nan,
            "avg_goals_per_90":       round(avg_goals,   4) if not np.isnan(avg_goals)   else np.nan,
            "avg_assists_per_90":     round(avg_assists, 4) if not np.isnan(avg_assists) else np.nan,
            "avg_international_caps": round(avg_caps,    1) if not np.isnan(avg_caps)    else np.nan,
            "n_players_matched":      n_matched,
            "n_players_total":        n_total,
        })

    result = pd.DataFrame(rows).sort_values("team_canonical").reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def build_player_stats(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_path: Path = DEFAULT_OUTPUT,
    seasons: list[str] = DEFAULT_SEASONS,
) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("Building WC2026 player stats by team")
    log.info("Temporadas: %s", seasons)
    log.info("=" * 60)

    tm_dir = data_dir / "transfermarkt"

    # Cargar fuentes
    squads       = _load_squads(data_dir)
    profiles     = _load_profiles(tm_dir)
    performances = _load_performances(tm_dir, seasons)
    national     = _load_national_performances(tm_dir)

    # Matching player_ids
    squads_with_ids = _match_player_ids(squads, profiles)

    # Agregar por selección
    result = aggregate_by_team(squads_with_ids, performances, national)

    # Guardar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    log.info("Guardado en %s", output_path)

    # Resumen
    _print_summary(result)
    return result


def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("STATS DE JUGADORES POR SELECCIÓN WC2026")
    print(f"{'Selección':<25} {'Amarillas/90':>12} {'Rojas/90':>10} {'Goles/90':>10} {'Caps':>8} {'Match':>6}")
    print("-" * 70)
    for _, row in df.sort_values("avg_yellow_per_90", ascending=False).iterrows():
        y  = f"{row['avg_yellow_per_90']:.3f}" if pd.notna(row['avg_yellow_per_90']) else "  N/A"
        r  = f"{row['avg_red_per_90']:.3f}"    if pd.notna(row['avg_red_per_90'])    else "  N/A"
        g  = f"{row['avg_goals_per_90']:.3f}"  if pd.notna(row['avg_goals_per_90'])  else "  N/A"
        c  = f"{row['avg_international_caps']:.0f}" if pd.notna(row['avg_international_caps']) else "N/A"
        n  = f"{int(row['n_players_matched'])}/{int(row['n_players_total'])}"
        print(f"  {row['team_canonical']:<23} {y:>12} {r:>10} {g:>10} {c:>8} {n:>6}")
    print("=" * 70)
    covered = df["avg_yellow_per_90"].notna().sum()
    print(f"\nSelecciones con datos: {covered}/{len(df)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agrega stats de jugadores Transfermarkt por selección WC2026."
    )
    parser.add_argument("--data-dir",      default="data")
    parser.add_argument("--output",        default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seasons",       nargs="+", default=DEFAULT_SEASONS,
                        help="Temporadas a incluir, ej: 23/24 24/25 25/26")
    parser.add_argument("--performances",  default=None,
                        help="Ruta alternativa a player_performances.csv")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args()
    build_player_stats(
        data_dir    = Path(args.data_dir),
        output_path = Path(args.output),
        seasons     = args.seasons,
    )


if __name__ == "__main__":
    main()
     