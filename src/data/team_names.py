"""
src/data/team_names.py
======================
Normalización canónica de nombres de selección.

Problema que resuelve
---------------------
Los nombres de equipo llegan de fuentes distintas con grafías distintas:

  - Fixtures WC2026 (group_stage / knockout): "United States", "Curacao", "DR Congo"
  - Kaggle martj42 (unified.csv):             variantes con acentos o nombres largos
  - The Odds API:                             "USA", "Bosnia & Herzegovina", ...

Si no se canonizan, los joins por nombre fallan EN SILENCIO y dejan NaN en:
  - forma reciente            (feature_engineering._get_team_history)
  - elo_diff                  (builder._as_of_rating / elo_loader)
  - same_confederation        (feature_engineering.CONFEDERATION_MAP)
  - cruce modelo vs cuotas    (evaluation.betting, join por nombre)

La grafía CANÓNICA es la del fixture WC2026 (la de group_stage_wc2026.csv y
base_camps_wc2026.csv), porque es la que ya comparten la mayoría de tablas
estáticas del proyecto.

Uso
---
    from src.data.team_names import canonicalize, add_canonical_columns

    canonicalize("USA")                  # -> "United States"
    canonicalize("Bosnia & Herzegovina") # -> "Bosnia and Herzegovina"
    canonicalize("Equipo Marciano")      # -> "Equipo Marciano"  (no toca lo desconocido)
    canonicalize("Equipo Marciano", strict=True)  # -> None

    df = add_canonical_columns(df, ["home_team", "away_team"])
    # añade home_team_canon, away_team_canon

    # Para joins sin depender de quién es local/visitante (sede neutral):
    pair_key("USA", "Canada") == pair_key("Canada", "USA")  # True
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Nombres canónicos: las 48 selecciones del fixture WC2026
# ---------------------------------------------------------------------------

CANONICAL_TEAMS: tuple[str, ...] = (
    "Algeria", "Argentina", "Australia", "Austria", "Belgium",
    "Bosnia and Herzegovina", "Brazil", "Canada", "Cape Verde", "Colombia",
    "Croatia", "Curacao", "Czech Republic", "DR Congo", "Ecuador",
    "Egypt", "England", "France", "Germany", "Ghana",
    "Haiti", "Iran", "Iraq", "Ivory Coast", "Japan",
    "Jordan", "Mexico", "Morocco", "Netherlands", "New Zealand",
    "Norway", "Panama", "Paraguay", "Portugal", "Qatar",
    "Saudi Arabia", "Scotland", "Senegal", "South Africa", "South Korea",
    "Spain", "Sweden", "Switzerland", "Tunisia", "Turkey",
    "United States", "Uruguay", "Uzbekistan",
)


# ---------------------------------------------------------------------------
# Alias conocidos hacia el nombre canónico.
#
# La clave es la grafía CRUDA tal cual la escupe alguna fuente; el valor es
# el canónico. La normalización (acentos, mayúsculas, &) se aplica luego, así
# que aquí solo hace falta cubrir variantes que NO se resuelven solo quitando
# acentos (p.ej. "Curaçao" -> "Curacao" ya sale gratis y no necesita alias).
#
# Cuando el script de cobertura te liste nombres "sin resolver", añádelos aquí.
# ---------------------------------------------------------------------------

_ALIASES_RAW: dict[str, str] = {
    # Nombres en español (TyC Sports / Wikipedia)
    "Alemania": "Germany",
    "Argelia": "Algeria",
    "Arabia Saudita": "Saudi Arabia",
    "Arabia Saudí": "Saudi Arabia",
    "Australia": "Australia",
    "Austria": "Austria",
    "Belgica": "Belgium",
    "Bélgica": "Belgium",
    "Bosnia y Herzegovina": "Bosnia and Herzegovina",
    "Brasil": "Brazil",
    "Canada": "Canada",
    "Canadá": "Canada",
    "Cabo Verde": "Cape Verde",
    "Colombia": "Colombia",
    "Corea del Sur": "South Korea",
    "Costa de Marfil": "Ivory Coast",
    "Croacia": "Croatia",
    "Curazao": "Curacao",
    "Ecuador": "Ecuador",
    "Egipto": "Egypt",
    "Escocia": "Scotland",
    "España": "Spain",
    "Estados Unidos": "United States",
    "Francia": "France",
    "Ghana": "Ghana",
    "Haiti": "Haiti",
    "Haití": "Haiti",
    "Inglaterra": "England",
    "Iran": "Iran",
    "Irán": "Iran",
    "Irak": "Iraq",
    "Japón": "Japan",
    "Japon": "Japan",
    "Jordania": "Jordan",
    "Marruecos": "Morocco",
    "Mexico": "Mexico",
    "México": "Mexico",
    "Noruega": "Norway",
    "Nueva Zelanda": "New Zealand",
    "Paises Bajos": "Netherlands",
    "Países Bajos": "Netherlands",
    "Panama": "Panama",
    "Panamá": "Panama",
    "Paraguay": "Paraguay",
    "Portugal": "Portugal",
    "Qatar": "Qatar",
    "Catar": "Qatar",
    "RD Congo": "DR Congo",
    "R. D. Congo": "DR Congo",
    "Republica Democratica del Congo": "DR Congo",
    "República Democrática del Congo": "DR Congo",
    "Republica Checa": "Czech Republic",
    "República Checa": "Czech Republic",
    "Senegal": "Senegal",
    "Sudafrica": "South Africa",
    "Sudáfrica": "South Africa",
    "Suecia": "Sweden",
    "Suiza": "Switzerland",
    "Tunez": "Tunisia",
    "Túnez": "Tunisia",
    "Turquia": "Turkey",
    "Turquía": "Turkey",
    "Uruguay": "Uruguay",
    "Uzbekistan": "Uzbekistan",
    "Uzbekistán": "Uzbekistan",
    # Estados Unidos
    "USA": "United States",
    "US": "United States",
    "United States of America": "United States",
    "USMNT": "United States",
    # Corea del Sur
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Korea, Republic of": "South Korea",
    "South Korea": "South Korea",
    # Bosnia
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia": "Bosnia and Herzegovina",
    "Bosnia and Herzegovina IF": "Bosnia and Herzegovina",
    # Costa de Marfil
    "Cote d'Ivoire": "Ivory Coast",
    "Cote dIvoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Ivory Coast (Cote d'Ivoire)": "Ivory Coast",
    # RD del Congo
    "Congo DR": "DR Congo",
    "DR Congo (Congo)": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "Congo-Kinshasa": "DR Congo",
    "R.D. Congo": "DR Congo",
    "R. D. Congo": "DR Congo",
    "RD Congo": "DR Congo",
    # Cabo Verde
    "Cabo Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
    # Turquía
    "Turkiye": "Turkey",
    "Türkiye": "Turkey",
    # Chequia
    "Czechia": "Czech Republic",
    # Curaçao (también sale por acentos, pero lo dejamos explícito)
    "Curaçao": "Curacao",
    # Irán
    "IR Iran": "Iran",
    "Iran (Islamic Republic of)": "Iran",
    # Otros que a veces aparecen con sufijos en datasets
    "United States USA": "United States",
}


# ---------------------------------------------------------------------------
# Normalización y lookup
# ---------------------------------------------------------------------------

def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _norm(name: object) -> str:
    """
    Clave de comparación robusta: sin acentos, minúsculas, '&' -> 'and',
    sin puntuación, espacios colapsados.
    """
    if name is None:
        return ""
    text = str(name).strip()
    if not text:
        return ""
    text = _strip_accents(text)
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[._,]", " ", text)
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Lookup precomputado: clave normalizada -> nombre canónico
_LOOKUP: dict[str, str] = {}
for _canon in CANONICAL_TEAMS:
    _LOOKUP[_norm(_canon)] = _canon
for _raw, _canon in _ALIASES_RAW.items():
    if _canon not in CANONICAL_TEAMS:
        raise ValueError(
            f"Alias '{_raw}' apunta a '{_canon}', que no es un nombre canónico. "
            "Revisa CANONICAL_TEAMS."
        )
    _LOOKUP[_norm(_raw)] = _canon


def canonicalize(name: object, *, strict: bool = False) -> Optional[str]:
    """
    Devuelve el nombre canónico de una selección.

    Parámetros
    ----------
    name : nombre crudo en cualquier grafía/fuente.
    strict : si True devuelve None cuando no reconoce el nombre.
             Si False (default) devuelve el nombre original sin tocar,
             para no romper equipos históricos fuera del WC2026.
    """
    key = _norm(name)
    if key in _LOOKUP:
        return _LOOKUP[key]
    if strict:
        return None
    return None if name is None else str(name).strip()


def is_known(name: object) -> bool:
    """True si el nombre resuelve a un canónico del WC2026."""
    return _norm(name) in _LOOKUP


def unknown_names(names: Iterable[object]) -> list[str]:
    """
    Devuelve, ordenados y sin duplicar, los nombres crudos que NO resuelven
    a ningún canónico. Estos son los alias que faltan por añadir.
    """
    seen: dict[str, str] = {}
    for n in names:
        if n is None:
            continue
        raw = str(n).strip()
        if not raw:
            continue
        if not is_known(raw):
            seen.setdefault(_norm(raw), raw)
    return sorted(seen.values(), key=lambda s: s.lower())


def pair_key(home: object, away: object) -> tuple[str, str]:
    """
    Clave de emparejamiento independiente de la orientación local/visitante.
    Útil para unir predicciones con cuotas en sede neutral, donde The Odds API
    puede invertir home/away respecto a nuestro fixture.

    pair_key("USA", "Canada") == pair_key("Canada", "USA")
    """
    a = canonicalize(home) or ""
    b = canonicalize(away) or ""
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def add_canonical_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    suffix: str = "_canon",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Añade una columna canónica por cada columna de equipo indicada.

    Por defecto NO sobrescribe: crea <col><suffix>. Si suffix == "" reemplaza
    la columna original (útil para canonizar in situ antes de un merge).
    """
    out = df if inplace else df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        target = f"{col}{suffix}" if suffix else col
        out[target] = out[col].map(lambda v: canonicalize(v))
    return out


def add_pair_key_column(
    df: pd.DataFrame,
    home_col: str = "home_team",
    away_col: str = "away_team",
    *,
    out_col: str = "pair_key",
) -> pd.DataFrame:
    """Añade una columna con la clave de par no ordenada (string 'A|B')."""
    out = df.copy()
    if home_col not in out.columns or away_col not in out.columns:
        return out
    out[out_col] = [
        "|".join(pair_key(h, a))
        for h, a in zip(out[home_col], out[away_col])
    ]
    return out


__all__ = [
    "CANONICAL_TEAMS",
    "canonicalize",
    "is_known",
    "unknown_names",
    "pair_key",
    "add_canonical_columns",
    "add_pair_key_column",
]
