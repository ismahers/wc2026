"""
TyC Sports squad scraper for World Cup 2026 player lists.

The raw article is useful for early player-prop work before final official
lineups exist. This module writes both the raw TyC extraction and a normalized
`players` table compatible with `src/data/schemas.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.data.schemas import table_columns
from src.data.team_names import canonicalize


logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DEFAULT_TYC_URL = (
    "https://www.tycsports.com/mundial/"
    "una-por-una-las-listas-confirmadas-para-el-mundial-2026-de-cada-seleccion-y-las-que-faltan-id731187.html"
)


class TycSportsScraper:
    """
    Parse TyC Sports article text into squad rows.

    TyC writes positions in Spanish and often includes clubs inside
    parentheses. We split player lists only on commas outside parentheses so
    club-country hints such as `(Bologna, ITA)` stay attached to the player.
    """

    def __init__(self, url: str = DEFAULT_TYC_URL):
        self.url = url
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
        self.pos_map = {
            "arquero": "Goalkeeper",
            "portero": "Goalkeeper",
            "defensor": "Defender",
            "mediocampista": "Midfielder",
            "volante": "Midfielder",
            "delantero": "Forward",
            "atacante": "Forward",
        }

    def collect(self) -> pd.DataFrame | None:
        log.info("Descargando artículo desde: %s ...", self.url)
        try:
            response = requests.get(self.url, headers=self.headers, timeout=30)
            response.encoding = "utf-8"
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
        except Exception as exc:
            log.error("Fallo de conexión: %s", exc)
            return None

        players_data: list[dict[str, str]] = []
        current_team = "Desconocido"

        for block in soup.find_all(["h2", "h3", "p", "li"]):
            text = block.get_text(" ", strip=True)
            if not text:
                continue

            if block.name in {"h2", "h3"} and len(text.split()) <= 4:
                current_team = text.replace(":", "").strip()
                continue

            matched_pos = self._match_position(text)
            if not matched_pos:
                continue

            clean_text = re.sub(r"^(.*?)[-:]", "", text).strip().rstrip(".")
            for token in split_player_list(clean_text):
                player_name, club = split_player_and_club(token)
                if not player_name:
                    continue
                players_data.append({
                    "team": current_team,
                    "team_canonical": canonicalize(current_team),
                    "player_name": player_name,
                    "club": club,
                    "position_broad": matched_pos,
                    "source": "tyc_sports",
                })

        df = pd.DataFrame(players_data)
        if df.empty:
            log.warning("No se encontraron jugadores. Es posible que el formato de la noticia sea distinto.")
            return None

        df = df.drop_duplicates(["team_canonical", "player_name", "position_broad"]).reset_index(drop=True)
        log.info("Éxito: %d jugadores extraídos de TyC Sports.", len(df))
        log.info("Selecciones encontradas: %s", ", ".join(sorted(df["team_canonical"].dropna().unique())))
        return df

    def run(self, output_path: str, players_output_path: str | None = None) -> pd.DataFrame | None:
        df = self.collect()
        if df is None:
            return None

        raw_path = Path(output_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False)
        log.info("Raw squads guardado en %s", raw_path)

        if players_output_path:
            players = normalize_tyc_squads_to_players(df)
            players_path = Path(players_output_path)
            players_path.parent.mkdir(parents=True, exist_ok=True)
            players.to_csv(players_path, index=False)
            log.info("Players normalizado guardado en %s", players_path)

        return df

    def _match_position(self, text: str) -> str | None:
        lower_text = text.lower()
        for es_pos, en_pos in self.pos_map.items():
            plural = f"{es_pos}s"
            if (lower_text.startswith(es_pos) or lower_text.startswith(plural)) and (
                ":" in text[:25] or "-" in text[:25]
            ):
                return en_pos
        return None


def split_player_list(text: str) -> list[str]:
    """Split on commas/semicolons/last conjunction outside parentheses."""
    tokens: list[str] = []
    current: list[str] = []
    depth = 0
    idx = 0

    while idx < len(text):
        char = text[idx]
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1

        is_conjunction = (
            depth == 0
            and char == " "
            and text[idx:idx + 3].lower() in {" y ", " e "}
            and current
        )
        if (char in {",", ";"} and depth == 0) or is_conjunction:
            token = "".join(current).strip()
            if token:
                tokens.append(token)
            current = []
            idx += 3 if is_conjunction else 1
            continue
        else:
            current.append(char)
        idx += 1

    token = "".join(current).strip()
    if token:
        tokens.append(token)
    return [token for token in tokens if token]


def split_player_and_club(token: str) -> tuple[str, str | None]:
    """Extract `Name (Club)` into name and club when the article provides it."""
    cleaned = re.sub(r"\s+", " ", token).strip()
    match = re.match(r"^(?P<name>.+?)\s*\((?P<club>[^()]+)\)$", cleaned)
    if not match:
        return cleaned, None
    return match.group("name").strip(), match.group("club").strip()


def make_player_id(team_name: str, player_name: str, club: object = "") -> str:
    club_part = "" if pd.isna(club) else str(club).strip()
    key = f"{canonicalize(team_name)}|{player_name}|{club_part}".lower().encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()[:12]
    return f"tyc_{digest}"


def make_team_id(team_name: str) -> str:
    canonical = canonicalize(team_name)
    slug = re.sub(r"[^a-z0-9]+", "_", str(canonical).lower()).strip("_")
    return f"team_{slug}"


def normalize_tyc_squads_to_players(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TyC squad rows to the normalized `players` table contract."""
    rows = []
    for _, row in df.iterrows():
        team_name = row.get("team_canonical") or canonicalize(row.get("team"))
        player_name = str(row.get("player_name", "")).strip()
        if not team_name or not player_name:
            continue
        rows.append({
            "player_id": make_player_id(team_name, player_name, row.get("club")),
            "player_name": player_name,
            "team_id": make_team_id(team_name),
            "club": row.get("club"),
            "position": row.get("position_broad"),
            "birth_date": pd.NA,
            "preferred_foot": pd.NA,
            "height_cm": pd.NA,
            "market_value_eur": pd.NA,
        })

    players = pd.DataFrame(rows)
    if players.empty:
        return pd.DataFrame(columns=table_columns("players"))
    players = players.drop_duplicates("player_id").sort_values(["team_id", "position", "player_name"])
    return players.reindex(columns=table_columns("players")).reset_index(drop=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape WC2026 squads from TyC Sports.")
    parser.add_argument("--url", default=DEFAULT_TYC_URL)
    parser.add_argument("--raw-output", default="data/raw/squads_wc2026.csv")
    parser.add_argument("--players-output", default="data/processed/players_wc2026.csv")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    scraper = TycSportsScraper(args.url)
    scraper.run(args.raw_output, args.players_output)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2])
    main()
