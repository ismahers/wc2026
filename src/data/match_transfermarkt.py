"""
Match WC2026 player identities to a local Transfermarkt players dataset.

The script expects a CSV exported from a Transfermarkt dataset, for example
data/raw/transfermarkt/players.csv. It does not scrape the web. The goal is to
produce reviewable source links and profile fields without mutating the official
squad CSV.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.player_master import (
    DEFAULT_MASTER_OUTPUT,
    DEFAULT_SOURCE_MATCHES_OUTPUT,
    build_player_source_matches,
    normalize_identity_text,
)


log = logging.getLogger(__name__)

DEFAULT_TRANSFERMARKT_INPUT = "data/raw/transfermarkt/players.csv.gz"
DEFAULT_SOURCE_MATCHES_TRANSFERMARKT_OUTPUT = "data/processed/player_source_matches.csv"
DEFAULT_PROFILES_OUTPUT = "data/processed/transfermarkt_player_profiles.csv"
DEFAULT_REVIEW_OUTPUT = "data/processed/transfermarkt_match_review.csv"

AUTO_MATCH_THRESHOLD = 0.92
REVIEW_THRESHOLD = 0.84

TRANSFERMARKT_PROFILE_COLUMNS = [
    "player_key",
    "transfermarkt_player_id",
    "transfermarkt_name",
    "transfermarkt_url",
    "date_of_birth",
    "age",
    "height_cm",
    "foot",
    "position_detail",
    "main_position",
    "other_positions",
    "market_value_eur",
    "market_value_date",
    "contract_until",
    "agent",
    "current_club_transfermarkt",
    "source",
]

REVIEW_COLUMNS = [
    "player_key",
    "team_canonical",
    "player_name",
    "club_from_squad",
    "position_broad",
    "transfermarkt_name",
    "current_club_transfermarkt",
    "transfermarkt_url",
    "match_confidence",
    "match_method",
    "review_reason",
]

POSITION_TO_BROAD = {
    "attack": "Forward",
    "goalkeeper": "Goalkeeper",
    "keeper": "Goalkeeper",
    "defence": "Defender",
    "defense": "Defender",
    "defender": "Defender",
    "centre back": "Defender",
    "center back": "Defender",
    "left back": "Defender",
    "right back": "Defender",
    "midfielder": "Midfielder",
    "midfield": "Midfielder",
    "defensive midfield": "Midfielder",
    "central midfield": "Midfielder",
    "attacking midfield": "Midfielder",
    "forward": "Forward",
    "striker": "Forward",
    "centre forward": "Forward",
    "center forward": "Forward",
    "left winger": "Forward",
    "right winger": "Forward",
    "winger": "Forward",
}

COUNTRY_ALIASES = {
    "bosnia herzegovina": "bosnia and herzegovina",
    "bosnia and herzegovina": "bosnia and herzegovina",
    "cape verde": "cape verde",
    "cote d ivoire": "ivory coast",
    "curacao": "curacao",
    "czechia": "czech republic",
    "czech republic": "czech republic",
    "dr congo": "dr congo",
    "korea south": "south korea",
    "south korea": "south korea",
    "turkiye": "turkey",
    "turkey": "turkey",
    "united states": "united states",
    "usa": "united states",
}


@dataclass(frozen=True)
class ColumnMap:
    player_id: str | None
    name: str
    url: str | None
    club: str | None
    nationality: str | None
    date_of_birth: str | None
    age: str | None
    height_cm: str | None
    foot: str | None
    position_detail: str | None
    main_position: str | None
    other_positions: str | None
    market_value_eur: str | None
    market_value_date: str | None
    contract_until: str | None
    agent: str | None


def _first_existing(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    available = {col.casefold(): col for col in columns}
    for alias in aliases:
        found = available.get(alias.casefold())
        if found is not None:
            return found
    return None


def infer_transfermarkt_columns(df: pd.DataFrame) -> ColumnMap:
    columns = df.columns
    name = _first_existing(columns, [
        "player_name", "name", "pretty_name", "player", "transfermarkt_name",
    ])
    if name is None:
        raise ValueError(
            "Transfermarkt CSV must include a player name column. "
            "Accepted aliases: player_name, name, pretty_name, player, transfermarkt_name"
        )

    return ColumnMap(
        player_id=_first_existing(columns, ["player_id", "transfermarkt_player_id", "transfermarkt_id"]),
        name=name,
        url=_first_existing(columns, ["transfermarkt_url", "player_url", "url", "profile_url"]),
        club=_first_existing(columns, [
            "current_club_name", "club_name", "current_club", "club", "team_name",
        ]),
        nationality=_first_existing(columns, [
            "country_of_citizenship", "nationality", "country", "citizenship",
        ]),
        date_of_birth=_first_existing(columns, ["date_of_birth", "birth_date", "dob"]),
        age=_first_existing(columns, ["age"]),
        height_cm=_first_existing(columns, ["height_cm", "height_in_cm", "height"]),
        foot=_first_existing(columns, ["foot", "preferred_foot"]),
        position_detail=_first_existing(columns, ["position_detail", "sub_position", "position"]),
        main_position=_first_existing(columns, ["main_position", "position_group"]),
        other_positions=_first_existing(columns, ["other_positions", "positions"]),
        market_value_eur=_first_existing(columns, [
            "market_value_eur", "market_value_in_eur", "market_value",
        ]),
        market_value_date=_first_existing(columns, ["market_value_date", "last_season"]),
        contract_until=_first_existing(columns, ["contract_until", "contract_expiration_date"]),
        agent=_first_existing(columns, ["agent", "agent_name"]),
    )


def _value(row: pd.Series, col: str | None, default=pd.NA):
    if col is None:
        return default
    value = row.get(col, default)
    return default if pd.isna(value) else value


def _age_from_birth_date(value) -> object:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return pd.NA
    birth_date = pd.to_datetime(value, errors="coerce")
    if pd.isna(birth_date):
        return pd.NA
    today = pd.Timestamp.today().normalize()
    return int((today - birth_date).days // 365.2425)


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _token_sort(value: str) -> str:
    return " ".join(sorted(_tokens(value)))


def _name_similarity(a: str, b: str) -> float:
    return max(
        _similarity(a, b),
        _similarity(_token_sort(a), _token_sort(b)),
    )


def _canonical_country_text(value: str) -> str:
    normalized = normalize_identity_text(value)
    token_sorted = _token_sort(normalized)
    return COUNTRY_ALIASES.get(normalized, COUNTRY_ALIASES.get(token_sorted, normalized))


def _tokens(value: str) -> set[str]:
    return {token for token in normalize_identity_text(value).split() if token}


def _position_to_broad(value: str) -> str:
    normalized = normalize_identity_text(value)
    for needle, broad in POSITION_TO_BROAD.items():
        if needle in normalized:
            return broad
    return ""


def prepare_transfermarkt_candidates(path: str | Path) -> tuple[pd.DataFrame, ColumnMap]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"No se encontró el CSV de Transfermarkt: {input_path}. "
            "Descarga o coloca el dataset en data/raw/transfermarkt/players.csv"
        )

    df = pd.read_csv(input_path, dtype=str, low_memory=False).fillna("")
    colmap = infer_transfermarkt_columns(df)
    df = df.copy()
    df["_tm_name"] = df[colmap.name].astype(str).str.strip()
    df["_tm_name_norm"] = df["_tm_name"].map(normalize_identity_text)
    df["_tm_club"] = df[colmap.club].astype(str).str.strip() if colmap.club else ""
    df["_tm_club_norm"] = df["_tm_club"].map(normalize_identity_text)
    df["_tm_nationality_norm"] = (
        df[colmap.nationality].map(_canonical_country_text) if colmap.nationality else ""
    )
    position_source = colmap.main_position or colmap.position_detail
    df["_tm_position_broad"] = (
        df[position_source].map(_position_to_broad) if position_source else ""
    )
    df = df[df["_tm_name_norm"].ne("")].reset_index(drop=True)
    return df, colmap


def _candidate_pool(master_row: pd.Series, candidates: pd.DataFrame) -> pd.DataFrame:
    name_norm = master_row["player_name_normalized"]
    name_tokens = [token for token in name_norm.split() if token]
    if not name_tokens:
        return candidates.iloc[0:0]

    exact = candidates[candidates["_tm_name_norm"].eq(name_norm)]
    if not exact.empty:
        return exact

    first = name_tokens[0]
    last = name_tokens[-1]
    mask = (
        candidates["_tm_name_norm"].str.contains(first, regex=False)
        | candidates["_tm_name_norm"].str.contains(last, regex=False)
    )
    return candidates[mask].copy()


def score_candidate(master_row: pd.Series, candidate: pd.Series) -> tuple[float, str]:
    name_score = _name_similarity(master_row["player_name_normalized"], candidate["_tm_name_norm"])
    club_score = _similarity(master_row["club_normalized"], candidate["_tm_club_norm"])
    team_score = _similarity(
        _canonical_country_text(master_row["team_canonical"]),
        candidate["_tm_nationality_norm"],
    )
    position_match = (
        bool(candidate["_tm_position_broad"])
        and candidate["_tm_position_broad"] == master_row["position_broad"]
    )

    score = name_score * 0.78
    if club_score == 1.0:
        score += 0.14
    elif club_score >= 0.88:
        score += 0.08
    elif club_score >= 0.75:
        score += 0.04

    if team_score == 1.0:
        score += 0.08
    elif team_score >= 0.85:
        score += 0.03

    if position_match:
        score += 0.05

    method = "fuzzy_name"
    raw_name_exact = master_row["player_name_normalized"] == candidate["_tm_name_norm"]
    token_name_exact = _token_sort(master_row["player_name_normalized"]) == _token_sort(candidate["_tm_name_norm"])
    if raw_name_exact and club_score == 1.0:
        method = "exact_name_club"
    elif raw_name_exact:
        method = "exact_name"
    elif token_name_exact:
        method = "exact_name_tokens"
    elif name_score >= 0.90:
        method = "near_name"

    return min(score, 1.0), method


def find_best_match(master_row: pd.Series, candidates: pd.DataFrame) -> tuple[pd.Series | None, float, str]:
    pool = _candidate_pool(master_row, candidates)
    if pool.empty:
        return None, 0.0, "unmatched"

    best_row = None
    best_score = -1.0
    best_method = "unmatched"
    for _, candidate in pool.iterrows():
        score, method = score_candidate(master_row, candidate)
        if score > best_score:
            best_row = candidate
            best_score = score
            best_method = method
    return best_row, round(float(best_score), 4), best_method


def _profile_row(master_row: pd.Series, candidate: pd.Series, colmap: ColumnMap) -> dict[str, object]:
    date_of_birth = _value(candidate, colmap.date_of_birth)
    age = _value(candidate, colmap.age)
    if pd.isna(age):
        age = _age_from_birth_date(date_of_birth)

    return {
        "player_key": master_row["player_key"],
        "transfermarkt_player_id": _value(candidate, colmap.player_id),
        "transfermarkt_name": candidate["_tm_name"],
        "transfermarkt_url": _value(candidate, colmap.url),
        "date_of_birth": date_of_birth,
        "age": age,
        "height_cm": _value(candidate, colmap.height_cm),
        "foot": _value(candidate, colmap.foot),
        "position_detail": _value(candidate, colmap.position_detail),
        "main_position": _value(candidate, colmap.main_position),
        "other_positions": _value(candidate, colmap.other_positions),
        "market_value_eur": _value(candidate, colmap.market_value_eur),
        "market_value_date": _value(candidate, colmap.market_value_date),
        "contract_until": _value(candidate, colmap.contract_until),
        "agent": _value(candidate, colmap.agent),
        "current_club_transfermarkt": candidate["_tm_club"],
        "source": "transfermarkt_dataset",
    }


def match_transfermarkt(
    master: pd.DataFrame,
    source_matches: pd.DataFrame,
    candidates: pd.DataFrame,
    colmap: ColumnMap,
    *,
    auto_match_threshold: float = AUTO_MATCH_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = source_matches.copy()
    source = source.set_index("player_key", drop=False)
    profile_rows = []
    review_rows = []

    for _, master_row in master.iterrows():
        candidate, confidence, method = find_best_match(master_row, candidates)
        player_key = master_row["player_key"]

        if candidate is None:
            review_rows.append({
                **{col: master_row.get(col, pd.NA) for col in [
                    "player_key", "team_canonical", "player_name",
                    "club_from_squad", "position_broad",
                ]},
                "transfermarkt_name": pd.NA,
                "current_club_transfermarkt": pd.NA,
                "transfermarkt_url": pd.NA,
                "match_confidence": 0.0,
                "match_method": "unmatched",
                "review_reason": "no_candidate_found",
            })
            continue

        transfermarkt_url = _value(candidate, colmap.url)
        if confidence >= REVIEW_THRESHOLD:
            needs_review = confidence < auto_match_threshold
            source.loc[player_key, "transfermarkt_name"] = candidate["_tm_name"]
            source.loc[player_key, "transfermarkt_url"] = transfermarkt_url
            source.loc[player_key, "match_confidence"] = confidence
            source.loc[player_key, "match_method"] = method
            source.loc[player_key, "needs_manual_review"] = needs_review

            profile_rows.append(_profile_row(master_row, candidate, colmap))

            if needs_review:
                review_rows.append({
                    "player_key": player_key,
                    "team_canonical": master_row["team_canonical"],
                    "player_name": master_row["player_name"],
                    "club_from_squad": master_row["club_from_squad"],
                    "position_broad": master_row["position_broad"],
                    "transfermarkt_name": candidate["_tm_name"],
                    "current_club_transfermarkt": candidate["_tm_club"],
                    "transfermarkt_url": transfermarkt_url,
                    "match_confidence": confidence,
                    "match_method": method,
                    "review_reason": "below_auto_match_threshold",
                })
        else:
            review_rows.append({
                "player_key": player_key,
                "team_canonical": master_row["team_canonical"],
                "player_name": master_row["player_name"],
                "club_from_squad": master_row["club_from_squad"],
                "position_broad": master_row["position_broad"],
                "transfermarkt_name": candidate["_tm_name"],
                "current_club_transfermarkt": candidate["_tm_club"],
                "transfermarkt_url": transfermarkt_url,
                "match_confidence": confidence,
                "match_method": method,
                "review_reason": "low_confidence_candidate",
            })

    profiles = pd.DataFrame(profile_rows, columns=TRANSFERMARKT_PROFILE_COLUMNS)
    review = pd.DataFrame(review_rows, columns=REVIEW_COLUMNS)
    return source.reset_index(drop=True), profiles, review


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Match WC2026 players to a local Transfermarkt dataset.")
    parser.add_argument("--master-input", default=DEFAULT_MASTER_OUTPUT)
    parser.add_argument("--source-matches-input", default=DEFAULT_SOURCE_MATCHES_OUTPUT)
    parser.add_argument("--transfermarkt-input", default=DEFAULT_TRANSFERMARKT_INPUT)
    parser.add_argument("--source-matches-output", default=DEFAULT_SOURCE_MATCHES_TRANSFERMARKT_OUTPUT)
    parser.add_argument("--profiles-output", default=DEFAULT_PROFILES_OUTPUT)
    parser.add_argument("--review-output", default=DEFAULT_REVIEW_OUTPUT)
    parser.add_argument("--auto-match-threshold", type=float, default=AUTO_MATCH_THRESHOLD)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args()

    master = pd.read_csv(args.master_input, dtype=str).fillna("")
    source_path = Path(args.source_matches_input)
    if source_path.exists():
        source_matches = pd.read_csv(source_path, dtype=str).fillna("")
    else:
        source_matches = build_player_source_matches(master)

    candidates, colmap = prepare_transfermarkt_candidates(args.transfermarkt_input)
    log.info("Loaded %d Transfermarkt candidates from %s", len(candidates), args.transfermarkt_input)

    source, profiles, review = match_transfermarkt(
        master,
        source_matches,
        candidates,
        colmap,
        auto_match_threshold=args.auto_match_threshold,
    )

    outputs = [
        (source, Path(args.source_matches_output), "player source matches"),
        (profiles, Path(args.profiles_output), "Transfermarkt profiles"),
        (review, Path(args.review_output), "Transfermarkt review rows"),
    ]
    for frame, output, label in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        log.info("Saved %d %s to %s", len(frame), label, output)

    matched = source["transfermarkt_url"].replace("", pd.NA).notna().sum()
    auto_matched = source["needs_manual_review"].astype(str).str.lower().eq("false").sum()
    log.info(
        "Transfermarkt matching summary: matched=%d/%d auto_matched=%d review=%d",
        matched,
        len(source),
        auto_matched,
        len(review),
    )


if __name__ == "__main__":
    main()
