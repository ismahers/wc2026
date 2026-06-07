"""
Extract and compare the official FIFA WC2026 squad-list PDF.

The PDF is the audit source of truth. This script does not overwrite the
project's current squad CSV; it writes a FIFA-extracted CSV plus a comparison
report so manual corrections are explicit.

Usage:
    python -m src.data.fifa_squads_pdf
"""

from __future__ import annotations

import argparse
import difflib
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.data.normalize_squads import DEFAULT_INPUT, VALID_POSITIONS, validate_official_squads
from src.data.player_master import normalize_identity_text
from src.data.team_names import canonicalize


log = logging.getLogger(__name__)

DEFAULT_PDF = "docs/SquadLists-English.pdf"
DEFAULT_OUTPUT = "data/raw/squads_wc2026_fifa_official.csv"
DEFAULT_REPORT = "data/processed/fifa_squad_diff.csv"
DEFAULT_SUMMARY = "data/processed/fifa_squad_summary.csv"
DEFAULT_REAL_CHANGES = "data/processed/fifa_real_player_changes.csv"

POS_MAP = {
    "GK": "Goalkeeper",
    "DF": "Defender",
    "MF": "Midfielder",
    "FW": "Forward",
}

ROW_RE = re.compile(
    r"^\s*(?P<number>\d{1,2})\s*(?P<pos>GK|DF|MF|FW)\s+"
    r"(?P<body>.+?)"
    r"(?P<dob>\d{2}/\d{2}/\d{4})"
    r"(?P<club_height>.+?)\s+(?P<height>\d{3})\s*$"
)
TEAM_RE = re.compile(r"^\s*(?P<team>.+?)\s+\((?P<code>[A-Z]{3})\)\s*$")


def _clean_pdf_text(value: str) -> str:
    value = value.replace("\x00", "fi")
    value = unicodedata.normalize("NFC", value)
    value = re.sub(r"\bM\s+(?=[A-Z])", "M", value)
    return re.sub(r"\s+", " ", value).strip()


def _title_preserving_particles(value: str) -> str:
    particles = {"da", "de", "del", "dos", "du", "el", "la", "le", "van", "von"}
    words = []
    for word in value.split():
        if word.casefold() in particles:
            words.append(word.casefold())
        elif word.isupper():
            words.append(word.title())
        else:
            words.append(word)
    return " ".join(words)


def _display_name_from_fifa_name(fifa_player_name: str, first_names: str = "", last_names: str = "") -> str:
    """
    Convert FIFA's "LAST First" display into the project's "First Last" style.

    The PDF extraction sometimes loses cells, so we prefer the explicit
    FIRST/LAST columns when available and fall back to splitting PLAYER NAME.
    """
    first_names = _clean_pdf_text(first_names)
    last_names = _clean_pdf_text(last_names)
    name = _clean_pdf_text(fifa_player_name)
    parts = name.split()
    if len(parts) <= 1:
        return _title_preserving_particles(name)

    split_at = None
    for idx, token in enumerate(parts):
        if not token.isupper():
            split_at = idx
            break
    if split_at is None:
        return _title_preserving_particles(name)

    last = " ".join(parts[:split_at])
    first = " ".join(parts[split_at:])
    return _title_preserving_particles(f"{first} {last}")


def _load_pypdf():
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "pypdf is required to parse the FIFA squad PDF. "
            "Install project requirements or run: pip install pypdf"
        ) from exc
    return PdfReader


def _extract_team_name(page_text: str) -> tuple[str, str]:
    for line in page_text.splitlines():
        match = TEAM_RE.match(line)
        if not match:
            continue
        raw_team = _clean_pdf_text(match.group("team"))
        if raw_team.lower() in {"fifa world cup 2026", "squad list"}:
            continue
        return canonicalize(raw_team), match.group("code")
    raise ValueError("Could not find team header in PDF page")


def _parse_player_row(line: str) -> dict[str, object] | None:
    match = ROW_RE.match(line)
    if not match:
        return None

    body = match.group("body")
    body_parts = [part.strip() for part in re.split(r"\s{2,}", body) if part.strip()]
    if not body_parts:
        return None

    fifa_player_name = _clean_pdf_text(body_parts[0])
    first_names = _clean_pdf_text(body_parts[1]) if len(body_parts) > 1 else ""
    last_names = _clean_pdf_text(body_parts[2]) if len(body_parts) > 2 else ""
    name_on_shirt = _clean_pdf_text(body_parts[3]) if len(body_parts) > 3 else ""
    club = _clean_pdf_text(match.group("club_height"))

    return {
        "squad_number": int(match.group("number")),
        "position_code": match.group("pos"),
        "position_broad": POS_MAP[match.group("pos")],
        "fifa_player_name": fifa_player_name,
        "player_name": _display_name_from_fifa_name(fifa_player_name, first_names, last_names),
        "full_player_name": (
            _title_preserving_particles(f"{first_names} {last_names}")
            if first_names and last_names else pd.NA
        ),
        "first_names": first_names,
        "last_names": last_names,
        "name_on_shirt": name_on_shirt,
        "dob": match.group("dob"),
        "club": club,
        "height_cm": int(match.group("height")),
    }


def extract_fifa_squads(pdf_path: str | Path = DEFAULT_PDF) -> pd.DataFrame:
    PdfReader = _load_pypdf()
    reader = PdfReader(str(pdf_path))
    rows = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text(extraction_mode="layout")
        team_name, team_code = _extract_team_name(text)
        page_rows = []
        for line in text.splitlines():
            parsed = _parse_player_row(line)
            if parsed is None:
                continue
            parsed.update({
                "team_canonical": team_name,
                "team_code": team_code,
                "pdf_page": page_number,
            })
            page_rows.append(parsed)

        if len(page_rows) != 26:
            raise ValueError(f"{team_name}: expected 26 players on page {page_number}, got {len(page_rows)}")
        rows.extend(page_rows)

    df = pd.DataFrame(rows)
    ordered = [
        "team_canonical", "team_code", "squad_number", "position_code",
        "position_broad", "player_name", "fifa_player_name", "full_player_name",
        "first_names", "last_names", "name_on_shirt", "dob", "club", "height_cm", "pdf_page",
    ]
    return df[ordered].sort_values(["team_canonical", "squad_number"]).reset_index(drop=True)


def _strip_nickname(value: str) -> str:
    """Remove quoted nickname from player names: 'Ahmed Sayed "Zizo"' → 'Ahmed Sayed'."""
    return re.sub(r'\s*["\u201c\u201d\']+[^"\u201c\u201d\']+["\u201c\u201d\']+\s*', " ", str(value)).strip()


def _extract_nickname(value: str) -> str | None:
    """Extract the quoted nickname: 'Ahmed Sayed "Zizo"' → 'Zizo'."""
    match = re.search(r'["\u201c\u201d\']+([^"\u201c\u201d\']+)["\u201c\u201d\']+', str(value))
    return match.group(1).strip() if match else None


def _name_key(value: str) -> str:
    tokens = normalize_identity_text(_strip_nickname(value)).split()
    return " ".join(sorted(tokens))


def _name_compact_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_identity_text(_strip_nickname(value)))


def _name_tokens(value: str) -> set[str]:
    return set(normalize_identity_text(_strip_nickname(value)).split())


def _name_similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(
        None,
        normalize_identity_text(_strip_nickname(left)),
        normalize_identity_text(_strip_nickname(right)),
    ).ratio()


def _club_key(value: str) -> str:
    value = str(value or "")
    # Strip FIFA country code suffix: "(ENG)", "(ESP)", etc.
    value = re.sub(r"\s*\([A-Za-z]{3}\)\s*$", "", value)
    # Strip our own country suffix: "-GER", "-MEX", "-BRA", etc.
    value = re.sub(r"\s*-[A-Z]{3}\s*$", "", value)
    # Strip Transfermarkt-style parenthetical: "(Alemania)", "(España)", etc.
    value = re.sub(r"\s*\([^)]+\)\s*$", "", value)
    # Strip leading ordinal: "1." from "1. FSV Mainz 05"
    value = re.sub(r"^\s*\d+\.\s*", "", value)
    value = normalize_identity_text(value)

    # --- Specific club aliases (sorted by frequency of mismatch) ---
    _CLUB_ALIASES = {
        # German clubs
        r"\bbayern\s+munchen\b": "bayern",
        r"\bbayern\s+munich\b": "bayern",
        r"\bbayer\s+04\s+leverkusen\b": "bayer leverkusen",
        r"\btsg\s+hoffenheim\b": "hoffenheim",
        r"\bvfb\s+stuttgart\b": "stuttgart",
        r"\bsv\s+werder\s+bremen\b": "werder bremen",
        r"\bborussia\s+monchengladbach\b": "gladbach",
        r"\bborussia\s+moenchengladbach\b": "gladbach",
        r"\bborussia\s+m\s+gladbach\b": "gladbach",
        r"\b1\s+fsv\s+mainz\s+05\b": "mainz",
        r"\bmainz\s+05\b": "mainz",
        r"\brb\s+leipzig\b": "leipzig",
        r"\brasen\s+ballsport\s+leipzig\b": "leipzig",
        # English clubs
        r"\btottenham\s+hotspur\b": "tottenham",
        r"\bbrighton\s+(?:\&|and)\s+hove\s+albion\b": "brighton",
        r"\bcrystal\s+palace\b": "crystal palace",
        r"\bwest\s+ham\s+united\b": "west ham",
        r"\bnottingham\s+forest\b": "nottingham forest",
        r"\bwolverhampton\s+wanderers\b": "wolves",
        # French clubs
        r"\bparis\s+saint\s*-?\s*germain\b": "psg",
        r"\bolympique\s+de\s+marseille\b": "marseille",
        r"\bolympique\s+marseille\b": "marseille",
        r"\bolympique\s+lyonnais\b": "lyon",
        r"\bolympique\s+lyon\b": "lyon",
        r"\bstade\s+rennais\b": "rennes",
        r"\brc\s+lens\b": "lens",
        r"\brc\s+strasbourg\b": "strasbourg",
        r"\baj\s+auxerre\b": "auxerre",
        r"\bangers\s+sco\b": "angers",
        # Spanish clubs
        r"\breal\s+madrid\s+c\s*\.?\s*f\s*\.?\b": "real madrid",
        r"\batletico\s+de\s+madrid\b": "atletico madrid",
        r"\bathletic\s+club\b": "athletic bilbao",
        r"\bathletic\s+bilbao\b": "athletic bilbao",
        r"\brc\s+celta\s+vigo\b": "celta vigo",
        r"\brc\s+celta\b": "celta vigo",
        r"\breal\s+betis\b": "betis",
        # Italian clubs
        r"\binternationale\s+milano\b": "inter",
        r"\binternazionale\s+milano\b": "inter",
        r"\binternazionale\b": "inter",
        r"\binter\s+milan\b": "inter",
        r"\batalanta\s+bergamo\b": "atalanta",
        r"\bnapoli\b": "napoli",
        # Portuguese clubs
        r"\bsl\s+benfica\b": "benfica",
        r"\bsporting\s+cp\b": "sporting",
        r"\bsporting\s+de\s+lisboa\b": "sporting",
        r"\bsporting\s+lisbon\b": "sporting",
        r"\bporto\b": "porto",
        # Dutch clubs
        r"\bafc\s+ajax\b": "ajax",
        r"\bpsv\s+eindhoven\b": "psv",
        r"\bfeyenoord\s+rotterdam\b": "feyenoord",
        r"\balmere\s+city\b": "almere",
        # Belgian clubs
        r"\bsporting\s+charleroi\b": "charleroi",
        # Danish clubs
        r"\bkobenhavn\b": "copenhagen",
        r"\bbrondby\s+if\b": "brondby",
        # Czech clubs
        r"\bslavia\s+praha\b": "slavia prague",
        r"\bslavia\s+prague\b": "slavia prague",
        r"\bsparta\s+praha\b": "sparta prague",
        r"\bsparta\s+prague\b": "sparta prague",
        # Swedish clubs
        r"\baik\s+stockholm\b": "aik",
        # Brazilian clubs
        r"\bcr\s+flamengo\b": "flamengo",
        r"\bse\s+palmeiras\b": "palmeiras",
        r"\batletico\s+mineiro\b": "atletico mineiro",
        r"\bathletico\s+paranaense\b": "athletico paranaense",
        # Argentine clubs
        r"\bca\s+river\s+plate\b": "river plate",
        # Tunisian clubs
        r"\besperance\s+de\s+tunisie\b": "esperance",
        # Swiss clubs
        r"\byoung\s+boys\s+bern\b": "young boys",
        # Mexican clubs
        r"\bcd\s+guadalajara\b": "guadalajara",
        r"\bchivas\b": "guadalajara",
        # Korean clubs
        r"\bjeonbuk\s+hyundai\s+motors\b": "jeonbuk hyundai",
        r"\bdaejeon\s+hana\s+citizen\b": "daejeon hana",
        # Jordanian clubs
        r"\bal\s+wahdat\b": "al wehdat",
        r"\bal\s+hussein\b": "al hussein",
        r"\bal\s+faisaly\b": "al faisaly",
        # Saudi clubs
        r"\bal\s+ettifaq\b": "al ettifaq",
        r"\bal\s+etiffaq\b": "al ettifaq",
        r"\bal\s+qadsiah\b": "al qadisiyah",
        r"\bal\s+qadisiyah\b": "al qadisiyah",
        # Qatari clubs
        r"\bal\s+sailiya\b": "al sailiaya",
        r"\bal\s+sailiaya\b": "al sailiaya",
    }

    for pattern, replacement in _CLUB_ALIASES.items():
        value = re.sub(pattern, replacement, value)

    # Strip common organisational suffixes/prefixes
    value = re.sub(
        r"\b(fc|cf|sc|afc|ac|cd|ca|club|se|ss|ssc|ogc|osc|bsc|hsc|fk|sk|sl|sv|vfb|tsg|rc|cr|us|ks|hnk|if|jk)\b",
        " ",
        value,
    )
    value = re.sub(r"\b(futbol|football|soccer|deportivo|united)\b", " ", value)
    # Strip trailing "c f" or "c.f." remnants
    value = re.sub(r"\s+c\s*\.?\s*f\s*\.?\s*$", "", value)
    return re.sub(r"\s+", " ", value).strip()



def _raw_text_match(left: object, right: object) -> bool:
    return str(left or "").casefold().strip() == str(right or "").casefold().strip()


def _build_report_row(
    team: str,
    current_row: pd.Series | None,
    fifa_row: pd.Series | None,
    match_method: str,
    name_issue: str | None = None,
) -> dict[str, object]:
    current_name = current_row.get("player_name", "") if current_row is not None else ""
    fifa_name = fifa_row.get("player_name", "") if fifa_row is not None else ""
    similarity = _name_similarity(current_name, fifa_name) if current_row is not None and fifa_row is not None else pd.NA

    if current_row is None:
        issue = "right_only"
        club_raw_match = pd.NA
        club_norm_match = pd.NA
        position_match = pd.NA
    elif fifa_row is None:
        issue = "left_only"
        club_raw_match = pd.NA
        club_norm_match = pd.NA
        position_match = pd.NA
    else:
        current_club = current_row.get("club", "")
        fifa_club = fifa_row.get("club", "")
        current_position = current_row.get("position_broad", "")
        fifa_position = fifa_row.get("position_broad", "")
        club_raw_match = _raw_text_match(current_club, fifa_club)
        club_norm_match = _club_key(current_club) == _club_key(fifa_club)
        position_match = current_position == fifa_position

        if not club_norm_match and not position_match:
            issue = "club_and_position_mismatch"
        elif not club_norm_match:
            issue = "club_mismatch"
        elif not position_match:
            issue = "position_mismatch"
        elif name_issue:
            issue = name_issue
        elif not club_raw_match:
            issue = "club_suffix_only"
        else:
            issue = "match"

    return {
        "team_canonical": team,
        "issue": issue,
        "match_method": match_method,
        "name_similarity": similarity,
        "player_name_current": current_name or pd.NA,
        "player_name_fifa": fifa_name or pd.NA,
        "fifa_player_name_raw": fifa_row.get("fifa_player_name") if fifa_row is not None else pd.NA,
        "club_current": current_row.get("club") if current_row is not None else pd.NA,
        "club_fifa": fifa_row.get("club") if fifa_row is not None else pd.NA,
        "position_current": current_row.get("position_broad") if current_row is not None else pd.NA,
        "position_fifa": fifa_row.get("position_broad") if fifa_row is not None else pd.NA,
        "squad_number_fifa": fifa_row.get("squad_number") if fifa_row is not None else pd.NA,
        "dob_fifa": fifa_row.get("dob") if fifa_row is not None else pd.NA,
        "height_cm_fifa": fifa_row.get("height_cm") if fifa_row is not None else pd.NA,
        "club_raw_match": club_raw_match,
        "club_norm_match": club_norm_match,
        "club_match": club_norm_match,
        "position_match": position_match,
    }


def _best_fuzzy_match(
    current_row: pd.Series,
    fifa_candidates: pd.DataFrame,
    threshold: float,
    require_context: bool = False,
) -> int | None:
    """Find the best fuzzy name match among FIFA candidates.

    When *require_context* is True the match also needs at least one contextual
    signal (same position OR same normalised club) to be accepted.  This allows
    using a lower *threshold* without introducing false positives.
    """
    if fifa_candidates.empty:
        return None
    scored = []
    for idx, fifa_row in fifa_candidates.iterrows():
        base_score = _name_similarity(current_row["player_name"], fifa_row["player_name"])
        pos_match = current_row.get("position_broad") == fifa_row.get("position_broad")
        club_match = _club_key(current_row.get("club", "")) == _club_key(fifa_row.get("club", ""))
        bonus = (0.03 if pos_match else 0) + (0.05 if club_match else 0)
        final_score = min(base_score + bonus, 1.0)
        has_context = pos_match or club_match
        scored.append((final_score, base_score, has_context, idx))
    final_score, base_score, has_context, idx = max(scored, key=lambda item: item[0])
    if final_score < threshold:
        return None
    if require_context and not has_context:
        return None
    return idx


def compare_squads(current: pd.DataFrame, fifa: pd.DataFrame, fuzzy_threshold: float = 0.80) -> pd.DataFrame:
    current = current.copy().reset_index(drop=True)
    fifa = fifa.copy().reset_index(drop=True)

    current["team_canonical"] = current["team_canonical"].map(canonicalize)
    fifa["team_canonical"] = fifa["team_canonical"].map(canonicalize)
    for df in (current, fifa):
        df["name_key"] = df["player_name"].map(_name_key)
        df["name_norm"] = df["player_name"].map(normalize_identity_text)
        df["name_compact"] = df["player_name"].map(_name_compact_key)
        df["name_tokens"] = df["player_name"].map(_name_tokens)

    rows = []
    teams = sorted(set(current["team_canonical"]) | set(fifa["team_canonical"]))
    for team in teams:
        current_team = current[current["team_canonical"] == team]
        fifa_team = fifa[fifa["team_canonical"] == team]
        unmatched_current = set(current_team.index)
        unmatched_fifa = set(fifa_team.index)

        def match_by_column(column: str, method: str, name_issue: str | None = None) -> None:
            for current_idx in list(unmatched_current):
                current_row = current.loc[current_idx]
                candidates = [
                    idx for idx in unmatched_fifa
                    if fifa.at[idx, column] == current_row[column]
                ]
                if len(candidates) != 1:
                    continue
                fifa_idx = candidates[0]
                rows.append(_build_report_row(team, current_row, fifa.loc[fifa_idx], method, name_issue))
                unmatched_current.remove(current_idx)
                unmatched_fifa.remove(fifa_idx)

        for current_idx in list(unmatched_current):
            current_row = current.loc[current_idx]
            candidates = [
                idx for idx in unmatched_fifa
                if fifa.at[idx, "name_key"] == current_row["name_key"]
            ]
            if len(candidates) != 1:
                continue
            fifa_idx = candidates[0]
            fifa_row = fifa.loc[fifa_idx]
            raw_match = _raw_text_match(current_row["player_name"], fifa_row["player_name"])
            if raw_match:
                method = "exact"
                name_issue = None
            elif current_row["name_norm"] == fifa_row["name_norm"]:
                method = "accent"
                name_issue = "accent_diff"
            else:
                method = "name_format"
                name_issue = "name_format_diff"
            rows.append(_build_report_row(team, current_row, fifa_row, method, name_issue))
            unmatched_current.remove(current_idx)
            unmatched_fifa.remove(fifa_idx)

        match_by_column("name_norm", "accent", "accent_diff")
        match_by_column("name_compact", "name_format", "name_format_diff")

        for current_idx in list(unmatched_current):
            current_row = current.loc[current_idx]
            current_tokens = current_row["name_tokens"]
            candidates = []
            for fifa_idx in unmatched_fifa:
                fifa_tokens = fifa.at[fifa_idx, "name_tokens"]
                if not current_tokens or not fifa_tokens:
                    continue
                smaller, larger = sorted([current_tokens, fifa_tokens], key=len)
                if smaller.issubset(larger) and len(smaller) / len(larger) >= 0.5:
                    candidates.append(fifa_idx)
            if len(candidates) != 1:
                continue
            fifa_idx = candidates[0]
            rows.append(_build_report_row(team, current_row, fifa.loc[fifa_idx], "partial", "name_partial"))
            unmatched_current.remove(current_idx)
            unmatched_fifa.remove(fifa_idx)

        # Stage: nickname matching — try quoted nickname against FIFA mononyms
        for current_idx in list(unmatched_current):
            current_row = current.loc[current_idx]
            nickname = _extract_nickname(current_row["player_name"])
            if not nickname:
                continue
            nick_norm = normalize_identity_text(nickname)
            if len(nick_norm) < 3:
                continue
            candidates = [
                idx for idx in unmatched_fifa
                if normalize_identity_text(fifa.at[idx, "player_name"]) == nick_norm
            ]
            if len(candidates) != 1:
                # Try compact match (no spaces)
                nick_compact = re.sub(r"[^a-z0-9]+", "", nick_norm)
                candidates = [
                    idx for idx in unmatched_fifa
                    if re.sub(r"[^a-z0-9]+", "", normalize_identity_text(fifa.at[idx, "player_name"])) == nick_compact
                ]
            if len(candidates) != 1:
                continue
            fifa_idx = candidates[0]
            rows.append(_build_report_row(team, current_row, fifa.loc[fifa_idx], "nickname", "name_nickname"))
            unmatched_current.remove(current_idx)
            unmatched_fifa.remove(fifa_idx)

        # Stage: standard fuzzy (threshold 0.80)
        for current_idx in list(unmatched_current):
            current_row = current.loc[current_idx]
            fifa_idx = _best_fuzzy_match(current_row, fifa.loc[list(unmatched_fifa)], fuzzy_threshold)
            if fifa_idx is None:
                continue
            rows.append(_build_report_row(team, current_row, fifa.loc[fifa_idx], "fuzzy", "name_fuzzy_match"))
            unmatched_current.remove(current_idx)
            unmatched_fifa.remove(fifa_idx)

        # Stage: lenient fuzzy (threshold 0.70) but requires club OR position match
        lenient_threshold = max(fuzzy_threshold - 0.10, 0.65)
        for current_idx in list(unmatched_current):
            current_row = current.loc[current_idx]
            fifa_idx = _best_fuzzy_match(
                current_row, fifa.loc[list(unmatched_fifa)],
                lenient_threshold, require_context=True,
            )
            if fifa_idx is None:
                continue
            rows.append(_build_report_row(team, current_row, fifa.loc[fifa_idx], "fuzzy_lenient", "name_fuzzy_match"))
            unmatched_current.remove(current_idx)
            unmatched_fifa.remove(fifa_idx)

        for current_idx in sorted(unmatched_current):
            rows.append(_build_report_row(team, current.loc[current_idx], None, "unmatched"))
        for fifa_idx in sorted(unmatched_fifa):
            rows.append(_build_report_row(team, None, fifa.loc[fifa_idx], "unmatched"))

    report = pd.DataFrame(rows)
    issue_order = {
        "left_only": 0,
        "right_only": 1,
        "club_and_position_mismatch": 2,
        "position_mismatch": 3,
        "club_mismatch": 4,
        "club_suffix_only": 5,
        "name_fuzzy_match": 6,
        "name_nickname": 7,
        "name_partial": 8,
        "name_format_diff": 9,
        "accent_diff": 10,
        "match": 11,
    }
    report["_issue_order"] = report["issue"].map(issue_order).fillna(99)
    return (
        report
        .sort_values(["_issue_order", "team_canonical", "player_name_current", "player_name_fifa"], na_position="last")
        .drop(columns=["_issue_order"])
        .reset_index(drop=True)
    )


def build_official_squad_csv(fifa: pd.DataFrame) -> pd.DataFrame:
    out = fifa[["team_canonical", "player_name", "club", "position_broad"]].copy()
    return out.sort_values(["team_canonical", "position_broad", "player_name"]).reset_index(drop=True)


def validate_fifa_extract(fifa: pd.DataFrame) -> dict[str, object]:
    official_shape = build_official_squad_csv(fifa)
    report = validate_official_squads(official_shape)
    invalid_codes = sorted(set(fifa["position_code"]) - set(POS_MAP))
    report["invalid_position_codes"] = invalid_codes
    report["duplicate_team_squad_numbers"] = int(fifa.duplicated(["team_canonical", "squad_number"]).sum())
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract official FIFA WC2026 squad PDF and compare it.")
    parser.add_argument("--pdf", default=DEFAULT_PDF)
    parser.add_argument("--current", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report-output", default=DEFAULT_REPORT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY)
    parser.add_argument("--real-changes-output", default=DEFAULT_REAL_CHANGES)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.80)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args()

    fifa = extract_fifa_squads(args.pdf)
    validation = validate_fifa_extract(fifa)
    failures = {
        key: value for key, value in validation.items()
        if key in {
            "teams_not_26", "missing_teams", "extra_teams", "unknown_teams",
            "invalid_positions", "invalid_position_codes", "duplicate_team_player_rows",
            "duplicate_team_squad_numbers", "empty_clubs",
        } and value
    }
    if failures:
        raise ValueError(f"FIFA PDF squad validation failed: {failures}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_official_squad_csv(fifa).to_csv(output, index=False)
    log.info("Saved FIFA official squad CSV to %s", output)

    current = pd.read_csv(args.current, dtype=str).fillna("")
    report = compare_squads(current, fifa, fuzzy_threshold=args.fuzzy_threshold)
    report_output = Path(args.report_output)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_output, index=False)
    log.info("Saved FIFA/current diff report to %s", report_output)

    real_changes = report[report["issue"].isin(["left_only", "right_only"])].copy()
    real_changes_output = Path(args.real_changes_output)
    real_changes_output.parent.mkdir(parents=True, exist_ok=True)
    real_changes.to_csv(real_changes_output, index=False)
    log.info("Saved likely real player changes to %s", real_changes_output)

    summary = (
        report.groupby("issue", dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    summary_output = Path(args.summary_output)
    summary.to_csv(summary_output, index=False)
    log.info("Saved FIFA diff summary to %s", summary_output)
    print(summary.to_string(index=False))

    by_team = (
        report.pivot_table(
            index="team_canonical",
            columns="issue",
            values="match_method",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )
    by_team_output = summary_output.with_name(f"{summary_output.stem}_by_team{summary_output.suffix}")
    by_team.to_csv(by_team_output, index=False)
    log.info("Saved FIFA diff team summary to %s", by_team_output)


if __name__ == "__main__":
    main()
