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


def _name_key(value: str) -> str:
    tokens = normalize_identity_text(value).split()
    return " ".join(sorted(tokens))


def _club_key(value: str) -> str:
    value = normalize_identity_text(value)
    value = re.sub(r"\b(fc|cf|sc|afc|ac|cd|ca|club)\b", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def compare_squads(current: pd.DataFrame, fifa: pd.DataFrame) -> pd.DataFrame:
    current = current.copy()
    fifa = fifa.copy()

    current["team_canonical"] = current["team_canonical"].map(canonicalize)
    fifa["team_canonical"] = fifa["team_canonical"].map(canonicalize)
    current["name_key"] = current["player_name"].map(_name_key)
    fifa["name_key"] = fifa["player_name"].map(_name_key)

    merged = current.merge(
        fifa,
        on=["team_canonical", "name_key"],
        how="outer",
        suffixes=("_current", "_fifa"),
        indicator=True,
    )

    rows = []
    for _, row in merged.iterrows():
        status = row["_merge"]
        issue = status
        club_match = pd.NA
        position_match = pd.NA

        if status == "both":
            current_club = row.get("club_current", "")
            fifa_club = row.get("club_fifa", "")
            current_position = row.get("position_broad_current", "")
            fifa_position = row.get("position_broad_fifa", "")
            club_match = _club_key(current_club) == _club_key(fifa_club)
            position_match = current_position == fifa_position
            if club_match and position_match:
                issue = "match"
            elif not club_match and not position_match:
                issue = "club_and_position_mismatch"
            elif not club_match:
                issue = "club_mismatch"
            else:
                issue = "position_mismatch"

        rows.append({
            "team_canonical": row["team_canonical"],
            "issue": issue,
            "player_name_current": row.get("player_name_current"),
            "player_name_fifa": row.get("player_name_fifa"),
            "fifa_player_name_raw": row.get("fifa_player_name"),
            "club_current": row.get("club_current"),
            "club_fifa": row.get("club_fifa"),
            "position_current": row.get("position_broad_current"),
            "position_fifa": row.get("position_broad_fifa"),
            "squad_number_fifa": row.get("squad_number"),
            "dob_fifa": row.get("dob"),
            "height_cm_fifa": row.get("height_cm"),
            "club_match": club_match,
            "position_match": position_match,
        })

    report = pd.DataFrame(rows)
    issue_order = {
        "left_only": 0,
        "right_only": 1,
        "club_and_position_mismatch": 2,
        "position_mismatch": 3,
        "club_mismatch": 4,
        "match": 5,
    }
    report["_issue_order"] = report["issue"].map(issue_order).fillna(99)
    return (
        report
        .sort_values(["_issue_order", "team_canonical", "player_name_current", "player_name_fifa"])
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
    report = compare_squads(current, fifa)
    report_output = Path(args.report_output)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_output, index=False)
    log.info("Saved FIFA/current diff report to %s", report_output)

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


if __name__ == "__main__":
    main()
