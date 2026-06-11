"""
Probe soccerdata FBref league identifiers before scraping.

Usage:
  python -m src.data.fbref_probe_leagues \
    --leagues "USA-Major League Soccer" "NED-Eredivisie" "POR-Primeira Liga"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether soccerdata recognizes FBref league keys.")
    parser.add_argument("--leagues", nargs="+", required=True)
    args = parser.parse_args()

    os.environ.setdefault("SOCCERDATA_DIR", str(Path(".soccerdata").resolve()))
    try:
        import soccerdata as sd
    except ImportError as exc:
        raise ImportError("Falta soccerdata. Ejecuta `pip install -r requirements.txt`.") from exc

    available = set(sd.FBref.available_leagues())

    print("\nFBref leagues available in soccerdata:")
    for league in sorted(available):
        print(f"  - {league}")

    print("\nRequested leagues check:")
    missing = []
    for league in args.leagues:
        status = "OK" if league in available else "MISSING"
        print(f"  [{status}] {league}")
        if status == "MISSING":
            missing.append(league)

    if missing:
        print("\nMissing leagues must be added to SOCCERDATA_DIR/config/league_dict.json before scraping:")
        for league in missing:
            print(f"  - {league}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
