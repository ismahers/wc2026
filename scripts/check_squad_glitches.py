"""
scripts/check_squad_glitches.py
===============================
Detecta nombres de jugador sospechosos de venir DEFORMADOS de la extracción
del PDF de FIFA (run-on en mayúsculas tipo "ABDELMONEIMMohamed", token
repetido, o demasiado largos).

NO detecta apodos oficiales dispares (Cabo Verde: "Roberto Lopes" -> "Pico
Lopes"): esos no son glitches, son nombres oficiales y se dejan como vengan de
FIFA salvo que rompan el match con Transfermarkt.

Uso
---
    python scripts/check_squad_glitches.py

    python scripts/check_squad_glitches.py \
        --squads data/raw/squads_wc2026_final_official_corrected.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd

# Permitir ejecutar como `python scripts/check_squad_glitches.py` desde la raíz
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

MAX_LEN = 28


def _is_glitch(name: object) -> bool:
    s = str(name).strip()
    if not s:
        return True
    return (
        bool(re.search(r"[A-Z]{4,}[a-z]", s))              # run-on MAYUS+minus pegados
        or bool(re.search(r"\b(\w+)\b.*\b\1\b", s, re.I))   # token repetido
        or len(s) > MAX_LEN                                 # sospechosamente largo
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Detecta nombres deformados en el squad CSV.")
    parser.add_argument(
        "--squads",
        default="data/raw/squads_wc2026_final_official_corrected.csv",
        help="CSV de convocatorias a revisar.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.squads):
        print(f"✗ No se encontró {args.squads}")
        sys.exit(2)

    df = pd.read_csv(args.squads)
    if "player_name" not in df.columns:
        print("✗ El CSV no tiene columna 'player_name'")
        sys.exit(2)

    flagged = df[df["player_name"].map(_is_glitch)].copy()

    print("=" * 72)
    print("NOMBRES SOSPECHOSOS DE GLITCH DE EXTRACCIÓN")
    print(f"Archivo: {args.squads}")
    print("=" * 72)

    if flagged.empty:
        print("✓ Ninguno. No hay nombres deformados que arreglar.")
        sys.exit(0)

    cols = [c for c in ["team_canonical", "player_name", "club", "position_broad"] if c in flagged.columns]
    print(flagged[cols].to_string(index=False))
    print("-" * 72)
    print(f"{len(flagged)} nombres a revisar a mano en {args.squads}")
    print("(Apunta cuáles tocas: si re-extraes el PDF, los glitches reaparecen.)")

    # Código de salida útil para CI: 1 si hay glitches.
    sys.exit(1)


if __name__ == "__main__":
    main()
    
