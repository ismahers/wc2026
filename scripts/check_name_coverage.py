"""
scripts/check_name_coverage.py
==============================
Diagnóstico de cobertura de nombres de equipo ANTES del primer run end-to-end.

Para cada una de las 48 selecciones del fixture WC2026 comprueba:
  - cuántos partidos históricos tiene en unified.csv (vía nombre canónico)
  - si aparece en el flat de cuotas (The Odds API)

Los nombres de las fuentes que NO resuelven a un canónico se separan en dos:
  - CANDIDATOS A ALIAS: nombres que se parecen mucho a un equipo del Mundial
    (p.ej. "Cabo Verde" -> "Cape Verde"). Estos sí van a _ALIASES_RAW.
  - Rivales históricos ajenos al Mundial (Italia, Yorkshire, Yugoslavia...).
    Estos se DEJAN como están; no son alias. Solo se cuentan.

Uso
---
    python scripts/check_name_coverage.py

    python scripts/check_name_coverage.py \
        --unified data/unified.csv \
        --odds data/processed/odds_current_worldcup_flat.csv \
        --report data/processed/name_coverage.csv \
        --show-all-unresolved        # opcional: lista los rivales ajenos también
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys

import pandas as pd

# Permitir ejecutar como `python scripts/check_name_coverage.py` desde la raíz
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data.team_names import (  # noqa: E402
    CANONICAL_TEAMS,
    canonicalize,
    unknown_names,
)

# Umbral de similitud para considerar un nombre "casi" un equipo del Mundial.
NEAR_MISS_CUTOFF = 0.84

_CANON_LOWER = {t.lower(): t for t in CANONICAL_TEAMS}


def _team_series(df: pd.DataFrame) -> pd.Series:
    """Concatena home_team y away_team (lo que exista) en una sola Serie."""
    cols = [c for c in ("home_team", "away_team") if c in df.columns]
    if not cols:
        return pd.Series(dtype=object)
    parts = [df[c].dropna().astype(str) for c in cols]
    return pd.concat(parts, ignore_index=True)


def _load(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"  ⚠ {label}: no encontrado en {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        print(f"  ✓ {label}: {len(df)} filas  ({path})")
        return df
    except Exception as exc:  # pragma: no cover - diagnóstico
        print(f"  ✗ {label}: error leyendo {path}: {exc}")
        return pd.DataFrame()


def _near_canonical(name: str) -> str | None:
    """Devuelve el canónico más parecido si supera el umbral, si no None."""
    match = difflib.get_close_matches(
        name.lower(), list(_CANON_LOWER), n=1, cutoff=NEAR_MISS_CUTOFF
    )
    return _CANON_LOWER[match[0]] if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Cobertura de nombres de equipo WC2026.")
    parser.add_argument("--unified", default="data/unified.csv",
                        help="Histórico unificado (Kaggle + StatsBomb).")
    parser.add_argument("--odds", default="data/processed/odds_current_worldcup_flat.csv",
                        help="Flat de cuotas actuales de The Odds API.")
    parser.add_argument("--report", default=None,
                        help="Ruta opcional para guardar el informe CSV por equipo.")
    parser.add_argument("--show-all-unresolved", action="store_true",
                        help="Listar también los rivales históricos ajenos al Mundial.")
    args = parser.parse_args()

    print("=" * 64)
    print("COBERTURA DE NOMBRES DE EQUIPO — WC2026")
    print("=" * 64)
    print(f"Selecciones canónicas en el fixture: {len(CANONICAL_TEAMS)}")
    print()
    print("Cargando fuentes...")
    unified = _load(args.unified, "unified.csv")
    odds = _load(args.odds, "odds flat")
    print()

    # --- Conteo histórico por equipo canónico --------------------------------
    hist_counts: dict[str, int] = {t: 0 for t in CANONICAL_TEAMS}
    for raw in _team_series(unified):
        canon = canonicalize(raw)
        if canon in hist_counts:
            hist_counts[canon] += 1

    # --- Presencia en odds ----------------------------------------------------
    odds_canon_present = {
        c for c in (canonicalize(r) for r in _team_series(odds)) if c in hist_counts
    }

    # --- Tabla por equipo -----------------------------------------------------
    report = pd.DataFrame(
        [{"team": t, "hist_matches": hist_counts[t], "in_odds": t in odds_canon_present}
         for t in CANONICAL_TEAMS]
    ).sort_values("hist_matches").reset_index(drop=True)

    missing_hist = report[report["hist_matches"] == 0]["team"].tolist()
    thin_hist = report[(report["hist_matches"] > 0) &
                       (report["hist_matches"] < 20)]["team"].tolist()

    print("-" * 64)
    print("COBERTURA HISTÓRICA (unified.csv)")
    print("-" * 64)
    print(f"  Con historial:  {len(CANONICAL_TEAMS) - len(missing_hist)}/{len(CANONICAL_TEAMS)}")
    if missing_hist:
        print(f"  ✗ SIN historial ({len(missing_hist)}): {', '.join(missing_hist)}")
        print("    → o el equipo no está en Kaggle, o su nombre no canoniza bien.")
    if thin_hist:
        print(f"  ⚠ Historial escaso (<20 partidos): {', '.join(thin_hist)}")
    print()

    print("-" * 64)
    print("COBERTURA EN CUOTAS (odds flat)")
    print("-" * 64)
    if odds.empty:
        print("  (sin archivo de cuotas; omitido)")
    else:
        missing_odds = [t for t in CANONICAL_TEAMS if t not in odds_canon_present]
        print(f"  Con cuota:  {len(odds_canon_present)}/{len(CANONICAL_TEAMS)}")
        if missing_odds:
            print(f"  Sin cuota ({len(missing_odds)}): {', '.join(missing_odds)}")
    print()

    # --- Nombres sin resolver: candidatos a alias vs rivales ajenos -----------
    print("-" * 64)
    print("NOMBRES SIN RESOLVER")
    print("-" * 64)
    for label, df in (("unified.csv", unified), ("odds flat", odds)):
        if df.empty:
            continue
        unresolved = unknown_names(_team_series(df))
        candidates = [(n, _near_canonical(n)) for n in unresolved]
        alias_candidates = [(n, c) for n, c in candidates if c]
        foreign = [n for n, c in candidates if not c]

        print(f"\n  {label}: {len(unresolved)} sin resolver "
              f"({len(alias_candidates)} candidatos a alias, "
              f"{len(foreign)} rivales ajenos al Mundial)")

        if alias_candidates:
            print("    \u25ba Candidatos a alias (revisa y a\u00f1ade a _ALIASES_RAW):")
            for name, canon in alias_candidates:
                print(f'        "{name}": "{canon}",')
        else:
            print("    \u2713 Ning\u00fan candidato a alias: lo no resuelto son rivales ajenos.")

        if args.show_all_unresolved and foreign:
            print("    \u00b7 Rivales ajenos (se dejan como est\u00e1n):")
            for name in foreign:
                print(f"        {name}")
    print()

    # --- Guardar informe ------------------------------------------------------
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        report.to_csv(args.report, index=False)
        print(f"Informe por equipo guardado en {args.report}")

    print("=" * 64)
    # Código de salida útil para CI: 1 si falta historial de algún equipo del Mundial.
    sys.exit(1 if missing_hist else 0)


if __name__ == "__main__":
    main()