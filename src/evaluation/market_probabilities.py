"""
src/evaluation/market_probabilities.py
======================================
Build a multi-market probability radar from WC2026 ensemble predictions.

This module does not call any odds API. It converts model probabilities into
fair odds so we can review more markets before we have full bookmaker coverage.

Markets:
  - 1X2
  - Double chance
  - Draw no bet
  - Match totals 1.5 / 2.5 / 3.5
  - Both teams to score
  - Team goals 0.5 / 1.5 / 2.5
  - Clean sheet
  - Total corners 8.5 / 9.5 / 10.5 (Poisson approximation from expected corners)

Usage:
    python -m src.evaluation.market_probabilities
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.data.team_names import canonicalize


DEFAULT_ENSEMBLE_INPUT = "outputs/wc2026_ensemble_predictions.csv"
DEFAULT_RELIABILITY_INPUT = "outputs/wc2026_reliability.csv"
DEFAULT_OUTPUT = "outputs/wc2026_market_probabilities.csv"

ALL_MARKETS = {
    "h2h",
    "double_chance",
    "draw_no_bet",
    "totals",
    "btts",
    "team_goals",
    "clean_sheet",
    "corners",
}

CONFIDENCE_RANK = {
    "low": 1,
    "medium": 2,
    "high": 3,
}


def _safe_prob(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    return float(np.clip(float(value), 0.0001, 0.9999))


def _fair_odds(probability: float) -> float:
    if pd.isna(probability) or probability <= 0:
        return np.nan
    return round(1.0 / max(float(probability), 0.0001), 3)


def _confidence_floor(*values: object) -> str:
    ranks = []
    for value in values:
        key = str(value or "").strip().lower()
        if key in CONFIDENCE_RANK:
            ranks.append(CONFIDENCE_RANK[key])
    if not ranks:
        return "derived"
    rank = min(ranks)
    for label, label_rank in CONFIDENCE_RANK.items():
        if label_rank == rank:
            return label
    return "derived"


def _poisson_over(mean: float, line: float) -> float:
    if pd.isna(mean) or mean <= 0:
        return np.nan
    return float(1.0 - poisson.cdf(math.floor(line), float(mean)))


def _poisson_under(mean: float, line: float) -> float:
    if pd.isna(mean) or mean <= 0:
        return np.nan
    return float(poisson.cdf(math.floor(line), float(mean)))


def _selection_probability(row: pd.Series, side: str) -> float:
    return _safe_prob(row.get(f"final_prob_{side}"))


def _add_row(
    rows: list[dict[str, object]],
    row: pd.Series,
    *,
    market: str,
    selection: str,
    probability: float,
    line: float | str | None = None,
    source_model: str,
    model_confidence: str,
    notes: str = "",
) -> None:
    probability = _safe_prob(probability)
    if pd.isna(probability):
        return

    rows.append({
        "date": row.get("date"),
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "market": market,
        "selection": selection,
        "line": line,
        "model_probability": round(float(probability), 4),
        "fair_odds": _fair_odds(probability),
        "min_odds_ev5": round(1.05 / probability, 3),
        "min_odds_ev10": round(1.10 / probability, 3),
        "min_odds_ev15": round(1.15 / probability, 3),
        "source_model": source_model,
        "model_confidence": model_confidence,
        "fiabilidad_pct": row.get("fiabilidad_pct"),
        "fiabilidad_nivel": row.get("fiabilidad_nivel"),
        "model_regime": row.get("model_regime"),
        "poisson_weight": row.get("poisson_weight"),
        "notes": notes,
    })


def _build_reliability_lookup(reliability_path: str) -> dict[frozenset[str], dict[str, object]]:
    if not os.path.exists(reliability_path):
        return {}

    reliability = pd.read_csv(reliability_path)
    required = {"home_team", "away_team"}
    if not required.issubset(reliability.columns):
        return {}

    lookup = {}
    for _, row in reliability.iterrows():
        key = frozenset((canonicalize(row["home_team"]), canonicalize(row["away_team"])))
        lookup[key] = {
            "fiabilidad_pct": row.get("fiabilidad_pct"),
            "fiabilidad_nivel": row.get("fiabilidad_nivel"),
        }
    return lookup


def _attach_reliability(ensemble: pd.DataFrame, reliability_path: str) -> pd.DataFrame:
    lookup = _build_reliability_lookup(reliability_path)
    out = ensemble.copy()
    fiabilidad_pct = []
    fiabilidad_nivel = []

    for _, row in out.iterrows():
        key = frozenset((canonicalize(row["home_team"]), canonicalize(row["away_team"])))
        values = lookup.get(key, {})
        fiabilidad_pct.append(values.get("fiabilidad_pct", np.nan))
        fiabilidad_nivel.append(values.get("fiabilidad_nivel", "N/A"))

    out["fiabilidad_pct"] = fiabilidad_pct
    out["fiabilidad_nivel"] = fiabilidad_nivel
    return out


def _parse_markets(markets: str | Iterable[str]) -> set[str]:
    if isinstance(markets, str):
        values = {part.strip().lower() for part in markets.split(",") if part.strip()}
    else:
        values = {str(part).strip().lower() for part in markets if str(part).strip()}

    if not values or "all" in values:
        return set(ALL_MARKETS)

    unknown = values - ALL_MARKETS
    if unknown:
        raise ValueError(f"Mercados no soportados: {sorted(unknown)}")
    return values


def build_market_probabilities(
    ensemble_path: str = DEFAULT_ENSEMBLE_INPUT,
    reliability_path: str = DEFAULT_RELIABILITY_INPUT,
    output_path: str = DEFAULT_OUTPUT,
    markets: str | Iterable[str] = "all",
) -> pd.DataFrame:
    if not os.path.exists(ensemble_path):
        raise FileNotFoundError(
            f"{ensemble_path} no existe. Ejecuta antes: python -m src.evaluation.model_ensemble"
        )

    selected_markets = _parse_markets(markets)
    ensemble = pd.read_csv(ensemble_path)
    ensemble = _attach_reliability(ensemble, reliability_path)

    rows: list[dict[str, object]] = []
    for _, row in ensemble.iterrows():
        home = str(row["home_team"])
        away = str(row["away_team"])
        p_h = _selection_probability(row, "H")
        p_d = _selection_probability(row, "D")
        p_a = _selection_probability(row, "A")

        if "h2h" in selected_markets:
            _add_row(
                rows, row, market="h2h", selection=home, probability=p_h,
                source_model="ensemble_1x2", model_confidence=row.get("confidence_H"),
            )
            _add_row(
                rows, row, market="h2h", selection="Draw", probability=p_d,
                source_model="ensemble_1x2", model_confidence=row.get("confidence_D"),
                notes="No usar como apuesta core hasta recalibrar empates.",
            )
            _add_row(
                rows, row, market="h2h", selection=away, probability=p_a,
                source_model="ensemble_1x2", model_confidence=row.get("confidence_A"),
            )

        if "double_chance" in selected_markets:
            _add_row(
                rows, row, market="double_chance", selection=f"{home} or Draw",
                probability=p_h + p_d, source_model="derived_from_1x2",
                model_confidence=_confidence_floor(row.get("confidence_H"), row.get("confidence_D")),
            )
            _add_row(
                rows, row, market="double_chance", selection=f"{away} or Draw",
                probability=p_a + p_d, source_model="derived_from_1x2",
                model_confidence=_confidence_floor(row.get("confidence_A"), row.get("confidence_D")),
            )
            _add_row(
                rows, row, market="double_chance", selection=f"{home} or {away}",
                probability=p_h + p_a, source_model="derived_from_1x2",
                model_confidence=_confidence_floor(row.get("confidence_H"), row.get("confidence_A")),
            )

        if "draw_no_bet" in selected_markets:
            non_draw = p_h + p_a
            if pd.notna(non_draw) and non_draw > 0:
                _add_row(
                    rows, row, market="draw_no_bet", selection=home,
                    probability=p_h / non_draw, source_model="derived_from_1x2",
                    model_confidence=_confidence_floor(row.get("confidence_H"), row.get("confidence_A")),
                    notes="Probabilidad condicionada a que no haya empate; empate void.",
                )
                _add_row(
                    rows, row, market="draw_no_bet", selection=away,
                    probability=p_a / non_draw, source_model="derived_from_1x2",
                    model_confidence=_confidence_floor(row.get("confidence_H"), row.get("confidence_A")),
                    notes="Probabilidad condicionada a que no haya empate; empate void.",
                )

        if "totals" in selected_markets:
            total_lambda = float(row.get("lambda_home", np.nan)) + float(row.get("lambda_away", np.nan))
            for line in (1.5, 2.5, 3.5):
                if line == 2.5 and pd.notna(row.get("final_prob_over25")):
                    p_over = _safe_prob(row.get("final_prob_over25"))
                    source = "ensemble_over25"
                    confidence = row.get("confidence_over25")
                else:
                    p_over = _poisson_over(total_lambda, line)
                    source = "poisson_total_goals"
                    confidence = row.get("confidence_over25", "derived")
                _add_row(
                    rows, row, market="total_goals", selection="Over", line=line,
                    probability=p_over, source_model=source, model_confidence=confidence,
                )
                _add_row(
                    rows, row, market="total_goals", selection="Under", line=line,
                    probability=1.0 - p_over, source_model=source, model_confidence=confidence,
                )

        if "btts" in selected_markets and pd.notna(row.get("final_prob_btts")):
            p_btts = _safe_prob(row.get("final_prob_btts"))
            _add_row(
                rows, row, market="btts", selection="Yes", probability=p_btts,
                source_model="ensemble_btts", model_confidence=row.get("confidence_btts"),
            )
            _add_row(
                rows, row, market="btts", selection="No", probability=1.0 - p_btts,
                source_model="ensemble_btts", model_confidence=row.get("confidence_btts"),
            )

        if "team_goals" in selected_markets:
            for team, team_side, mean in (
                (home, "home", row.get("lambda_home")),
                (away, "away", row.get("lambda_away")),
            ):
                mean = float(mean) if pd.notna(mean) else np.nan
                for line in (0.5, 1.5, 2.5):
                    p_over = _poisson_over(mean, line)
                    _add_row(
                        rows, row, market="team_goals", selection=f"{team} Over",
                        line=line, probability=p_over, source_model=f"poisson_{team_side}_goals",
                        model_confidence="derived",
                    )
                    _add_row(
                        rows, row, market="team_goals", selection=f"{team} Under",
                        line=line, probability=1.0 - p_over,
                        source_model=f"poisson_{team_side}_goals", model_confidence="derived",
                    )

        if "clean_sheet" in selected_markets:
            lambda_home = float(row.get("lambda_home", np.nan))
            lambda_away = float(row.get("lambda_away", np.nan))
            p_home_clean = _poisson_under(lambda_away, 0.5)
            p_away_clean = _poisson_under(lambda_home, 0.5)
            _add_row(
                rows, row, market="clean_sheet", selection=f"{home} Yes",
                probability=p_home_clean, source_model="poisson_opponent_goals_zero",
                model_confidence="derived",
            )
            _add_row(
                rows, row, market="clean_sheet", selection=f"{home} No",
                probability=1.0 - p_home_clean, source_model="poisson_opponent_goals_zero",
                model_confidence="derived",
            )
            _add_row(
                rows, row, market="clean_sheet", selection=f"{away} Yes",
                probability=p_away_clean, source_model="poisson_opponent_goals_zero",
                model_confidence="derived",
            )
            _add_row(
                rows, row, market="clean_sheet", selection=f"{away} No",
                probability=1.0 - p_away_clean, source_model="poisson_opponent_goals_zero",
                model_confidence="derived",
            )

        if "corners" in selected_markets and pd.notna(row.get("final_pred_corners")):
            mean_corners = float(row.get("final_pred_corners"))
            for line in (8.5, 9.5, 10.5):
                p_over = _poisson_over(mean_corners, line)
                _add_row(
                    rows, row, market="total_corners", selection="Over", line=line,
                    probability=p_over, source_model="poisson_approx_from_expected_corners",
                    model_confidence="low",
                    notes="Aproximacion: corners tienen sobredispersion; usar como radar, no como apuesta core.",
                )
                _add_row(
                    rows, row, market="total_corners", selection="Under", line=line,
                    probability=1.0 - p_over, source_model="poisson_approx_from_expected_corners",
                    model_confidence="low",
                    notes="Aproximacion: corners tienen sobredispersion; usar como radar, no como apuesta core.",
                )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No se generaron probabilidades de mercado.")

    out = out.sort_values(["date", "home_team", "away_team", "market", "line", "selection"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def summarize_market_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("market", dropna=False)
        .agg(
            rows=("market", "size"),
            avg_probability=("model_probability", "mean"),
            avg_fair_odds=("fair_odds", "mean"),
            min_fair_odds=("fair_odds", "min"),
            max_fair_odds=("fair_odds", "max"),
        )
        .round(3)
        .reset_index()
        .sort_values("market")
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build WC2026 multi-market model probabilities.")
    parser.add_argument("--ensemble", default=DEFAULT_ENSEMBLE_INPUT)
    parser.add_argument("--reliability", default=DEFAULT_RELIABILITY_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--markets",
        default="all",
        help="Comma-separated list: all,h2h,double_chance,draw_no_bet,totals,btts,team_goals,clean_sheet,corners",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    df = build_market_probabilities(
        ensemble_path=args.ensemble,
        reliability_path=args.reliability,
        output_path=args.output,
        markets=args.markets,
    )
    print(f"Guardado en {args.output} ({len(df)} filas)")
    print(summarize_market_probabilities(df).to_string(index=False))


if __name__ == "__main__":
    main()
