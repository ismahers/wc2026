"""
Backtest WC2022 using the current XGB/Poisson ensemble without Odds API calls.

This script reuses the already-downloaded odds/result rows from
outputs/backtest2022_bets.csv and replaces only the model probabilities.

Leakage control:
  - Base XGB and Poisson models are trained with matches before --cutoff.
  - Ensemble weights/calibration are fitted on a pre-cutoff validation window
    by default: train < 2018, validate 2018 <= year < cutoff.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from src.data.team_names import canonicalize
from src.evaluation.model_ensemble import (
    ONE_X_TWO,
    _apply_calibration_matrix,
    _blend_validation_probs,
    _build_validation_predictions,
    _poisson_weight,
    fit_ensemble_calibration,
    load_calibration_config,
    load_weight_params,
    optimize_ensemble_weights,
)
from src.evaluation.value_filters import apply_value_filters
from src.models.poisson_model import PoissonGoalModel, _add_goal_targets
from src.models.xgb_baseline import MARKETS, MarketModel


DEFAULT_INPUT = "outputs/backtest2022_bets.csv"
DEFAULT_OUTPUT = "outputs/backtest2022_ensemble_bets.csv"
DEFAULT_FILTERED_OUTPUT = "outputs/backtest2022_ensemble_filtered_bets.csv"
DEFAULT_SUMMARY_OUTPUT = "outputs/backtest2022_ensemble_summary.csv"
DEFAULT_WEIGHT_OUTPUT = "outputs/backtest2022_ensemble_weight_params.json"
DEFAULT_CALIBRATION_OUTPUT = "outputs/backtest2022_ensemble_calibration.json"


def _score_to_result(score: object) -> str:
    home, away = str(score).split("-", maxsplit=1)
    home_goals = int(home)
    away_goals = int(away)
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _is_win(market: str, score: object) -> bool:
    result = _score_to_result(score)
    if market == "1X2-H":
        return result == "H"
    if market == "1X2-D":
        return result == "D"
    if market == "1X2-A":
        return result == "A"

    home, away = str(score).split("-", maxsplit=1)
    total_goals = int(home) + int(away)
    if market == "Over2.5":
        return total_goals > 2.5
    if market == "Under2.5":
        return total_goals < 2.5
    raise ValueError(f"Mercado no soportado: {market}")


def _profit(win: bool, odds: float) -> float:
    return float(odds) - 1.0 if win else -1.0


def _match_key(home_team: object, away_team: object) -> frozenset[str]:
    return frozenset((canonicalize(home_team), canonicalize(away_team)))


def _load_backtest_matches(odds_rows: pd.DataFrame, features_path: str, cutoff: int) -> pd.DataFrame:
    """Return one feature row per WC2022 match present in the odds CSV."""
    features = pd.read_csv(features_path, parse_dates=["date"])
    features = _add_goal_targets(features)
    features["date"] = pd.to_datetime(features["date"], errors="coerce")

    wanted = {}
    for _, row in odds_rows.drop_duplicates(["home_team", "away_team"]).iterrows():
        wanted[_match_key(row["home_team"], row["away_team"])] = str(row["resultado_real"])

    rows = []
    for idx, row in features.iterrows():
        if pd.isna(row.get("date")) or row["date"].year != cutoff:
            continue
        key = _match_key(row["home_team"], row["away_team"])
        score = wanted.get(key)
        if score is None:
            continue
        if pd.isna(row.get("home_score")) or pd.isna(row.get("away_score")):
            continue
        actual_score = f"{int(row['home_score'])}-{int(row['away_score'])}"
        if actual_score != score:
            continue
        rows.append(idx)

    matches = features.loc[rows].copy().sort_values("date").reset_index(drop=True)
    if matches.empty:
        raise ValueError("No se pudo cruzar el CSV de backtest con features WC2022.")
    return matches


def _train_xgb_probs(train: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    result_model = MarketModel(MARKETS["result_1x2"]).fit(
        train[train[MARKETS["result_1x2"].target_col].notna()].copy()
    )
    result_probs = result_model.predict_proba(matches)
    result_idx = {label: idx for idx, label in enumerate(result_model.label_encoder.classes_)}

    over_model = MarketModel(MARKETS["over25"]).fit(
        train[train[MARKETS["over25"].target_col].notna()].copy()
    )
    over_probs = over_model.predict_proba(matches)[:, 1]

    out = matches[["date", "home_team", "away_team", "rating_diff", "elo_diff"]].copy()
    for side in ONE_X_TWO:
        out[f"xgb_prob_{side}"] = result_probs[:, result_idx[side]]
    out["xgb_prob_over25"] = over_probs
    return out


def _train_poisson_probs(train: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    model = PoissonGoalModel().fit(train)
    model.fit_dixon_coles_rho(train)
    pred = model.predict_markets(matches)

    out = matches[["date", "home_team", "away_team", "rating_diff", "elo_diff"]].copy()
    for side in ONE_X_TWO:
        out[f"poisson_prob_{side}"] = pred[f"prob_{side}"].to_numpy(dtype=float)
    out["poisson_prob_over25"] = pred["prob_over25"].to_numpy(dtype=float)
    out["dixon_coles_rho"] = model.dixon_coles_rho
    return out


def _fit_or_load_blend(
    features_path: str,
    weight_path: str,
    calibration_path: str,
    *,
    train_cutoff: int,
    cutoff: int,
    maxiter: int,
    reuse_existing: bool,
):
    if reuse_existing and os.path.exists(weight_path) and os.path.exists(calibration_path):
        weight_params, _ = load_weight_params(weight_path)
        calibration = load_calibration_config(calibration_path)
        return weight_params, calibration

    weight_params, _, validation = optimize_ensemble_weights(
        features_path,
        weight_path,
        train_cutoff=train_cutoff,
        val_cutoff=cutoff,
        maxiter=maxiter,
    )
    calibration, _ = fit_ensemble_calibration(validation, weight_params, calibration_path)
    return weight_params, calibration


def _build_probability_table(
    features_path: str,
    odds_rows: pd.DataFrame,
    *,
    cutoff: int,
    weight_train_cutoff: int,
    weight_path: str,
    calibration_path: str,
    optimizer_maxiter: int,
    reuse_existing_fit: bool,
) -> pd.DataFrame:
    matches = _load_backtest_matches(odds_rows, features_path, cutoff)
    all_features = pd.read_csv(features_path, parse_dates=["date"])
    all_features = _add_goal_targets(all_features)
    all_features["date"] = pd.to_datetime(all_features["date"], errors="coerce")
    train = all_features[all_features["date"].dt.year < cutoff].copy()
    if train.empty:
        raise ValueError(f"No hay train antes de {cutoff}.")

    xgb_probs = _train_xgb_probs(train, matches)
    poisson_probs = _train_poisson_probs(train, matches)
    probs = xgb_probs.merge(
        poisson_probs,
        on=["date", "home_team", "away_team", "rating_diff", "elo_diff"],
        how="inner",
    )

    weight_params, calibration = _fit_or_load_blend(
        features_path,
        weight_path,
        calibration_path,
        train_cutoff=weight_train_cutoff,
        cutoff=cutoff,
        maxiter=optimizer_maxiter,
        reuse_existing=reuse_existing_fit,
    )

    ensemble_raw = []
    weights = []
    for _, row in probs.iterrows():
        poisson_weight, _, _ = _poisson_weight(
            row.get("rating_diff"),
            row.get("elo_diff"),
            params=weight_params,
        )
        weights.append(poisson_weight)
        ensemble_raw.append([
            (1.0 - poisson_weight) * row[f"xgb_prob_{side}"] + poisson_weight * row[f"poisson_prob_{side}"]
            for side in ONE_X_TWO
        ])
    ensemble_raw = np.asarray(ensemble_raw, dtype=float)
    ensemble_raw = np.clip(ensemble_raw, 1e-6, 1.0)
    ensemble_raw = ensemble_raw / ensemble_raw.sum(axis=1, keepdims=True)
    ensemble_cal = _apply_calibration_matrix(ensemble_raw, calibration)

    for idx, side in enumerate(ONE_X_TWO):
        probs[f"ensemble_raw_prob_{side}"] = ensemble_raw[:, idx]
        probs[f"ensemble_calibrated_prob_{side}"] = ensemble_cal[:, idx]
    probs["ensemble_raw_prob_over25"] = [
        (1.0 - w) * xgb + w * poi
        for w, xgb, poi in zip(weights, probs["xgb_prob_over25"], probs["poisson_prob_over25"])
    ]
    probs["ensemble_calibrated_prob_over25"] = probs["ensemble_raw_prob_over25"]
    probs["poisson_weight"] = weights
    return probs


def _prob_lookup(probabilities: pd.DataFrame) -> dict:
    lookup = {}
    for _, row in probabilities.iterrows():
        lookup[_match_key(row["home_team"], row["away_team"])] = row
    return lookup


def _prob_for_market(row: pd.Series, bet: pd.Series, variant: str) -> float:
    market = str(bet["mercado"])
    if market in {"1X2-H", "1X2-D", "1X2-A"}:
        if market == "1X2-D":
            return float(row[f"{variant}_prob_D"])

        bet_home = canonicalize(bet["home_team"])
        model_home = canonicalize(row["home_team"])
        if bet_home == model_home:
            side = "H" if market == "1X2-H" else "A"
        else:
            side = "A" if market == "1X2-H" else "H"
        return float(row[f"{variant}_prob_{side}"])

    if market == "Over2.5":
        return float(row[f"{variant}_prob_over25"])
    if market == "Under2.5":
        return 1.0 - float(row[f"{variant}_prob_over25"])
    raise ValueError(f"Mercado no soportado: {market}")


def build_backtest_rows(
    odds_rows: pd.DataFrame,
    probabilities: pd.DataFrame,
) -> pd.DataFrame:
    lookup = _prob_lookup(probabilities)
    rows = []
    variants = {
        "xgb_clean": "xgb",
        "poisson_dixon_coles": "poisson",
        "ensemble_raw": "ensemble_raw",
        "ensemble_calibrated": "ensemble_calibrated",
    }
    for _, bet in odds_rows.iterrows():
        prob_row = lookup.get(_match_key(bet["home_team"], bet["away_team"]))
        if prob_row is None:
            continue
        market = str(bet["mercado"])
        odds = float(bet["cuota"])
        win = _is_win(market, bet["resultado_real"])
        for model_variant, prefix in variants.items():
            probability = _prob_for_market(prob_row, bet, prefix)
            ev_pct = (probability * odds - 1.0) * 100.0
            rows.append({
                "model_variant": model_variant,
                "home_team": bet["home_team"],
                "away_team": bet["away_team"],
                "mercado": market,
                "seleccion": bet["seleccion"],
                "cuota": round(odds, 3),
                "prob_modelo": round(probability, 4),
                "ev_pct": round(ev_pct, 2),
                "win": bool(win),
                "profit": round(_profit(win, odds), 3),
                "resultado_real": bet["resultado_real"],
            })
    return pd.DataFrame(rows)


def summarize_strategy(filtered: pd.DataFrame) -> pd.DataFrame:
    rows = []
    placed = filtered[filtered["strategy_bet_allowed"].astype(bool)].copy()

    def add_row(model_variant: str, segment: str, group: pd.DataFrame) -> None:
        n = len(group)
        profit = float(group["profit_strategy"].sum()) if n else 0.0
        wins = int(group["win"].sum()) if n else 0
        rows.append({
            "model_variant": model_variant,
            "segment": segment,
            "bets": n,
            "wins": wins,
            "hit_rate_pct": round(wins / n * 100.0, 2) if n else 0.0,
            "profit_units": round(profit, 3),
            "roi_pct": round(profit / n * 100.0, 2) if n else 0.0,
            "avg_odds": round(float(group["cuota"].mean()), 3) if n else 0.0,
            "min_odds": round(float(group["cuota"].min()), 3) if n else 0.0,
            "max_odds": round(float(group["cuota"].max()), 3) if n else 0.0,
            "avg_ev_pct": round(float(group["ev_pct"].mean()), 2) if n else 0.0,
        })

    for model_variant, model_group in placed.groupby("model_variant"):
        add_row(model_variant, "overall", model_group)
        for market, market_group in model_group.groupby("mercado"):
            add_row(model_variant, str(market), market_group)

    if not rows:
        return pd.DataFrame(columns=[
            "model_variant", "segment", "bets", "wins", "hit_rate_pct",
            "profit_units", "roi_pct", "avg_odds", "avg_ev_pct",
        ])
    return pd.DataFrame(rows).sort_values(["segment", "roi_pct"], ascending=[True, False])


def run(
    input_path: str = DEFAULT_INPUT,
    features_path: str = "data/processed/features.csv",
    output_path: str = DEFAULT_OUTPUT,
    filtered_output_path: str = DEFAULT_FILTERED_OUTPUT,
    summary_output_path: str = DEFAULT_SUMMARY_OUTPUT,
    weight_path: str = DEFAULT_WEIGHT_OUTPUT,
    calibration_path: str = DEFAULT_CALIBRATION_OUTPUT,
    cutoff: int = 2022,
    weight_train_cutoff: int = 2018,
    min_ev_pct: float = 10.0,
    max_ev_pct: float = 40.0,
    min_odds: float = 1.50,
    max_odds: float = 2.50,
    optimizer_maxiter: int = 50,
    reuse_existing_fit: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)
    odds_rows = pd.read_csv(input_path)
    probabilities = _build_probability_table(
        features_path,
        odds_rows,
        cutoff=cutoff,
        weight_train_cutoff=weight_train_cutoff,
        weight_path=weight_path,
        calibration_path=calibration_path,
        optimizer_maxiter=optimizer_maxiter,
        reuse_existing_fit=reuse_existing_fit,
    )
    bets = build_backtest_rows(odds_rows, probabilities)
    if bets.empty:
        raise ValueError("No se generaron filas de backtest.")

    filtered_parts = []
    for _, group in bets.groupby("model_variant"):
        filtered = apply_value_filters(
            group,
            min_ev_pct=min_ev_pct,
            max_ev_pct=max_ev_pct,
        )
        filtered["strategy_odds_ok"] = filtered["cuota"].between(min_odds, max_odds, inclusive="both")
        odds_mask = filtered["strategy_bet_allowed"] & ~filtered["strategy_odds_ok"]
        filtered.loc[odds_mask, "strategy_reason"] = "odds_outside_range"
        filtered["strategy_bet_allowed"] = filtered["strategy_bet_allowed"] & filtered["strategy_odds_ok"]
        filtered_parts.append(filtered)
    filtered_bets = pd.concat(filtered_parts, ignore_index=True)
    filtered_bets["profit_strategy"] = 0.0
    mask = filtered_bets["strategy_bet_allowed"].astype(bool)
    filtered_bets.loc[mask, "profit_strategy"] = filtered_bets.loc[mask, "profit"]
    summary = summarize_strategy(filtered_bets)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    bets.to_csv(output_path, index=False)
    filtered_bets.to_csv(filtered_output_path, index=False)
    summary.to_csv(summary_output_path, index=False)
    return bets, filtered_bets, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compara ROI WC2022 entre XGB, Poisson y ensemble calibrado.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--features", default="data/processed/features.csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--filtered-output", default=DEFAULT_FILTERED_OUTPUT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--weight-output", default=DEFAULT_WEIGHT_OUTPUT)
    parser.add_argument("--calibration-output", default=DEFAULT_CALIBRATION_OUTPUT)
    parser.add_argument("--cutoff", type=int, default=2022)
    parser.add_argument("--weight-train-cutoff", type=int, default=2018)
    parser.add_argument("--min-ev-pct", type=float, default=10.0)
    parser.add_argument("--max-ev-pct", type=float, default=40.0)
    parser.add_argument("--min-odds", type=float, default=1.50)
    parser.add_argument("--max-odds", type=float, default=2.50)
    parser.add_argument("--optimizer-maxiter", type=int, default=50)
    parser.add_argument("--reuse-existing-fit", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    _, _, summary = run(
        input_path=args.input,
        features_path=args.features,
        output_path=args.output,
        filtered_output_path=args.filtered_output,
        summary_output_path=args.summary_output,
        weight_path=args.weight_output,
        calibration_path=args.calibration_output,
        cutoff=args.cutoff,
        weight_train_cutoff=args.weight_train_cutoff,
        min_ev_pct=args.min_ev_pct,
        max_ev_pct=args.max_ev_pct,
        min_odds=args.min_odds,
        max_odds=args.max_odds,
        optimizer_maxiter=args.optimizer_maxiter,
        reuse_existing_fit=args.reuse_existing_fit,
    )
    print(summary.to_string(index=False))
    print(f"\nGuardado: {args.output}")
    print(f"Guardado: {args.filtered_output}")
    print(f"Guardado: {args.summary_output}")


if __name__ == "__main__":
    main()
