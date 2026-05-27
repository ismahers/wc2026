"""
Stable XGBoost training entry point.

This module keeps the user-facing command small while reusing the existing
market-specific baseline implementation in xgb_baseline.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.models.xgb_baseline import MARKETS, XGBBaselinePipeline


@dataclass(frozen=True)
class XGBoostRunConfig:
    """File paths and split settings for one XGBoost run."""

    features_path: Path = Path("data/processed/features_train.csv")
    wc_features_path: Path = Path("data/processed/features_wc2026.csv")
    output_dir: Path = Path("outputs")
    train_cutoff: int = 2018
    val_cutoff: int = 2023
    markets: tuple[str, ...] | None = None


def _parse_markets(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    markets = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = [market for market in markets if market not in MARKETS]
    if unknown:
        known = ", ".join(sorted(MARKETS))
        raise ValueError(f"Mercados desconocidos: {unknown}. Mercados disponibles: {known}")
    return markets


def _load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de features: {path}")
    return pd.read_csv(path, parse_dates=["date"])


def run_xgboost_pipeline(config: XGBoostRunConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train market models and optionally predict WC2026 fixtures.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Metrics summary and WC2026 predictions. Predictions can be empty when
        the WC2026 features file is not present.
    """
    df_train = _load_features(config.features_path)
    pipeline = XGBBaselinePipeline(
        train_cutoff=config.train_cutoff,
        val_cutoff=config.val_cutoff,
        markets=list(config.markets) if config.markets else None,
    )
    metrics = pipeline.run(df_train, output_dir=str(config.output_dir))

    predictions = pd.DataFrame()
    if config.wc_features_path.exists():
        df_wc = _load_features(config.wc_features_path)
        predictions = pipeline.predict_wc2026(df_wc, output_dir=str(config.output_dir))

    return metrics, predictions


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the WC2026 XGBoost market pipeline.")
    parser.add_argument("--features", default="data/processed/features_train.csv", help="Training features CSV.")
    parser.add_argument("--wc-features", default="data/processed/features_wc2026.csv", help="WC2026 features CSV.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for metrics and predictions.")
    parser.add_argument("--train-cutoff", type=int, default=2018, help="Train with matches before this year.")
    parser.add_argument("--val-cutoff", type=int, default=2023, help="Validate from train cutoff up to this year.")
    parser.add_argument(
        "--markets",
        default=None,
        help=f"Comma-separated market keys. Available: {', '.join(sorted(MARKETS))}",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = XGBoostRunConfig(
        features_path=Path(args.features),
        wc_features_path=Path(args.wc_features),
        output_dir=Path(args.output_dir),
        train_cutoff=args.train_cutoff,
        val_cutoff=args.val_cutoff,
        markets=_parse_markets(args.markets),
    )
    metrics, predictions = run_xgboost_pipeline(config)
    print(f"Métricas generadas: {len(metrics)} mercados")
    if not predictions.empty:
        print(f"Predicciones WC2026 generadas: {len(predictions)} partidos")


if __name__ == "__main__":
    main()
