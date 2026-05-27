"""Model entry points for WC2026 market prediction."""

from .xgboost_model import XGBoostRunConfig, run_xgboost_pipeline

__all__ = ["XGBoostRunConfig", "run_xgboost_pipeline"]
