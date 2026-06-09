"""
src/evaluation/paper_tracker.py
===============================
Local paper tracker for non-v1 markets.

This tracker is deliberately separate from bet_tracker.py. It stores no real
stake recommendations; it is for CLV/result tracking before a market can be
promoted from paper to staking.

Usage:
    python -m src.evaluation.paper_tracker
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os

import pandas as pd


DEFAULT_PAPER_INPUT = "outputs/wc2026_multi_market_paper_shortlist.csv"
DEFAULT_REVIEW_INPUT = "outputs/wc2026_multi_market_review.csv"
DEFAULT_TRACKER_OUTPUT = "data/tracking/wc2026_paper_tracker.csv"
DEFAULT_PAPER_STAKE_UNITS = 1.0

PAPER_TRACKER_COLUMNS = [
    "paper_signal_id",
    "first_seen_utc",
    "last_seen_utc",
    "match_date",
    "home_team",
    "away_team",
    "market",
    "registry_market_key",
    "registry_display_name",
    "selection",
    "line",
    "bookmaker",
    "bucket",
    "tracking_action",
    "validation_status",
    "betting_status",
    "stake_allowed",
    "first_odds",
    "latest_odds",
    "model_prob",
    "fair_odds",
    "ev_pct",
    "fiabilidad_pct",
    "fiabilidad_nivel",
    "model_confidence",
    "paper_stake_units",
    "paper_status",
    "closing_odds",
    "clv_pct",
    "result",
    "paper_profit_units",
    "notes",
]


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _paper_signal_id(row: pd.Series) -> str:
    raw = "|".join([
        str(row.get("date", "")),
        str(row.get("home_team", "")),
        str(row.get("away_team", "")),
        str(row.get("market", "")),
        str(row.get("selection", "")),
        str(row.get("line", "")),
        str(row.get("bookmaker", "")),
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _load_signals(path: str, *, bucket: str, default_action: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()

    return pd.DataFrame({
        "paper_signal_id": df.apply(_paper_signal_id, axis=1),
        "match_date": df.get("date"),
        "home_team": df.get("home_team"),
        "away_team": df.get("away_team"),
        "market": df.get("market"),
        "registry_market_key": df.get("registry_market_key"),
        "registry_display_name": df.get("registry_display_name"),
        "selection": df.get("selection"),
        "line": df.get("line"),
        "bookmaker": df.get("bookmaker"),
        "bucket": bucket,
        "tracking_action": df.get("tracking_action", default_action),
        "validation_status": df.get("validation_status"),
        "betting_status": df.get("betting_status"),
        "stake_allowed": df.get("stake_allowed"),
        "latest_odds": df.get("odds_decimal"),
        "model_prob": df.get("model_probability"),
        "fair_odds": df.get("fair_odds"),
        "ev_pct": df.get("ev_pct"),
        "fiabilidad_pct": df.get("fiabilidad_pct"),
        "fiabilidad_nivel": df.get("fiabilidad_nivel"),
        "model_confidence": df.get("model_confidence"),
    })


def load_current_paper_signals(paper_input: str, review_input: str | None = None) -> pd.DataFrame:
    frames = [
        _load_signals(paper_input, bucket="paper", default_action="paper_track"),
    ]
    if review_input:
        frames.append(_load_signals(review_input, bucket="review", default_action="review_only"))

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=PAPER_TRACKER_COLUMNS)
    signals = pd.concat(frames, ignore_index=True)
    return signals.drop_duplicates("paper_signal_id", keep="first").reset_index(drop=True)


def _empty_tracker() -> pd.DataFrame:
    return pd.DataFrame(columns=PAPER_TRACKER_COLUMNS)


def _compute_clv(row: pd.Series) -> object:
    first = pd.to_numeric(row.get("first_odds"), errors="coerce")
    closing = pd.to_numeric(row.get("closing_odds"), errors="coerce")
    if pd.isna(first) or pd.isna(closing) or first <= 1.0 or closing <= 1.0:
        return pd.NA
    return round((first / closing - 1.0) * 100.0, 2)


def _compute_paper_profit(row: pd.Series) -> object:
    raw_result = row.get("result")
    if raw_result is None or pd.isna(raw_result):
        return pd.NA
    result = str(raw_result).strip().casefold()
    stake = pd.to_numeric(row.get("paper_stake_units"), errors="coerce")
    odds = pd.to_numeric(row.get("first_odds"), errors="coerce")
    if pd.isna(stake) or stake <= 0 or pd.isna(odds) or odds <= 1.0:
        return pd.NA
    if result in {"win", "won", "gana", "ganada", "true", "1"}:
        return round((odds - 1.0) * stake, 3)
    if result in {"loss", "lost", "pierde", "perdida", "false", "0"}:
        return round(-stake, 3)
    if result in {"void", "push", "nula", "cancelled", "cancelada"}:
        return 0.0
    return pd.NA


def update_paper_tracker(
    signals: pd.DataFrame,
    *,
    tracker_path: str = DEFAULT_TRACKER_OUTPUT,
    seen_at_utc: str | None = None,
    paper_stake_units: float = DEFAULT_PAPER_STAKE_UNITS,
) -> pd.DataFrame:
    seen_at_utc = seen_at_utc or _now_utc()
    if os.path.exists(tracker_path):
        tracker = pd.read_csv(tracker_path).reindex(columns=PAPER_TRACKER_COLUMNS)
    else:
        tracker = _empty_tracker()

    tracker = tracker.copy()
    for col in ["paper_status", "result", "paper_profit_units", "notes", "tracking_action"]:
        if col in tracker.columns:
            tracker[col] = tracker[col].astype("object")
    tracker = tracker.set_index("paper_signal_id", drop=False) if not tracker.empty else tracker
    existing_ids = set(tracker.index) if not tracker.empty else set()
    new_rows: list[dict[str, object]] = []

    for _, signal in signals.iterrows():
        signal_id = signal["paper_signal_id"]
        default_stake = paper_stake_units if str(signal.get("tracking_action")) == "paper_track" else 0.0

        if signal_id not in existing_ids:
            row = {col: pd.NA for col in PAPER_TRACKER_COLUMNS}
            row.update(signal.to_dict())
            row["first_seen_utc"] = seen_at_utc
            row["last_seen_utc"] = seen_at_utc
            row["first_odds"] = signal.get("latest_odds")
            row["paper_stake_units"] = default_stake
            row["paper_status"] = "paper_candidate"
            row["closing_odds"] = pd.NA
            row["clv_pct"] = pd.NA
            row["result"] = pd.NA
            row["paper_profit_units"] = pd.NA
            row["notes"] = pd.NA
            new_rows.append(row)
            existing_ids.add(signal_id)
            continue

        preserve_cols = {
            "paper_signal_id", "first_seen_utc", "first_odds", "paper_stake_units",
            "paper_status", "closing_odds", "result", "paper_profit_units", "notes",
        }
        for col in signal.index:
            if col not in preserve_cols:
                tracker.at[signal_id, col] = signal[col]
        tracker.at[signal_id, "last_seen_utc"] = seen_at_utc
        if str(tracker.at[signal_id, "paper_status"]) == "paper_candidate":
            current_stake = pd.to_numeric(tracker.at[signal_id, "paper_stake_units"], errors="coerce")
            if pd.isna(current_stake):
                tracker.at[signal_id, "paper_stake_units"] = default_stake

    if new_rows:
        new_frame = pd.DataFrame(new_rows, columns=PAPER_TRACKER_COLUMNS).set_index("paper_signal_id", drop=False)
        tracker = new_frame if tracker.empty else pd.concat([tracker, new_frame])

    if tracker.empty:
        out = _empty_tracker()
    else:
        out = tracker.reset_index(drop=True).reindex(columns=PAPER_TRACKER_COLUMNS)
        out["clv_pct"] = out.apply(_compute_clv, axis=1)
        out["paper_profit_units"] = out.apply(_compute_paper_profit, axis=1)
        out = out.sort_values(["bucket", "match_date", "home_team", "away_team", "market"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(tracker_path) or ".", exist_ok=True)
    out.to_csv(tracker_path, index=False)
    return out


def summarize_paper_tracker(tracker: pd.DataFrame) -> pd.DataFrame:
    if tracker.empty:
        return pd.DataFrame([{
            "rows": 0,
            "paper_track": 0,
            "review_only": 0,
            "paper_stake_units_total": 0.0,
            "settled": 0,
            "paper_profit_units": 0.0,
        }])
    profit = pd.to_numeric(tracker["paper_profit_units"], errors="coerce").fillna(0.0)
    return pd.DataFrame([{
        "rows": len(tracker),
        "paper_track": int(tracker["tracking_action"].eq("paper_track").sum()),
        "review_only": int(tracker["tracking_action"].eq("review_only").sum()),
        "paper_stake_units_total": round(float(pd.to_numeric(tracker["paper_stake_units"], errors="coerce").fillna(0).sum()), 3),
        "settled": int(tracker["result"].notna().sum()),
        "paper_profit_units": round(float(profit.sum()), 3),
    }])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update local paper tracker for non-v1 markets.")
    parser.add_argument("--paper-input", default=DEFAULT_PAPER_INPUT)
    parser.add_argument("--review-input", default=DEFAULT_REVIEW_INPUT)
    parser.add_argument("--tracker-output", default=DEFAULT_TRACKER_OUTPUT)
    parser.add_argument("--seen-at-utc", default=None)
    parser.add_argument("--paper-stake-units", type=float, default=DEFAULT_PAPER_STAKE_UNITS)
    parser.add_argument("--no-review", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    signals = load_current_paper_signals(
        args.paper_input,
        None if args.no_review else args.review_input,
    )
    tracker = update_paper_tracker(
        signals,
        tracker_path=args.tracker_output,
        seen_at_utc=args.seen_at_utc,
        paper_stake_units=args.paper_stake_units,
    )
    print(summarize_paper_tracker(tracker).to_string(index=False))
    print(f"Guardado en {args.tracker_output}")


if __name__ == "__main__":
    main()
