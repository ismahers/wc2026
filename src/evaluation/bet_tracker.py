"""
Local tracker for WC2026 value signals.

The tracker stores one row per stable signal and preserves the first detected
odds while updating the latest odds on repeated runs. It is intentionally a
local CSV for v1; the same columns can later be moved to Supabase.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os

import pandas as pd


DEFAULT_CORE_INPUT = "outputs/wc2026_ev_h2h_shortlist.csv"
DEFAULT_REVIEW_INPUT = "outputs/wc2026_ev_h2h_manual_review.csv"
DEFAULT_TRACKER_OUTPUT = "data/tracking/wc2026_bet_tracker.csv"

TRACKER_COLUMNS = [
    "signal_id",
    "first_seen_utc",
    "last_seen_utc",
    "match_date",
    "home_team",
    "away_team",
    "market",
    "selection",
    "bookmaker",
    "bucket",
    "recommended_action",
    "review_reason",
    "first_odds",
    "latest_odds",
    "model_prob",
    "fair_odds",
    "ev_pct",
    "fiabilidad_pct",
    "fiabilidad_nivel",
    "stake_units",
    "bet_status",
    "closing_odds",
    "clv_pct",
    "result",
    "profit_units",
    "notes",
]


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _signal_id(row: pd.Series) -> str:
    raw = "|".join([
        str(row.get("date", "")),
        str(row.get("home_team", "")),
        str(row.get("away_team", "")),
        str(row.get("mercado", "")),
        str(row.get("seleccion", "")),
        str(row.get("bookmaker", "")),
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _load_signals(path: str, *, bucket: str, default_action: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "signal_id": df.apply(_signal_id, axis=1),
        "match_date": df.get("date"),
        "home_team": df.get("home_team"),
        "away_team": df.get("away_team"),
        "market": df.get("mercado"),
        "selection": df.get("seleccion"),
        "bookmaker": df.get("bookmaker"),
        "bucket": bucket,
        "recommended_action": df.get("review_action", default_action),
        "review_reason": df.get("review_reason", ""),
        "latest_odds": df.get("cuota"),
        "model_prob": df.get("prob_modelo"),
        "fair_odds": df.get("cuota_justa"),
        "ev_pct": df.get("ev_pct"),
        "fiabilidad_pct": df.get("fiabilidad_pct"),
        "fiabilidad_nivel": df.get("fiabilidad_nivel"),
    })
    if bucket == "core":
        out["recommended_action"] = default_action
        out["review_reason"] = "core_odds_range"
    return out


def load_current_signals(core_input: str, review_input: str) -> pd.DataFrame:
    frames = [
        _load_signals(core_input, bucket="core", default_action="core_candidate"),
        _load_signals(review_input, bucket="manual_review", default_action="paper_only"),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=TRACKER_COLUMNS)
    signals = pd.concat(frames, ignore_index=True)
    return signals.drop_duplicates("signal_id", keep="first").reset_index(drop=True)


def _empty_tracker() -> pd.DataFrame:
    return pd.DataFrame(columns=TRACKER_COLUMNS)


def _compute_clv(row: pd.Series) -> object:
    first = pd.to_numeric(row.get("first_odds"), errors="coerce")
    closing = pd.to_numeric(row.get("closing_odds"), errors="coerce")
    if pd.isna(first) or pd.isna(closing) or first <= 1.0:
        return pd.NA
    return round((first / closing - 1.0) * 100.0, 2)


def update_tracker(
    signals: pd.DataFrame,
    *,
    tracker_path: str = DEFAULT_TRACKER_OUTPUT,
    seen_at_utc: str | None = None,
) -> pd.DataFrame:
    seen_at_utc = seen_at_utc or _now_utc()
    if os.path.exists(tracker_path):
        tracker = pd.read_csv(tracker_path).reindex(columns=TRACKER_COLUMNS)
    else:
        tracker = _empty_tracker()

    tracker = tracker.copy()
    tracker = tracker.set_index("signal_id", drop=False) if not tracker.empty else tracker
    new_rows: list[dict[str, object]] = []
    existing_ids = set(tracker.index) if not tracker.empty else set()

    for _, signal in signals.iterrows():
        signal_id = signal["signal_id"]
        if signal_id not in existing_ids:
            row = {col: pd.NA for col in TRACKER_COLUMNS}
            row.update(signal.to_dict())
            row["first_seen_utc"] = seen_at_utc
            row["last_seen_utc"] = seen_at_utc
            row["first_odds"] = signal.get("latest_odds")
            row["stake_units"] = 0.0
            row["bet_status"] = "candidate"
            row["closing_odds"] = pd.NA
            row["clv_pct"] = pd.NA
            row["result"] = pd.NA
            row["profit_units"] = pd.NA
            row["notes"] = pd.NA
            new_rows.append(row)
            existing_ids.add(signal_id)
            continue

        preserve_cols = {
            "signal_id", "first_seen_utc", "first_odds", "stake_units",
            "bet_status", "closing_odds", "result", "profit_units", "notes",
        }
        for col in signal.index:
            if col not in preserve_cols:
                tracker.at[signal_id, col] = signal[col]
        tracker.at[signal_id, "last_seen_utc"] = seen_at_utc

    if new_rows:
        new_frame = pd.DataFrame(new_rows, columns=TRACKER_COLUMNS).set_index("signal_id", drop=False)
        tracker = new_frame if tracker.empty else pd.concat([tracker, new_frame])

    if tracker.empty:
        out = _empty_tracker()
    else:
        out = tracker.reset_index(drop=True).reindex(columns=TRACKER_COLUMNS)
        out["clv_pct"] = out.apply(_compute_clv, axis=1)
        out = out.sort_values(["bucket", "match_date", "home_team", "away_team"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(tracker_path) or ".", exist_ok=True)
    out.to_csv(tracker_path, index=False)
    return out


def summarize_tracker(tracker: pd.DataFrame) -> pd.DataFrame:
    if tracker.empty:
        return pd.DataFrame([{"rows": 0, "core": 0, "manual_review": 0, "paper_only": 0}])
    return pd.DataFrame([{
        "rows": len(tracker),
        "core": int(tracker["bucket"].eq("core").sum()),
        "manual_review": int(tracker["recommended_action"].eq("manual_check").sum()),
        "paper_only": int(tracker["recommended_action"].eq("paper_only").sum()),
        "candidates": int(tracker["bet_status"].eq("candidate").sum()),
    }])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update local WC2026 bet tracker from strategy outputs.")
    parser.add_argument("--core-input", default=DEFAULT_CORE_INPUT)
    parser.add_argument("--review-input", default=DEFAULT_REVIEW_INPUT)
    parser.add_argument("--tracker-output", default=DEFAULT_TRACKER_OUTPUT)
    parser.add_argument("--seen-at-utc", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    signals = load_current_signals(args.core_input, args.review_input)
    tracker = update_tracker(signals, tracker_path=args.tracker_output, seen_at_utc=args.seen_at_utc)
    print(summarize_tracker(tracker).to_string(index=False))
    print(f"Guardado en {args.tracker_output}")


if __name__ == "__main__":
    main()
