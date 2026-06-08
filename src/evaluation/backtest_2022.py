"""
src/evaluation/backtest_2022.py
================================
Backtest honesto sobre el Mundial 2022 (prueba de concepto, 12 partidos).

Mide si la estrategia de value betting (apostar cuando EV = p_modelo*cuota - 1
supera un umbral) habria dado beneficio, usando:
  - Modelo entrenado SOLO con datos < cutoff (sin fugas; NO usa el refit con
    todos los datos de xgb_baseline, que veria los partidos a predecir).
  - Cuotas de CIERRE reales de The Odds API (snapshot ~1h antes del kickoff).
  - Resultado real de cada partido (home_score/away_score de features).

Mercados: 1X2 (h2h) y Over/Under 2.5 (totals).

⚠ COSTE: el endpoint historico gasta 10 creditos por llamada. 12 partidos =
~120 creditos. Usa cache (use_cache=True) para no repetir gasto en re-runs.

Uso:
    python -m src.evaluation.backtest_2022
    python -m src.evaluation.backtest_2022 --min-ev 0.08 --bookmaker pinnacle
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import timedelta

import numpy as np
import pandas as pd

from src.data.team_names import canonicalize
from src.data.historical_odds_collector import (
    HistoricalOddsCollector,
    _parse_utc,
    _iso_z,
)
from src.models.xgb_baseline import MarketModel, MARKETS

log = logging.getLogger(__name__)


# ---------- funciones puras (validadas) ----------
def result_from_score(hs, a_s):
    if hs > a_s: return "H"
    if hs < a_s: return "A"
    return "D"

def ev(p, cuota):
    return p * cuota - 1.0

def profit(win, cuota):
    return (cuota - 1.0) if win else -1.0

def over_win(hs, a_s, side):
    total = hs + a_s
    return total > 2.5 if side == "over" else total < 2.5


def _train_clean_models(df_all, cutoff, markets):
    """Entrena un MarketModel por mercado SOLO con < cutoff. Sin refit."""
    df_all = df_all.copy()
    df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
    train = df_all[df_all["date"].dt.year < cutoff].copy()
    log.info("Train limpio: %d partidos (< %d)", len(train), cutoff)

    models = {}
    for mk in markets:
        config = MARKETS[mk]
        m = MarketModel(config)
        m.fit(train[train[config.target_col].notna()].copy())
        models[mk] = m
    return models


def _model_probs(models, bt):
    """Devuelve dict por indice de bt -> probabilidades del modelo."""
    out = {idx: {} for idx in bt.index}

    # 1X2
    if "result_1x2" in models:
        m = models["result_1x2"]
        proba = m.predict_proba(bt)
        classes = list(m.label_encoder.classes_)  # orden de columnas
        col = {c: i for i, c in enumerate(classes)}
        for n, idx in enumerate(bt.index):
            out[idx]["H"] = float(proba[n, col["H"]])
            out[idx]["D"] = float(proba[n, col["D"]])
            out[idx]["A"] = float(proba[n, col["A"]])

    # Over 2.5
    if "over25" in models:
        proba = models["over25"].predict_proba(bt)
        for n, idx in enumerate(bt.index):
            out[idx]["over25"] = float(proba[n, 1])

    return out


def _closing_odds_for_event(collector, event_id, commence_time, bookmaker, use_cache):
    """Snapshot ~1h antes del kickoff. Devuelve filas planas de cuotas."""
    snap_dt = _parse_utc(commence_time) - timedelta(hours=1)
    result = collector.get_historical_event_snapshot(
        event_id,
        _iso_z(snap_dt),
        markets="h2h,totals",
        bookmakers=(bookmaker,),
        regions=("eu",),
        use_cache=use_cache,
    )
    return collector._flatten_snapshot(result, "closing")


def run(features_path, events_path, cutoff, min_ev, bookmaker, output_path,
        env_path, data_dir, use_cache):
    df_all = pd.read_csv(features_path)

    events = pd.read_csv(events_path)
    if not {"event_id", "commence_time", "home_team", "away_team"}.issubset(events.columns):
        raise ValueError(f"{events_path} no tiene las columnas esperadas de eventos.")

    # Conjunto a predecir: filas de features que casan con los eventos (por pareja canon)
    ev_keys = {}
    for _, e in events.iterrows():
        key = frozenset((canonicalize(e["home_team"]), canonicalize(e["away_team"])))
        ev_keys[key] = {"event_id": e["event_id"], "commence_time": e["commence_time"]}

    df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
    bt_rows = []
    for idx, r in df_all.iterrows():
        key = frozenset((canonicalize(r["home_team"]), canonicalize(r["away_team"])))
        if key in ev_keys and pd.notna(r.get("home_score")) and pd.notna(r.get("away_score")):
            # quedarnos con la fila cuya fecha coincide con el evento (evita partidos repetidos historicos)
            ev_date = _parse_utc(ev_keys[key]["commence_time"]).date()
            if r["date"].date() == ev_date:
                bt_rows.append(idx)
    bt = df_all.loc[bt_rows].copy()
    if bt.empty:
        raise SystemExit("No se cruzo ningun evento con features. Revisa fechas/nombres.")
    log.info("Partidos a backtestear: %d", len(bt))

    # Entrenar modelos limpios y predecir
    models = _train_clean_models(df_all, cutoff, ["result_1x2", "over25"])
    probs = _model_probs(models, bt)

    # Cuotas + EV + scoring
    collector = HistoricalOddsCollector(data_dir=data_dir, env_path=env_path)
    bets = []
    no_odds = []
    for idx, r in bt.iterrows():
        key = frozenset((canonicalize(r["home_team"]), canonicalize(r["away_team"])))
        meta = ev_keys[key]
        rows = _closing_odds_for_event(
            collector, meta["event_id"], meta["commence_time"], bookmaker, use_cache
        )
        if not rows:
            no_odds.append(f"{r['home_team']} vs {r['away_team']}")
            continue

        ens_home = canonicalize(r["home_team"])
        ens_away = canonicalize(r["away_team"])
        hs, a_s = float(r["home_score"]), float(r["away_score"])
        real_1x2 = result_from_score(hs, a_s)

        for o in rows:
            mk = str(o.get("market")).lower()
            sel = o.get("selection")
            cuota = o.get("odds_decimal")
            if cuota is None or float(cuota) <= 1.0:
                continue
            cuota = float(cuota)
            sel_canon = canonicalize(sel) if sel else None

            p = None; etiqueta = None; win = None
            if mk == "h2h":
                if sel_canon == ens_home:
                    p, etiqueta, win = probs[idx]["H"], "1X2-H", (real_1x2 == "H")
                elif sel_canon == ens_away:
                    p, etiqueta, win = probs[idx]["A"], "1X2-A", (real_1x2 == "A")
                elif str(sel).strip().lower() in ("draw", "empate"):
                    p, etiqueta, win = probs[idx]["D"], "1X2-D", (real_1x2 == "D")
            elif mk == "totals":
                line = o.get("line")
                if line is None or abs(float(line) - 2.5) > 1e-6:
                    continue
                s = str(sel).strip().lower()
                if s == "over":
                    p, etiqueta, win = probs[idx]["over25"], "Over2.5", over_win(hs, a_s, "over")
                elif s == "under":
                    p, etiqueta, win = 1.0 - probs[idx]["over25"], "Under2.5", over_win(hs, a_s, "under")

            if p is None:
                continue
            e = ev(p, cuota)
            bets.append({
                "home_team": r["home_team"], "away_team": r["away_team"],
                "mercado": etiqueta, "seleccion": sel, "cuota": round(cuota, 3),
                "prob_modelo": round(p, 4), "ev_pct": round(e * 100, 2),
                "apostada": e >= min_ev,
                "win": bool(win),
                "profit": round(profit(win, cuota), 3) if e >= min_ev else 0.0,
                "resultado_real": f"{int(hs)}-{int(a_s)}",
            })

    if not bets:
        print("No se genero ninguna apuesta. ¿Cuotas vacias o sin cruce?")
        if no_odds:
            print("Sin cuotas para:", no_odds)
        return

    df = pd.DataFrame(bets)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    placed = df[df["apostada"]].copy()
    print("\n" + "=" * 78)
    print(f"BACKTEST WC2022  (umbral EV >= {min_ev*100:.0f}%, libro={bookmaker}, cutoff<{cutoff})")
    print("=" * 78)
    print(f"Partidos: {bt.shape[0]}   |   Cuotas cruzadas: {df['home_team'].nunique()}   |   "
          f"Sin cuotas: {len(no_odds)}")
    print(f"Apuestas con value (EV>={min_ev*100:.0f}%): {len(placed)} de {len(df)} posibles\n")

    if placed.empty:
        print("Ninguna apuesta supero el umbral. Prueba --min-ev mas bajo.")
    else:
        n = len(placed)
        wins = int(placed["win"].sum())
        staked = n * 1.0
        ret = placed["profit"].sum()
        roi = ret / staked * 100
        print(f"  Apostadas:   {n}")
        print(f"  Acertadas:   {wins}  ({wins/n*100:.1f}%)")
        print(f"  Stake total: {staked:.0f}u   |   Beneficio: {ret:+.2f}u   |   ROI: {roi:+.1f}%\n")

        print("  Por mercado:")
        for mk, g in placed.groupby("mercado"):
            r_ = g["profit"].sum() / len(g) * 100
            print(f"    {mk:<10} n={len(g):<3} acierto={g['win'].mean()*100:4.0f}%  ROI={r_:+6.1f}%")

        print("\n  Detalle de apuestas:")
        for _, b in placed.sort_values("ev_pct", ascending=False).iterrows():
            res = "GANA" if b["win"] else "PIERDE"
            print(f"    {b['home_team'][:12]:<12} v {b['away_team'][:12]:<12} {b['mercado']:<9} "
                  f"@{b['cuota']:<5} EV={b['ev_pct']:+5.1f}% -> {res:<6} {b['profit']:+.2f}u  ({b['resultado_real']})")

    print("=" * 78)
    print(f"Guardado: {output_path}")
    if no_odds:
        print(f"⚠ Sin cuotas ({bookmaker}) para {len(no_odds)} partidos: {no_odds}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backtest value betting sobre WC2022.")
    p.add_argument("--features", default="data/processed/features.csv")
    p.add_argument("--events", default="data/processed/odds_events.csv")
    p.add_argument("--cutoff", type=int, default=2022, help="Entrena solo con partidos < este anio.")
    p.add_argument("--min-ev", type=float, default=0.05, help="Umbral de EV para apostar (0.05 = 5%%).")
    p.add_argument("--bookmaker", default="pinnacle", help="Libro para las cuotas de cierre (sharp por defecto).")
    p.add_argument("--output", default="outputs/backtest2022_bets.csv")
    p.add_argument("--env-path", default=".env")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--no-cache", action="store_true", help="Ignora cache (vuelve a gastar creditos).")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = build_arg_parser().parse_args()
    run(
        features_path=a.features, events_path=a.events, cutoff=a.cutoff,
        min_ev=a.min_ev, bookmaker=a.bookmaker, output_path=a.output,
        env_path=a.env_path, data_dir=a.data_dir, use_cache=not a.no_cache,
    )


if __name__ == "__main__":
    main()
    