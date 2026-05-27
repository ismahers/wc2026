"""
src/evaluation/betting.py
==========================
Detección de valor esperado y simulación de ROI en apuestas.

Funciones principales:
  - expected_value: calcula VE para cada resultado y mercado
  - find_value_bets: filtra apuestas con VE positivo
  - simulate_roi: simula apuestas en un conjunto histórico
  - calibration_report: genera reliability diagram y ECE
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── Valor Esperado ────────────────────────────────────────────────────────────

def expected_value(prob: float, odds: float) -> float:
    """
    Calcula el valor esperado de una apuesta.

    VE = P_modelo × cuota_decimal - 1

    VE > 0 → apuesta con valor positivo (la casa subestima la probabilidad)
    VE < 0 → apuesta sin valor (la casa sobreestima)

    Parámetros
    ----------
    prob : probabilidad estimada por el modelo (0-1)
    odds : cuota decimal ofrecida por la casa (ej: 2.50)
    """
    return prob * odds - 1


def add_no_vig_probability(
    odds: pd.DataFrame,
    group_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Add margin-adjusted probabilities within each market group.

    The default grouping treats each match/bookmaker/market/line as one book.
    Example: h2h has home/draw/away outcomes; totals has over/under at a line.
    """
    if odds.empty:
        return odds.copy()

    group_cols = group_cols or ["match_id", "bookmaker", "market", "line"]
    result = odds.copy()
    result["implied_probability"] = 1 / result["odds_decimal"].astype(float)
    denom = result.groupby(group_cols, dropna=False)["implied_probability"].transform("sum")
    result["no_vig_probability"] = result["implied_probability"] / denom
    result["book_overround"] = denom - 1
    return result


def prepare_h2h_odds_matrix(odds: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot flat h2h odds into one row per match with H/D/A odds and no-vig probs.

    Labels follow the model convention:
      H = home team
      D = draw
      A = away team
    """
    if odds.empty:
        return pd.DataFrame()

    h2h = odds[odds["market"] == "h2h"].copy()
    if h2h.empty:
        return pd.DataFrame()

    h2h = add_no_vig_probability(h2h)

    def _side(row: pd.Series) -> str | None:
        selection = str(row["selection"])
        if selection == str(row["home_team"]):
            return "H"
        if selection == str(row["away_team"]):
            return "A"
        if selection.lower() == "draw":
            return "D"
        return None

    h2h["side"] = h2h.apply(_side, axis=1)
    h2h = h2h.dropna(subset=["side"])
    index_cols = ["match_id", "match_date", "home_team", "away_team", "bookmaker"]
    odds_wide = h2h.pivot_table(
        index=index_cols,
        columns="side",
        values="odds_decimal",
        aggfunc="first",
    ).rename(columns={"H": "odds_H", "D": "odds_D", "A": "odds_A"})
    prob_wide = h2h.pivot_table(
        index=index_cols,
        columns="side",
        values="no_vig_probability",
        aggfunc="first",
    ).rename(columns={"H": "prob_market_H", "D": "prob_market_D", "A": "prob_market_A"})
    overround = h2h.groupby(index_cols, dropna=False)["book_overround"].first().rename("book_overround")
    return odds_wide.join(prob_wide).join(overround).reset_index()


def prepare_totals_odds_matrix(odds: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot totals odds into one row per match/bookmaker/line with Over/Under.
    """
    if odds.empty:
        return pd.DataFrame()

    totals = odds[odds["market"] == "totals"].copy()
    if totals.empty:
        return pd.DataFrame()

    totals = add_no_vig_probability(totals)
    totals["side"] = totals["selection"].astype(str).str.lower().map({"over": "Over", "under": "Under"})
    totals = totals.dropna(subset=["side"])

    index_cols = ["match_id", "match_date", "home_team", "away_team", "bookmaker", "line"]
    odds_wide = totals.pivot_table(
        index=index_cols,
        columns="side",
        values="odds_decimal",
        aggfunc="first",
    ).rename(columns={"Over": "odds_over", "Under": "odds_under"})
    prob_wide = totals.pivot_table(
        index=index_cols,
        columns="side",
        values="no_vig_probability",
        aggfunc="first",
    ).rename(columns={"Over": "prob_market_over", "Under": "prob_market_under"})
    overround = totals.groupby(index_cols, dropna=False)["book_overround"].first().rename("book_overround")
    return odds_wide.join(prob_wide).join(overround).reset_index()


def compare_1x2_predictions_to_odds(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    *,
    min_ev: float = 0.0,
) -> pd.DataFrame:
    """
    Compare model 1X2 probabilities against flat h2h odds and return EV rows.

    Expected prediction columns: date, home_team, away_team, prob_H, prob_D, prob_A.
    """
    if predictions.empty or odds.empty:
        return pd.DataFrame()

    h2h = prepare_h2h_odds_matrix(odds)
    if h2h.empty:
        return pd.DataFrame()

    pred = predictions.copy()
    pred["date"] = pd.to_datetime(pred["date"], errors="coerce", utc=True).dt.date
    h2h["date"] = pd.to_datetime(h2h["match_date"], errors="coerce", utc=True).dt.date
    merged = pred.merge(h2h, on=["date", "home_team", "away_team"], how="inner", suffixes=("_model", "_odds"))

    rows = []
    for _, row in merged.iterrows():
        for side in ("H", "D", "A"):
            prob = row.get(f"prob_{side}")
            odds_decimal = row.get(f"odds_{side}")
            if pd.isna(prob) or pd.isna(odds_decimal):
                continue
            ev = expected_value(float(prob), float(odds_decimal))
            if ev < min_ev:
                continue
            rows.append({
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "bookmaker": row["bookmaker"],
                "selection": side,
                "prob_model": float(prob),
                "prob_market_no_vig": row.get(f"prob_market_{side}"),
                "odds_decimal": float(odds_decimal),
                "expected_value": ev,
                "book_overround": row.get("book_overround"),
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("expected_value", ascending=False).reset_index(drop=True)
    return result


def compare_over_predictions_to_odds(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    *,
    model_prob_col: str = "prob_over25",
    target_line: float = 2.5,
    min_ev: float = 0.0,
) -> pd.DataFrame:
    """
    Compare model Over probability against totals odds at a target line.

    Expected prediction columns: date, home_team, away_team, and model_prob_col.
    """
    if predictions.empty or odds.empty:
        return pd.DataFrame()

    totals = prepare_totals_odds_matrix(odds)
    if totals.empty:
        return pd.DataFrame()

    totals = totals[np.isclose(totals["line"].astype(float), float(target_line))].copy()
    if totals.empty:
        return pd.DataFrame()

    pred = predictions.copy()
    pred["date"] = pd.to_datetime(pred["date"], errors="coerce", utc=True).dt.date
    totals["date"] = pd.to_datetime(totals["match_date"], errors="coerce", utc=True).dt.date
    merged = pred.merge(totals, on=["date", "home_team", "away_team"], how="inner", suffixes=("_model", "_odds"))

    rows = []
    for _, row in merged.iterrows():
        prob = row.get(model_prob_col)
        odds_decimal = row.get("odds_over")
        if pd.isna(prob) or pd.isna(odds_decimal):
            continue
        ev = expected_value(float(prob), float(odds_decimal))
        if ev < min_ev:
            continue
        rows.append({
            "date": row["date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "bookmaker": row["bookmaker"],
            "market": f"over_{target_line:g}",
            "prob_model": float(prob),
            "prob_market_no_vig": row.get("prob_market_over"),
            "odds_decimal": float(odds_decimal),
            "expected_value": ev,
            "book_overround": row.get("book_overround"),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("expected_value", ascending=False).reset_index(drop=True)
    return result


def find_value_bets(
    proba_matrix: np.ndarray,
    odds_matrix: np.ndarray,
    labels: list,
    min_ve: float = 0.05,
    min_prob: float = 0.10,
) -> pd.DataFrame:
    """
    Encuentra apuestas con valor esperado positivo.

    Parámetros
    ----------
    proba_matrix : (n_matches, n_outcomes) — probabilidades del modelo
    odds_matrix  : (n_matches, n_outcomes) — cuotas decimales de la casa
    labels       : nombres de los outcomes (ej. ["A", "D", "H"])
    min_ve       : umbral mínimo de VE para considerar apuesta (default 5%)
    min_prob     : probabilidad mínima para apostar (evitar longshots)

    Retorna
    -------
    DataFrame con apuestas de valor, ordenado por VE descendente
    """
    rows = []
    for i in range(proba_matrix.shape[0]):
        for j, label in enumerate(labels):
            p    = proba_matrix[i, j]
            odd  = odds_matrix[i, j]
            if np.isnan(p) or np.isnan(odd) or odd <= 1:
                continue
            ve = expected_value(p, odd)
            if ve >= min_ve and p >= min_prob:
                rows.append({
                    "match_idx":  i,
                    "outcome":    label,
                    "prob_model": round(p, 4),
                    "prob_house": round(1 / odd, 4),
                    "odds":       round(odd, 2),
                    "ve":         round(ve, 4),
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("ve", ascending=False).reset_index(drop=True)
    return df


# ── Simulación de ROI ─────────────────────────────────────────────────────────

def simulate_roi(
    value_bets: pd.DataFrame,
    actual_results: np.ndarray,
    labels: list,
    stake: float = 1.0,
    strategy: str = "flat",
) -> dict:
    """
    Simula el resultado de apostar en todas las apuestas con valor detectadas.

    Parámetros
    ----------
    value_bets     : DataFrame de find_value_bets
    actual_results : array con el resultado real de cada partido (índice = match_idx)
    labels         : lista de outcomes posibles
    stake          : importe apostado por apuesta (flat betting)
    strategy       : "flat" (stake fijo) o "kelly" (Kelly criterion)

    Retorna
    -------
    dict con métricas: n_bets, total_staked, profit, roi, win_rate
    """
    if value_bets.empty:
        return {"n_bets": 0, "roi": 0.0, "profit": 0.0, "win_rate": 0.0}

    profits = []
    for _, bet in value_bets.iterrows():
        idx    = int(bet["match_idx"])
        actual = actual_results[idx] if idx < len(actual_results) else None
        won    = (actual == bet["outcome"])

        if strategy == "kelly":
            p   = bet["prob_model"]
            b   = bet["odds"] - 1
            k   = max(0, (b * p - (1 - p)) / b)
            s   = stake * k
        else:
            s = stake

        profit = s * (bet["odds"] - 1) if won else -s
        profits.append({"stake": s, "profit": profit, "won": won})

    profits_df    = pd.DataFrame(profits)
    total_staked  = profits_df["stake"].sum()
    total_profit  = profits_df["profit"].sum()
    roi           = total_profit / total_staked if total_staked > 0 else 0.0
    win_rate      = profits_df["won"].mean()

    return {
        "n_bets":       len(profits_df),
        "total_staked": round(total_staked, 2),
        "profit":       round(total_profit, 2),
        "roi":          round(roi, 4),
        "win_rate":     round(win_rate, 4),
    }


# ── Calibración ───────────────────────────────────────────────────────────────

def calibration_report(
    proba: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
    label: str = "Modelo",
    save_path: Optional[str] = None,
) -> dict:
    """
    Genera reliability diagram y calcula ECE (Expected Calibration Error).

    Parámetros
    ----------
    proba   : probabilidades predichas para la clase positiva (1D array)
    y_true  : etiquetas binarias reales (1D array)
    n_bins  : número de bins para el diagrama

    Retorna
    -------
    dict con ece y fraction_positives por bin
    """
    import matplotlib.pyplot as plt

    bins       = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    ece = 0.0
    bin_data = []

    for lower, upper in zip(bin_lowers, bin_uppers):
        mask = (proba >= lower) & (proba < upper)
        if mask.sum() == 0:
            continue
        avg_conf  = proba[mask].mean()
        avg_acc   = y_true[mask].mean()
        bin_size  = mask.sum()
        ece      += (bin_size / len(proba)) * abs(avg_conf - avg_acc)
        bin_data.append({
            "confidence": avg_conf,
            "accuracy":   avg_acc,
            "count":      bin_size,
        })

    # Reliability diagram
    fig, ax = plt.subplots(figsize=(6, 6))
    if bin_data:
        confs = [b["confidence"] for b in bin_data]
        accs  = [b["accuracy"]   for b in bin_data]
        ax.bar(confs, accs, width=1/n_bins, alpha=0.7, label=label, align="center")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectamente calibrado")
    ax.set_xlabel("Confianza del modelo")
    ax.set_ylabel("Fracción de positivos")
    ax.set_title(f"Reliability Diagram — ECE: {ece:.4f}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  → Reliability diagram guardado en {save_path}")
    plt.show()

    return {"ece": round(ece, 4), "bin_data": bin_data}


# ── Script standalone ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Ejemplo de uso:")
    print()

    # Simulación sencilla
    np.random.seed(42)
    n = 100
    proba  = np.random.dirichlet([2, 1, 3], size=n)   # (n, 3) — A, D, H
    odds   = 1 / proba * 1.05                          # cuotas con margen del 5%
    actual = np.random.choice(["A", "D", "H"], size=n, p=[0.27, 0.25, 0.48])

    # Detectar valor (ninguna debería tener VE>0 con margen del 5%)
    vbets = find_value_bets(proba, odds, labels=["A", "D", "H"], min_ve=0.0)
    print(f"Apuestas con VE >= 0: {len(vbets)}")

    # Simular ROI
    roi_result = simulate_roi(vbets, actual, labels=["A", "D", "H"])
    print(f"ROI simulado: {roi_result['roi']:.2%}")
    print(f"Beneficio:    {roi_result['profit']:.2f} unidades")
