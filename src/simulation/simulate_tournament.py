# -*- coding: utf-8 -*-
"""
simulate_tournament.py
======================
Simula el Mundial 2026 completo y predice UN ganador (modo determinista),
pensado para divulgacion (post). NO toca nada del pipeline: solo LEE outputs
ya generados.

Que usa:
  - outputs/wc2026_ensemble_predictions.csv : las 72 predicciones reales del
        ensemble (final_prob_H/D/A + lambda_home/lambda_away por partido).
  - data/raw/group_stage_wc2026.csv          : grupos y emparejamientos.
  - data/raw/knockout_wc2026.csv             : estructura del cuadro (slots).

Como decide:
  - FASE DE GRUPOS: usa directamente la prediccion del modelo de cada partido
        real (resultado mas probable). Clasificacion por puntos; desempate por
        diferencia de goles esperada (de las lambdas) y luego goles esperados.
  - ELIMINATORIAS: los cruces no existen en ningun CSV (son slots), asi que la
        probabilidad de cada cruce se calcula con la FUERZA del propio modelo:
        ataque/defensa de cada equipo derivados de las lambdas de sus partidos
        de grupo, combinados en un Poisson neutral (campo neutral, sin empate:
        en eliminatoria el empate se reparte segun fuerza -> prorroga/penaltis).

Salida:
  - Consola: grupos resueltos + cuadro completo hasta el campeon.
  - outputs/tournament_prediction.csv : todos los partidos con su ganador.

Uso:
    python src/simulation/simulate_tournament.py
    python src/simulation/simulate_tournament.py --mc 10000   (control opcional)

El modo --mc NO cambia la prediccion del post: solo dice en que % de
simulaciones aleatorias sale ese mismo campeon (robustez interna).
"""

import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

# ---- Rutas (relativas a la raiz del repo) ----
ENSEMBLE_CSV = "outputs/wc2026_ensemble_predictions.csv"
GROUPS_CSV = "data/raw/group_stage_wc2026.csv"
KNOCKOUT_CSV = "data/raw/knockout_wc2026.csv"
OUT_CSV = "outputs/tournament_prediction.csv"

MAX_GOALS = 10  # truncado del Poisson para la matriz de marcadores


# ===========================================================================
# Carga de datos
# ===========================================================================
def load_inputs():
    for path in (ENSEMBLE_CSV, GROUPS_CSV, KNOCKOUT_CSV):
        if not os.path.exists(path):
            sys.exit(
                "[ERROR] No encuentro '%s'. Ejecuta este script desde la raiz "
                "del repo (la carpeta que contiene 'outputs/' y 'data/')." % path
            )

    ens = pd.read_csv(ENSEMBLE_CSV)
    groups = pd.read_csv(GROUPS_CSV)
    knockout = pd.read_csv(KNOCKOUT_CSV)

    need = {"home_team", "away_team", "final_prob_H", "final_prob_D",
            "final_prob_A", "lambda_home", "lambda_away"}
    missing = need - set(ens.columns)
    if missing:
        sys.exit("[ERROR] Al ensemble le faltan columnas: %s" % sorted(missing))

    return ens, groups, knockout


def build_match_lookup(ens):
    """(home, away) -> dict con probabilidades y lambdas del partido real."""
    lut = {}
    for _, r in ens.iterrows():
        lut[(r["home_team"], r["away_team"])] = {
            "pH": float(r["final_prob_H"]),
            "pD": float(r["final_prob_D"]),
            "pA": float(r["final_prob_A"]),
            "lh": float(r["lambda_home"]),
            "la": float(r["lambda_away"]),
        }
    return lut


# ===========================================================================
# Fuerza por equipo (para eliminatorias) derivada de las lambdas del modelo
# ===========================================================================
def derive_team_strength(ens):
    """
    attack[t]  = goles esperados que t MARCA (media de sus lambdas ofensivas)
    defense[t] = goles esperados que t ENCAJA (media de sus lambdas defensivas)
    Cuanto menor 'defense', mejor defensa.
    """
    scored = defaultdict(list)
    conceded = defaultdict(list)
    for _, r in ens.iterrows():
        h, a = r["home_team"], r["away_team"]
        lh, la = float(r["lambda_home"]), float(r["lambda_away"])
        scored[h].append(lh)
        conceded[h].append(la)
        scored[a].append(la)
        conceded[a].append(lh)

    attack, defense = {}, {}
    for t in scored:
        attack[t] = float(np.mean(scored[t]))
        defense[t] = float(np.mean(conceded[t]))

    all_lams = []
    for _, r in ens.iterrows():
        all_lams.append(float(r["lambda_home"]))
        all_lams.append(float(r["lambda_away"]))
    global_mean = float(np.mean(all_lams)) if all_lams else 1.3

    return attack, defense, global_mean


def neutral_lambdas(team_a, team_b, attack, defense, global_mean):
    """Goles esperados de A y B en campo neutral (modelo multiplicativo)."""
    gm = global_mean if global_mean > 1e-6 else 1.3
    lam_a = attack.get(team_a, gm) * defense.get(team_b, gm) / gm
    lam_b = attack.get(team_b, gm) * defense.get(team_a, gm) / gm
    # cota de seguridad
    lam_a = float(np.clip(lam_a, 0.2, 5.0))
    lam_b = float(np.clip(lam_b, 0.2, 5.0))
    return lam_a, lam_b


def poisson_outcome_probs(lam_a, lam_b):
    """P(gana A), P(empate), P(gana B) con dos Poisson independientes."""
    ka = np.arange(0, MAX_GOALS + 1)
    fact = np.array([math.factorial(int(k)) for k in ka], dtype=float)
    pa = np.exp(-lam_a) * lam_a ** ka / fact
    pb = np.exp(-lam_b) * lam_b ** ka / fact
    pa /= pa.sum()
    pb /= pb.sum()
    mat = np.outer(pa, pb)
    p_a = np.tril(mat, -1).sum()   # filas (A) > columnas (B)
    p_b = np.triu(mat, 1).sum()
    p_draw = np.trace(mat)
    return float(p_a), float(p_draw), float(p_b)


def knockout_advance_prob(p_a, p_draw, p_b):
    """En eliminatoria no hay empate: se reparte segun fuerza relativa."""
    base = p_a + p_b
    if base < 1e-9:
        return 0.5
    return p_a + p_draw * (p_a / base)


# ===========================================================================
# Fase de grupos
# ===========================================================================
def resolve_groups(groups, lut, rng=None):
    """
    Devuelve:
      standings: group -> lista ordenada de dicts {team, pts, egd, egf}
    Si rng is None: determinista (resultado mas probable de cada partido).
    Si rng dado: muestrea el resultado de cada partido (para Montecarlo).
    """
    pts = defaultdict(float)
    egd = defaultdict(float)   # diferencia de goles esperada (tiebreak)
    egf = defaultdict(float)   # goles esperados a favor (tiebreak 2)
    team_group = {}

    for _, m in groups.iterrows():
        g = str(m["group"])
        h, a = m["home_team"], m["away_team"]
        team_group[h] = g
        team_group[a] = g
        info = lut.get((h, a))
        if info is None:
            # por si el fixture lista el orden invertido respecto al ensemble
            inv = lut.get((a, h))
            if inv is not None:
                info = {"pH": inv["pA"], "pD": inv["pD"], "pA": inv["pH"],
                        "lh": inv["la"], "la": inv["lh"]}
        if info is None:
            continue

        # diferencia de goles esperada del partido
        egd[h] += info["lh"] - info["la"]
        egd[a] += info["la"] - info["lh"]
        egf[h] += info["lh"]
        egf[a] += info["la"]

        if rng is None:
            outcome = int(np.argmax([info["pH"], info["pD"], info["pA"]]))
        else:
            outcome = rng.choice(3, p=_norm3(info["pH"], info["pD"], info["pA"]))

        if outcome == 0:      # gana local
            pts[h] += 3
        elif outcome == 1:    # empate
            pts[h] += 1
            pts[a] += 1
        else:                 # gana visitante
            pts[a] += 3

    standings = defaultdict(list)
    for team, g in team_group.items():
        standings[g].append({
            "team": team, "pts": pts[team],
            "egd": round(egd[team], 3), "egf": round(egf[team], 3),
        })
    for g in standings:
        standings[g].sort(key=lambda d: (d["pts"], d["egd"], d["egf"]), reverse=True)
    return standings


def _norm3(a, b, c):
    s = a + b + c
    if s <= 0:
        return [1 / 3, 1 / 3, 1 / 3]
    return [a / s, b / s, c / s]


def best_thirds(standings, n=8):
    """Top-n terceros de los 12 grupos."""
    thirds = []
    for g, lst in standings.items():
        if len(lst) >= 3:
            d = dict(lst[2])
            d["group"] = g
            thirds.append(d)
    thirds.sort(key=lambda d: (d["pts"], d["egd"], d["egf"]), reverse=True)
    return thirds[:n]


# ===========================================================================
# Asignacion de terceros a slots (respeta los grupos permitidos de cada slot)
# ===========================================================================
def assign_thirds_to_slots(third_slots, qualified_thirds):
    """
    third_slots: lista de (match_number, side, allowed_groups_set)
    qualified_thirds: lista de dicts con 'team' y 'group'
    Devuelve dict slot_key -> team via backtracking (matching valido).
    slot_key = (match_number, side)
    """
    slots = list(third_slots)
    thirds = list(qualified_thirds)
    assignment = {}
    used = [False] * len(thirds)

    # ordenar slots por nº de grupos permitidos (mas restrictivos primero)
    order = sorted(range(len(slots)), key=lambda i: len(slots[i][2]))

    def backtrack(pos):
        if pos == len(order):
            return True
        si = order[pos]
        mnum, side, allowed = slots[si]
        for ti, th in enumerate(thirds):
            if not used[ti] and th["group"] in allowed:
                used[ti] = True
                assignment[(mnum, side)] = th["team"]
                if backtrack(pos + 1):
                    return True
                used[ti] = False
                del assignment[(mnum, side)]
        return False

    ok = backtrack(0)
    if not ok:
        # fallback: asignacion directa por orden (no deberia hacer falta)
        free = [th for i, th in enumerate(thirds) if not used[i]]
        for si in order:
            mnum, side, _ = slots[si]
            if (mnum, side) not in assignment and free:
                assignment[(mnum, side)] = free.pop(0)["team"]
    return assignment


# ===========================================================================
# Resolucion del cuadro
# ===========================================================================
def parse_slot(slot):
    """Clasifica un slot. Devuelve ('pos', pos, group) | ('third', groups) |
       ('winner', mnum) | ('loser', mnum)."""
    s = str(slot).strip()
    if s.startswith("Winner "):
        return ("winner", int(s.split()[1]))
    if s.startswith("Loser "):
        return ("loser", int(s.split()[1]))
    if "/" in s:  # tercero: '3A/B/C/D/F'
        head = s[0]  # '3'
        groups = set()
        body = s[1:]
        for part in body.split("/"):
            groups.add(part.strip())
        return ("third", groups)
    # '1A', '2B'
    return ("pos", int(s[0]), s[1:])


def resolve_bracket(knockout, standings, thirds, lut, attack, defense, gm,
                    rng=None):
    """Resuelve todo el cuadro. Devuelve (results, rows)."""
    group_winner = {g: standings[g][0]["team"] for g in standings if standings[g]}
    group_runner = {g: standings[g][1]["team"] for g in standings if len(standings[g]) > 1}

    # slots de terceros del CSV
    third_slots = []
    for _, m in knockout.iterrows():
        for side in ("home_slot", "away_slot"):
            kind = parse_slot(m[side])
            if kind[0] == "third":
                third_slots.append((int(m["match_number"]), side, kind[1]))
    third_assign = assign_thirds_to_slots(third_slots, thirds)

    results = {}  # match_number -> {"winner":, "loser":, "home":, "away":, ...}
    rows = []

    kk = knockout.sort_values("match_number")
    for _, m in kk.iterrows():
        mnum = int(m["match_number"])
        stage = m["stage"]

        home = _resolve_side(m["home_slot"], group_winner, group_runner,
                             third_assign, results, mnum, "home_slot")
        away = _resolve_side(m["away_slot"], group_winner, group_runner,
                             third_assign, results, mnum, "away_slot")

        if home is None or away is None:
            # algo no resoluble; saltar con marcador vacio
            results[mnum] = {"winner": home or away, "loser": None,
                             "home": home, "away": away, "p": None}
            continue

        # probabilidad de que pase 'home' (neutral, fuerza del modelo)
        lam_h, lam_a = neutral_lambdas(home, away, attack, defense, gm)
        p_h, p_d, p_a = poisson_outcome_probs(lam_h, lam_a)
        p_home_adv = knockout_advance_prob(p_h, p_d, p_a)

        if rng is None:
            home_wins = p_home_adv >= 0.5
        else:
            home_wins = rng.random() < p_home_adv

        winner = home if home_wins else away
        loser = away if home_wins else home

        results[mnum] = {"winner": winner, "loser": loser,
                         "home": home, "away": away,
                         "p": round(p_home_adv if home_wins else 1 - p_home_adv, 3)}
        rows.append({
            "stage": stage, "match_number": mnum,
            "home": home, "away": away, "winner": winner,
            "win_prob": results[mnum]["p"],
        })

    return results, rows


def _resolve_side(slot, gw, gr, third_assign, results, mnum, side):
    kind = parse_slot(slot)
    if kind[0] == "pos":
        pos, group = kind[1], kind[2]
        return gw.get(group) if pos == 1 else gr.get(group)
    if kind[0] == "third":
        return third_assign.get((mnum, side))
    if kind[0] == "winner":
        ref = results.get(kind[1])
        return ref["winner"] if ref else None
    if kind[0] == "loser":
        ref = results.get(kind[1])
        return ref["loser"] if ref else None
    return None


# ===========================================================================
# Salida
# ===========================================================================
def print_report(standings, thirds, results, knockout):
    print("\n" + "=" * 60)
    print("  SIMULACION MUNDIAL 2026  -  prediccion del modelo")
    print("=" * 60)

    print("\n--- FASE DE GRUPOS ---")
    for g in sorted(standings.keys()):
        print("\nGrupo %s:" % g)
        for i, d in enumerate(standings[g], 1):
            mark = "  (clasifica)" if i <= 2 else ""
            print("  %d. %-22s %2.0f pts  (dif esp %+.2f)%s"
                  % (i, d["team"], d["pts"], d["egd"], mark))

    print("\n--- MEJORES TERCEROS (clasifican 8) ---")
    for i, d in enumerate(thirds, 1):
        print("  %d. %-22s (Grupo %s) %2.0f pts" % (i, d["team"], d["group"], d["pts"]))

    print("\n--- ELIMINATORIAS ---")
    stage_order = ["Round of 32", "Round of 16", "Quarter-finals",
                   "Semi-finals", "Third-place play-off", "Final"]
    kk = knockout.sort_values("match_number")
    by_stage = defaultdict(list)
    for _, m in kk.iterrows():
        by_stage[m["stage"]].append(int(m["match_number"]))

    for st in stage_order:
        if st not in by_stage:
            continue
        print("\n%s:" % st)
        for mnum in by_stage[st]:
            r = results.get(mnum)
            if not r or not r["home"] or not r["away"]:
                continue
            print("  %-22s vs %-22s -> %-22s (%.0f%%)"
                  % (r["home"], r["away"], r["winner"],
                     100 * (r["p"] or 0.5)))

    final = results.get(104)
    if final and final["winner"]:
        print("\n" + "=" * 60)
        print("  CAMPEON DEL MUNDO PREDICHO:  %s" % final["winner"].upper())
        print("=" * 60 + "\n")


def write_csv(rows_groups, rows_bracket):
    df = pd.DataFrame(rows_groups + rows_bracket)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print("[OK] Guardado %s (%d filas)." % (OUT_CSV, len(df)))


# ===========================================================================
# Montecarlo de control (opcional)
# ===========================================================================
def run_montecarlo(ens, groups, knockout, lut, attack, defense, gm, n):
    rng = np.random.default_rng(12345)
    champions = defaultdict(int)
    for _ in range(n):
        st = resolve_groups(groups, lut, rng=rng)
        th = best_thirds(st)
        res, _ = resolve_bracket(knockout, st, th, lut, attack, defense, gm, rng=rng)
        champ = res.get(104, {}).get("winner")
        if champ:
            champions[champ] += 1
    ranking = sorted(champions.items(), key=lambda kv: kv[1], reverse=True)
    print("\n--- CONTROL MONTECARLO (%d simulaciones) ---" % n)
    for team, c in ranking[:10]:
        print("  %-22s %5.1f%%" % (team, 100 * c / n))
    return ranking


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Simula el Mundial 2026 y predice el campeon.")
    ap.add_argument("--mc", type=int, default=0,
                    help="Nº de simulaciones Montecarlo de control (0 = solo determinista).")
    args = ap.parse_args()

    ens, groups, knockout = load_inputs()
    lut = build_match_lookup(ens)
    attack, defense, gm = derive_team_strength(ens)

    # --- Prediccion determinista (la del post) ---
    standings = resolve_groups(groups, lut, rng=None)
    thirds = best_thirds(standings)
    results, rows_bracket = resolve_bracket(
        knockout, standings, thirds, lut, attack, defense, gm, rng=None)

    print_report(standings, thirds, results, knockout)

    # filas de grupos para el CSV
    rows_groups = []
    for g in sorted(standings.keys()):
        for pos, d in enumerate(standings[g], 1):
            rows_groups.append({
                "stage": "Group %s" % g, "match_number": "",
                "home": d["team"], "away": "", "winner": "",
                "group_pos": pos, "group_pts": d["pts"], "exp_gd": d["egd"],
            })
    write_csv(rows_groups, rows_bracket)

    if args.mc and args.mc > 0:
        run_montecarlo(ens, groups, knockout, lut, attack, defense, gm, args.mc)


if __name__ == "__main__":
    main()
    