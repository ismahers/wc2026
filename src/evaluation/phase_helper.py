"""
Modulo auxiliar para calcular la fase (grupo o stage) de cada partido.
Importado por push_signals_supabase.py y market_radar_push.py.
Copiar a src/evaluation/phase_helper.py
"""
import pandas as pd
import os

_GS_PATH = "data/raw/group_stage_wc2026.csv"
_KO_PATH = "data/raw/knockout_wc2026.csv"

_phase_cache = None

def _build_phase_map():
    global _phase_cache
    if _phase_cache is not None:
        return _phase_cache
    
    lookup = {}  # (home, away) -> phase
    
    # Fase de grupos: "Grupo A", "Grupo B"...
    if os.path.exists(_GS_PATH):
        gs = pd.read_csv(_GS_PATH)
        for _, r in gs.iterrows():
            key = (str(r.get("home_team","")), str(r.get("away_team","")))
            group = r.get("group","")
            phase = f"Grupo {group}" if group and str(group) != "nan" else "Fase de grupos"
            lookup[key] = phase
    
    # Eliminatorias: usar el stage directamente
    if os.path.exists(_KO_PATH):
        ko = pd.read_csv(_KO_PATH)
        for _, r in ko.iterrows():
            home = str(r.get("home_team", r.get("home_slot","")))
            away = str(r.get("away_team", r.get("away_slot","")))
            stage = str(r.get("stage","Eliminatorias"))
            key = (home, away)
            lookup[key] = stage if stage != "nan" else "Eliminatorias"
    
    _phase_cache = lookup
    return lookup

def get_phase(home_team, away_team):
    """Devuelve la fase del partido: 'Grupo A', 'Round of 32', etc."""
    lookup = _build_phase_map()
    key = (str(home_team or ""), str(away_team or ""))
    # Intentar orden normal y luego invertido
    return lookup.get(key) or lookup.get((key[1], key[0])) or "Eliminatorias"
