"""
data_collector.py
=================
Script de recolección de datos para el Pronosticador del Mundial 2026.

Requisitos:
  pip install requests pandas beautifulsoup4 lxml
"""

import os
import time
import logging
import requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup

# ── Configuración ─────────────────────────────────────────────────────────────

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# Pausa entre peticiones HTTP para no saturar los servidores
REQUEST_DELAY = 2.0  # segundos


# ── 1. Kaggle — Resultados internacionales históricos ────────────────────────

# Dataset de Kaggle — descarga manual requerida
# URL: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
# Descarga el archivo results.csv y colócalo en ./data/results_raw.csv
KAGGLE_LOCAL_PATH = os.path.join(DATA_DIR, "results_raw.csv")

# Alternativa: repositorio GitHub del mismo autor (requiere acceso a internet sin restricciones)
KAGGLE_GITHUB_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/results.csv"
)

def fetch_kaggle_results() -> pd.DataFrame:
    """
    Carga el dataset de resultados internacionales históricos.

    Prioridad:
      1. Archivo local en ./data/results_raw.csv  (descarga manual desde Kaggle)
      2. Descarga directa desde GitHub del autor
         → https://github.com/martj42/international_results

    El dataset contiene todos los partidos internacionales desde 1872.
    Se filtra para quedarse con partidos relevantes desde 1990.
    """
    log.info("Cargando resultados internacionales históricos...")

    df = None

    # Intento 1: archivo local
    if os.path.exists(KAGGLE_LOCAL_PATH):
        log.info(f"  → Usando archivo local: {KAGGLE_LOCAL_PATH}")
        df = pd.read_csv(KAGGLE_LOCAL_PATH, parse_dates=["date"])
    else:
        # Intento 2: descarga desde GitHub
        log.info("  → Archivo local no encontrado, intentando descarga desde GitHub...")
        log.info(f"     URL: {KAGGLE_GITHUB_URL}")
        try:
            r = requests.get(KAGGLE_GITHUB_URL, headers=HEADERS, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text), parse_dates=["date"])
        except Exception as e:
            log.error(
                f"  ✗ No se pudo descargar el dataset: {e}\n"
                "    → Descarga manual:\n"
                "      1. Ve a https://www.kaggle.com/datasets/martj42/"
                "international-football-results-from-1872-to-2017\n"
                "      2. Descarga results.csv\n"
                "      3. Colócalo en ./data/results_raw.csv\n"
                "      4. Vuelve a ejecutar el script"
            )
            return pd.DataFrame()

    # # Filtrar partidos relevantes (Mundiales y clasificatorias desde 1990)
    # wc_tournaments = [
    #     "FIFA World Cup", "FIFA World Cup qualification",
    #     "Friendly", "AFC Asian Cup", "UEFA Euro",
    #     "Copa América", "Africa Cup of Nations",
    # ]
    # df = df[df["tournament"].isin(wc_tournaments) & (df["date"].dt.year >= 1990)]

    # Filtrar solo por año — todos los torneos cuentan para el Elo
    # (amistosos K=20, Mundiales K=60, etc.)
    df = df[df["date"].dt.year >= 1990]

    df = df.reset_index(drop=True)

    out = os.path.join(DATA_DIR, "results.csv")
    df.to_csv(out, index=False)
    log.info(f"  → {len(df)} partidos guardados en {out}")
    return df


# ── 2. football-data.co.uk — Córners, tarjetas y cuotas ─────────────────────

# Partidos internacionales disponibles en football-data.co.uk
# La URL base es http://www.football-data.co.uk/mmz4281/{season}/{league}.csv
# ── 2. StatsBomb Open Data — Stats avanzadas de Mundiales ───────────────────
#
# Repositorio público: https://github.com/statsbomb/open-data
# Competiciones masculinas senior disponibles (verificado en competitions.json,
# jun 2026):
#   - FIFA World Cup 2018:  competition_id=43,   season_id=3
#   - FIFA World Cup 2022:  competition_id=43,   season_id=106
#   - UEFA Euro 2020:       competition_id=55,   season_id=43
#   - UEFA Euro 2024:       competition_id=55,   season_id=282
#   - Copa America 2024:    competition_id=223,  season_id=282   (32 partidos)
#   - AFCON 2023:           competition_id=1267, season_id=107   (52 partidos)
#   - WC históricos sueltos: competition_id=43, season_id ∈
#       {269(1958), 270(1962), 272(1970), 51(1974), 54(1986), 55(1990)}
#       (19 partidos en total, con evento completo; córners parseables)
#
# NO existen en StatsBomb Open Data: Nations League ni Confederations Cup.
#
# Por cada partido se agregan desde los eventos:
#   córners, tarjetas amarillas/rojas, tiros, xG, posesión estimada

SB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

SB_COMPETITIONS = [
    {"name": "World Cup 2018",    "competition_id": 43,   "season_id": 3},
    {"name": "World Cup 2022",    "competition_id": 43,   "season_id": 106},
    {"name": "Euro 2020",         "competition_id": 55,   "season_id": 43},
    {"name": "Euro 2024",         "competition_id": 55,   "season_id": 282},
    # ── Nuevas (verificadas jun 2026) ──
    {"name": "Copa America 2024", "competition_id": 223,  "season_id": 282},
    {"name": "AFCON 2023",        "competition_id": 1267, "season_id": 107},
    # ── WC históricos sueltos (19 partidos). Descomenta los que quieras ──
    # {"name": "World Cup 1970",  "competition_id": 43,   "season_id": 272},
    # {"name": "World Cup 1974",  "competition_id": 43,   "season_id": 51},
    # {"name": "World Cup 1986",  "competition_id": 43,   "season_id": 54},
    # {"name": "World Cup 1958",  "competition_id": 43,   "season_id": 269},
    # {"name": "World Cup 1962",  "competition_id": 43,   "season_id": 270},
    # {"name": "World Cup 1990",  "competition_id": 43,   "season_id": 55},
]


def _sb_get(url: str) -> dict | list:
    """Descarga y parsea un JSON de StatsBomb."""
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def _aggregate_events(events: list) -> dict:
    """
    Agrega los eventos de un partido en estadísticas por equipo.
    Nombres de eventos según el esquema real de StatsBomb:
      - Córners:  Pass con pass.type.name == "Corner"
      - Amarillas: Foul Committed con foul_committed.card.name == "Yellow Card"
                   o Bad Behaviour con bad_behaviour.card.name == "Yellow Card"
      - Rojas:    ídem con "Red Card" o "Second Yellow"
      - Tiros:    tipo "Shot"
      - xG:       shot.statsbomb_xg dentro de eventos Shot
    """
    stats = {
        "corners_home": 0, "corners_away": 0,
        "yellow_home":  0, "yellow_away":  0,
        "red_home":     0, "red_away":     0,
        "shots_home":   0, "shots_away":   0,
        "xg_home":      0.0, "xg_away":    0.0,
        "home_team_id": None, "away_team_id": None,
    }

    # Identificar equipo local desde el primer Starting XI
    for ev in events:
        if ev.get("type", {}).get("name") == "Starting XI":
            if stats["home_team_id"] is None:
                stats["home_team_id"] = ev["team"]["id"]
            elif ev["team"]["id"] != stats["home_team_id"]:
                stats["away_team_id"] = ev["team"]["id"]
                break

    for ev in events:
        team_id = ev.get("team", {}).get("id")
        is_home = (team_id == stats["home_team_id"])
        suffix  = "home" if is_home else "away"
        etype   = ev.get("type", {}).get("name", "")

        # Córners: pases de tipo Corner
        if etype == "Pass":
            pass_type = ev.get("pass", {}).get("type", {}).get("name", "")
            if pass_type == "Corner":
                stats[f"corners_{suffix}"] += 1

        # Tarjetas en faltas
        elif etype == "Foul Committed":
            card = ev.get("foul_committed", {}).get("card", {}).get("name", "")
            if card == "Yellow Card":
                stats[f"yellow_{suffix}"] += 1
            elif card in ("Red Card", "Second Yellow"):
                stats[f"red_{suffix}"] += 1

        # Tarjetas por conducta violenta (expulsión directa sin falta)
        elif etype == "Bad Behaviour":
            card = ev.get("bad_behaviour", {}).get("card", {}).get("name", "")
            if card == "Yellow Card":
                stats[f"yellow_{suffix}"] += 1
            elif card in ("Red Card", "Second Yellow"):
                stats[f"red_{suffix}"] += 1

        # Tiros y xG
        elif etype == "Shot":
            stats[f"shots_{suffix}"] += 1
            xg = ev.get("shot", {}).get("statsbomb_xg", 0.0) or 0.0
            stats[f"xg_{suffix}"] += xg

    stats["xg_home"] = round(stats["xg_home"], 3)
    stats["xg_away"] = round(stats["xg_away"], 3)
    return stats


def fetch_statsbomb_data() -> pd.DataFrame:
    """
    Descarga datos de StatsBomb Open Data para Mundiales y Eurocopas.
    Por cada partido agrega: córners, tarjetas, tiros, xG local y visitante.
    Guarda el resultado en ./data/statsbomb_matches.csv
    """
    try:
        from src.data.statsbomb_collector import StatsBombInternationalCollector

        collector = StatsBombInternationalCollector(request_delay_seconds=0.3)
        df = collector.collect()
        collector.save(
            df,
            raw_output=os.path.join(DATA_DIR, "raw", "international_match_stats.csv"),
            statsbomb_output=os.path.join(DATA_DIR, "statsbomb_matches.csv"),
        )
        return df
    except Exception as e:
        log.warning(
            "No se pudo usar src.data.statsbomb_collector (%s). "
            "Fallback a collector legado en data_collector.py",
            e,
        )

    log.info("Descargando datos de StatsBomb Open Data...")
    all_rows = []

    for comp in SB_COMPETITIONS:
        cid, sid, cname = comp["competition_id"], comp["season_id"], comp["name"]
        matches_url = f"{SB_BASE}/matches/{cid}/{sid}.json"

        try:
            time.sleep(REQUEST_DELAY)
            matches = _sb_get(matches_url)
            log.info(f"  → {cname}: {len(matches)} partidos encontrados")
        except Exception as e:
            log.warning(f"  ✗ No se pudieron descargar partidos de {cname}: {e}")
            continue

        for match in matches:
            match_id   = match["match_id"]
            match_date = match.get("match_date", "")
            home_team  = match.get("home_team", {}).get("home_team_name", "")
            away_team  = match.get("away_team", {}).get("away_team_name", "")
            home_score = match.get("home_score")
            away_score = match.get("away_score")
            stage      = match.get("competition_stage", {}).get("name", "")
            stadium    = match.get("stadium", {}).get("name", "")

            # Descargar eventos del partido
            events_url = f"{SB_BASE}/events/{match_id}.json"
            try:
                time.sleep(0.3)  # delay suave para no saturar GitHub
                events = _sb_get(events_url)
                agg    = _aggregate_events(events)
            except Exception as e:
                log.warning(f"    ✗ Eventos {match_id} ({home_team} vs {away_team}): {e}")
                agg = {}

            row = {
                "date":          match_date,
                "home_team":     home_team,
                "away_team":     away_team,
                "home_score":    home_score,
                "away_score":    away_score,
                "tournament":    cname,
                "stage":         stage,
                "stadium":       stadium,
                "match_id":      match_id,
                "corners_home":  agg.get("corners_home"),
                "corners_away":  agg.get("corners_away"),
                "corners_total": (agg.get("corners_home", 0) or 0) + (agg.get("corners_away", 0) or 0),
                "yellow_home":   agg.get("yellow_home"),
                "yellow_away":   agg.get("yellow_away"),
                "yellow_total":  (agg.get("yellow_home", 0) or 0) + (agg.get("yellow_away", 0) or 0),
                "red_home":      agg.get("red_home"),
                "red_away":      agg.get("red_away"),
                "shots_home":    agg.get("shots_home"),
                "shots_away":    agg.get("shots_away"),
                "xg_home":       agg.get("xg_home"),
                "xg_away":       agg.get("xg_away"),
            }
            all_rows.append(row)

        log.info(f"    ✓ {cname} completado")

    if not all_rows:
        log.error("No se obtuvo ningún dato de StatsBomb")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["result"] = df.apply(
        lambda r: "H" if (r["home_score"] or 0) > (r["away_score"] or 0)
                  else ("A" if (r["home_score"] or 0) < (r["away_score"] or 0) else "D"),
        axis=1,
    )

    out = os.path.join(DATA_DIR, "statsbomb_matches.csv")
    df.to_csv(out, index=False)
    log.info(f"  → {len(df)} partidos con stats guardados en {out}")
    return df


# ── 4. Transfermarkt — Perfil de árbitros ────────────────────────────────────

# Lista oficial de los 52 árbitros del Mundial 2026 (FIFA, mayo 2026)
# Formato: (nombre_display, slug_transfermarkt, nacionalidad)
# El slug se usa para búsqueda: https://www.transfermarkt.es/schiedsrichter/suche?query={slug}
WC2026_REFEREES = [
    ("AL JASSIM Abdulrahman", "abdulrahman-al-jassim", "QAT"),
    ("AL TURAIS Khalid",      "khalid-al-turais",      "KSA"),
    ("ARAKI Yusuke",          "yusuke-araki",          "JPN"),
    ("ARTAN Omar Abdulkadir", "omar-artan",             "SOM"),
    ("ATCHO Pierre",          "pierre-atcho",           "GAB"),
    ("BARTON Ivan",           "ivan-barton",            "SLV"),
    ("BEIDA Dahane",          "dahane-beida",           "MTN"),
    ("BENITEZ Juan Gabriel",  "juan-gabriel-benitez",  "PAR"),
    ("CALDERON Juan",         "juan-calderon",          "CRC"),
    ("CLAUS Raphael",         "raphael-claus",          "BRA"),
    ("ELFATH Ismail",         "ismail-elfath",          "USA"),
    ("ESKAS Espen",           "espen-eskas",            "NOR"),
    ("FAGHANI Alireza",       "alireza-faghani",        "AUS"),
    ("FALCON PEREZ Yael",     "yael-falcon-perez",      "ARG"),
    ("FISCHER Drew",          "drew-fischer",           "CAN"),
    ("GARAY Cristian",        "cristian-garay",         "CHI"),
    ("GARCIA Katia",          "katia-garcia",           "MEX"),
    ("GHORBAL Mustapha",      "mustapha-ghorbal",       "ALG"),
    ("HERNANDEZ Alejandro",   "alejandro-hernandez",    "ESP"),
    ("HERRERA Dario",         "dario-herrera",          "ARG"),
    ("JAYED Jalal",           "jalal-jayed",            "MAR"),
    ("KAWANA-WAUGH Campbell-Kirk", "campbell-kirk-kawana-waugh", "NZL"),
    ("KOVACS Istvan",         "istvan-kovacs",          "ROU"),
    ("LETEXIER Francois",     "francois-letexier",      "FRA"),
    ("MA Ning",               "ma-ning",                "CHN"),
    ("MAKHADMEH Adham",       "adham-makhadmeh",        "JOR"),
    ("MAKKELIE Danny",        "danny-makkelie",         "NED"),
    ("MARCINIAK Szymon",      "szymon-marciniak",       "POL"),
    ("MARIANI Maurizio",      "maurizio-mariani",       "ITA"),
    ("MARTINEZ Hector Said",  "hector-said-martinez",   "HON"),
    ("MOHAMED Amin",          "amin-mohamed",           "EGY"),
    ("NATION Oshane",         "oshane-nation",          "JAM"),
    ("NYBERG Glenn",          "glenn-nyberg",           "SWE"),
    ("OLIVER Michael",        "michael-oliver",         "ENG"),
    ("OMAR AL ALI",           "omar-al-ali",            "UAE"),
    ("ORTEGA Kevin",          "kevin-ortega",           "PER"),
    ("PENSO Tori",            "tori-penso",             "USA"),
    ("PINHEIRO Joao",         "joao-pinheiro",          "POR"),
    ("RAMON ABATTI",          "ramon-abatti",           "BRA"),
    ("RAMOS Cesar",           "cesar-ramos",            "MEX"),
    ("ROJAS Andres",          "andres-rojas",           "COL"),
    ("SCHAERER Sandro",       "sandro-schaerer",        "SUI"),
    ("TANTASHEV Ilgiz",       "ilgiz-tantashev",        "UZB"),
    ("TAYLOR Anthony",        "anthony-taylor",         "ENG"),
    ("TEJERA Gustavo",        "gustavo-tejera",         "URU"),
    ("TELLO Facundo",         "facundo-tello",          "ARG"),
    ("TOM Abongile",          "abongile-tom",           "RSA"),
    ("TURPIN Clement",        "clement-turpin",         "FRA"),
    ("VALENZUELA Jesus",      "jesus-valenzuela",       "VEN"),
    ("VINCIC Slavko",         "slavko-vincic",          "SVN"),
    ("WILTON SAMPAIO",        "wilton-sampaio",         "BRA"),
    ("ZWAYER Felix",          "felix-zwayer",           "GER"),
]

TM_SEARCH_URL  = "https://www.transfermarkt.es/schnellsuche/ergebnis/schnellsuche?query={query}&Schiedsrichter_page=0"
TM_PROFILE_URL = "https://www.transfermarkt.es{path}/statistik/schiedsrichter/{id}"
TM_HEADERS     = {**HEADERS, "Accept-Language": "es-ES,es;q=0.9", "Referer": "https://www.transfermarkt.es/"}


def _search_referee_id(name: str) -> tuple[str, str] | None:
    """
    Busca el árbitro en transfermarkt y devuelve (slug, id) de su perfil.
    Ejemplo: ("szymon-marciniak", "1964")
    """
    query = name.lower().replace(" ", "+")
    url   = TM_SEARCH_URL.format(query=query)
    try:
        r = requests.get(url, headers=TM_HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        # El primer resultado de árbitros tiene un enlace con /schiedsrichter/ en la URL
        link = soup.find("a", href=lambda h: h and "/schiedsrichter/" in h and "/profil/" in h)
        if not link:
            return None
        href = link["href"]          # ej: /szymon-marciniak/profil/schiedsrichter/1964
        parts = href.strip("/").split("/")
        # parts = [slug, 'profil', 'schiedsrichter', id]
        if len(parts) >= 4:
            return parts[0], parts[-1]
    except Exception:
        pass
    return None


def _scrape_referee_stats(slug: str, ref_id: str) -> dict:
    """
    Extrae estadísticas de carrera del perfil de un árbitro en transfermarkt.
    Devuelve dict con matches, yellow_cards, yellow_red, red_cards y promedios.
    """
    url = f"https://www.transfermarkt.es/{slug}/statistik/schiedsrichter/{ref_id}"
    r   = requests.get(url, headers=TM_HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    # Tabla de estadísticas totales (última fila = totales de carrera)
    stats = {"matches": None, "yellow_cards": None, "yellow_red": None, "red_cards": None}
    table = soup.find("table", {"class": "items"})
    if not table:
        return stats

    rows = table.find_all("tr")
    if not rows:
        return stats

    # Última fila con datos numéricos = totales
    for tr in reversed(rows):
        cols = tr.find_all("td")
        if len(cols) >= 5:
            try:
                def _num(s):
                    return int(s.replace(".", "").replace("-", "0").strip() or 0)
                stats["matches"]      = _num(cols[1].get_text(strip=True))
                stats["yellow_cards"] = _num(cols[2].get_text(strip=True))
                stats["yellow_red"]   = _num(cols[3].get_text(strip=True))
                stats["red_cards"]    = _num(cols[4].get_text(strip=True))
                break
            except (ValueError, IndexError):
                continue
    return stats


def fetch_referee_profiles() -> pd.DataFrame:
    """
    Lee el perfil de árbitros desde ./data/referees.csv (relleno manualmente).

    Si el archivo no existe, genera la plantilla vacía con los 52 árbitros
    del Mundial 2026 para rellenar a mano desde transfermarkt.es.

    Columnas esperadas en el CSV:
      referee, nationality, tm_id, matches,
      yellow_cards, yellow_red, red_cards,
      yellow_per_match, red_per_match
    """
    ref_path = os.path.join(DATA_DIR, "referees.csv")
    log.info("Cargando perfil de árbitros...")

    if os.path.exists(ref_path):
        df = pd.read_csv(ref_path)
        # Calcular promedios si faltan pero hay datos brutos
        if "yellow_per_match" not in df.columns and "yellow_cards" in df.columns:
            df["yellow_per_match"] = (df["yellow_cards"] / df["matches"]).round(3)
            df["red_per_match"]    = ((df["yellow_red"] + df["red_cards"]) / df["matches"]).round(3)
            df.to_csv(ref_path, index=False)
        filled = df["yellow_per_match"].notna().sum() if "yellow_per_match" in df.columns else 0
        log.info(f"  → referees.csv cargado: {len(df)} árbitros, {filled} con stats completas")
        return df
    else:
        log.warning(
            "  ✗ ./data/referees.csv no encontrado.\n"
            "    → Generando plantilla vacía con los 52 árbitros.\n"
            "    → Rellénala manualmente desde transfermarkt.es y vuelve a ejecutar."
        )
        df = pd.DataFrame([
            {"referee": name, "nationality": nat,
             "tm_id": "", "matches": "", "yellow_cards": "",
             "yellow_red": "", "red_cards": "",
             "yellow_per_match": "", "red_per_match": ""}
            for name, _, nat in WC2026_REFEREES
        ])
        df.to_csv(ref_path, index=False)
        log.info(f"  → Plantilla guardada en {ref_path}")
        return pd.DataFrame()


# ── 5. Unificación del dataset ────────────────────────────────────────────────

def unify_datasets(
    results: pd.DataFrame,
    statsbomb: pd.DataFrame,
    referees: pd.DataFrame,
) -> pd.DataFrame:
    """
    Une los datasets en un único DataFrame listo para modelado.

    - results:    base histórica de Kaggle (19k partidos desde 1990)
    - statsbomb:  stats avanzadas de Mundiales y Eurocopas (córners, tarjetas, xG)
    - referees:   perfil histórico de árbitros

    Estrategia de merge:
      1. Base = results (todos los partidos)
      2. Para partidos que están en statsbomb, enriquecer con sus stats
      3. Merge con referees por nombre de árbitro (cuando esté disponible)
    """
    log.info("Unificando datasets...")

    if results.empty:
        log.error(
            "El dataset base (resultados históricos) está vacío. "
            "Descarga results.csv de Kaggle y colócalo en ./data/results_raw.csv"
        )
        return pd.DataFrame()

    # Base: resultados de Kaggle — quitar partidos sin resultado
    df = results.copy()
    df = df[df["home_score"].notna() & df["away_score"].notna()].reset_index(drop=True)
    df["result"] = df.apply(
        lambda r: "H" if r["home_score"] > r["away_score"]
                  else ("A" if r["home_score"] < r["away_score"] else "D"),
        axis=1,
    )

    # Merge con StatsBomb por fecha + equipos
    if not statsbomb.empty:
        sb_cols = [
            "date", "home_team", "away_team",
            "corners_home", "corners_away", "corners_total",
            "yellow_home",  "yellow_away",  "yellow_total",
            "red_home",     "red_away",     "red_total",
            "shots_home",   "shots_away",   "shots_total",
            "shots_on_target_home", "shots_on_target_away", "shots_on_target_total",
            "xg_home",      "xg_away",      "xg_total",
            "stage",        "referee",
        ]
        sb_sub = statsbomb[[c for c in sb_cols if c in statsbomb.columns]].copy()
        sb_sub["date"] = pd.to_datetime(sb_sub["date"], errors="coerce")
        df["date"]     = pd.to_datetime(df["date"], errors="coerce")

        from src.data.team_names import add_canonical_columns

        df = add_canonical_columns(df, ["home_team", "away_team"], suffix="_merge")
        sb_sub = add_canonical_columns(sb_sub, ["home_team", "away_team"], suffix="_merge")

        # StatsBomb usa fecha UTC en algunas competiciones y el histórico base
        # suele usar fecha local. En Copa America 2024 eso desplaza muchos
        # partidos un día y el merge exacto pierde córners/tarjetas. Creamos
        # candidatos exacto, -1 y +1 día, manteniendo siempre el exacto primero.
        df["date_merge"] = df["date"].dt.normalize()
        sb_sub["statsbomb_date"] = sb_sub["date"].dt.normalize()
        sb_candidates = []
        for priority, offset_days in enumerate((0, -1, 1)):
            candidate = sb_sub.copy()
            candidate["date_merge"] = candidate["statsbomb_date"] + pd.to_timedelta(offset_days, unit="D")
            candidate["date_merge_priority"] = priority
            sb_candidates.append(candidate)
        sb_sub = pd.concat(sb_candidates, ignore_index=True, sort=False)

        swap_cols = [
            ("home_team_merge", "away_team_merge"),
            ("corners_home", "corners_away"),
            ("yellow_home", "yellow_away"),
            ("red_home", "red_away"),
            ("shots_home", "shots_away"),
            ("shots_on_target_home", "shots_on_target_away"),
            ("xg_home", "xg_away"),
        ]
        sb_swapped = sb_sub.copy()
        for left, right in swap_cols:
            if left in sb_swapped.columns and right in sb_swapped.columns:
                sb_swapped[[left, right]] = sb_swapped[[right, left]].to_numpy()

        sb_merge = pd.concat([sb_sub, sb_swapped], ignore_index=True, sort=False)
        sb_merge = sb_merge.sort_values("date_merge_priority")
        sb_merge = sb_merge.drop_duplicates(["date_merge", "home_team_merge", "away_team_merge"], keep="first")
        df = df.merge(
            sb_merge.drop(columns=["date", "home_team", "away_team"]),
            on=["date_merge", "home_team_merge", "away_team_merge"],
            how="left",
        )
        df = df.drop(
            columns=[
                "home_team_merge", "away_team_merge", "date_merge",
                "statsbomb_date", "date_merge_priority",
            ],
            errors="ignore",
        )
        n_matched = df["xg_home"].notna().sum()
        log.info(f"  → StatsBomb enriqueció {n_matched} partidos con stats avanzadas")

    # Merge con árbitros
    if not referees.empty and "referee" in df.columns:
        ref_sub = referees[
            [c for c in ["referee", "yellow_per_match", "red_per_match"] if c in referees.columns]
        ].copy()
        df = df.merge(ref_sub, on="referee", how="left")

    out = os.path.join(DATA_DIR, "unified.csv")
    df.to_csv(out, index=False)
    log.info(f"  → Dataset unificado: {len(df)} partidos, {len(df.columns)} columnas → {out}")
    return df


# ── 6. Resumen del dataset ────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("RESUMEN DEL DATASET UNIFICADO")
    print("=" * 60)
    if df.empty:
        print("  Dataset vacío — revisa los errores anteriores.")
        print("=" * 60 + "\n")
        return
    print(f"  Partidos totales:        {len(df)}")
    print(f"  Rango temporal:          {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Columnas disponibles:    {len(df.columns)}")
    print(f"  Valores nulos (%):")
    null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    for col, pct in null_pct[null_pct > 0].head(10).items():
        print(f"    {col:<30} {pct:.1f}%")
    if "result" in df.columns:
        print(f"\n  Distribución de resultados:")
        for label, count in df["result"].value_counts().items():
            print(f"    {label}: {count} ({count/len(df)*100:.1f}%)")
    if "corners_total" in df.columns:
        print(f"\n  Córners totales — media: {df['corners_total'].mean():.2f}, "
              f"std: {df['corners_total'].std():.2f}")
    if "yellow_total" in df.columns:
        print(f"  Tarjetas amarillas — media: {df['yellow_total'].mean():.2f}, "
              f"std: {df['yellow_total'].std():.2f}")
    print("=" * 60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("Iniciando recolección de datos para el Pronosticador Mundial 2026")
    log.info(f"Directorio de salida: {os.path.abspath(DATA_DIR)}")

    results   = fetch_kaggle_results()
    statsbomb = fetch_statsbomb_data()
    referees  = fetch_referee_profiles()
    unified   = unify_datasets(results, statsbomb, referees)

    print_summary(unified)
    log.info("Recolección completada. Revisa la carpeta ./data/")


if __name__ == "__main__":
    main()
