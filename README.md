# ⚽ Pronosticador Mundial 2026

Sistema de predicción de partidos y detección de valor en apuestas para el Mundial de Fútbol 2026.

## Estructura del proyecto

```
wc2026/
├── data/                   # Datos crudos y procesados (no subir a Git)
│   ├── results.csv         # Resultados filtrados (generado por script)
│   ├── statsbomb_matches.csv  # Stats avanzadas de Mundiales y Eurocopas
│   ├── referees.csv        # Perfil de árbitros — rellenar manualmente
│   └── unified.csv         # Dataset final unificado
│
├── notebooks/              # Jupyter notebooks de exploración
│   ├── 01_eda.ipynb        # Análisis exploratorio
│   ├── 02_features.ipynb   # Feature engineering
│   └── 03_modelos.ipynb    # Entrenamiento y evaluación
│
├── src/
│   ├── features/
│   │   └── feature_engineering.py   # Construcción de features
│   ├── models/
│   │   ├── ordinal_model.py         # Pipeline sin redes (Frank&Hall / SVOR)
│   │   └── neural_model.py          # Pipeline con redes (MLP / siamesa)
│   └── evaluation/
│       └── betting.py               # Cálculo de valor esperado y ROI
│
├── outputs/                # Modelos entrenados, métricas, gráficas
├── tests/                  # Tests unitarios
├── data_collector.py       # Script de recolección de datos
├── requirements.txt        # Dependencias Python
└── .gitignore
```

## Instalación

```bash
git clone https://github.com/TU_USUARIO/wc2026.git
cd wc2026
pip install -r requirements.txt
```

## Recolección de datos

```bash
# 1. Rellena ./data/referees.csv manualmente desde transfermarkt.es

# 2. Ejecuta el script (tarda ~15 min por StatsBomb)
python data_collector.py
```

StatsBomb Open Data también puede regenerarse de forma aislada:

```bash
python -m src.data.statsbomb_collector
```

Esto genera `data/raw/international_match_stats.csv` y
`data/statsbomb_matches.csv` con córners, tarjetas, tiros y xG de Mundial
2018/2022 y Euro 2020/2024. Es la cobertura gratuita disponible en StatsBomb
Open Data para selecciones; para ampliar córners/tarjetas a miles de partidos
hace falta otra fuente.

Convocatorias oficiales WC2026:

```bash
python -m src.data.normalize_squads
```

Este comando valida `data/raw/squads_wc2026_final_official_corrected.csv`
(48 selecciones, 26 jugadores por selección, posiciones válidas y sin
duplicados) y genera `data/processed/players_wc2026.csv` con el contrato
normalizado de `players`.

### Cuotas históricas y CLV

El collector de The Odds API descarga dos snapshots por partido, por defecto
apertura aproximada (`T-48h`) y cierre (`T-1h`), cachea el JSON crudo en
`data/raw/odds/the_odds_api/` y genera:

| Salida | Uso |
|---|---|
| `data/processed/odds_history.csv` | Tabla normalizada `odds` para joins y EV |
| `data/processed/odds_open_close.csv` | Apertura/cierre en la misma fila para CLV |

```bash
ODDS_API_KEY=tu_key python -m src.data.historical_odds_collector \
  --target-date 2022-12-18T15:00:00Z \
  --sport soccer_fifa_world_cup \
  --markets h2h,totals \
  --bookmakers pinnacle,bet365
```

Para props de jugador no uses un market genérico tipo `player_props`: hay que
pasar las claves concretas que ofrezca The Odds API para ese deporte/evento y
tu plan histórico. Esos mercados ampliados se piden por `eventId` histórico:

```bash
ODDS_API_KEY=tu_key python -m src.data.historical_odds_collector \
  --list-events-date 2024-07-14T19:00:00Z \
  --sport soccer_uefa_european_championship \
  --events-output data/processed/odds_events_euro_2024.csv

ODDS_API_KEY=tu_key python -m src.data.historical_odds_collector \
  --target-date 2024-07-14T19:00:00Z \
  --sport soccer_uefa_european_championship \
  --event-id EVENT_ID_DE_THE_ODDS_API \
  --markets player_shots_on_target,player_shots \
  --bookmakers pinnacle
```

## Arquitectura objetivo

El proyecto pasa de un predictor centrado en 1X2 a un sistema de mercados/eventos. La capa de datos debe normalizar entidades antes de entrenar modelos:

| Tabla | Uso principal |
|---|---|
| `matches` | Calendario, equipos, sede, descanso, viajes, clima, árbitro y resultado final |
| `teams` | Selecciones, confederación, ranking FIFA/Elo y atributos agregados |
| `team_ratings` | Ratings Elo/ranking por selección y fecha para joins temporales sin leakage |
| `players` | Convocados, posición, club y perfil físico/técnico |
| `lineups` | Titulares, suplentes, minutos y disponibilidad por partido |
| `player_match_stats` | Goles, asistencias, tiros, tiros a puerta, xG, corners, faltas y tarjetas por jugador |
| `venues` | Ciudad, país, coordenadas, altura, zona horaria, superficie y contexto ambiental |
| `base_camps` | Ciudad, hospedaje y campo de entrenamiento por selección para estimar viaje/fatiga |
| `weather_hourly` | Clima histórico/forecast por sede y hora: temperatura, humedad, sensación térmica, lluvia y viento |
| `referees` | Perfil disciplinario por árbitro y torneo/contexto: tarjetas, faltas y penaltis por partido |
| `odds` | Cuotas por casa, mercado, línea, selección, apertura/cierre y probabilidad sin margen |

Los contratos iniciales están definidos en `src/data/schemas.py`. Esta capa permite enchufar fuentes distintas (StatsBomb, Kaggle, convocatorias, clima y odds) sin acoplar los modelos al formato crudo de cada proveedor.

## Fuentes de datos

| Fuente | Encaje en el sistema | Nota operativa |
|---|---|---|
| FBref | Props de jugador: tiros, tiros a puerta, xG, xA, faltas, tarjetas y corners lanzados | Scraping HTML con pausa mínima de 3.5s entre peticiones |
| StatsBomb Open Data | Eventos con coordenadas y timestamps para construir targets de tiros, xG, corners, faltas y tarjetas | Fuente principal gratuita para Mundiales 2018/2022 y Eurocopas |
| API-Football | API REST para fixtures, alineaciones, estadios, árbitros, eventos y estadísticas de jugador | Freemium; útil como integración estructurada si compensa el coste |
| The Odds API | Historial de cuotas por timestamp para apertura, movimientos de mercado y cierre | API comercial; cachear respuestas y usar para EV/CLV |
| Transfermarkt Datasets | Alineaciones, minutos, posiciones, sustituciones y valor de mercado como proxy de calidad | Dataset comunitario CC0 en CSV/Parquet; preferible a scraping directo |
| WC 2026 Official Squads | Convocados oficiales por selección, club y posición | Archivo estático validado en `data/raw/squads_wc2026_final_official_corrected.csv` |
| WC 2026 Manual Venues | CSV local con las 16 sedes, coordenadas, altura, zona horaria, techo y superficie | Archivo estático en `data/raw/venues.csv` |
| WC 2026 Manual Referees | CSV local con árbitros, país, confederación y torneo | Archivo estático en `data/raw/referees_wc2026.csv` |
| WC 2026 Manual Base Camps | CSV local con campamento base, hospedaje y campo de entrenamiento por selección | Archivo estático en `data/raw/base_camps_wc2026.csv` |
| WC 2026 Manual Group Stage | CSV local con los 72 partidos de grupos A-L, fecha, equipos y sede | Archivo estático en `data/raw/group_stage_wc2026.csv` |
| WC 2026 Manual Knockout | CSV local con los 32 partidos eliminatorios, slots de bracket y sede | Archivo estático en `data/raw/knockout_wc2026.csv` |
| Open-Meteo API | Clima histórico horario y forecast a 14 días para cada estadio por lat/lon | Sin API key para uso no comercial; cachear por sede/hora |
| World Football Elo Ratings | Rating Elo histórico de selecciones por fecha | Más reactivo que FIFA ranking; usar join as-of antes del partido |
| WorldReferee / Football Referees Stats | Agregados históricos de árbitros por torneo para tarjetas, rojas y penaltis | Scraping con delays o CSV de Kaggle si está disponible |

El catálogo vive en `src/data/sources.py` y los primeros normalizadores hacia las tablas base viven en `src/data/normalizers.py`.

## Pipeline

```
Datos históricos + convocatorias + sedes + árbitros + odds
        ↓
Tablas normalizadas (matches, teams, team ratings, players, lineups, player stats, venues, base camps, weather, referees, odds)
        ↓
Feature engineering sin leakage (forma, descanso, viaje, clima, altura, árbitro, rol del jugador)
        ↓
Modelos por mercado
  - XGBoost para clasificación/regresión tabular
  - Modelos de conteo para tiros, corners, tarjetas y props
  - Modelos específicos de jugador condicionados a minutos/rol
        ↓
Calibración + comparación contra cuotas sharp
        ↓
Valor esperado, ROI simulado, closing line value y staking conservador
```

Entrenamiento del baseline XGBoost:

```bash
python -m src.models.xgboost_model \
  --features data/processed/features_train.csv \
  --wc-features data/processed/features_wc2026.csv \
  --markets result_1x2,over25,btts,total_goals
```

## Mercados predichos

| Mercado | Tipo |
|---|---|
| Resultado (1X2) | Ordinal / 3 clases |
| Over/Under 2.5 goles | Binario |
| Ambos equipos marcan (BTTS) | Binario |
| Resultado al descanso | 3 clases |
| Goles por equipo/jugador | Conteo / probabilidad |
| Asistencias de jugador | Conteo / probabilidad |
| Tiros y tiros a puerta por jugador | Conteo |
| Córners por equipo y totales | Conteo |
| Tarjetas por equipo/jugador | Conteo / binario, con árbitro como feature fuerte |
| Tarjetas rojas | Binario |
| Quién marca primero | Multinomial / survival simplificado |

## Evaluación

Simulación de apuestas en Mundiales 2018 y 2022 midiendo ROI, Brier Score y ECE.
