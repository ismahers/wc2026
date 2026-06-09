# ⚽ Pronosticador Mundial 2026

Sistema de predicción de partidos y detección de valor en apuestas para el Mundial de Fútbol 2026.

## Estructura del proyecto

```
wc2026/
├── data/
│   ├── raw/                         # Datos estáticos del torneo
│   │   ├── venues.csv               # 16 sedes con coordenadas y altitud
│   │   ├── referees_wc2026.csv      # 52 árbitros designados
│   │   ├── base_camps_wc2026.csv    # Campamentos base de las 48 selecciones
│   │   ├── group_stage_wc2026.csv   # 72 partidos de fase de grupos
│   │   ├── knockout_wc2026.csv      # 32 partidos eliminatorios
│   │   ├── squads_wc2026_final_official_corrected.csv  # Convocatorias oficiales corregidas
│   │   └── squads_wc2026_fifa_official.csv  # Extraído del PDF oficial FIFA
│   ├── transfermarkt/               # Datos de jugadores (descargar de Kaggle)
│   │   ├── player_profiles.csv      # Perfiles: nombre, posición, nacionalidad
│   │   ├── player_latest_market_value.csv  # Valor de mercado más reciente
│   │   └── player_national_performances.csv  # Stats en selección
│   │   # player_performances.csv (157MB) — descargar manualmente, no está en Git
│   └── processed/                   # Generados por los scripts
│       ├── players_master.csv       # Claves internas estables de jugadores
│       ├── player_source_matches.csv # Puente FBref/Transfermarkt/StatsBomb
│       ├── transfermarkt_player_profiles.csv # Perfiles enlazados
│       ├── transfermarkt_match_review.csv # Casos para revisión manual
│       ├── team_ratings_wc2026.csv  # Rating compuesto por selección (Elo+mercado+racha)
│       ├── team_player_stats_wc2026.csv  # Stats de jugadores agregadas por selección
│       ├── players_wc2026.csv       # Jugadores normalizados
│       ├── teams_wc2026.csv         # Selecciones normalizadas
│       ├── squad_summary_wc2026.csv # Resumen de convocatorias
│       └── market_registry.csv      # Estado operativo de cada mercado
│
├── src/
│   ├── data/
│   │   ├── builder.py               # Une todas las fuentes en matches_enriched.csv
│   │   ├── elo_loader.py            # Calcula Elo histórico desde resultados
│   │   ├── team_rating_wc2026.py    # Rating compuesto WC2026 (Elo+mercado+racha)
│   │   ├── player_stats_aggregator.py  # Agrega stats de jugadores por selección
│   │   ├── normalize_squads.py      # Normaliza convocatorias oficiales
│   │   ├── team_names.py            # Canonización de nombres de selección
│   │   └── schemas.py               # Contratos de tablas normalizadas
│   ├── features/
│   │   └── feature_engineering.py  # Features: forma, contexto, árbitro, rating, player stats
│   ├── models/
│   │   ├── xgboost_model.py         # Pipeline XGBoost por mercado
│   │   ├── xgb_baseline.py          # Implementación detallada XGBoost
│   │   ├── poisson_model.py         # Modelo Poisson de goles → mercados derivados
│   │   └── ordinal_model.py         # Frank&Hall + KMM (experimental)
│   └── evaluation/
│       ├── betting.py               # Cálculo de VE y ROI
│       ├── odds_converter.py        # Convierte probabilidades a cuotas decimales
│       ├── model_comparison.py      # Compara XGBoost vs Poisson con nivel de confianza
│       └── model_ensemble.py        # Combina XGBoost+Poisson en una cuota final
│
├── outputs/                         # Predicciones y métricas (subidas a Git)
│   ├── wc2026_predictions.csv       # Predicciones XGBoost (prob + cuotas)
│   ├── wc2026_predictions_with_odds.csv
│   ├── wc2026_poisson_predictions.csv  # Predicciones Poisson (lambdas + marcadores)
│   ├── wc2026_poisson_predictions_with_odds.csv
│   ├── wc2026_ensemble_predictions.csv # Predicción final con una cuota por mercado
│   ├── model_comparison.csv         # Comparación XGBoost vs Poisson con confianza
│   ├── xgb_baseline_metrics.json    # Métricas de evaluación XGBoost
│   └── poisson_metrics.json         # Métricas de evaluación Poisson
│
├── data_collector.py                # Recolección de datos históricos
├── requirements.txt
└── .gitignore
```

## Instalación

```bash
git clone https://github.com/ismahers/wc2026.git
cd wc2026
pip install -r requirements.txt
```

## Pipeline completo (orden de ejecución)

```bash
# 1. Recolectar datos históricos (Kaggle + StatsBomb)
python data_collector.py

# 2. Construir identidad de jugadores, normalizar convocatorias y calcular Elo
python -m src.data.fifa_squads_pdf
python -m src.data.player_master
python -m src.data.match_transfermarkt
python -m src.data.normalize_squads
python -m src.data.elo_loader

# 3. Calcular rating compuesto de selecciones
#    Requiere: data/transfermarkt/player_profiles.csv
#              data/transfermarkt/player_latest_market_value.csv
python -m src.data.team_rating_wc2026

# 4. Agregar stats de jugadores por selección
#    Requiere además: data/transfermarkt/player_performances.csv (descargar de Kaggle)
#                    data/transfermarkt/player_national_performances.csv
python -m src.data.player_stats_aggregator

# 5. Construir dataset enriquecido
python -m src.data.builder wc2026

# 6. Construir features
python src/features/feature_engineering.py

# 7. Entrenar modelos
python -m src.models.xgboost_model \
  --features data/processed/features_train.csv \
  --wc-features data/processed/features_wc2026.csv

python -m src.models.poisson_model \
  --features data/processed/features_train.csv \
  --wc-features data/processed/features_wc2026.csv

# 8. Convertir probabilidades a cuotas
python -m src.evaluation.odds_converter --input outputs/wc2026_predictions.csv
python -m src.evaluation.odds_converter --input outputs/wc2026_poisson_predictions.csv

# 9. Comparar modelos y ver nivel de confianza
python -m src.evaluation.model_comparison

# 10. Generar predicción final combinada
python -m src.evaluation.model_ensemble
```

La política completa de CSVs versionados e ignorados está en
`docs/data_inventory.md`.

## Rating compuesto de selecciones

Se calcula con tres componentes sobre las 48 selecciones del WC2026:

| Componente | Peso | Fuente |
|---|---|---|
| Valor de mercado (log-normalizado) | 60% | Transfermarkt por línea: ataque/medio/defensa |
| Elo histórico | 20% | Calculado desde 32k partidos desde 1990 |
| Racha ponderada | 20% | Últimos 10 partidos vs rivales WC2026 con Elo ≥ mediana |

La racha pondera cada resultado por el nivel del rival:
```
factor_rival = 0.5 * elo_norm_rival + 0.5 * valor_mercado_norm_rival
puntos_ponderados = resultado(0/1/3) × factor_rival
```

Output: `data/processed/team_ratings_wc2026.csv` con rating 0-100 y desglose por línea.

## Modelos de predicción

### XGBoost (6 mercados)

| Mercado | Tipo | Métricas (val 2018-2022) |
|---|---|---|
| Resultado 1X2 | Multiclase | acc=0.594, log_loss=0.885 |
| Over 2.5 goles | Binario | acc=0.582, AUC=0.610 |
| BTTS | Binario | acc=0.582, AUC=0.569 |
| Total goles | Poisson | MAE=1.406 |
| Córners totales | Poisson | MAE=3.064 (314 partidos StatsBomb) |
| Tarjetas amarillas | Analítico / pendiente | Depende de árbitros y tasas de equipo |

Split temporal obligatorio: train < 2018, val 2018-2022. Sin random split.

### Modelo Poisson (goles → mercados derivados)

Dos XGBoost Poisson independientes: `lambda_home` y `lambda_away`. De ahí se deriva:
- Distribución de marcadores P(i, j) para i,j ∈ 0..8
- 1X2, Over/Under 2.5 y BTTS de forma coherente (misma distribución)
- Top 5 marcadores más probables con probabilidad

Ventaja: las probabilidades de los tres mercados son siempre coherentes entre sí.

### Comparación de modelos

```bash
python -m src.evaluation.model_comparison
```

Calcula la diferencia porcentual relativa entre cuotas XGBoost y Poisson:
- **ALTA confianza** (diff < 10%): ambos modelos coinciden → apostar con más seguridad
- **MEDIA confianza** (diff 10-25%): divergencia moderada → revisar antes de apostar
- **BAJA confianza** (diff > 25%): modelos divergen → no apostar hasta tener más información

### Ensemble final

```bash
python -m src.evaluation.model_ensemble
```

Genera `outputs/wc2026_ensemble_predictions.csv`, que es el archivo recomendado
para comparar contra cuotas de casas. El peso de Poisson sube cuando la
diferencia entre selecciones es grande; XGBoost pesa más en partidos parejos.
El output incluye:

- `final_prob_H`, `final_prob_D`, `final_prob_A`
- `final_odds_H`, `final_odds_D`, `final_odds_A`
- `final_prob_over25`, `final_odds_over25`
- `final_prob_btts`, `final_odds_btts`
- `model_regime`: `xgb_lean`, `blend` o `poisson_lean`

## Datos de Transfermarkt

Los archivos ligeros están en el repo. El archivo pesado hay que descargarlo manualmente:

1. Ve a [Kaggle - football-datasets](https://www.kaggle.com/datasets/xfkzujqjvx97n/football-datasets)
2. Descarga `player_performances.csv` (157MB)
3. Colócalo en `data/transfermarkt/player_performances.csv`
4. Ejecuta `python -m src.data.player_stats_aggregator`

## Detección de valor esperado

```
VE = p_modelo × cuota_casa - 1
```

VE > 0 → hay valor (la casa subestima la probabilidad real).

Para calcular VE necesitáis las cuotas reales de una casa. El converter genera
las cuotas del modelo (sin margen) para comparar:

```bash
python -m src.evaluation.odds_converter --input outputs/wc2026_poisson_predictions.csv
```

### Estrategia depurada de value betting

El primer backtest con cuotas de cierre de Pinnacle para WC2022 detectó dos
fallos del modelo:

- No apostar empates 1X2: el modelo los sobreestima.
- No apostar EV extremo: `EV > 40%` suele ser error del modelo, no value real.
- Descartar EV bajo: el tramo 5-10% añade ruido y baja el beneficio del backtest.

La regla operativa actual es:

- Solo 1X2 local/visitante (`1X2-H`, `1X2-A`).
- Solo EV moderado: `10% <= EV <= 40%`.
- Para shortlist core: cuota `1.50-2.50` y fiabilidad mínima `MEDIA`.

Comando reproducible sin gastar créditos de API:

```bash
python -m src.evaluation.value_filters
```

Resultado con `outputs/backtest2022_bets.csv`:

| Segmento | Apuestas | Acierto | Beneficio | ROI |
|---|---:|---:|---:|---:|
| Total | 28 | 42.86% | +15.06u | +53.79% |
| 1X2-A | 12 | 33.33% | +7.58u | +63.17% |
| 1X2-H | 16 | 50.00% | +7.48u | +46.75% |

El calculador en vivo `src/evaluation/ev_calculator.py` ya aplica estos filtros
por defecto y guarda `strategy_bet_allowed` + `strategy_reason` en el CSV.
El modo más abierto `5% <= EV <= 40%` sigue disponible con `--min-ev 0.05`,
pero no es el perfil operativo por defecto.

Para convertir el EV vivo en shortlist revisable:

```bash
python -m src.evaluation.strategy_shortlist
```

Ese filtro evita que el sistema recomiende automáticamente underdogs de cuota
muy alta. En el snapshot actual de Pinnacle, la shortlist core queda en 6
apuestas, cuota media 1.83 y EV medio 16.15%.

El mismo comando genera una segunda lista de revisión manual:
`outputs/wc2026_ev_h2h_manual_review.csv`. Ahí entran selecciones que el modelo
ve posibles, pero que no pasan el perfil core por cuota alta o fiabilidad baja.
Esas señales se tratan como `manual_check` o `paper_only`, no como apuestas
automáticas.

Para registrar las señales en el tracker local:

```bash
python -m src.evaluation.bet_tracker
```

El tracker queda en `data/tracking/wc2026_bet_tracker.csv` y no se sube a Git
porque es estado vivo. Conserva `first_odds`, actualiza `latest_odds` en cada
run y deja columnas preparadas para `closing_odds`, `clv_pct`, resultado,
beneficio y notas manuales.

Staking v1:

- `core_candidate`: stake conservador recomendado de `0.5u` sobre banca base
  de `100u`.
- `manual_check`: stake `0u` hasta aprobación manual.
- `paper_only`: stake `0u`, solo seguimiento de CLV.

El tracker guarda `recommended_stake_units`, `stake_units`, `bankroll_units`,
`stake_method` y `risk_notes`. Si editáis manualmente una fila marcada como
`placed`, el updater conserva la decisión.

Comando único de v1 1X2:

```bash
# Gasta una llamada de cuotas actuales
python -m src.evaluation.run_1x2_v1

# No gasta crédito; reutiliza outputs/wc2026_ev_h2h_strategy.csv
python -m src.evaluation.run_1x2_v1 --skip-ev
```

Para cambiar banca/stake core:

```bash
python -m src.evaluation.run_1x2_v1 --bankroll-units 100 --core-stake-units 0.5
```

## Radar multi-mercado

La v1 operativa sigue limitada a 1X2 core. Para ampliar el análisis sin gastar
créditos ni forzar apuestas, `src/evaluation/market_probabilities.py` convierte
el ensemble en probabilidades y cuotas justas de varios mercados:

- 1X2, doble oportunidad y draw no bet.
- Totales 1.5 / 2.5 / 3.5.
- BTTS.
- Goles de equipo 0.5 / 1.5 / 2.5.
- Porterías a cero.
- Córners totales 8.5 / 9.5 / 10.5 como aproximación Poisson.

La fuente de verdad sobre qué mercados son apostables, paper o solo radar es:

```bash
python -m src.evaluation.market_registry
```

Genera `data/processed/market_registry.csv` con `model_status`,
`validation_status`, `betting_status`, `stake_allowed` y el siguiente paso de
cada mercado. El radar y el EV local arrastran esas columnas para evitar que un
mercado derivado parezca validado.

```bash
python -m src.evaluation.market_probabilities
```

Genera `outputs/wc2026_market_probabilities.csv`. Ese archivo no contiene
cuotas reales ni EV; es un radar para saber qué cuota mínima necesitaríamos ver
en una casa. Incluye `fair_odds` y umbrales `min_odds_ev5`, `min_odds_ev10` y
`min_odds_ev15`. Los córners salen marcados con confianza baja porque se derivan
de una media esperada, no de una distribución calibrada.

Si ya existe un CSV plano de cuotas actuales, se puede cruzar sin llamar a la
API:

```bash
python -m src.evaluation.multi_market_ev
```

Por defecto lee `data/processed/odds_current_worldcup_flat.csv` y guarda
`outputs/wc2026_multi_market_ev.csv`. Ese output contiene cuotas vivas y no se
versiona.

Para sacar una shortlist de seguimiento en papel de mercados no validados:

```bash
python -m src.evaluation.multi_market_shortlist
```

Por defecto solo considera `total_goals` y `btts`, líneas principales, EV
moderado, cuotas `1.50-2.50`, fiabilidad mínima `MEDIA` y confianza de modelo
al menos `medium`. El resultado se usa para medir CLV/resultado antes de
permitir apuestas reales.

Backtest dedicado de Over/Under 2.5, sin gastar API:

```bash
python -m src.evaluation.total_goals_backtest
```

Resultado actual con el backtest WC2022 disponible: la estrategia por defecto
(`5% <= EV <= 25%`, cuotas `1.50-2.50`) da 10 apuestas, 40% de acierto y ROI
`-23.5%`. Por tanto `total_goals_2_5` se mantiene como `paper_only`, no pasa a
stake real.

Tracker local de mercados paper:

```bash
python -m src.evaluation.paper_tracker
```

Guarda `data/tracking/wc2026_paper_tracker.csv`, con `first_odds`,
`latest_odds`, `closing_odds`, `clv_pct`, resultado y beneficio simulado. Ese
CSV es estado local y no se versiona.

Flujo completo de fase 2 en un comando, sin llamar a la API:

```bash
python -m src.evaluation.run_market_radar
```

## Features principales

| Feature | Cobertura | Descripción |
|---|---|---|
| `elo_diff` | 100% histórico | Diferencia de Elo entre equipos |
| `home/away_form_*_10` | 99% | Forma reciente (últimos 10 partidos) |
| `effective_home_adv` | 100% | 0.0 en WC2026 (sede neutral), 1.0 en partidos normales |
| `rating_diff` | 100% WC2026 | Diferencia de rating compuesto |
| `home/away_attack_norm` | 100% WC2026 | Valor de ataque normalizado |
| `home/away_defense_norm` | 100% WC2026 | Valor de defensa normalizado |
| `home/away_yellow_per_90` | 100% WC2026 | Propensión a tarjetas del equipo |
| `home/away_goals_per_90` | 100% WC2026 | Capacidad goleadora del equipo |
| `ref_yellow_per_match` | Parcial | Perfil disciplinario del árbitro |
| `altitude_m` | 100% WC2026 | Altitud de la sede (Ciudad de México: 2240m) |

## Trabajo pendiente

- Datos históricos de córners y tarjetas para selecciones (solo 165 partidos disponibles)
- Resultado al descanso (HT)
- Recalibrar la clase empate o mantenerla excluida de las apuestas
- Conectar EV y fiabilidad al tracker de apuestas en vivo
- Ampliar backtesting a más torneos con cache de The Odds API
- Árbitros: rellenar `data/raw/referees.csv` manualmente desde transfermarkt.es

## Evaluación

Simulación de apuestas en Mundiales 2018 y 2022 midiendo ROI, Brier Score y ECE.
Primer backtest WC2022 con cuotas de cierre de Pinnacle disponible en
`outputs/backtest2022_bets.csv`; la versión filtrada está en
`outputs/backtest2022_filtered_bets.csv`.
