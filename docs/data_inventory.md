# Inventario de Datos

Este documento cierra la politica de CSV del repo: que se versiona, que se genera y que queda fuera de Git.

## Versionados

Estos archivos son pequenos, revisables o contienen decisiones manuales del proyecto.

### Datos estaticos WC2026

| Archivo | Motivo |
|---|---|
| `data/raw/venues.csv` | Sedes, coordenadas, altitud, techo y superficie. |
| `data/raw/referees_wc2026.csv` | Lista de arbitros candidatos/designados para el torneo. |
| `data/raw/base_camps_wc2026.csv` | Campamentos base de las 48 selecciones. |
| `data/raw/group_stage_wc2026.csv` | Calendario de fase de grupos. |
| `data/raw/knockout_wc2026.csv` | Slots de eliminatorias. |
| `data/raw/squads_wc2026_final_official_corrected.csv` | Convocatorias oficiales corregidas. Es fuente base, no se sobreescribe. |

### Identidad y agregados de jugadores

| Archivo | Generador | Motivo |
|---|---|---|
| `data/processed/players_master.csv` | `python -m src.data.player_master` | Claves internas estables de los 1248 jugadores. |
| `data/processed/player_source_matches.csv` | `python -m src.data.player_master` y `python -m src.data.match_transfermarkt` | Tabla puente hacia FBref, Transfermarkt y StatsBomb. |
| `data/processed/transfermarkt_player_profiles.csv` | `python -m src.data.match_transfermarkt` | Perfiles Transfermarkt enlazados a `player_key`. |
| `data/processed/transfermarkt_match_review.csv` | `python -m src.data.match_transfermarkt` | Casos que requieren revision manual. |
| `data/processed/players_wc2026.csv` | `python -m src.data.normalize_squads` | Jugadores normalizados para el pipeline actual. |
| `data/processed/teams_wc2026.csv` | `python -m src.data.normalize_squads` | Selecciones normalizadas. |
| `data/processed/squad_summary_wc2026.csv` | `python -m src.data.normalize_squads` | Resumen por seleccion usado por `builder.py`. |
| `data/processed/team_player_stats_wc2026.csv` | `python -m src.data.player_stats_aggregator` | Stats agregadas por seleccion. |
| `data/processed/team_ratings_wc2026.csv` | `python -m src.data.team_rating_wc2026` | Rating compuesto para WC2026. |
| `data/processed/referees_with_stats.csv` | `src.features.referee_rates` / carga manual | Tasas de arbitros usadas por features. |
| `data/processed/name_coverage.csv` | `python scripts/check_name_coverage.py` | Reporte de cobertura de nombres canonicos. |

### Outputs versionados

| Archivo | Motivo |
|---|---|
| `outputs/wc2026_predictions.csv` | Predicciones XGBoost principales. |
| `outputs/wc2026_predictions_with_odds.csv` | Cuotas justas derivadas de XGBoost. |
| `outputs/wc2026_poisson_predictions.csv` | Predicciones Poisson de goles. |
| `outputs/wc2026_poisson_predictions_with_odds.csv` | Cuotas justas derivadas de Poisson. |
| `outputs/wc2026_ensemble_predictions.csv` | Prediccion final del ensemble, una unica cuota por mercado. |
| `outputs/model_comparison.csv` | Diagnostico XGBoost vs Poisson. |
| `outputs/xgb_baseline_metrics.json` | Metricas XGBoost. |
| `outputs/poisson_metrics.json` | Metricas Poisson. |

## Ignorados

Estos archivos son pesados, regenerables o dependen de snapshots/API.

| Patron | Motivo |
|---|---|
| `data/raw/*` salvo whitelisted WC2026 | Datos crudos externos o descargables. |
| `data/raw/international_match_stats.csv` | Generado por StatsBomb collector. |
| `data/results.csv`, `data/results_raw.csv` | Histórico externo descargable. |
| `data/unified.csv` | Generado por `data_collector.py`. |
| `data/statsbomb_matches.csv` | Generado por `src.data.statsbomb_collector`. |
| `data/processed/features*.csv` | Matrices grandes generadas por feature engineering. |
| `data/processed/matches_enriched.csv` | Dataset enriquecido generado por builder. |
| `data/processed/team_ratings.csv` | Elo histórico grande generado por `elo_loader.py`. |
| `data/processed/odds_current*.csv` | Snapshots de cuotas actuales con timestamp/API. |
| `outputs/*_check/` | Diagnosticos temporales de runs parciales. |
| `data/transfermarkt/player_performances.csv` | Archivo grande de Transfermarkt. |

## Comandos de regeneracion

Orden recomendado cuando cambian datos base:

```bash
python -m src.data.player_master
python -m src.data.match_transfermarkt
python -m src.data.normalize_squads
python -m src.data.elo_loader
python -m src.data.team_rating_wc2026
python -m src.data.player_stats_aggregator
python -m src.data.builder
python -m src.features.feature_engineering
python -m src.models.xgboost_model --markets result_1x2,over25,btts,total_goals,corners
python -m src.models.poisson_model
python -m src.evaluation.odds_converter --input outputs/wc2026_predictions.csv --output outputs/wc2026_predictions_with_odds.csv
python -m src.evaluation.odds_converter --input outputs/wc2026_poisson_predictions.csv --output outputs/wc2026_poisson_predictions_with_odds.csv
python -m src.evaluation.model_comparison
python -m src.evaluation.model_ensemble
```
