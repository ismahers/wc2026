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
| `data/raw/squads_wc2026_fifa_official.csv` | Convocatorias extraidas del PDF oficial de FIFA. Candidata a sustituir la fuente base tras revision. |
| `docs/SquadLists-English.pdf` | PDF oficial de FIFA usado como fuente auditable. |

### Identidad y agregados de jugadores

| Archivo | Generador | Motivo |
|---|---|---|
| `data/manual/player_expected_minutes_overrides_wc2026.csv` | Manual | Overrides de alineaciones, bajas y minutos esperados. No se sobreescribe al regenerar. |
| `data/processed/players_master.csv` | `python -m src.data.player_master` | Claves internas estables de los 1248 jugadores. |
| `data/processed/player_source_matches.csv` | `python -m src.data.player_master` y `python -m src.data.match_transfermarkt` | Tabla puente hacia FBref, Transfermarkt y StatsBomb. |
| `data/processed/transfermarkt_player_profiles.csv` | `python -m src.data.match_transfermarkt` | Perfiles Transfermarkt enlazados a `player_key`. |
| `data/processed/transfermarkt_match_review.csv` | `python -m src.data.match_transfermarkt` | Casos que requieren revision manual. |
| `data/processed/players_wc2026.csv` | `python -m src.data.normalize_squads` | Jugadores normalizados para el pipeline actual. |
| `data/processed/teams_wc2026.csv` | `python -m src.data.normalize_squads` | Selecciones normalizadas. |
| `data/processed/squad_summary_wc2026.csv` | `python -m src.data.normalize_squads` | Resumen por seleccion usado por `builder.py`. |
| `data/processed/team_player_stats_wc2026.csv` | `python -m src.data.player_stats_aggregator` | Stats agregadas por seleccion. |
| `data/processed/player_prop_features_wc2026.csv` | `python -m src.data.player_prop_features` | Features Transfermarkt a nivel jugador para props v0. |
| `data/processed/player_prop_readiness_summary.csv` | `python -m src.data.player_prop_features` | Cobertura por seleccion para props v0. |
| `data/processed/player_season_stats.parquet` | `python -m src.data.normalize_fbref` | Tabla comun player-season normalizada desde FBref/soccerdata. |
| `data/processed/player_season_stats.csv` | `python -m src.data.normalize_fbref` | Copia CSV revisable de la tabla player-season FBref. |
| `data/processed/fbref_player_features_wc2026.csv` | `python -m src.data.fbref_player_features` | Features FBref por convocado para tiros, SoT, tackles, faltas, tarjetas y porteros. |
| `data/processed/fbref_player_match_review.csv` | `python -m src.data.fbref_player_features` | Filas FBref sin matching conservador a `player_key`. |
| `data/processed/fbref_player_feature_summary.csv` | `python -m src.data.fbref_player_features` | Cobertura FBref por seleccion. |
| `data/processed/player_expected_minutes_wc2026.csv` | `python -m src.evaluation.player_minutes_projection` | Minutos esperados por jugador-partido, con base automatica y overrides manuales aplicados. |
| `data/processed/team_ratings_wc2026.csv` | `python -m src.data.team_rating_wc2026` | Rating compuesto para WC2026. |
| `data/processed/referees_with_stats.csv` | `src.features.referee_rates` / carga manual | Tasas de arbitros usadas por features. |
| `data/processed/name_coverage.csv` | `python scripts/check_name_coverage.py` | Reporte de cobertura de nombres canonicos. |
| `data/processed/fifa_squad_diff.csv` | `python -m src.data.fifa_squads_pdf` | Diferencias entre el CSV actual y el PDF FIFA. |
| `data/processed/fifa_squad_summary.csv` | `python -m src.data.fifa_squads_pdf` | Resumen por tipo de diferencia. |
| `data/processed/market_registry.csv` | `python -m src.evaluation.market_registry` | Estado operativo de cada mercado: entrenado, derivado, paper, live o bloqueado. |

### Outputs versionados

| Archivo | Motivo |
|---|---|
| `outputs/wc2026_predictions.csv` | Predicciones XGBoost principales. |
| `outputs/wc2026_predictions_with_odds.csv` | Cuotas justas derivadas de XGBoost. |
| `outputs/wc2026_poisson_predictions.csv` | Predicciones Poisson de goles. |
| `outputs/wc2026_poisson_predictions_with_odds.csv` | Cuotas justas derivadas de Poisson. |
| `outputs/wc2026_ensemble_predictions.csv` | Prediccion final del ensemble, una unica cuota por mercado. |
| `outputs/wc2026_market_probabilities.csv` | Radar multi-mercado con cuotas justas y estado del registry. |
| `outputs/model_comparison.csv` | Diagnostico XGBoost vs Poisson. |
| `outputs/xgb_baseline_metrics.json` | Metricas XGBoost. |
| `outputs/poisson_metrics.json` | Metricas Poisson. |
| `outputs/backtest2022_total_goals_v2.csv` | Backtest dedicado de Over/Under 2.5. |
| `outputs/backtest2022_total_goals_v2_summary.csv` | Resumen de ROI del backtest de Over/Under 2.5. |
| `outputs/backtest2022_total_goals_v2_grid.csv` | Sensibilidad por umbrales EV para Over/Under 2.5. |
| `outputs/backtest2022_strategy_summary.csv` | Diagnostico ampliado del backtest ensemble: ROI, hit-rate CI95 y drawdown. |
| `outputs/backtest2022_strategy_grid.csv` | Grid de sensibilidad EV/cuota para el backtest ensemble. |
| `outputs/backtest2022_strategy_filtered_bets.csv` | Filas del backtest ensemble con razones de filtro y profit de estrategia. |
| `outputs/backtest2022_legacy_strategy_summary.csv` | Diagnostico equivalente sobre el backtest legacy. |
| `outputs/backtest2022_legacy_strategy_grid.csv` | Grid de sensibilidad legacy para comparar con el resultado historico +53%. |
| `outputs/backtest2022_legacy_strategy_filtered_bets.csv` | Filas filtradas legacy con razones y profit de estrategia. |
| `outputs/btts_readiness_summary.csv` | Semaforo BTTS: estado del modelo, cobertura de cuotas y decision operativa. |
| `outputs/wc2026_btts_radar_candidates.csv` | Candidatos radar BTTS si aparecen cuotas suficientes, sin stake real. |
| `outputs/wc2026_market_availability.csv` | Disponibilidad local por mercado: probabilidades, odds, EV, tracker y accion recomendada. |
| `outputs/wc2026_market_availability_summary.csv` | Resumen por estado operativo de disponibilidad. |
| `outputs/wc2026_player_prop_radar.csv` | Radar de props de jugador sin odds, stake ni lineups confirmadas. |
| `outputs/wc2026_player_prop_radar_summary.csv` | Resumen por mercado/linea/calidad de dato del radar player props. |
| `outputs/wc2026_player_minutes_projection_summary.csv` | Resumen de minutos esperados por fuente, estado de alineacion y overrides. |
| `docs/supabase_player_prop_radar.sql` | SQL para crear la tabla movil `player_prop_radar` en Supabase. |

## Ignorados

Estos archivos son pesados, regenerables o dependen de snapshots/API.

| Patron | Motivo |
|---|---|
| `data/raw/*` salvo whitelisted WC2026 | Datos crudos externos o descargables. |
| `data/raw/fbref/player_season/*` | Crudos FBref/soccerdata, regenerables y sujetos a rate limits/cache local. |
| `data/raw/international_match_stats.csv` | Generado por StatsBomb collector. |
| `data/results.csv`, `data/results_raw.csv` | Histórico externo descargable. |
| `data/unified.csv` | Generado por `data_collector.py`. |
| `data/statsbomb_matches.csv` | Generado por `src.data.statsbomb_collector`. |
| `data/processed/features*.csv` | Matrices grandes generadas por feature engineering. |
| `data/processed/matches_enriched.csv` | Dataset enriquecido generado por builder. |
| `data/processed/team_ratings.csv` | Elo histórico grande generado por `elo_loader.py`. |
| `data/processed/odds_current*.csv` | Snapshots de cuotas actuales con timestamp/API. |
| `data/tracking/wc2026_paper_tracker.csv` | Tracker local de paper trading, estado vivo. |
| `outputs/*_check/` | Diagnosticos temporales de runs parciales. |
| `data/transfermarkt/player_performances.csv` | Archivo grande de Transfermarkt. |

## Comandos de regeneracion

Orden recomendado cuando cambian datos base:

```bash
python -m src.data.player_master
python -m src.data.fifa_squads_pdf
python -m src.data.match_transfermarkt
python -m src.data.normalize_squads
python -m src.data.elo_loader
python -m src.data.team_rating_wc2026
python -m src.data.player_stats_aggregator
python -m src.data.player_prop_features
# Opcional, solo cuando se quiera refrescar FBref:
python -m src.data.fbref_probe_leagues --leagues "USA-Major League Soccer" "NED-Eredivisie" "POR-Primeira Liga"
python -m src.data.fbref_collector
python -m src.data.normalize_fbref
python -m src.data.fbref_player_features
python -m src.data.builder
python -m src.features.feature_engineering
python -m src.models.xgboost_model --markets result_1x2,over25,btts,total_goals,corners
python -m src.models.poisson_model
python -m src.evaluation.odds_converter --input outputs/wc2026_predictions.csv --output outputs/wc2026_predictions_with_odds.csv
python -m src.evaluation.odds_converter --input outputs/wc2026_poisson_predictions.csv --output outputs/wc2026_poisson_predictions_with_odds.csv
python -m src.evaluation.model_comparison
python -m src.evaluation.model_ensemble
python -m src.evaluation.market_registry
python -m src.evaluation.run_market_radar
python -m src.evaluation.total_goals_backtest
python -m src.evaluation.btts_readiness
python -m src.evaluation.market_availability
python -m src.evaluation.paper_tracker
python -m src.evaluation.player_minutes_projection
python -m src.evaluation.player_prop_radar
python -m src.evaluation.player_prop_radar_push --dry-run
```

`player_prop_radar_push` sube por defecto solo las props accionables del dia
actual en horario Europe/Madrid. Para revisar otro dia o todo el torneo:

```bash
python -m src.evaluation.player_prop_radar_push --date 2026-06-12
python -m src.evaluation.player_prop_radar_push --match-number 1
python -m src.evaluation.player_prop_radar_push --all
```

## FBref Seguro

Antes de scrapear una liga nueva, comprobar que `soccerdata` reconoce el nombre:

```bash
python -m src.data.fbref_probe_leagues \
  --leagues "USA-Major League Soccer" "NED-Eredivisie" "POR-Primeira Liga"
```

Las ligas con temporada europea usan `2025-26`; MLS va por año natural y debe
probarse con `2026`. Para pruebas pequeñas, usar un directorio temporal y
`--no-overwrite`:

```bash
python -m src.data.fbref_collector \
  --leagues "NED-Eredivisie" \
  --seasons 2025-26 \
  --stat-types standard \
  --output-dir data/raw/fbref_probe \
  --no-overwrite

python -m src.data.fbref_collector \
  --leagues "USA-Major League Soccer" \
  --seasons 2026 \
  --stat-types standard \
  --output-dir data/raw/fbref_probe \
  --no-overwrite
```
