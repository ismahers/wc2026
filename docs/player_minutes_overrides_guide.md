# Guia rapida: overrides de minutos esperados

Esta capa sirve para ajustar el radar de props cuando tengamos noticias de
alineaciones probables, titulares confirmados, lesiones o sanciones.

## Archivos

| Archivo | Se edita a mano | Funcion |
|---|---:|---|
| `data/manual/player_expected_minutes_overrides_wc2026.csv` | Si | Overrides manuales. Nunca lo pisa el codigo. |
| `data/processed/player_expected_minutes_wc2026.csv` | No | Tabla final generada con base automatica + overrides. |
| `outputs/wc2026_player_minutes_projection_summary.csv` | No | Resumen de cobertura y estados. |
| `outputs/wc2026_player_prop_radar.csv` | No | Radar de props ya usando los minutos finales. |

## Flujo antes de un partido

1. Abrir `data/processed/player_expected_minutes_wc2026.csv` y filtrar por `match_number`.
2. Copiar al CSV manual solo los jugadores que queramos corregir.
3. Rellenar `lineup_status`, minutos o probabilidad de titular.
4. Regenerar:

```bash
python -m src.evaluation.player_minutes_projection
python -m src.evaluation.player_prop_radar
```

## Columnas del CSV manual

```csv
match_number,team,opponent,player_key,player_name,lineup_status,manual_expected_minutes,manual_start_probability,minutes_source,confidence,notes
```

Campos clave:

| Campo | Uso |
|---|---|
| `match_number` | Numero de partido en `data/raw/group_stage_wc2026.csv`. |
| `team` | Seleccion canonica, igual que en el resto del pipeline. |
| `player_key` | Clave interna del jugador. Es el identificador fuerte. |
| `lineup_status` | Estado manual de alineacion. |
| `manual_expected_minutes` | Minutos finales si queremos fijarlos directamente. |
| `manual_start_probability` | Probabilidad manual de titular si preferimos no fijar minutos. |
| `minutes_source` | Ejemplo: `probable_lineup`, `confirmed_lineup`, `injury_news`. |
| `confidence` | Confianza del ajuste, de 0 a 1. |
| `notes` | Motivo humano del cambio. |

## Estados validos

```text
locked_starter
probable_starter
rotation
bench_option
injury_doubt
suspended
out
confirmed_starter
confirmed_bench
not_in_squad
```

Reglas practicas:

| Caso | `lineup_status` | Minutos recomendados |
|---|---|---:|
| Titular oficial | `confirmed_starter` | 65-90 segun posicion |
| Titular casi seguro | `locked_starter` | 65-85 |
| Probable titular | `probable_starter` | 55-80 |
| Duda entre titular y suplente | `rotation` | 25-55 |
| Suplente probable | `bench_option` | 5-30 |
| Baja / sancionado | `out` o `suspended` | 0 |

## Ejemplos

Titular probable:

```csv
1,Mexico,South Africa,mexico_santiago_gimenez,Santiago Gimenez,probable_starter,72,0.85,probable_lineup,0.75,Delantero titular esperado
```

Baja confirmada:

```csv
1,Mexico,South Africa,mexico_johan_vasquez,Johan Vasquez,out,0,0,injury_news,0.95,Baja confirmada
```

## Como leer el radar

`stake_allowed` sigue siempre en `False`: esta capa no habilita apuesta real por
si sola. Para seguimiento en papel, usar `paper_tracking_allowed=True`.

Una senal de props solo deberia revisarse si:

- `paper_tracking_allowed=True`
- `data_quality_tier` es `A` o `B`
- `final_expected_minutes` / `expected_minutes` tiene sentido para el jugador
- la cuota real supera claramente la `fair_odds`
- la alineacion no contradice el mercado

