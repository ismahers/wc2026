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

## Pipeline

```
Datos históricos (Kaggle + StatsBomb)
        ↓
Corrección de distribución (KMM / concept drift)     ← T2
        ↓
Feature engineering (racha, ranking FIFA, árbitro...)
        ↓
┌─────────────────┬──────────────────────────────┐
│  Modelo clásico │      Red neuronal multi-output │
│  Frank&Hall /   │      MLP / siamesa             │
│  SVOR (ordinal) │      + Temperature Scaling     │
└────────┬────────┴──────────────┬───────────────┘
         └──────────┬────────────┘
              Calibración + Cuantificación (HDy/PAC)  ← T2
                        ↓
              Detección de valor esperado
              VE(r) = p̂(r) × cuota - 1
```

## Mercados predichos

| Mercado | Tipo |
|---|---|
| Resultado (1X2) | Ordinal / 3 clases |
| Over/Under 2.5 goles | Binario |
| Ambos equipos marcan (BTTS) | Binario |
| Resultado al descanso | 3 clases |
| Córners totales | Regresión |
| Tarjetas amarillas | Regresión / ordinal |
| Tarjetas rojas | Binario |
| Quién marca primero | Multinomial |

## Evaluación

Simulación de apuestas en Mundiales 2018 y 2022 midiendo ROI, Brier Score y ECE.
