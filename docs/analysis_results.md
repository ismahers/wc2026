# 🔍 Análisis Exhaustivo — Pronosticador WC2026

Análisis completo del repositorio tras revisar **todos** los módulos de datos, features, modelos, evaluación y estrategia de apuestas.

---

## 📊 Estado Actual del Sistema

| Componente | Estado | Observación |
|---|---|---|
| Pipeline de datos | ✅ Sólido | Builder + normalización + Elo + ratings compuestos |
| Feature engineering | ✅ Bueno | Sin data leakage, O(log n), buena cobertura |
| Modelo XGBoost | ⚠️ Aceptable | acc=0.594, AUC Over2.5=0.611, BTTS AUC=0.567 |
| Modelo Poisson | ✅ Bueno | acc=0.605 en 1X2, coherencia entre mercados |
| Ensemble | ⚠️ Mejorable | Pesos fijos, sin validación del peso óptimo |
| Calibración | ✅ Buena | ECE 1X2 ~0.013, Over2.5 ~0.017, BTTS ~0.021 |
| Backtest | ⚠️ Insuficiente | Solo 28 apuestas en WC2022, ROI +53% pero n muy bajo |
| Staking | ⚠️ Básico | Flat 0.5u, sin Kelly fraccional, sin bankroll management |
| Multi-mercado | 🔴 Incompleto | Solo 1X2 operativo, total_goals en paper, BTTS sin cuotas |

---

## 🐛 FALLOS DETECTADOS

### 1. CRÍTICO — Refit con todos los datos contamina la evaluación indirectamente

**Archivo**: [xgb_baseline.py](file:///Users/danifdez/Documents/Mundial/wc2026/src/models/xgb_baseline.py#L553-L561)

```python
# ── NUEVO: reentrenar con TODOS los datos para predecir WC2026 ──
full_df = df[df[config.target_col].notna()].copy()
refit_model = MarketModel(config)
refit_model.fit(full_df)
self.models[market_key] = refit_model
```

El modelo se evalúa primero en `df_val` (correcto), pero luego se **reemplaza** con un modelo reentrenado con TODOS los datos (incluyendo la validación). Esto es intencional para predicciones WC2026, pero:

- Los `feature_fill_values` del modelo final son diferentes a los del modelo evaluado (medianas de train vs medianas de todo).
- Las métricas reportadas en `xgb_baseline_metrics.json` **no corresponden** al modelo que genera las predicciones reales.
- Si alguien usa esas métricas para calibrar confianza, estaría usando métricas de un modelo distinto.

> [!WARNING]
> Las métricas de evaluación no corresponden al modelo que realmente predice WC2026. Solución: guardar también métricas del modelo refitted (walk-forward cross-validation).

### 2. CRÍTICO — Modelo Poisson asume independencia de goles (home, away)

**Archivo**: [poisson_model.py](file:///Users/danifdez/Documents/Mundial/wc2026/src/models/poisson_model.py#L116-L125)

```python
def score_matrix(lambda_home, lambda_away, max_goals=MAX_GOALS):
    home_probs = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    away_probs = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    return np.outer(home_probs, away_probs)  # ← Independencia total
```

La distribución **real** de goles en fútbol tiene **correlación positiva** (empates 0-0 y 1-1 son más frecuentes de lo que predice Poisson independiente). Esto causa:
- **Subestimación de empates** → confirma que el README dice "no apostar empates" porque el modelo los sobreestima. En realidad, el modelo XGBoost los sobreestima pero el Poisson los **subestima** ligeramente. El ensemble promedia, pero no soluciona la raíz del problema.
- **Over/Under sesgado** → afecta la precisión de totals.

### 3. IMPORTANTE — BTTS con AUC = 0.567 es prácticamente inútil

**Archivo**: [xgb_baseline_metrics.json](file:///Users/danifdez/Documents/Mundial/wc2026/outputs/xgb_baseline_metrics.json)

Un AUC de 0.567 está apenas por encima del azar (0.5). Las probabilidades BTTS que genera el modelo no contienen suficiente información para generar valor. Sin embargo, el sistema las usa en el ensemble y en `market_probabilities.py` para generar "cuotas justas" BTTS. Apostar con estas probabilidades sería como apostar al azar con una ligera ventaja teórica que se pierde por el vig.

### 4. IMPORTANTE — Kelly criterion implementado pero nunca usado

**Archivo**: [betting.py](file:///Users/danifdez/Documents/Mundial/wc2026/src/evaluation/betting.py#L326-L330)

```python
if strategy == "kelly":
    p = bet["prob_model"]
    b = bet["odds"] - 1
    k = max(0, (b * p - (1 - p)) / b)
    s = stake * k
```

El Kelly está implementado en `simulate_roi()` pero:
- **Nunca se invoca** con `strategy="kelly"` en ningún script del repo.
- El staking real es flat `0.5u` para todos los core_candidates.
- No hay Kelly fraccional (1/4 Kelly o 1/2 Kelly), que es lo recomendado en la práctica.

### 5. IMPORTANTE — Córners con solo 111 partidos de evaluación y MAE = 3.17

**Archivo**: [xgb_baseline_metrics.json](file:///Users/danifdez/Documents/Mundial/wc2026/outputs/xgb_baseline_metrics.json)

```json
{"market": "Córners Totales", "n_test": 111, "mae": 3.1705}
```

Con un MAE de 3.17 sobre una media de 9.22, el error relativo es del **34%**. Y con solo 111 muestras de test, las métricas no son fiables estadísticamente. Las predicciones de córners no deberían usarse para apostar.

### 6. MEDIO — Modelo ordinal (Frank & Hall) implementado pero no integrado

**Archivo**: [ordinal_model.py](file:///Users/danifdez/Documents/Mundial/wc2026/src/models/ordinal_model.py)

El modelo ordinal con KMM está completo pero:
- No se ejecuta en el pipeline.
- No se incluye en el ensemble.
- No se evalúa comparativamente con los otros dos modelos.
- El KMM es computacionalmente caro (O(n²)) y no escalará bien con 32k partidos.

### 7. MEDIO — `n_estimators` se muta del diccionario de params

**Archivo**: [poisson_model.py](file:///Users/danifdez/Documents/Mundial/wc2026/src/models/poisson_model.py#L230)

```python
n_est = self.params.pop("n_estimators", 300)  # ← Muta el dict
```

Esto modifica `self.params` in-place. Si se llama a `fit()` dos veces, la segunda vez `n_estimators` ya no existe en el dict. El fix al final (`self.params["n_estimators"] = n_est`) lo mitiga, pero es frágil.

### 8. BAJO — Backtest WC2022 con solo 12 partidos cruzados

**Archivo**: [backtest_2022.py](file:///Users/danifdez/Documents/Mundial/wc2026/src/evaluation/backtest_2022.py#L4)

El backtest reporta 28 apuestas y ROI +53%, pero:
- Solo 12 partidos (los que tenían cuotas de cierre de Pinnacle vía The Odds API).
- Con n=28 apuestas, el intervalo de confianza del ROI es enorme (aprox. ±40%).
- Es **insuficiente** para validar una estrategia de betting.

---

## 🚀 MEJORAS PROPUESTAS

### A. Modelos — Prioridad ALTA

#### A1. Bivariante Poisson corregida (Dixon-Coles)

Reemplazar el Poisson independiente por el modelo **Dixon-Coles** que añade un parámetro ρ (rho) de correlación para marcadores bajos (0-0, 1-0, 0-1, 1-1). Esto:
- Mejora la calibración de empates significativamente.
- Es el estándar de la industria para betting de fútbol.
- Se implementa añadiendo un factor multiplicativo a la matriz de scores para i,j ∈ {0,1}.

```python
def dixon_coles_adjust(matrix, lambda_home, lambda_away, rho):
    """Ajusta la matriz de Poisson independiente con correlación Dixon-Coles."""
    matrix[0,0] *= 1 - lambda_home * lambda_away * rho
    matrix[1,0] *= 1 + lambda_away * rho
    matrix[0,1] *= 1 + lambda_home * rho
    matrix[1,1] *= 1 - rho
    matrix /= matrix.sum()  # renormalizar
    return matrix
```

**Impacto estimado**: +2-4% accuracy en 1X2, mejor calibración de empates, posibilidad de apostar empates.

#### A2. Hyperparameter tuning con Optuna

Actualmente todos los hiperparámetros de XGBoost son **fijos manualmente** (max_depth=4-5, lr=0.05, n_estimators=200-300). Propuesta:

- Integrar **Optuna** con walk-forward cross-validation.
- Optimizar para log_loss (no accuracy) ya que lo que importa es la calibración de probabilidades.
- Incluir regularización como variable a optimizar (reg_alpha, reg_lambda, min_child_weight).

#### A3. Calibración post-hoc (Platt scaling / isotonic regression)

Las ECE actuales son buenas (~0.013-0.021), pero para betting cada décima de punto cuenta. Propuesta:
- Añadir **Platt scaling** al pipeline de predicción final.
- Usar `sklearn.calibration.CalibratedClassifierCV` con `method='isotonic'` sobre validación.
- El `calibration_report.py` ya detecta los bins donde el modelo falla → usar eso para corregir.

#### A4. Ensemble con pesos optimizados

**Archivo**: [model_ensemble.py](file:///Users/danifdez/Documents/Mundial/wc2026/src/evaluation/model_ensemble.py#L50-L82)

Actualmente los pesos XGB/Poisson se calculan con una función lineal de `rating_diff`:

```python
scale = (strength - close_rating) / (strong_rating - close_rating)
weight = min_weight + scale * (max_weight - min_weight)
```

Los parámetros (`close_rating=0.15`, `strong_rating=0.45`, `min_weight=0.25`, `max_weight=0.80`) son arbitrarios. Propuesta:
- Optimizar estos 4 parámetros minimizando el log_loss en validación WC2018+WC2022.
- Explorar **stacking** (logistic regression sobre las probabilidades de ambos modelos) en lugar de average ponderado.

---

### B. Features — Prioridad MEDIA-ALTA

#### B1. Expected Goals (xG) pre-torneo

Las features actuales usan goles brutos. El xG (Expected Goals) es **mucho más predictivo** que los goles reales porque elimina la varianza. Propuesta:
- Si StatsBomb tiene xG para los partidos históricos, usarlo como feature.
- Calcular un "xG diferencial" de los últimos 10 partidos como feature de forma.

#### B2. Head-to-head histórico entre parejas

No hay ninguna feature de H2H entre los dos equipos del partido. En fútbol internacional hay matchups que se repiten (ej. Argentina-Brasil, España-Italia) con patrones claros. Propuesta:
- `h2h_wins_home_last5`: proporción de victorias del home en los últimos 5 enfrentamientos.
- `h2h_total_goals_avg`: media de goles en los H2H recientes.
- `h2h_matches_played`: nº de partidos entre estas dos selecciones.

#### B3. Importancia del partido (pressure factor)

No hay ninguna feature que capture la presión del partido. Un partido de fase de grupos con clasificación decidida es muy diferente de un cuarto de final. Propuesta:
- `stage_ordinal` ya existe pero es lineal (0-6). Necesita ser **exponencial** (la final es mucho más presión que un cuarto).
- Añadir `elimination_pressure`: 0 para grupos, 1 para knockout.
- Considerar si los equipos necesitan ganar para clasificarse (requiere estado del grupo en tiempo real).

#### B4. Distancia de viaje para partidos históricos

`home_travel_km` y `away_travel_km` solo se calculan para WC2026 (requieren base_camps). En el histórico son NaN. Esto significa que el modelo **nunca ve la importancia del viaje en entrenamiento**. Propuesta:
- Estimar distancia desde la capital del país para partidos históricos como proxy.

#### B5. Momentum/tendencia (más allá de los últimos 10)

Las features de forma solo miran los últimos 10 partidos. Propuesta:
- Añadir ventana de 5 partidos (forma corta) y 20 partidos (forma larga).
- Feature de **tendencia**: ¿la forma reciente (5) es mejor o peor que la media (20)?

---

### C. Backtesting — Prioridad ALTA

#### C1. Ampliar el backtest a múltiples torneos

28 apuestas no son suficientes. Necesitáis backtestear en:
- **WC 2014**: entrenar con <2014, apostar en 2014 con cuotas de Pinnacle.
- **Euro 2016, 2020/2021**: más datos de torneos internacionales.
- **Copa América 2024**: datos recientes.
- **Copa Confederaciones** si hay cuotas disponibles.

El objetivo es tener **>200 apuestas** para que el ROI sea estadísticamente significativo (p < 0.05).

#### C2. Walk-forward cross-validation

En lugar de un único split train<2018 / val 2018-2022:
- Ventana deslizante: train 1990-2013 → test 2014, train 1990-2017 → test 2018, etc.
- Esto da **métricas más fiables** y permite detectar si el modelo pierde o gana rendimiento con el tiempo.

#### C3. Medir CLV (Closing Line Value) sistemáticamente

El `bet_tracker.py` tiene columnas para `closing_odds` y `clv_pct`, pero no hay pipeline automático para medirlo. El CLV es **el mejor predictor de rentabilidad a largo plazo** en betting:

```
CLV = (prob_modelo_cierre / prob_modelo_apertura - 1) × 100
```

Si vuestra línea de apertura gana consistentemente CLV positivo, estáis en el camino correcto aunque el ROI a corto plazo sea negativo.

---

### D. Staking y Gestión de Riesgo — Prioridad ALTA

#### D1. Kelly fraccional

Reemplazar flat betting (0.5u) por **1/4 Kelly** o **1/2 Kelly**:

```python
kelly_fraction = 0.25  # 1/4 Kelly
full_kelly = (prob_modelo * odds - 1) / (odds - 1)
stake = bankroll * kelly_fraction * max(0, full_kelly)
```

Ventajas:
- Apuesta más donde tiene más edge.
- Protege contra drawdowns severos.
- Teóricamente maximiza el crecimiento del bankroll.

#### D2. Simulación de Monte Carlo para el torneo

Correr 10,000 simulaciones del torneo completo usando las lambdas del Poisson para:
- Estimar probabilidades de pasar ronda para cada equipo.
- Detectar value en futuros (ganador del grupo, ganador del torneo).
- Generar distribuciones de bankroll para diferentes estrategias de staking.

#### D3. Max drawdown y stop-loss

No hay control de riesgo. Propuesta:
- Definir un **max drawdown** (ej. -30% del bankroll → parar de apostar).
- Reducir stakes progresivamente si el drawdown supera -15%.
- Simular esto sobre el backtest para ver cuánto protege.

---

### E. Infraestructura — Prioridad MEDIA

#### E1. Tests unitarios

**No hay ni un solo test** en el repositorio. Esto es muy peligroso para un sistema de betting. Propuesta mínima:
- Tests para `expected_value()`, `score_matrix()`, `derive_markets()`.
- Test de coherencia: `prob_H + prob_D + prob_A ≈ 1.0` para todas las predicciones.
- Test de no-leakage: features de un partido no usan datos del futuro.
- Test de reproducibilidad: mismos datos → mismas predicciones.

#### E2. Pipeline reproducible

Los scripts `pipeline.ps1` y `regen_fifa.ps1` son PowerShell (Windows). Propuesta:
- Crear un `Makefile` o `justfile` multiplataforma.
- Añadir `dvc` (Data Version Control) para rastrear los CSVs procesados.

#### E3. Logging estructurado

Muchos `print()` mezclados con `log.info()`. Propuesta:
- Migrar todo a logging con niveles apropiados.
- Guardar logs en `outputs/logs/` con timestamp.

---

## ❌ QUÉ FALTA PARA APOSTAR MAXIMIZANDO GANANCIAS

### 1. 🔴 Backtest estadísticamente significativo
El n=28 actual no permite saber si el ROI +53% es real o suerte. **Esto es bloqueante**.

### 2. 🔴 Dixon-Coles en lugar de Poisson independiente
Sin esto, las probabilidades de empate y los marcadores bajos están sesgados. Afecta a todos los mercados derivados.

### 3. 🔴 Kelly fraccional para staking
Flat betting con 0.5u ignora completamente la magnitud del edge. Estáis dejando dinero sobre la mesa (o asumiendo demasiado riesgo en apuestas con bajo edge).

### 4. 🟡 Más mercados operativos
Solo 1X2 está en stake real. `total_goals` en paper con ROI -23.5% indica que necesita recalibración. BTTS necesita cuotas reales para evaluar. La mayoría del value en betting viene de mercados secundarios (totals, handicap asiático), no de 1X2.

### 5. 🟡 Cuotas de múltiples casas
Solo usáis `winamax_fr` como bookmaker por defecto. Para maximizar ganancias:
- Comparar cuotas de **Pinnacle** (sharp), **Betfair Exchange** (sin margen), y 3-4 soft books.
- El edge real es la diferencia entre vuestra cuota justa y **la mejor cuota disponible**, no la de una sola casa.

### 6. 🟡 Features de xG y H2H
Sin xG os falta la feature más predictiva en fútbol moderno. Sin H2H os perdéis patrones de rivalidad.

### 7. 🟢 Automatización de ejecución
Ahora mismo hay que ejecutar 10+ comandos en orden. Un `make all` o `python -m src.run_pipeline` simplificaría mucho.

### 8. 🟢 Alertas de steam moves
Si una cuota cae de 2.50 a 2.20 en pocas horas, es una "steam move" — dinero informado entrando. Detectar esto y cruzarlo con vuestro modelo daría señales de alta calidad.

---

## 📈 Priorización de Impacto

| # | Acción | Impacto en ROI | Esfuerzo | Prioridad |
|---|---|---|---|---|
| 1 | Dixon-Coles | ⬆️⬆️⬆️ | Medio | **P0** |
| 2 | Backtest ampliado (>200 bets) | ⬆️⬆️⬆️ | Alto | **P0** |
| 3 | Kelly fraccional | ⬆️⬆️ | Bajo | **P0** |
| 4 | Optuna tuning + calibración post-hoc | ⬆️⬆️ | Medio | **P1** |
| 5 | Ensemble con pesos optimizados/stacking | ⬆️⬆️ | Medio | **P1** |
| 6 | Walk-forward CV | ⬆️⬆️ | Medio | **P1** |
| 7 | Multi-bookmaker comparison | ⬆️⬆️ | Bajo | **P1** |
| 8 | Features xG + H2H | ⬆️ | Medio | **P2** |
| 9 | Tests unitarios | ⬆️ (evita pérdidas por bugs) | Medio | **P2** |
| 10 | Monte Carlo del torneo | ⬆️ | Medio | **P2** |
| 11 | Ventanas múltiples de forma (5/10/20) | ⬆️ | Bajo | **P2** |
| 12 | CLV tracking automático | ⬆️ (diagnóstico) | Bajo | **P3** |
| 13 | Pipeline reproducible (Makefile) | Operativo | Bajo | **P3** |

---

## 🎯 Resumen Ejecutivo

El sistema es **sólido en arquitectura e ingeniería de datos** — sin data leakage, con buena modularización y un pipeline completo de EV → shortlist → tracker. Pero tiene **tres carencias críticas** para apostar con confianza:

1. **El Poisson independiente sesga las probabilidades** de empate y marcadores bajos → Dixon-Coles.
2. **El backtest es demasiado pequeño** (n=28) para saber si funciona → ampliar a >200 apuestas.
3. **El staking es demasiado básico** (flat 0.5u) → Kelly fraccional para maximizar crecimiento.

Resolver estos tres puntos convertiría el sistema de un "prototipo prometedor" a un "modelo operativo para apostar".
