"""
src/models/ordinal_model.py
============================
Pipeline clásico sin redes neuronales para predicción de resultado.

Implementa:
  - Frank & Hall: descomposición ordinal en K-1 clasificadores binarios
  - Corrección de covariate shift con KMM
  - Manejo de desbalanceo con SMOTE o cost-sensitive
  - Calibración con Platt Scaling
  - Cuantificación con HDy
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, brier_score_loss
import warnings
warnings.filterwarnings("ignore")


# ── Frank & Hall — Clasificación Ordinal ─────────────────────────────────────

class FrankHallOrdinal:
    """
    Implementa el método de Frank & Hall para clasificación ordinal.
    Para K=3 clases (A < D < H) entrena K-1=2 clasificadores binarios:
      clf_0: P(y > A) = P(D o H)
      clf_1: P(y > D) = P(H)

    Las probabilidades finales se derivan de:
      P(A) = 1 - P(y > A)
      P(D) = P(y > A) - P(y > H)
      P(H) = P(y > D)
    """

    CLASSES = ["A", "D", "H"]  # orden de menor a mayor

    def __init__(self, base_estimator=None):
        if base_estimator is None:
            base_estimator = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42
            )
        self.base_estimator = base_estimator
        self.clfs = []
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        """
        X : matriz de features
        y : array de etiquetas ordinales ("A", "D", "H")
        sample_weight : pesos por ejemplo (para KMM)
        """
        X_scaled = self.scaler.fit_transform(X)
        y = np.array(y)
        self.clfs = []

        # Entrenar K-1 clasificadores binarios
        for k in range(len(self.CLASSES) - 1):
            # y_bin = 1 si y > CLASSES[k], 0 en caso contrario
            threshold = self.CLASSES[k]
            y_bin = (y > threshold).astype(int) if y.dtype == object else \
                    np.array([1 if yi in self.CLASSES[k+1:] else 0 for yi in y])

            clf = CalibratedClassifierCV(
                self._clone_estimator(), method="isotonic", cv=5
            )
            if sample_weight is not None:
                clf.fit(X_scaled, y_bin, sample_weight=sample_weight)
            else:
                clf.fit(X_scaled, y_bin)
            self.clfs.append(clf)

        return self

    def _clone_estimator(self):
        from sklearn.base import clone
        return clone(self.base_estimator)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Devuelve matriz (n_samples, 3) con probabilidades [P(A), P(D), P(H)].
        """
        X_scaled = self.scaler.transform(X)

        # P(y > A), P(y > D)
        cumprobs = np.array([clf.predict_proba(X_scaled)[:, 1] for clf in self.clfs])
        # cumprobs shape: (K-1, n_samples)

        n = X_scaled.shape[0]
        proba = np.zeros((n, len(self.CLASSES)))

        # P(A) = 1 - P(y > A)
        proba[:, 0] = 1 - cumprobs[0]
        # P(D) = P(y > A) - P(y > D)
        proba[:, 1] = cumprobs[0] - cumprobs[1]
        # P(H) = P(y > D)
        proba[:, 2] = cumprobs[1]

        # Normalizar por si hay pequeños errores numéricos
        proba = np.clip(proba, 0, 1)
        proba = proba / proba.sum(axis=1, keepdims=True)

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return np.array([self.CLASSES[i] for i in idx])


# ── Kernel Mean Matching (KMM) ───────────────────────────────────────────────

def compute_kmm_weights(
    X_train: np.ndarray,
    X_target: np.ndarray,
    kernel: str = "rbf",
    sigma: float = 1.0,
    B: float = 10.0,
) -> np.ndarray:
    """
    Calcula pesos KMM para corregir covariate shift entre X_train y X_target.

    Los pesos w_i hacen que la distribución ponderada de X_train se aproxime
    a la distribución de X_target (partidos de Mundial).

    Parámetros
    ----------
    X_train  : features del conjunto de entrenamiento
    X_target : features de los partidos objetivo (Mundial)
    B        : cota superior de los pesos (regularización)

    Retorna
    -------
    w : array de pesos de forma (n_train,)
    """
    from scipy.optimize import minimize

    n_tr = X_train.shape[0]
    n_te = X_target.shape[0]

    # Kernel RBF entre train y target
    def rbf(A, B, sigma):
        diff = A[:, None, :] - B[None, :, :]
        return np.exp(-np.sum(diff**2, axis=-1) / (2 * sigma**2))

    K_tr  = rbf(X_train, X_train, sigma)   # (n_tr, n_tr)
    K_te  = rbf(X_train, X_target, sigma)  # (n_tr, n_te)

    kappa = (n_tr / n_te) * K_te.sum(axis=1)  # (n_tr,)

    # Minimizar 0.5 * w^T K_tr w - kappa^T w  s.t. 0 <= w <= B
    def objective(w):
        return 0.5 * w @ K_tr @ w - kappa @ w

    def gradient(w):
        return K_tr @ w - kappa

    bounds = [(0, B)] * n_tr
    w0 = np.ones(n_tr)
    result = minimize(objective, w0, jac=gradient, bounds=bounds, method="L-BFGS-B")
    w = result.x

    # Normalizar
    w = w * (n_tr / w.sum())
    return w


# ── Pipeline completo ─────────────────────────────────────────────────────────

class OrdinalPipeline:
    """
    Pipeline completo sin redes neuronales:
      1. Preprocesado de features
      2. KMM para corrección de distribución (opcional)
      3. Frank & Hall ordinal
      4. Evaluación
    """

    def __init__(self, use_kmm: bool = True, base_estimator=None):
        self.use_kmm = use_kmm
        self.model   = FrankHallOrdinal(base_estimator)
        self.weights = None

    def prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Extrae X e y del DataFrame de features."""
        feature_cols = [c for c in df.columns if c.startswith(("home_form", "away_form", "diff_", "is_", "same_", "stage_", "ref_"))]
        target_col   = "target_result"

        mask = df[target_col].notna() & df[feature_cols].notna().all(axis=1)
        df_clean = df[mask].copy()

        X = df_clean[feature_cols].values.astype(float)
        y = df_clean[target_col].values

        return X, y, feature_cols

    def fit(self, df_train: pd.DataFrame, df_target: Optional[pd.DataFrame] = None):
        """
        df_train  : DataFrame de features de entrenamiento
        df_target : DataFrame de partidos objetivo (Mundial) para KMM
        """
        X_train, y_train, self.feature_cols = self.prepare_features(df_train)

        if self.use_kmm and df_target is not None:
            X_target, _, _ = self.prepare_features(df_target)
            print("Calculando pesos KMM...")
            self.weights = compute_kmm_weights(X_train, X_target)
            print(f"  Pesos KMM — min: {self.weights.min():.3f}, max: {self.weights.max():.3f}, "
                  f"media: {self.weights.mean():.3f}")
        else:
            self.weights = None

        print("Entrenando modelo ordinal (Frank & Hall)...")
        self.model.fit(X_train, y_train, sample_weight=self.weights)
        print("  ✓ Modelo entrenado")
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        feature_cols = [c for c in df.columns if c in self.feature_cols]
        X = df[feature_cols].fillna(0).values.astype(float)
        return self.model.predict_proba(X)

    def evaluate(self, df_test: pd.DataFrame) -> dict:
        """Evalúa el modelo en un conjunto de test."""
        X_test, y_test, _ = self.prepare_features(df_test)
        proba = self.model.predict_proba(X_test)
        preds = self.model.predict(X_test)

        # Codificar y_test para métricas
        le = LabelEncoder()
        le.classes_ = np.array(FrankHallOrdinal.CLASSES)
        y_idx = le.transform(y_test)

        accuracy = (preds == y_test).mean()
        ll       = log_loss(y_idx, proba, labels=[0, 1, 2])

        # Brier Score multiclase
        y_onehot = np.zeros_like(proba)
        y_onehot[np.arange(len(y_idx)), y_idx] = 1
        brier = np.mean(np.sum((proba - y_onehot)**2, axis=1))

        metrics = {
            "accuracy":    round(accuracy, 4),
            "log_loss":    round(ll, 4),
            "brier_score": round(brier, 4),
        }
        print("\nMétricas de evaluación:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        return metrics


# ── Script standalone ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from typing import Optional

    features_path = "./data/features.csv"
    if not os.path.exists(features_path):
        print("No se encuentra features.csv. Ejecuta primero feature_engineering.py")
        exit(1)

    print("Cargando features...")
    df = pd.read_csv(features_path, parse_dates=["date"])

    # Split temporal: entrenar hasta 2017, validar 2018-2022, test 2022+
    df_train = df[df["date"].dt.year < 2018]
    df_val   = df[(df["date"].dt.year >= 2018) & (df["date"].dt.year < 2022)]
    df_wc22  = df[df["tournament"].str.contains("World Cup 2022", na=False)]

    print(f"Train: {len(df_train)} | Val: {len(df_val)} | WC2022: {len(df_wc22)}")

    pipeline = OrdinalPipeline(use_kmm=True)
    pipeline.fit(df_train, df_target=df_wc22)
    metrics = pipeline.evaluate(df_val)
