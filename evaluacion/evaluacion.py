import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, roc_auc_score,
    confusion_matrix, classification_report,
)


class Evaluacion:
    """Métricas de un modelo ya entrenado a partir de y_real vs y_predicho.

    Calcula el conjunto de métricas adecuado según el tipo de problema. El tipo
    ('regresion' o 'clasificacion') puede pasarse explícitamente; si se omite se
    infiere de los valores observados. `metricas()` devuelve el conjunto correcto
    sin que quien llama tenga que elegir.
    """

    def __init__(self, modelo: object, y_real: np.ndarray, y_predicho: np.ndarray,
                 tipo: str | None = None):
        self.modelo = modelo
        self.y_real = np.asarray(y_real)
        self.y_predicho = np.asarray(y_predicho)
        self.tipo = tipo or self._detectar_tipo()

    # ── Detección de tipo ──────────────────────────────────────────────
    def _detectar_tipo(self) -> str:
        """Infiere 'regresion' o 'clasificacion' a partir del target observado.

        Un target flotante con muchos valores distintos se trata como regresión;
        enteros, texto o pocos valores distintos se tratan como clasificación.
        """
        y = self.y_real
        if y.dtype.kind == "f" and np.unique(y).size > 20:
            return "regresion"
        return "clasificacion"

    def es_regresion(self) -> bool:
        return self.tipo == "regresion"

    def _n_predictores(self) -> int:
        """Número de predictores (k) para el R² ajustado."""
        features = getattr(self.modelo, "features", None)
        if features:
            return len(features)
        X_test = getattr(self.modelo, "X_test", None)
        if X_test is not None:
            arr = np.asarray(X_test)
            return arr.shape[1] if arr.ndim > 1 else 1
        return 1

    # ── Métricas de regresión ──────────────────────────────────────────
    def r2(self) -> float:
        return float(r2_score(self.y_real, self.y_predicho))

    def r2_ajustado(self) -> float:
        """R² ajustado: penaliza por número de predictores.

        1 - [(1-R²)(n-1)/(n-k-1)], con n = observaciones y k = predictores.
        Devuelve NaN cuando no hay grados de libertad suficientes (n - k - 1 <= 0).
        """
        n = self.y_real.size
        k = self._n_predictores()
        denom = n - k - 1
        if denom <= 0:
            return float("nan")
        return float(1 - (1 - self.r2()) * (n - 1) / denom)

    def mae(self) -> float:
        """Error Absoluto Medio."""
        return float(mean_absolute_error(self.y_real, self.y_predicho))

    def error_cuadratico_medio(self) -> float:
        """MSE: Error Cuadrático Medio (base del RMSE)."""
        return float(mean_squared_error(self.y_real, self.y_predicho))

    def rmse(self) -> float:
        """Raíz del Error Cuadrático Medio."""
        return float(np.sqrt(self.error_cuadratico_medio()))

    def error_porcentual_absoluto_medio(self) -> float | None:
        """MAPE: Error Porcentual Absoluto Medio, en porcentaje.

        No está definido si el target tiene ceros (dividiría entre cero y daría
        valores absurdos); en ese caso devuelve None para que no se reporte.
        """
        if np.any(self.y_real == 0):
            return None
        return float(mean_absolute_percentage_error(self.y_real, self.y_predicho) * 100)

    def metricas_regresion(self) -> dict:
        mse = self.error_cuadratico_medio()
        metricas = {
            "r2": self.r2(),
            "r2_ajustado": self.r2_ajustado(),
            "mae": self.mae(),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
        }
        mape = self.error_porcentual_absoluto_medio()
        if mape is not None:
            metricas["mape"] = mape
        return metricas

    # ── Métricas de clasificación ──────────────────────────────────────
    def accuracy(self) -> float:
        return float(accuracy_score(self.y_real, self.y_predicho))

    def precision(self) -> float:
        return float(precision_score(self.y_real, self.y_predicho, average="weighted", zero_division=0))

    def recall(self) -> float:
        return float(recall_score(self.y_real, self.y_predicho, average="weighted", zero_division=0))

    def f1_score(self) -> float:
        return float(f1_score(self.y_real, self.y_predicho, average="weighted", zero_division=0))

    def matriz_confusion(self) -> np.ndarray:
        return confusion_matrix(self.y_real, self.y_predicho)

    def reporte_clasificacion(self) -> str:
        return classification_report(self.y_real, self.y_predicho, zero_division=0)

    def roc_auc(self) -> float | None:
        """ROC-AUC si el estimador soporta predict_proba; None en otro caso.

        Usa las probabilidades del estimador sobre el conjunto de prueba guardado
        por el modelo. Binario: área de la clase positiva; multiclase: one-vs-rest
        ponderado. Cualquier fallo (clase única en test, etc.) devuelve None.
        """
        estimador = getattr(self.modelo, "modelo", None)
        X_test = getattr(self.modelo, "X_test", None)
        if estimador is None or X_test is None or not hasattr(estimador, "predict_proba"):
            return None
        try:
            proba = estimador.predict_proba(X_test)
            clases = np.unique(self.y_real)
            if clases.size == 2:
                return float(roc_auc_score(self.y_real, proba[:, 1]))
            return float(roc_auc_score(self.y_real, proba, multi_class="ovr", average="weighted"))
        except Exception:
            return None

    def metricas_clasificacion(self) -> dict:
        metricas = {
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1_score(),
            "confusion_matrix": self.matriz_confusion(),
            "reporte": self.reporte_clasificacion(),
        }
        roc = self.roc_auc()
        if roc is not None:
            metricas["roc_auc"] = roc
        return metricas

    # ── Despacho según el tipo de problema ─────────────────────────────
    def metricas(self) -> dict:
        return self.metricas_regresion() if self.es_regresion() else self.metricas_clasificacion()

    # Alias retrocompatible: el resto de la app llama obtener_metricas().
    def obtener_metricas(self) -> dict:
        return self.metricas()
