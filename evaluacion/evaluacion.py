import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, r2_score, mean_squared_error,
    confusion_matrix, classification_report,
)


class Evaluacion:
    """Métricas de un modelo ya entrenado a partir de y_real vs y_predicho."""

    def __init__(self, modelo: object, y_real: np.ndarray, y_predicho: np.ndarray):
        self.modelo = modelo
        self.y_real = np.asarray(y_real)
        self.y_predicho = np.asarray(y_predicho)

    def accuracy(self) -> float:
        return float(accuracy_score(self.y_real, self.y_predicho))

    def precision(self) -> float:
        return float(precision_score(self.y_real, self.y_predicho, average="weighted", zero_division=0))

    def recall(self) -> float:
        return float(recall_score(self.y_real, self.y_predicho, average="weighted", zero_division=0))

    def f1_score(self) -> float:
        return float(f1_score(self.y_real, self.y_predicho, average="weighted", zero_division=0))

    def r2(self) -> float:
        return float(r2_score(self.y_real, self.y_predicho))

    def error_cuadratico_medio(self) -> float:
        return float(mean_squared_error(self.y_real, self.y_predicho))

    def matriz_confusion(self) -> np.ndarray:
        return confusion_matrix(self.y_real, self.y_predicho)

    def reporte_clasificacion(self) -> str:
        return classification_report(self.y_real, self.y_predicho, zero_division=0)

    def obtener_metricas(self) -> dict:
        try:
            return {
                "accuracy": self.accuracy(),
                "precision": self.precision(),
                "recall": self.recall(),
                "f1": self.f1_score(),
                "confusion_matrix": self.matriz_confusion(),
                "reporte": self.reporte_clasificacion(),
            }
        except Exception:
            mse = self.error_cuadratico_medio()
            return {
                "r2": self.r2(),
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
            }
