import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, r2_score, mean_squared_error,
                             confusion_matrix, classification_report)


class Evaluacion:
    def __init__(self, modelo: object, y_real: np.ndarray, y_predicho: np.ndarray):
        self.modelo = modelo
        self.y_real = y_real
        self.y_predicho = y_predicho

    def accuracy(self) -> float:
        pass

    def precision(self) -> float:
        pass

    def recall(self) -> float:
        pass

    def f1_score(self) -> float:
        pass

    def r2(self) -> float:
        pass

    def error_cuadratico_medio(self) -> float:
        pass

    def matriz_confusion(self) -> np.ndarray:
        pass

    def reporte_clasificacion(self) -> str:
        pass

    def obtener_metricas(self) -> dict:
        pass
