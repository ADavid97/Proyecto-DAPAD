from sklearn.neighbors import KNeighborsRegressor
from .modelo import ModeloSupervisado


class ModeloKNNRegresion(ModeloSupervisado):
    """K-Nearest Neighbors para regresión: predice un número como promedio
    (ponderado o no) de los vecinos más cercanos. Sensible a la escala."""

    es_clasificacion = False
    scoring_cv = "r2"

    def __init__(self, datos, k: int = 5, weights: str = "uniform", metric: str = "minkowski"):
        super().__init__(datos)
        self.k = k
        self.weights = weights  # "uniform" (todos igual) o "distance" (los más cercanos pesan más)
        self.metric = metric

    def _crear_estimador(self) -> KNeighborsRegressor:
        return KNeighborsRegressor(n_neighbors=self.k, weights=self.weights, metric=self.metric)

    def _rejilla_busqueda(self) -> dict:
        return {
            "n_neighbors": [3, 5, 7, 11, 15],
            "weights": ["uniform", "distance"],
            "metric": ["minkowski", "manhattan"],
        }
