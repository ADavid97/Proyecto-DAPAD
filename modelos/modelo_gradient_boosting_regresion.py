from sklearn.ensemble import HistGradientBoostingRegressor
from .modelo import ModeloSupervisado


class ModeloGradientBoostingRegresion(ModeloSupervisado):
    """Gradient Boosting basado en histogramas para regresión (HistGradientBoosting).

    Mismo algoritmo que GradientBoosting clásico pero implementado con histogramas:
    10-100× más rápido, soporta paralelismo (n_jobs) y maneja nulos nativamente.
    No necesita escalado y expone importancia de features por permutación.
    """

    es_clasificacion = False
    scoring_cv = "r2"

    def __init__(self, datos, n_estimators: int = 100, learning_rate: float = 0.1,
                 profundidad_max: int | None = None):
        super().__init__(datos)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.profundidad_max = profundidad_max

    def _crear_estimador(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_iter=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.profundidad_max,
            random_state=42,
        )

    def _rejilla_busqueda(self) -> dict:
        return {
            "max_iter": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [None, 3, 5],
        }
