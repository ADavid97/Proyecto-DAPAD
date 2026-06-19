from sklearn.ensemble import RandomForestRegressor
from .modelo import ModeloSupervisado


class ModeloRandomForestRegresion(ModeloSupervisado):
    """Bosque aleatorio para regresión: promedia muchos árboles y suele superar a
    un árbol único. No necesita escalado y expone importancia de features nativa.
    """

    es_clasificacion = False
    scoring_cv = "r2"

    def __init__(self, datos, n_estimators: int = 100, profundidad_max: int | None = None):
        super().__init__(datos)
        self.n_estimators = n_estimators  # número de árboles del bosque
        self.profundidad_max = profundidad_max

    def _crear_estimador(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.profundidad_max,
            random_state=42,
            n_jobs=-1,
        )

    def _rejilla_busqueda(self) -> dict:
        return {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 5, 10, 20],
            "min_samples_leaf": [1, 2, 5],
        }
