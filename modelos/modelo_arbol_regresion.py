from sklearn.tree import DecisionTreeRegressor
from .modelo import ModeloSupervisado


class ModeloArbolRegresion(ModeloSupervisado):
    """Árbol de decisión para regresión; expone la importancia nativa de features."""

    es_clasificacion = False
    scoring_cv = "r2"

    def __init__(self, datos, profundidad_max: int | None = None, min_samples_leaf: int = 1):
        super().__init__(datos)
        self.profundidad_max = profundidad_max
        self.min_samples_leaf = min_samples_leaf  # mínimo de muestras por hoja: sube para evitar sobreajuste

    def _crear_estimador(self) -> DecisionTreeRegressor:
        return DecisionTreeRegressor(
            max_depth=self.profundidad_max,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )

    def _rejilla_busqueda(self) -> dict:
        return {
            "max_depth": [None, 3, 5, 10, 20],
            "min_samples_leaf": [1, 2, 5, 10],
        }
