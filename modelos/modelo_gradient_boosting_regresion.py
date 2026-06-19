from sklearn.ensemble import GradientBoostingRegressor
from .modelo import ModeloSupervisado


class ModeloGradientBoostingRegresion(ModeloSupervisado):
    """Gradient Boosting para regresión: árboles encadenados que corrigen el error
    del anterior. Suele dar la mejor exactitud en datos tabulares, a cambio de más
    cómputo y sensibilidad a learning_rate × n_estimators. No necesita escalado.
    """

    es_clasificacion = False
    scoring_cv = "r2"

    def __init__(self, datos, n_estimators: int = 100, learning_rate: float = 0.1,
                 profundidad_max: int = 3):
        super().__init__(datos)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate  # cuánto aporta cada árbol; menor = más lento pero más fino
        self.profundidad_max = profundidad_max

    def _crear_estimador(self) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.profundidad_max,
            random_state=42,
        )

    def _rejilla_busqueda(self) -> dict:
        return {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [2, 3, 5],
        }
