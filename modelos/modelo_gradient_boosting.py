from sklearn.ensemble import GradientBoostingClassifier
from .modelo import ModeloSupervisado


class ModeloGradientBoosting(ModeloSupervisado):
    """Gradient Boosting: árboles encadenados que corrigen el error del anterior.

    Suele dar la mejor exactitud en datos tabulares, a cambio de más cómputo y de
    ser más sensible a los hiperparámetros (learning_rate × n_estimators). No
    necesita escalado y expone importancia de features nativa.
    """

    es_clasificacion = True
    scoring_cv = "accuracy"

    def __init__(self, datos, n_estimators: int = 100, learning_rate: float = 0.1,
                 profundidad_max: int = 3):
        super().__init__(datos)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate  # cuánto aporta cada árbol; menor = más lento pero más fino
        self.profundidad_max = profundidad_max

    def _crear_estimador(self) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(
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
