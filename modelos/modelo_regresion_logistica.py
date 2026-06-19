from sklearn.linear_model import LogisticRegression
from .modelo import ModeloSupervisado


class ModeloRegresionLogistica(ModeloSupervisado):
    """Regresión logística (clasificación) con split estratificado y escalado opcional."""

    es_clasificacion = True
    scoring_cv = "accuracy"

    def __init__(self, datos, max_iter: int = 200, C: float = 1.0,
                 class_weight: str | None = None):
        super().__init__(datos)
        self.max_iter = max_iter
        self.C = C  # inverso de la regularización: menor C = más regularización
        self.class_weight = class_weight  # None o "balanced" (compensa clases desbalanceadas)

    def _crear_estimador(self) -> LogisticRegression:
        return LogisticRegression(max_iter=self.max_iter, C=self.C, class_weight=self.class_weight)

    def _rejilla_busqueda(self) -> dict:
        return {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "class_weight": [None, "balanced"],
        }
