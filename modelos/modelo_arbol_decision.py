from sklearn.tree import DecisionTreeClassifier
from .modelo import ModeloSupervisado


class ModeloArbolDecision(ModeloSupervisado):
    """Árbol de decisión para clasificación; expone la importancia nativa de features."""

    es_clasificacion = True
    scoring_cv = "accuracy"

    def __init__(self, datos, profundidad_max: int | None = None,
                 min_samples_leaf: int = 1, criterion: str = "gini"):
        super().__init__(datos)
        self.profundidad_max = profundidad_max
        self.min_samples_leaf = min_samples_leaf  # mínimo de muestras por hoja: sube para evitar sobreajuste
        self.criterion = criterion  # "gini" o "entropy"

    def _crear_estimador(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            max_depth=self.profundidad_max,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            random_state=42,
        )

    def _rejilla_busqueda(self) -> dict:
        return {
            "max_depth": [None, 3, 5, 10, 20],
            "min_samples_leaf": [1, 2, 5, 10],
            "criterion": ["gini", "entropy"],
        }
