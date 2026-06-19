from sklearn.neural_network import MLPClassifier
from .modelo import ModeloSupervisado


class ModeloRedNeuronal(ModeloSupervisado):
    """Red neuronal (perceptrón multicapa) para clasificación.

    Aprende relaciones no lineales mediante capas ocultas. Es MUY sensible a la
    escala: entrénala con "Escalar features" activado. En datasets pequeños suele
    perder contra los árboles y puede no converger (sube el máximo de iteraciones).
    """

    es_clasificacion = True
    scoring_cv = "accuracy"

    def __init__(self, datos, hidden_layer_sizes: tuple = (100,), max_iter: int = 300):
        super().__init__(datos)
        self.hidden_layer_sizes = hidden_layer_sizes  # neuronas por capa oculta
        self.max_iter = max_iter
        self.no_convergio = False  # True si agotó las iteraciones sin converger

    def _crear_estimador(self) -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            random_state=42,
        )

    def _rejilla_busqueda(self) -> dict:
        return {
            "hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "alpha": [0.0001, 0.001, 0.01],
        }

    def _post_entrenar(self) -> None:
        # n_iter_ == max_iter ⇒ tocó el techo sin converger
        self.no_convergio = getattr(self.modelo, "n_iter_", 0) >= self.max_iter
