from sklearn.neural_network import MLPRegressor
from .modelo import ModeloSupervisado


class ModeloRedNeuronalRegresion(ModeloSupervisado):
    """Red neuronal (perceptrón multicapa) para regresión: predice un número
    aprendiendo relaciones no lineales mediante capas ocultas.

    Mismas advertencias que la versión de clasificación: MUY sensible a la escala
    (entrena con "Escalar features" activado) y en datasets pequeños puede no
    converger o perder contra Random Forest / Gradient Boosting.
    """

    es_clasificacion = False
    scoring_cv = "r2"

    def __init__(self, datos, hidden_layer_sizes: tuple = (100,), max_iter: int = 300):
        super().__init__(datos)
        self.hidden_layer_sizes = hidden_layer_sizes  # neuronas por capa oculta
        self.max_iter = max_iter
        self.no_convergio = False  # True si agotó las iteraciones sin converger

    def _crear_estimador(self) -> MLPRegressor:
        return MLPRegressor(
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
