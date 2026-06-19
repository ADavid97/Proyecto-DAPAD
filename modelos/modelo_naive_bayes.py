from sklearn.naive_bayes import GaussianNB
from .modelo import ModeloSupervisado


class ModeloNaiveBayes(ModeloSupervisado):
    """Naive Bayes gaussiano: clasificador probabilístico que asume independencia
    entre features. Muy rápido y sin hiperparámetros que ajustar; sirve de buena
    línea de base para clasificación."""

    es_clasificacion = True
    scoring_cv = "accuracy"

    def _crear_estimador(self) -> GaussianNB:
        return GaussianNB()
