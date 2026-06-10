import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Modelo(ABC):
    """Base de todos los modelos: define el contrato entrenar / predecir / evaluar.

    `scaler` lo asignan las subclases cuando se entrena con `escalar=True`;
    `predecir` debe aplicarlo si existe. Los modelos supervisados además
    implementan `_crear_estimador()` para habilitar la validación cruzada.
    """

    # Métrica que usa cross_val_score; las subclases de regresión la cambian a "r2"
    scoring_cv = "accuracy"

    def __init__(self, datos: pd.DataFrame):
        self.datos = datos
        self.modelo = None
        self.scaler = None
        self.resultados = {}

    @abstractmethod
    def entrenar(self) -> None:
        pass

    @abstractmethod
    def predecir(self, datos: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def evaluar(self) -> dict:
        pass

    def _crear_estimador(self):
        raise NotImplementedError("Este modelo no define un estimador para validación cruzada.")

    def validacion_cruzada(self, cv: int = 5, escalar: bool = False) -> dict:
        """Validación cruzada k-fold sobre todo el dataset con un estimador nuevo.

        Si `escalar` es True el StandardScaler entra al pipeline, de modo que se
        ajusta dentro de cada fold (sin fuga de datos entre folds).
        """
        df = self.datos if isinstance(self.datos, pd.DataFrame) else pd.DataFrame(self.datos)
        # to_numpy(): con pandas 3 .values devuelve ArrowStringArray para texto y sklearn no lo acepta
        X = df[self.features].to_numpy()
        y = df[self.target].to_numpy()
        estimador = self._crear_estimador()
        if escalar:
            estimador = make_pipeline(StandardScaler(), estimador)
        scores = cross_val_score(estimador, X, y, cv=cv, scoring=self.scoring_cv)
        return {
            "metrica": self.scoring_cv,
            "scores": [float(s) for s in scores],
            "media": float(scores.mean()),
            "desviacion": float(scores.std()),
        }
