import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from .modelo import Modelo


class ModeloArbolDecision(Modelo):
    def __init__(self, datos: np.ndarray):
        super().__init__(datos)
        self.target = None
        self.features = []
        self.profundidad_max = None

    def entrenar(self) -> None:
        pass

    def predecir(self, datos: pd.DataFrame) -> np.ndarray:
        pass

    def evaluar(self) -> dict:
        pass
