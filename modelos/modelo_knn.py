import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from .modelo import Modelo


class ModeloKNN(Modelo):
    def __init__(self, datos: np.ndarray):
        super().__init__(datos)
        self.target = None
        self.features = []
        self.k = 5

    def entrenar(self) -> None:
        pass

    def predecir(self, datos: pd.DataFrame) -> np.ndarray:
        pass

    def evaluar(self) -> dict:
        pass
