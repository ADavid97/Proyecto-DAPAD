import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from .modelo import Modelo


class ModeloKMeans(Modelo):
    def __init__(self, datos: np.ndarray):
        super().__init__(datos)
        self.n_clusters = 3
        self.centroides = None
        self.inercias = []

    def entrenar(self) -> None:
        pass

    def predecir(self, datos: pd.DataFrame) -> np.ndarray:
        pass

    def evaluar(self) -> dict:
        pass

    def calcular_codo(self, k_max: int = 10) -> list:
        pass

    def obtener_segmentos(self) -> dict:
        pass
