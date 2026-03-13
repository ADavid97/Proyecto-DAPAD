import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


class Modelos:
    def __init__(self, datos: np.ndarray):
        self.datos = datos
        self.modelo = None
        self.resultados = {}
        self.tipo_modelo = None

    def kmeans(self, n_clusters: int = 3) -> dict:
        pass

    def regresion_lineal(self, target: str, features: list) -> dict:
        pass

    def regresion_logistica(self, target: str, features: list) -> dict:
        pass

    def arbol_decision(self, target: str, features: list) -> dict:
        pass

    def knn(self, target: str, features: list, k: int = 5) -> dict:
        pass

    def calcular_codo(self, k_max: int = 10) -> list:
        pass

    def entrenar(self, target: str, features: list) -> None:
        pass

    def predecir(self, datos: pd.DataFrame) -> np.ndarray:
        pass
