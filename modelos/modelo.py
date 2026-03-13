import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class Modelo(ABC):
    def __init__(self, datos: np.ndarray):
        self.datos = datos
        self.modelo = None
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

    def obtener_metricas(self) -> dict:
        pass
