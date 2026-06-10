import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class Modelo(ABC):
    """Base de todos los modelos: define el contrato entrenar / predecir / evaluar.

    `scaler` lo asignan las subclases cuando se entrena con `escalar=True`;
    `predecir` debe aplicarlo si existe.
    """

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
