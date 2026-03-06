import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


class Preprocesamiento:
    def __init__(self, datos: pd.DataFrame):
        self.datos = datos.copy()
        self.datos_procesados = None
        self.scaler = None

    def seleccionar_columnas(self, columnas: list) -> pd.DataFrame:
        pass

    def eliminar_nulos(self) -> pd.DataFrame:
        pass

    def rellenar_nulos(self, estrategia: str = "media") -> pd.DataFrame:
        pass

    def eliminar_duplicados(self) -> pd.DataFrame:
        pass

    def filtrar_filas(self, columna: str, valor: object) -> pd.DataFrame:
        pass

    def normalizar_standard(self) -> np.ndarray:
        pass

    def normalizar_minmax(self) -> np.ndarray:
        pass

    def codificar_categoricas(self) -> pd.DataFrame:
        pass

    def obtener_datos_procesados(self) -> pd.DataFrame:
        pass
