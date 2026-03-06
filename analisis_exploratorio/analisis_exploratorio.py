import pandas as pd


class AnalisisExploratorio:
    def __init__(self, datos: pd.DataFrame):
        self.datos = datos

    def estadisticas_descriptivas(self) -> pd.DataFrame:
        pass

    def distribucion_columna(self, columna: str) -> dict:
        pass

    def detectar_outliers(self, columna: str) -> pd.DataFrame:
        pass

    def matriz_correlacion(self) -> pd.DataFrame:
        pass

    def valores_unicos(self, columna: str) -> dict:
        pass

    def conteo_nulos(self) -> dict:
        pass

    def resumen_general(self) -> dict:
        pass
