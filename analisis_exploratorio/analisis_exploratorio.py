import pandas as pd


class AnalisisExploratorio:
    def __init__(self, datos: pd.DataFrame):
        self.datos = datos

    def estadisticas_descriptivas(self) -> pd.DataFrame:
        return self.datos.describe(include="all")

    def distribucion_columna(self, columna: str) -> dict:
        return self.datos[columna].value_counts().to_dict()

    def detectar_outliers(self, columna: str) -> pd.DataFrame:
        Q1 = self.datos[columna].quantile(0.25)
        Q3 = self.datos[columna].quantile(0.75)
        IQR = Q3 - Q1
        return self.datos[
            (self.datos[columna] < Q1 - 1.5 * IQR) |
            (self.datos[columna] > Q3 + 1.5 * IQR)
        ]

    def matriz_correlacion(self) -> pd.DataFrame:
        return self.datos.select_dtypes(include="number").corr()

    def valores_unicos(self, columna: str) -> dict:
        return {
            "total": self.datos[columna].nunique(),
            "valores": self.datos[columna].unique().tolist()
        }

    def conteo_nulos(self) -> dict:
        return self.datos.isnull().sum().to_dict()

    def resumen_general(self) -> dict:
        return {
            "filas": self.datos.shape[0],
            "columnas": self.datos.shape[1],
            "tipos": self.datos.dtypes.astype(str).to_dict(),
            "nulos_totales": int(self.datos.isnull().sum().sum()),
            "duplicados": int(self.datos.duplicated().sum())
        }
