import pandas as pd


class AnalisisExploratorio:
    """Análisis exploratorio de un DataFrame: resúmenes, distribuciones, outliers y correlación."""

    def __init__(self, datos: pd.DataFrame):
        self.datos = datos

    def estadisticas_descriptivas(self) -> pd.DataFrame:
        """describe() de todas las columnas; convierte objects a str para evitar errores de Arrow."""
        df = self.datos.copy()
        for col in df.select_dtypes(include=["object", "str"]).columns:
            df[col] = df[col].astype(str)
        return df.describe(include="all")

    def distribucion_columna(self, columna: str) -> dict:
        """Frecuencia de cada valor de la columna: {valor: conteo}."""
        return self.datos[columna].astype(str).value_counts().to_dict()

    def detectar_outliers(self, columna: str) -> pd.DataFrame:
        """Filas fuera del rango [Q1 - 1.5·IQR, Q3 + 1.5·IQR] de una columna numérica."""
        Q1 = self.datos[columna].quantile(0.25)
        Q3 = self.datos[columna].quantile(0.75)
        IQR = Q3 - Q1
        return self.datos[
            (self.datos[columna] < Q1 - 1.5 * IQR) |
            (self.datos[columna] > Q3 + 1.5 * IQR)
        ]

    def matriz_correlacion(self) -> pd.DataFrame:
        """Correlación de Pearson entre las columnas numéricas."""
        return self.datos.select_dtypes(include="number").corr()

    def valores_unicos(self, columna: str) -> dict:
        """Total y lista de valores únicos de la columna (como strings)."""
        serie = self.datos[columna].astype(str)
        return {
            "total": serie.nunique(),
            "valores": serie.unique().tolist()
        }

    def conteo_nulos(self) -> dict:
        """Nulos por columna: {columna: conteo}."""
        return self.datos.isnull().sum().to_dict()

    def resumen_general(self) -> dict:
        """Dimensiones, tipos, nulos totales y duplicados del dataset."""
        try:
            duplicados = int(self.datos.duplicated().sum())
        except TypeError:  # celdas no hashables (listas/dicts provenientes de JSON)
            duplicados = int(self.datos.astype(str).duplicated().sum())
        return {
            "filas": self.datos.shape[0],
            "columnas": self.datos.shape[1],
            "tipos": self.datos.dtypes.astype(str).to_dict(),
            "nulos_totales": int(self.datos.isnull().sum().sum()),
            "duplicados": duplicados,
        }
