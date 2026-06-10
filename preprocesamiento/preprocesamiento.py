import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


class Preprocesamiento:
    """Transformaciones del dataset: limpieza, filtrado, escalado y codificación.

    Cada operación parte del último resultado (`datos_procesados`) o de los datos
    originales si aún no hay transformaciones, por lo que son encadenables.
    """

    def __init__(self, datos: pd.DataFrame):
        self.datos = datos.copy()
        self.datos_procesados: pd.DataFrame | None = None
        self.scaler = None

    def _base(self) -> pd.DataFrame:
        return self.datos_procesados if self.datos_procesados is not None else self.datos

    def seleccionar_columnas(self, columnas: list) -> pd.DataFrame:
        """Conserva solo las columnas indicadas."""
        self.datos_procesados = self._base()[columnas].copy()
        return self.datos_procesados

    def eliminar_nulos(self) -> pd.DataFrame:
        """Elimina toda fila que contenga al menos un valor nulo."""
        self.datos_procesados = self._base().dropna().reset_index(drop=True)
        return self.datos_procesados

    def rellenar_nulos(self, estrategia: str = "media", valor_constante: float = 0) -> pd.DataFrame:
        """Rellena nulos según la estrategia: media, mediana, moda o constante."""
        df = self._base().copy()
        numericas = df.select_dtypes(include="number").columns
        categoricas = df.select_dtypes(exclude="number").columns
        if estrategia == "media":
            df[numericas] = df[numericas].fillna(df[numericas].mean())
        elif estrategia == "mediana":
            df[numericas] = df[numericas].fillna(df[numericas].median())
        elif estrategia == "moda":
            for col in df.columns:
                moda = df[col].mode()
                if not moda.empty:
                    df[col] = df[col].fillna(moda.iloc[0])
        elif estrategia == "constante":
            df[numericas] = df[numericas].fillna(valor_constante)
            df[categoricas] = df[categoricas].fillna("")
        self.datos_procesados = df
        return self.datos_procesados

    def eliminar_duplicados(self) -> pd.DataFrame:
        """Elimina filas duplicadas (compara como texto si hay celdas no hashables)."""
        base = self._base()
        try:
            self.datos_procesados = base.drop_duplicates().reset_index(drop=True)
        except TypeError:  # celdas no hashables (listas/dicts provenientes de JSON)
            mask = ~base.astype(str).duplicated()
            self.datos_procesados = base[mask].reset_index(drop=True)
        return self.datos_procesados

    def filtrar_filas(self, columna: str, valor: object, operador: str = "==") -> pd.DataFrame:
        """Filtra filas comparando una columna contra un valor (==, !=, >, <, >=, <=)."""
        base = self._base()
        ops = {
            "==": base[columna] == valor,
            "!=": base[columna] != valor,
            ">":  base[columna] > valor,
            "<":  base[columna] < valor,
            ">=": base[columna] >= valor,
            "<=": base[columna] <= valor,
        }
        mask = ops.get(operador, base[columna] == valor)
        self.datos_procesados = base[mask].reset_index(drop=True)
        return self.datos_procesados

    def normalizar_standard(self) -> pd.DataFrame:
        """Escala las columnas numéricas a media 0 y desviación 1 (todo el dataset).

        Ojo: si después se entrena un modelo, es preferible escalar dentro de
        `entrenar(escalar=True)` para no filtrar información de test al scaler.
        """
        df = self._base().copy()
        numericas = df.select_dtypes(include="number").columns
        self.scaler = StandardScaler()
        df[numericas] = self.scaler.fit_transform(df[numericas])
        self.datos_procesados = df
        return self.datos_procesados

    def normalizar_minmax(self) -> pd.DataFrame:
        """Escala las columnas numéricas al rango [0, 1] (todo el dataset).

        Misma advertencia de fuga de datos que `normalizar_standard`.
        """
        df = self._base().copy()
        numericas = df.select_dtypes(include="number").columns
        self.scaler = MinMaxScaler()
        df[numericas] = self.scaler.fit_transform(df[numericas])
        self.datos_procesados = df
        return self.datos_procesados

    def codificar_categoricas(self, columnas: list | None = None) -> pd.DataFrame:
        """Convierte columnas object/category a enteros con LabelEncoder (impone un orden artificial)."""
        df = self._base().copy()
        categoricas = columnas if columnas else df.select_dtypes(include=["object", "category"]).columns
        for col in categoricas:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        self.datos_procesados = df
        return self.datos_procesados

    def codificar_onehot(self, columnas: list | None = None) -> pd.DataFrame:
        """One-Hot: una columna binaria 0/1 por categoría, sin imponer orden entre categorías."""
        df = self._base().copy()
        categoricas = columnas if columnas else df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.datos_procesados = pd.get_dummies(df, columns=categoricas, dtype=int)
        return self.datos_procesados

    def convertir_tipo(self, columna: str, dtype: str) -> pd.DataFrame:
        """Convierte una columna a float64/int64/str/datetime; lanza ValueError si no es posible."""
        df = self._base().copy()
        try:
            if dtype == "datetime":
                df[columna] = pd.to_datetime(df[columna], errors="coerce")
            else:
                df[columna] = df[columna].astype(dtype)
        except (ValueError, TypeError) as e:
            raise ValueError(f"No se pudo convertir '{columna}' a {dtype}: {e}")
        self.datos_procesados = df
        return self.datos_procesados

    def obtener_datos_procesados(self) -> pd.DataFrame:
        """Copia del último resultado, o de los datos originales si no hay transformaciones."""
        return self.datos_procesados.copy() if self.datos_procesados is not None else self.datos.copy()
