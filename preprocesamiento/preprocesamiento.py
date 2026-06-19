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
        # {columna: {categoria: codigo}} de la última codificación LabelEncoder,
        # para poder mostrar nombres de categoría en la predicción interactiva.
        self.mapeos_label: dict[str, dict] = {}
        # {columna_original: {categoria: nombre_columna_dummy}} de la última
        # codificación One-Hot, para reconstruir las dummies en la predicción.
        self.mapeos_onehot: dict[str, dict] = {}

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

    def columnas_vacias(self, umbral: float = 1.0) -> list:
        """Nombres de columnas cuya proporción de nulos es >= umbral (1.0 = 100% vacías)."""
        base = self._base()
        if len(base) == 0:
            return []
        frac = base.isnull().mean()
        return frac[frac >= umbral].index.tolist()

    def eliminar_columnas_vacias(self, umbral: float = 1.0) -> pd.DataFrame:
        """Elimina las columnas con proporción de nulos >= umbral (1.0 = 100% vacías).

        Pensado para limpiar las columnas sin datos que a veces deja el web
        scraping, antes de arrastrarlas al resto del preprocesamiento.
        """
        base = self._base()
        self.datos_procesados = base.drop(columns=self.columnas_vacias(umbral)).copy()
        return self.datos_procesados

    def rellenar_nulos(self, estrategia: str = "media", valor_constante: float | str = 0,
                       columnas: list | None = None) -> pd.DataFrame:
        """Rellena nulos según la estrategia: media, mediana, moda o constante.

        Si se indican `columnas`, solo se rellenan esas (el resto queda intacto);
        así el usuario no se ve forzado a imputar binarias, identificadores o el
        target. Media y mediana solo aplican a columnas numéricas (no tienen
        sentido en texto); para texto usa moda o constante.
        """
        df = self._base().copy()
        cols = columnas if columnas is not None else df.columns.tolist()
        for col in cols:
            es_num = pd.api.types.is_numeric_dtype(df[col])
            if estrategia == "media" and es_num:
                df[col] = df[col].fillna(df[col].mean())
            elif estrategia == "mediana" and es_num:
                df[col] = df[col].fillna(df[col].median())
            elif estrategia == "moda":
                moda = df[col].mode()
                if not moda.empty:
                    df[col] = df[col].fillna(moda.iloc[0])
            elif estrategia == "constante":
                # El valor se aplica tal cual: número en columnas numéricas, texto
                # en categóricas. La UI garantiza que el tipo del valor encaje con
                # las columnas elegidas (no mezcla numéricas y de texto a la vez).
                df[col] = df[col].fillna(valor_constante)
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

    def normalizar_standard(self, columnas: list | None = None) -> pd.DataFrame:
        """Escala a media 0 y desviación 1 las columnas indicadas (o todas las
        numéricas si no se indican).

        Solo deberían escalarse features numéricas continuas: escalar binarias,
        dummies, identificadores o el target les quita su significado. Ojo: si
        después se entrena un modelo, es preferible escalar dentro de
        `entrenar(escalar=True)` para no filtrar información de test al scaler.
        """
        df = self._base().copy()
        cols = columnas if columnas is not None else df.select_dtypes(include="number").columns.tolist()
        self.scaler = StandardScaler()
        df[cols] = self.scaler.fit_transform(df[cols])
        self.datos_procesados = df
        return self.datos_procesados

    def normalizar_minmax(self, columnas: list | None = None) -> pd.DataFrame:
        """Escala al rango [0, 1] las columnas indicadas (o todas las numéricas si
        no se indican). Misma recomendación y advertencia que `normalizar_standard`.
        """
        df = self._base().copy()
        cols = columnas if columnas is not None else df.select_dtypes(include="number").columns.tolist()
        self.scaler = MinMaxScaler()
        df[cols] = self.scaler.fit_transform(df[cols])
        self.datos_procesados = df
        return self.datos_procesados

    def codificar_categoricas(self, columnas: list | None = None) -> pd.DataFrame:
        """Convierte columnas object/category a enteros con LabelEncoder (impone un orden artificial).

        Los nulos se conservan como NaN: no se codifican como una categoría más
        (el viejo `.astype(str)` convertía NaN en el texto "nan" y le daba un
        código), para que el resto de la app los siga tratando como dato faltante.
        """
        df = self._base().copy()
        categoricas = columnas if columnas else df.select_dtypes(include=["object", "category"]).columns
        self.mapeos_label = {}
        for col in categoricas:
            presentes = df[col].notna()
            if not presentes.any():
                self.mapeos_label[col] = {}
                continue
            le = LabelEncoder()
            codigos = le.fit_transform(df.loc[presentes, col].astype(str))
            df[col] = np.nan  # las filas que eran nulas se quedan en NaN
            df.loc[presentes, col] = codigos
            # le.classes_ está ordenado y el código de cada clase es su índice
            self.mapeos_label[col] = {cat: i for i, cat in enumerate(le.classes_)}
        self.datos_procesados = df
        return self.datos_procesados

    def codificar_onehot(self, columnas: list | None = None) -> pd.DataFrame:
        """One-Hot: una columna binaria 0/1 por categoría, sin imponer orden entre categorías.

        Las filas que eran nulas no representan ninguna categoría: sus dummies
        quedan como NaN (no como 0, que get_dummies pondría de forma silenciosa),
        para que el control de nulos las siga detectando antes de entrenar.
        """
        df = self._base().copy()
        categoricas = columnas if columnas else df.select_dtypes(include=["object", "category"]).columns.tolist()
        resultado = pd.get_dummies(df, columns=categoricas, dtype=int)
        # Mapea cada columna original a sus dummies (col_categoria) para poder
        # ofrecer la categoría por nombre en la predicción interactiva.
        self.mapeos_onehot = {}
        for col in categoricas:
            # Enumerar las categorías reales de ESTA columna, no filtrar por
            # prefijo: si otra columna se llamara "col_algo", sus dummies también
            # empezarían por "col_" y se mezclarían en este grupo. get_dummies
            # nombra cada dummy como f"{col}_{categoria}".
            mapa = {
                str(cat): f"{col}_{cat}" for cat in df[col].dropna().unique()
                if f"{col}_{cat}" in resultado.columns
            }
            self.mapeos_onehot[col] = mapa
            dummies = list(mapa.values())
            nulos = df[col].isna().to_numpy()
            if nulos.any():
                # asignar NaN promueve esas dummies a float (int no admite NaN)
                resultado.loc[nulos, dummies] = np.nan
        self.datos_procesados = resultado
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
