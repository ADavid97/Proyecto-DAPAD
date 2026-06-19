import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from .modelo import Modelo


class ModeloKMeans(Modelo):
    """Clustering K-Means; evalúa con inercia y silhouette, e incluye la curva del codo.

    K-Means se basa en distancias euclídeas, así que sin escalar las columnas de
    mayor magnitud dominan la formación de clusters. Por eso `entrenar(escalar=True)`
    ajusta un StandardScaler; la matriz resultante se reutiliza en `evaluar()` para
    no recomputarla.
    """

    def __init__(self, datos, n_clusters: int = 3):
        super().__init__(datos)
        self.n_clusters = n_clusters
        self.features: list = []
        self.centroides: np.ndarray | None = None
        self.inercias: list = []
        self.etiquetas: np.ndarray | None = None
        # Matriz (ya escalada si aplica) usada para entrenar; la reutiliza evaluar().
        self._X_modelo: np.ndarray | None = None

    def _X(self) -> np.ndarray:
        df = self._df()
        return df[self.features].to_numpy() if self.features else df.select_dtypes(include="number").to_numpy()

    def entrenar(self, escalar: bool = False) -> None:
        X = self._X()
        if escalar:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        self._X_modelo = X
        self.modelo = KMeans(n_clusters=self.n_clusters, random_state=42, n_init="auto")
        self.modelo.fit(X)
        self.etiquetas = self.modelo.labels_
        centros = self.modelo.cluster_centers_
        # Centroides de vuelta al espacio original para graficarlos sobre las
        # features sin escalar; el modelo conserva los escalados para predecir.
        self.centroides = self.scaler.inverse_transform(centros) if self.scaler is not None else centros

    def predecir(self, datos: pd.DataFrame) -> np.ndarray:
        X = datos[self.features].to_numpy() if isinstance(datos, pd.DataFrame) else datos
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.modelo.predict(X)

    # Silhouette construye una matriz de distancias O(n²); por encima de este
    # tamaño se calcula sobre una muestra para no agotar memoria ni colgar la app.
    _MAX_SILHOUETTE = 5000

    def evaluar(self) -> dict:
        X = self._X_modelo if self._X_modelo is not None else self._X()
        if len(set(self.etiquetas)) <= 1:
            sil = 0.0
        elif len(X) > self._MAX_SILHOUETTE:
            sil = float(silhouette_score(
                X, self.etiquetas,
                sample_size=self._MAX_SILHOUETTE, random_state=42,
            ))
        else:
            sil = float(silhouette_score(X, self.etiquetas))
        return {
            "inercia": float(self.modelo.inertia_),
            "silhouette": sil,
            "n_clusters": self.n_clusters,
        }

    def calcular_codo(self, k_max: int = 10, escalar: bool = False) -> list:
        X = self._X()
        if escalar:
            X = StandardScaler().fit_transform(X)
        self.inercias = []
        for k in range(1, k_max + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            km.fit(X)
            self.inercias.append(float(km.inertia_))
        return self.inercias

    def obtener_segmentos(self) -> dict:
        df = self._df().copy()
        df["Cluster"] = self.etiquetas
        return {int(k): grp.drop(columns=["Cluster"]) for k, grp in df.groupby("Cluster")}

    def obtener_metricas(self) -> dict:
        return self.evaluar()
