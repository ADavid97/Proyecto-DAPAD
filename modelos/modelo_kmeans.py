import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .modelo import Modelo


class ModeloKMeans(Modelo):
    """Clustering K-Means; evalúa con inercia y silhouette, e incluye la curva del codo."""

    def __init__(self, datos, n_clusters: int = 3):
        super().__init__(datos)
        self.n_clusters = n_clusters
        self.features: list = []
        self.centroides: np.ndarray | None = None
        self.inercias: list = []
        self.etiquetas: np.ndarray | None = None

    def _X(self) -> np.ndarray:
        df = self.datos if isinstance(self.datos, pd.DataFrame) else pd.DataFrame(self.datos)
        return df[self.features].to_numpy() if self.features else df.select_dtypes(include="number").to_numpy()

    def entrenar(self) -> None:
        X = self._X()
        self.modelo = KMeans(n_clusters=self.n_clusters, random_state=42, n_init="auto")
        self.modelo.fit(X)
        self.etiquetas = self.modelo.labels_
        self.centroides = self.modelo.cluster_centers_

    def predecir(self, datos: pd.DataFrame) -> np.ndarray:
        X = datos[self.features].to_numpy() if isinstance(datos, pd.DataFrame) else datos
        return self.modelo.predict(X)

    def evaluar(self) -> dict:
        X = self._X()
        sil = float(silhouette_score(X, self.etiquetas)) if len(set(self.etiquetas)) > 1 else 0.0
        return {
            "inercia": float(self.modelo.inertia_),
            "silhouette": sil,
            "n_clusters": self.n_clusters,
        }

    def calcular_codo(self, k_max: int = 10) -> list:
        X = self._X()
        self.inercias = []
        for k in range(1, k_max + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            km.fit(X)
            self.inercias.append(float(km.inertia_))
        return self.inercias

    def obtener_segmentos(self) -> dict:
        df = self.datos if isinstance(self.datos, pd.DataFrame) else pd.DataFrame(self.datos)
        df = df.copy()
        df["Cluster"] = self.etiquetas
        return {int(k): grp.drop(columns=["Cluster"]) for k, grp in df.groupby("Cluster")}

    def obtener_metricas(self) -> dict:
        return self.evaluar()
