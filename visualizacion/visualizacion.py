import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizacion:
    def __init__(self, datos: pd.DataFrame):
        self.datos = datos

    def histograma(self, columna: str) -> None:
        pass

    def boxplot(self, columna: str) -> None:
        pass

    def scatter(self, col_x: str, col_y: str) -> None:
        pass

    def heatmap_correlacion(self) -> None:
        pass

    def grafica_barras(self, col_x: str, col_y: str) -> None:
        pass

    def grafica_lineas(self, col_x: str, col_y: str) -> None:
        pass

    def grafica_pastel(self, columna: str) -> None:
        pass

    def grafica_codo(self, inercias: list) -> None:
        pass

    def grafica_clusters(self, datos: pd.DataFrame, etiquetas: np.ndarray, centroides: np.ndarray) -> None:
        pass

    def grafica_regresion(self, y_real: np.ndarray, y_predicho: np.ndarray) -> None:
        pass

    def grafica_confusion(self, matriz: np.ndarray) -> None:
        pass
