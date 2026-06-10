import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from .modelo import Modelo


class ModeloRegresionLineal(Modelo):
    """Regresión lineal (target numérico); evalúa con R², MSE y RMSE."""

    scoring_cv = "r2"

    def __init__(self, datos):
        super().__init__(datos)
        self.target: str | None = None
        self.features: list = []
        self.coeficientes: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_test: np.ndarray | None = None
        self.y_pred: np.ndarray | None = None

    def _crear_estimador(self) -> LinearRegression:
        return LinearRegression()

    def entrenar(self, test_size: float = 0.2, random_state: int = 42, escalar: bool = False) -> None:
        df = self.datos if isinstance(self.datos, pd.DataFrame) else pd.DataFrame(self.datos)
        # to_numpy(): con pandas 3 .values devuelve ArrowStringArray para texto y sklearn no lo acepta
        X = df[self.features].to_numpy()
        y = df[self.target].to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        if escalar:
            # El scaler se ajusta solo con train para evitar fuga de datos hacia test
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        self.modelo = self._crear_estimador()
        self.modelo.fit(self.X_train, self.y_train)
        self.coeficientes = self.modelo.coef_
        self.y_pred = self.modelo.predict(self.X_test)

    def predecir(self, datos: pd.DataFrame) -> np.ndarray:
        X = datos[self.features].to_numpy() if isinstance(datos, pd.DataFrame) else datos
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.modelo.predict(X)

    def evaluar(self) -> dict:
        mse = float(mean_squared_error(self.y_test, self.y_pred))
        return {
            "r2": float(r2_score(self.y_test, self.y_pred)),
            "mse": mse,
            "rmse": math.sqrt(mse),
        }

    def obtener_metricas(self) -> dict:
        return self.evaluar()
