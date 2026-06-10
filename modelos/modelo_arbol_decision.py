import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from .modelo import Modelo


class ModeloArbolDecision(Modelo):
    """Árbol de decisión para clasificación; expone importancia de features en evaluar()."""

    def __init__(self, datos, profundidad_max: int | None = None):
        super().__init__(datos)
        self.target: str | None = None
        self.features: list = []
        self.profundidad_max = profundidad_max
        self.X_test: np.ndarray | None = None
        self.y_test: np.ndarray | None = None
        self.y_pred: np.ndarray | None = None

    def entrenar(self, test_size: float = 0.2, random_state: int = 42, escalar: bool = False) -> None:
        df = self.datos if isinstance(self.datos, pd.DataFrame) else pd.DataFrame(self.datos)
        X = df[self.features].values
        y = df[self.target].values
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        except ValueError:
            # stratify falla si alguna clase tiene menos de 2 muestras
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        if escalar:
            # El scaler se ajusta solo con train para evitar fuga de datos hacia test
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        self.modelo = DecisionTreeClassifier(
            max_depth=self.profundidad_max, random_state=random_state
        )
        self.modelo.fit(self.X_train, self.y_train)
        self.y_pred = self.modelo.predict(self.X_test)

    def predecir(self, datos: pd.DataFrame) -> np.ndarray:
        X = datos[self.features].values if isinstance(datos, pd.DataFrame) else datos
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.modelo.predict(X)

    def evaluar(self) -> dict:
        return {
            "accuracy": float(accuracy_score(self.y_test, self.y_pred)),
            "precision": float(precision_score(self.y_test, self.y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(self.y_test, self.y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(self.y_test, self.y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(self.y_test, self.y_pred),
            "importancias": dict(zip(self.features, self.modelo.feature_importances_.tolist())),
        }

    def obtener_metricas(self) -> dict:
        return self.evaluar()
