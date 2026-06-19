from sklearn.linear_model import LinearRegression, Ridge, Lasso
from .modelo import ModeloSupervisado


class ModeloRegresionLineal(ModeloSupervisado):
    """Regresión lineal (target numérico) con regularización opcional (Ridge/Lasso).

    - "ninguna": mínimos cuadrados ordinarios (OLS).
    - "ridge": penaliza coeficientes grandes (útil con features colineales).
    - "lasso": además lleva coeficientes a 0 (selección de features).
    """

    es_clasificacion = False
    scoring_cv = "r2"

    def __init__(self, datos, regularizacion: str = "ninguna", alpha: float = 1.0):
        super().__init__(datos)
        self.regularizacion = regularizacion
        self.alpha = alpha  # fuerza de la regularización (solo Ridge/Lasso)
        self.coeficientes = None

    def _crear_estimador(self):
        if self.regularizacion == "ridge":
            return Ridge(alpha=self.alpha)
        if self.regularizacion == "lasso":
            return Lasso(alpha=self.alpha)
        return LinearRegression()

    def _rejilla_busqueda(self) -> dict:
        # OLS no tiene nada que ajustar; Ridge/Lasso sí (alpha).
        if self.regularizacion in ("ridge", "lasso"):
            return {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
        return {}

    def _post_entrenar(self) -> None:
        self.coeficientes = self.modelo.coef_
