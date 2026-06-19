import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, r2_score

from evaluacion import Evaluacion


class Modelo(ABC):
    """Contrato base de todos los modelos: entrenar / predecir / evaluar.

    `scaler` lo asignan las subclases cuando se entrena con `escalar=True`;
    `predecir` debe aplicarlo si existe.
    """

    def __init__(self, datos: pd.DataFrame):
        self.datos = datos
        self.modelo = None
        self.scaler = None
        self.resultados = {}

    def _df(self) -> pd.DataFrame:
        """Los datos como DataFrame (acepta también arrays/dicts en el constructor)."""
        return self.datos if isinstance(self.datos, pd.DataFrame) else pd.DataFrame(self.datos)

    @abstractmethod
    def entrenar(self) -> None:
        pass

    @abstractmethod
    def predecir(self, datos: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def evaluar(self) -> dict:
        pass


class ModeloSupervisado(Modelo):
    """Base de los modelos con target (clasificación o regresión).

    Concentra el flujo común —split, escalado sin fuga, ajuste, predicción,
    métricas, diagnóstico de sobreajuste, línea base, importancias y validación
    cruzada— de modo que cada subclase solo declara su estimador y si es
    clasificación o regresión:

        class ModeloX(ModeloSupervisado):
            es_clasificacion = True            # o False para regresión
            def _crear_estimador(self): return EstimadorDeSklearn(...)
    """

    # Las subclases de regresión ponen es_clasificacion = False y scoring_cv = "r2".
    es_clasificacion: bool = True
    scoring_cv: str = "accuracy"

    def __init__(self, datos):
        super().__init__(datos)
        self.target: str | None = None
        self.features: list = []
        self.X_train: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.y_test: np.ndarray | None = None
        self.y_pred: np.ndarray | None = None
        # Mejores hiperparámetros hallados por GridSearch (None si no se buscó).
        self.mejores_hiperparametros: dict | None = None
        # True si se pidió tuning pero la búsqueda falló y se entrenó por defecto.
        self.tuning_fallo: bool = False

    # Rejilla de hiperparámetros para "buscar mejores parámetros" (GridSearch).
    # Vacía = el modelo no ofrece búsqueda automática. La sobrescriben las subclases.
    def _rejilla_busqueda(self) -> dict:
        return {}

    # ── A definir por cada subclase ────────────────────────────────────
    def _crear_estimador(self):
        raise NotImplementedError("La subclase debe definir _crear_estimador().")

    def _post_entrenar(self) -> None:
        """Hook opcional tras ajustar el modelo (p. ej. guardar coeficientes)."""

    # ── Entrenamiento y predicción ─────────────────────────────────────
    def entrenar(self, test_size: float = 0.2, random_state: int = 42,
                 escalar: bool = False, tuning: bool = False) -> None:
        df = self._df()
        # to_numpy(): con pandas 3 .values devuelve ArrowStringArray para texto y sklearn no lo acepta
        X = df[self.features].to_numpy()
        y = df[self.target].to_numpy()
        if self.es_clasificacion:
            try:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
            except ValueError:
                # stratify falla si alguna clase tiene menos de 2 muestras
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        if escalar:
            # El scaler se ajusta solo con train para evitar fuga de datos hacia test
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        self.mejores_hiperparametros = None
        self.tuning_fallo = False
        rejilla = self._rejilla_busqueda() if tuning else {}
        self.modelo = None
        if rejilla:
            # Búsqueda de hiperparámetros por CV SOLO sobre train; el test queda
            # intacto para una evaluación honesta del modelo ya ajustado. Si la
            # búsqueda falla (p. ej. clases con menos muestras que folds), se cae
            # con elegancia a un entrenamiento normal en vez de tumbar todo.
            particion = (
                StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                if self.es_clasificacion
                else KFold(n_splits=5, shuffle=True, random_state=42)
            )
            try:
                busqueda = GridSearchCV(
                    self._crear_estimador(), rejilla,
                    cv=particion, scoring=self.scoring_cv, n_jobs=-1,
                )
                busqueda.fit(self.X_train, self.y_train)
                self.modelo = busqueda.best_estimator_  # ya reajustado sobre todo el train
                self.mejores_hiperparametros = busqueda.best_params_
            except Exception:
                self.tuning_fallo = True
        if self.modelo is None:
            self.modelo = self._crear_estimador()
            self.modelo.fit(self.X_train, self.y_train)
        self.y_pred = self.modelo.predict(self.X_test)
        self._post_entrenar()

    def predecir(self, datos: pd.DataFrame) -> np.ndarray:
        X = datos[self.features].to_numpy() if isinstance(datos, pd.DataFrame) else datos
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.modelo.predict(X)

    def predecir_proba(self, datos: pd.DataFrame):
        """(clases, probabilidades) si el estimador soporta predict_proba; None si no."""
        if not hasattr(self.modelo, "predict_proba"):
            return None
        X = datos[self.features].to_numpy() if isinstance(datos, pd.DataFrame) else datos
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.modelo.classes_, self.modelo.predict_proba(X)

    # ── Evaluación ─────────────────────────────────────────────────────
    def evaluar(self) -> dict:
        tipo = "clasificacion" if self.es_clasificacion else "regresion"
        metricas = Evaluacion(self, self.y_test, self.y_pred, tipo=tipo).obtener_metricas()
        metricas["train_test"] = self._train_vs_test()
        metricas["baseline"] = self._baseline()
        metricas.update(self._importancias())
        coef = self._coeficientes()
        if coef is not None:
            metricas["coeficientes"] = coef
        return metricas

    def obtener_metricas(self) -> dict:
        return self.evaluar()

    def _metrica_principal(self, y_real, y_pred) -> float:
        """Accuracy en clasificación, R² en regresión: la métrica que se compara
        entre train y test y contra la línea base."""
        if self.es_clasificacion:
            return float(accuracy_score(y_real, y_pred))
        return float(r2_score(y_real, y_pred))

    def _train_vs_test(self) -> dict:
        """Métrica principal en train vs test: si train ≫ test hay sobreajuste."""
        return {
            "metrica": "accuracy" if self.es_clasificacion else "R²",
            "train": self._metrica_principal(self.y_train, self.modelo.predict(self.X_train)),
            "test": self._metrica_principal(self.y_test, self.y_pred),
        }

    def _baseline(self) -> dict:
        """Modelo trivial de referencia: si tu modelo no lo supera, no aporta."""
        if self.es_clasificacion:
            dummy = DummyClassifier(strategy="most_frequent")
            estrategia, metrica = "predecir siempre la clase más frecuente", "accuracy"
        else:
            dummy = DummyRegressor(strategy="mean")
            estrategia, metrica = "predecir siempre la media", "R²"
        dummy.fit(self.X_train, self.y_train)
        return {
            "metrica": metrica,
            "valor": self._metrica_principal(self.y_test, dummy.predict(self.X_test)),
            "estrategia": estrategia,
        }

    def _importancias(self) -> dict:
        """Importancia de cada feature: nativa si el modelo la expone (árboles),
        por permutación en cualquier otro caso (KNN, lineales, logística)."""
        if hasattr(self.modelo, "feature_importances_"):
            imp = dict(zip(self.features, self.modelo.feature_importances_.tolist()))
            return {"importancias": imp, "importancias_tipo": "impureza (nativa del modelo)"}
        try:
            r = permutation_importance(
                self.modelo, self.X_test, self.y_test,
                n_repeats=10, random_state=42, n_jobs=-1,
            )
            imp = dict(zip(self.features, r.importances_mean.tolist()))
            return {"importancias": imp, "importancias_tipo": "permutación"}
        except Exception:
            return {}

    def _coeficientes(self) -> dict | None:
        """Coeficientes por feature en modelos lineales (None en el resto).

        Multiclase: se reporta la magnitud (norma L2 entre clases) por feature.
        """
        coef = getattr(self.modelo, "coef_", None)
        if coef is None:
            return None
        coef = np.asarray(coef)
        if coef.ndim == 1:
            valores = coef
        elif coef.shape[0] == 1:
            valores = coef[0]
        else:
            valores = np.linalg.norm(coef, axis=0)
        return dict(zip(self.features, valores.tolist()))

    # ── Validación cruzada ─────────────────────────────────────────────
    def validacion_cruzada(self, cv: int = 5, escalar: bool = False) -> dict:
        """Validación cruzada k-fold sobre todo el dataset con un estimador nuevo.

        Si `escalar` es True el StandardScaler entra al pipeline, de modo que se
        ajusta dentro de cada fold (sin fuga de datos entre folds).
        """
        df = self._df()
        X = df[self.features].to_numpy()
        y = df[self.target].to_numpy()
        estimador = self._crear_estimador()
        if escalar:
            estimador = make_pipeline(StandardScaler(), estimador)
        # Barajar antes de partir: si el dataset viene ordenado (típico de SQL o
        # scraping) los folds sin shuffle quedan sesgados. Estratifica en
        # clasificación para conservar la proporción de clases en cada fold.
        if self.es_clasificacion:
            particion = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            particion = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(estimador, X, y, cv=particion, scoring=self.scoring_cv, n_jobs=-1)
        return {
            "metrica": self.scoring_cv,
            "scores": [float(s) for s in scores],
            "media": float(scores.mean()),
            "desviacion": float(scores.std()),
        }
