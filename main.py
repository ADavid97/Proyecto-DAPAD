import io
import os
import pickle
import threading

import joblib
import streamlit as st
import pandas as pd
from sklearn.pipeline import make_pipeline
from datos import Datos
from analisis_exploratorio import AnalisisExploratorio
from visualizacion import Visualizacion
from preprocesamiento import Preprocesamiento
from modelos import (
    ModeloKNN, ModeloKMeans, ModeloRegresionLineal, ModeloRegresionLogistica,
    ModeloArbolDecision, ModeloRandomForest, ModeloGradientBoosting,
    ModeloKNNRegresion, ModeloArbolRegresion, ModeloRandomForestRegresion,
    ModeloGradientBoostingRegresion,
    ModeloNaiveBayes, ModeloRedNeuronal, ModeloRedNeuronalRegresion,
)
from asistente import (
    chat_stream,
    mensajes_chat,
    ollama_disponible,
    prompt_explicacion,
    MODELO_POR_DEFECTO,
    SISTEMA_DATASET,
)
from styles import apply_styles
import chart_theme  # noqa: F401 — registers "daad" Plotly template on import

st.set_page_config(page_title="DAAD", page_icon="", layout="wide")
apply_styles()


def _hash_df(d: pd.DataFrame):
    """Clave de caché para un DataFrame, tolerante a celdas no hashables.

    st.cache_data hashea sus argumentos ANTES de ejecutar la función, y el hash
    por defecto de pandas revienta con celdas tipo lista/dict (frecuentes en JSON
    anidado). Aquí caemos a una representación textual para esas columnas, de modo
    que las secciones cacheadas no fallen con datasets provenientes de JSON.
    """
    try:
        return pd.util.hash_pandas_object(d).values.tobytes()
    except TypeError:
        return (d.shape, tuple(map(str, d.columns)), d.astype(str).to_numpy().tobytes())


# hash_funcs compartido por todos los helpers cacheados que reciben DataFrames
# (incluso anidados dentro de tuplas, como en la sección ② Conjunto).
_HASH_DF: dict = {pd.DataFrame: _hash_df}


@st.cache_data(hash_funcs=_HASH_DF)
def _csv_bytes(data: pd.DataFrame) -> bytes:
    return data.to_csv(index=False).encode("utf-8")


def _nombre_csv(nombre: str | None) -> str:
    """Nombre de archivo seguro para descargar, basado en el dataset activo."""
    base = nombre or "datos"
    for c in '\\/:*?"<>|':  # caracteres no válidos en nombres de archivo (Windows)
        base = base.replace(c, "_")
    base = base.strip() or "datos"
    return base if base.lower().endswith(".csv") else f"{base}.csv"


@st.cache_data(hash_funcs=_HASH_DF)
def _contar_duplicados(data: pd.DataFrame) -> int:
    try:
        return int(data.duplicated().sum())
    except TypeError:  # celdas no hashables (listas/dicts provenientes de JSON)
        return int(data.astype(str).duplicated().sum())


@st.cache_data(hash_funcs=_HASH_DF)
def _estadisticas_descriptivas(data: pd.DataFrame) -> pd.DataFrame:
    return AnalisisExploratorio(data).estadisticas_descriptivas()


@st.cache_data(hash_funcs=_HASH_DF)
def _matriz_correlacion(data: pd.DataFrame) -> pd.DataFrame:
    return AnalisisExploratorio(data).matriz_correlacion()


@st.cache_data(hash_funcs=_HASH_DF)
def _resumen_general(data: pd.DataFrame) -> dict:
    return AnalisisExploratorio(data).resumen_general()


# ── Helpers de la sección ② Conjunto / Comparar ──────────────────────────
# Reciben los datasets como tupla ((nombre, df), …) para que sean hashables y
# st.cache_data evite recalcular en cada interacción de un widget.
@st.cache_data(hash_funcs=_HASH_DF)
def _comp_resumen(items: tuple) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Dataset": n,
            "Filas": d.shape[0],
            "Columnas": d.shape[1],
            "Numéricas": d.select_dtypes(include="number").shape[1],
            "Nulos": int(d.isnull().sum().sum()),
        }
        for n, d in items
    ])


@st.cache_data(hash_funcs=_HASH_DF)
def _comp_stats(items: tuple, columna: str) -> pd.DataFrame:
    return pd.DataFrame({
        n: {
            "Conteo": int(d[columna].count()),
            "Media": d[columna].mean(),
            "Desv. est.": d[columna].std(),
            "Mín": d[columna].min(),
            "Mediana": d[columna].median(),
            "Máx": d[columna].max(),
            "Nulos": int(d[columna].isnull().sum()),
        }
        for n, d in items
    })


@st.cache_data(hash_funcs=_HASH_DF)
def _comp_largo(items: tuple, columna: str) -> pd.DataFrame:
    """Formato largo (valor, origen) para el histograma superpuesto."""
    return pd.concat(
        [pd.DataFrame({columna: d[columna], "origen": n}) for n, d in items],
        ignore_index=True,
    )


@st.cache_data(hash_funcs=_HASH_DF)
def _comp_props(items: tuple, columna: str, top: int = 20) -> pd.DataFrame:
    """Frecuencia relativa (%) por dataset de las top categorías de una columna."""
    props = pd.DataFrame({
        n: d[columna].astype(str).value_counts(normalize=True) for n, d in items
    }).fillna(0)
    props = props.loc[props.sum(axis=1).sort_values(ascending=False).index].head(top)
    return (props * 100).round(1)


@st.cache_data(hash_funcs=_HASH_DF)
def _hay_columnas_vacias(data: pd.DataFrame, umbral: float = 0.5) -> bool:
    """True si alguna columna supera el umbral de nulos. Cacheado: se evalúa en
    cada render del menú de ④ Preprocesamiento para decidir si ofrecer la opción."""
    if len(data) == 0:
        return False
    return bool((data.isnull().mean() >= umbral).any())


@st.cache_data(hash_funcs=_HASH_DF)
def _columnas_escalables(data: pd.DataFrame) -> tuple[list, list]:
    """(numéricas, recomendadas para escalar). Recomendadas = numéricas que NO son
    binarias (sus valores no se reducen a 0/1): escalar binarias, dummies o flags
    les quita su significado, así que se excluyen del default."""
    numericas = data.select_dtypes(include="number").columns.tolist()
    recomendadas = [c for c in numericas if not set(pd.unique(data[c].dropna())) <= {0, 1}]
    return numericas, recomendadas


MAX_HISTORIAL = 10  # pasos máximos que se pueden deshacer por dataset


# Tarea de cada modelo del selector de ⑥ Modelos. Es la ÚNICA fuente de verdad
# del tipo de problema: gobierna la validación del target y la vista de resultados
# en ⑦ Evaluación. Añadir un modelo nuevo = registrar aquí su tarea.
TAREA_MODELO = {
    "KNN — K-Nearest Neighbors": "clasificacion",
    "Árbol de Decisión": "clasificacion",
    "Random Forest": "clasificacion",
    "Gradient Boosting": "clasificacion",
    "Regresión Logística": "clasificacion",
    "Naive Bayes": "clasificacion",
    "Red Neuronal": "clasificacion",
    "Regresión Lineal": "regresion",
    "KNN (regresión)": "regresion",
    "Árbol de Decisión (regresión)": "regresion",
    "Random Forest (regresión)": "regresion",
    "Gradient Boosting (regresión)": "regresion",
    "Red Neuronal (regresión)": "regresion",
    "K-Means (clustering)": "clustering",
}


RUTA_SESION = os.path.join("resultados", "_sesion.pkl")
# Solo se persiste el trabajo del usuario; lo derivado de la carga
# (tablas/hojas/elementos del uploader o scraping) se queda fuera.
_CLAVES_SESION = (
    "datasets", "df", "df_original", "df_activo", "historiales",
    "modelo_entrenado", "modelo_tipo", "metricas", "cv_resultados",
    "label_mappings", "onehot_mappings",
)

# Serializar la sesión (todos los datasets + el modelo) puede ser pesado; se hace
# en un hilo aparte para no bloquear la UI. El lock serializa los volcados y el
# contador de versión descarta un guardado que llegue tarde y pisaría a otro más
# reciente. La escritura atómica (.tmp + replace) garantiza que nunca quede un
# .pkl a medio escribir aunque el proceso muera durante el guardado.
_LOCK_SESION = threading.Lock()
_VERSION_SESION = 0

# Carpeta de sesiones con nombre (snapshots explícitos del usuario), aparte del
# autoguardado de _sesion.pkl.
DIR_SESIONES = os.path.join("resultados", "sesiones")


def _snapshot_sesion() -> dict:
    """Instantánea del estado de trabajo (solo las claves persistibles)."""
    return {clave: st.session_state.get(clave) for clave in _CLAVES_SESION}


def guardar_sesion() -> None:
    """Persiste el estado de trabajo en disco de forma asíncrona y atómica.

    Se llama tras cada cambio real (no en cada rerun). Captura una instantánea
    del estado en el hilo principal y delega la escritura a un hilo daemon, de
    modo que la interacción no espera al volcado a disco. Nunca debe tumbar la
    app, así que cualquier error se ignora en silencio.
    """
    global _VERSION_SESION
    if not st.session_state.get("datasets"):
        return
    datos = _snapshot_sesion()
    _VERSION_SESION += 1
    mi_version = _VERSION_SESION

    def _escribir(payload: dict, version: int) -> None:
        with _LOCK_SESION:
            if version < _VERSION_SESION:
                return  # ya hay una instantánea más reciente en cola: no la pises
            try:
                os.makedirs(os.path.dirname(RUTA_SESION), exist_ok=True)
                tmp = RUTA_SESION + ".tmp"
                with open(tmp, "wb") as f:
                    pickle.dump(payload, f)
                os.replace(tmp, RUTA_SESION)  # atómico
            except Exception:
                pass

    threading.Thread(target=_escribir, args=(datos, mi_version), daemon=True).start()


def cargar_sesion() -> None:
    """Restaura el estado desde disco si existe. Si está corrupto, arranca limpio.

    Seguridad: usa pickle, que ejecuta código arbitrario al deserializar. Es
    aceptable porque el archivo lo escribe solo esta misma app en local
    (resultados/_sesion.pkl) y nadie más debería tocarlo. No cargues aquí un .pkl
    de origen ajeno; si esta app se expusiera en red, habría que migrar a un
    formato de datos (p. ej. parquet por dataset) en vez de pickle.
    """
    if not os.path.exists(RUTA_SESION):
        return
    try:
        with open(RUTA_SESION, "rb") as f:
            datos = pickle.load(f)
        for clave, valor in datos.items():
            st.session_state[clave] = valor
    except Exception:
        try:
            os.remove(RUTA_SESION)
        except OSError:
            pass


def borrar_sesion() -> None:
    """Elimina el guardado en disco y limpia el estado de trabajo (Nueva sesión)."""
    try:
        if os.path.exists(RUTA_SESION):
            os.remove(RUTA_SESION)
    except OSError:
        pass
    for clave in _CLAVES_SESION:
        st.session_state.pop(clave, None)


# ── Sesiones con nombre ────────────────────────────────────────────────────
def _nombre_sesion_seguro(nombre: str) -> str:
    """Nombre de archivo seguro para una sesión (sin caracteres inválidos ni rutas)."""
    limpio = nombre.strip()
    for c in '\\/:*?"<>|':  # inválidos en Windows + evita travesía de rutas
        limpio = limpio.replace(c, "_")
    return limpio.strip(". ")  # sin puntos/espacios al borde (".." , "." )


def listar_sesiones() -> list[str]:
    """Nombres de las sesiones guardadas en disco (orden alfabético)."""
    try:
        archivos = os.listdir(DIR_SESIONES)
    except OSError:
        return []
    return sorted(a[:-4] for a in archivos if a.endswith(".pkl"))


def guardar_sesion_nombre(nombre: str) -> bool:
    """Guarda el estado de trabajo actual como una sesión con nombre (atómico)."""
    if not st.session_state.get("datasets"):
        return False
    try:
        os.makedirs(DIR_SESIONES, exist_ok=True)
        ruta = os.path.join(DIR_SESIONES, f"{nombre}.pkl")
        tmp = ruta + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(_snapshot_sesion(), f)
        os.replace(tmp, ruta)  # atómico: nunca deja un .pkl a medio escribir
        return True
    except Exception:
        return False


def cargar_sesion_nombre(nombre: str) -> bool:
    """Restaura una sesión guardada como el estado de trabajo actual."""
    ruta = os.path.join(DIR_SESIONES, f"{nombre}.pkl")
    try:
        with open(ruta, "rb") as f:
            datos = pickle.load(f)
    except Exception:
        return False
    for clave, valor in datos.items():
        st.session_state[clave] = valor
    guardar_sesion()  # refleja la sesión cargada también en el autoguardado
    return True


def eliminar_sesion_nombre(nombre: str) -> None:
    """Borra del disco una sesión guardada."""
    try:
        os.remove(os.path.join(DIR_SESIONES, f"{nombre}.pkl"))
    except OSError:
        pass


def _guardar_df(nuevo_df: pd.DataFrame) -> None:
    """Actualiza df activo, sincroniza con el dict de datasets y persiste a disco."""
    st.session_state.df = nuevo_df
    if st.session_state.df_activo:
        st.session_state.datasets[st.session_state.df_activo] = nuevo_df
    guardar_sesion()


def _limpiar_modelo() -> None:
    """Olvida el modelo entrenado y sus resultados.

    Se llama al cambiar de dataset activo (cargar, combinar, seleccionar otro o
    borrar): un modelo entrenado sobre otras columnas no debe arrastrarse, o
    Evaluación intentaría graficar features que ya no existen en el df activo.
    """
    st.session_state.modelo_entrenado = None
    st.session_state.modelo_tipo = None
    st.session_state.metricas = None
    st.session_state.cv_resultados = None
    # Los mapeos de codificación pertenecen al dataset anterior: descártalos.
    st.session_state.label_mappings = {}
    st.session_state.onehot_mappings = {}


def _cargar_dataset(nombre: str, datos: pd.DataFrame) -> None:
    """Registra un dataset recién cargado como el activo y persiste la sesión."""
    st.session_state.df = datos
    st.session_state.datasets[nombre] = datos
    st.session_state.df_activo = nombre
    st.session_state.df_original = None
    st.session_state.historiales.pop(nombre, None)
    _limpiar_modelo()
    guardar_sesion()


def _clave_historial() -> str:
    return st.session_state.df_activo or "_actual"


def _sincronizar_mapeos_codificacion() -> None:
    """Descarta mapeos de codificación que ya no reflejan el df activo.

    Tras deshacer o restaurar, una columna puede volver a ser texto (se anuló su
    LabelEncoder) o desaparecer sus dummies (se anuló el One-Hot). Si no se podan,
    la predicción interactiva de ⑦ seguiría ofreciendo categorías para columnas
    que ya no están codificadas.
    """
    df = st.session_state.df
    if df is None:
        st.session_state.label_mappings = {}
        st.session_state.onehot_mappings = {}
        return
    cols = set(df.columns)
    # LabelEncoder: la columna debe seguir existiendo y ser numérica (codificada).
    st.session_state.label_mappings = {
        col: m for col, m in st.session_state.label_mappings.items()
        if col in cols and pd.api.types.is_numeric_dtype(df[col])
    }
    # One-Hot: todas las columnas dummy del grupo deben seguir presentes.
    st.session_state.onehot_mappings = {
        grupo: m for grupo, m in st.session_state.onehot_mappings.items()
        if m and all(dummy in cols for dummy in m.values())
    }


def _aplicar_transformacion(descripcion: str, nuevo_df: pd.DataFrame) -> bool:
    """Aplica una transformación guardando el estado previo para poder deshacerla.

    Si el resultado es idéntico al estado actual (operación sin efecto, p. ej.
    eliminar nulos cuando ya no hay), avisa al usuario, no la registra en el
    historial y devuelve False. Devuelve True si hubo un cambio real.
    """
    actual = st.session_state.df
    if actual is not None and nuevo_df.equals(actual):
        st.warning(
            "No hubo cambios: los datos ya estan en ese estado "
            "(sin nulos / sin duplicados, o la operacion ya se aplico)."
        )
        return False
    hist = st.session_state.historiales.setdefault(_clave_historial(), [])
    hist.append({"descripcion": descripcion, "df_antes": actual})
    del hist[:-MAX_HISTORIAL]
    _guardar_df(nuevo_df)
    return True


def mostrar_df(data: pd.DataFrame) -> None:
    # Solo las columnas de texto necesitan castearse a str (evita errores de
    # Arrow con celdas mixtas); assign reusa el resto sin copiar todo el frame.
    cols_obj = data.select_dtypes(include=["object", "str"]).columns
    if len(cols_obj) > 0:
        data = data.assign(**{col: data[col].astype(str) for col in cols_obj})
    st.dataframe(data, width="stretch")


def _page_header(icon: str, title: str, crumb: str) -> None:
    st.markdown(
        f'<div class="daad-page-header">'
        f'<span class="daad-page-icon">{icon}</span>'
        f'<span class="daad-page-title">{title}</span>'
        f'<span class="daad-page-crumb">{crumb}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _section_label(text: str) -> None:
    st.markdown(
        f'<span class="daad-section-label">{text}</span>',
        unsafe_allow_html=True,
    )


def _subheader(text: str) -> None:
    """Subtítulo de sección (estilo 'daad-subheader')."""
    st.markdown(f'<span class="daad-subheader">{text}</span>', unsafe_allow_html=True)


def _error_carga(mensaje: str, cargador: Datos) -> None:
    """Muestra un error de carga, añadiendo el detalle de `cargador.ultimo_error` si lo hay."""
    detalle = f" Detalle: {cargador.ultimo_error}" if cargador.ultimo_error else ""
    st.error(f"{mensaje}{detalle}")


def _crear_dataset_combinado(nombre: str, resultado: pd.DataFrame) -> None:
    """Valida el nombre y registra el dataset combinado como activo (concat/merge).

    Si el nombre está vacío o ya existe, avisa y no hace nada; si no, lo crea y
    relanza la app para reflejarlo en el resto de secciones.
    """
    if not nombre.strip():
        st.error("Ponle un nombre al dataset.")
    elif nombre in st.session_state.datasets:
        st.error(f"Ya existe un dataset llamado '{nombre}'. Elige otro nombre.")
    else:
        _cargar_dataset(nombre, resultado)
        st.success(f"Dataset '{nombre}' creado y activado. Ya puedes analizarlo en las demás secciones.")
        st.rerun()


# --- RESTAURAR SESIÓN ---
# Una sola vez por sesión de navegador: si hay un guardado en disco, se restaura
# antes de inicializar los valores por defecto para no pisarlo.
if "_sesion_cargada" not in st.session_state:
    cargar_sesion()
    st.session_state._sesion_cargada = True

# --- SESSION STATE ---
if "df" not in st.session_state:
    st.session_state.df = None
if "tablas_pg" not in st.session_state:
    st.session_state.tablas_pg = []
if "tablas_json" not in st.session_state:
    st.session_state.tablas_json = {}
if "elementos_url" not in st.session_state:
    st.session_state.elementos_url = {}
if "hojas_excel" not in st.session_state:
    st.session_state.hojas_excel = {}
if "df_original" not in st.session_state:
    st.session_state.df_original = None
if "datasets" not in st.session_state:
    st.session_state.datasets = {}
if "df_activo" not in st.session_state:
    st.session_state.df_activo = None
if "modelo_entrenado" not in st.session_state:
    st.session_state.modelo_entrenado = None
if "modelo_tipo" not in st.session_state:
    st.session_state.modelo_tipo = None
if "metricas" not in st.session_state:
    st.session_state.metricas = None
if "cv_resultados" not in st.session_state:
    st.session_state.cv_resultados = None
if "historiales" not in st.session_state:
    st.session_state.historiales = {}
if "label_mappings" not in st.session_state:
    st.session_state.label_mappings = {}
if "onehot_mappings" not in st.session_state:
    st.session_state.onehot_mappings = {}

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(
        '<div class="daad-chrome">'
        '<div class="daad-chrome-dots"><span></span><span></span><span></span></div>'
        '<span class="daad-chrome-url">datos.local</span>'
        '<span class="daad-chrome-mark">D</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    _section_label("Desarrollado por")
    st.markdown(
        '<div class="daad-authors">'
        "<p>Reyes Calva Angel David</p>"
        "<p>Ortiz Juarez Emiliano</p>"
        "<p>Hernandez Gaspar Andrei</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    _section_label("Cargar datos")
    tab_est, tab_noest = st.tabs(["Estructurados", "No estructurados"])

    with tab_est:
        tipo = st.radio("Formato", ["CSV", "TSV", "Excel", "PostgreSQL"], label_visibility="collapsed")

        if tipo in ("CSV", "TSV"):
            extensiones = ["csv"] if tipo == "CSV" else ["tsv", "txt"]
            archivo = st.file_uploader(f"Sube archivo {tipo}", type=extensiones, key=tipo)
            if archivo is not None:
                extension = archivo.name.rsplit(".", 1)[-1].lower()
                esperadas = {"CSV": ["csv"], "TSV": ["tsv", "txt"]}
                if extension not in esperadas[tipo]:
                    st.error(f"El archivo '{archivo.name}' no es un {tipo} válido. Sube un archivo con extensión {extensiones}.")
                elif st.session_state.get("_ultimo_archivo_id") != archivo.file_id:
                    # Solo cargar cuando el archivo es nuevo; el uploader sigue
                    # devolviendo el archivo en cada rerun, y recargarlo pisaría
                    # las transformaciones de preprocesamiento.
                    cargador = Datos()
                    df = cargador.cargar_csv(archivo) if tipo == "CSV" else cargador.cargar_tsv(archivo)
                    if df is not None:
                        st.session_state._ultimo_archivo_id = archivo.file_id
                        _cargar_dataset(archivo.name, df)
                        st.success(f"{df.shape[0]} filas × {df.shape[1]} columnas")
                    else:
                        _error_carga("Error al leer el archivo.", cargador)

        elif tipo == "Excel":
            archivo = st.file_uploader("Sube archivo Excel", type=["xlsx", "xls"], key="Excel")
            # Solo re-parsear cuando el archivo es nuevo; el uploader lo devuelve
            # en cada rerun y volver a leer todas las hojas es un desperdicio.
            if archivo is not None and st.session_state.get("_excel_file_id") != archivo.file_id:
                cargador = Datos()
                hojas = cargador.cargar_excel(archivo)
                if hojas:
                    st.session_state._excel_file_id = archivo.file_id
                    st.session_state.hojas_excel = hojas
                    st.success(f"{len(hojas)} hoja(s) encontrada(s)")
                else:
                    _error_carga("No se pudo leer el archivo Excel.", cargador)

            if st.session_state.hojas_excel:
                nombre = st.selectbox("Selecciona hoja", list(st.session_state.hojas_excel.keys()), key="sel_excel")
                if st.button("Cargar hoja", key="btn_excel", use_container_width=True):
                    df = st.session_state.hojas_excel[nombre]
                    nombre_ds = f"{archivo.name if archivo else 'excel'}[{nombre}]"
                    _cargar_dataset(nombre_ds, df)
                    st.success(f"{df.shape[0]} filas × {df.shape[1]} columnas")

        elif tipo == "PostgreSQL":
            host = st.text_input("Host", "localhost")
            puerto = st.number_input("Puerto", value=5432, step=1)
            base_datos = st.text_input("Base de datos")
            usuario = st.text_input("Usuario")
            contrasena = st.text_input("Contraseña", type="password")

            if st.button("Conectar", use_container_width=True):
                cargador = Datos()
                tablas = cargador.listar_tablas(host, int(puerto), base_datos, usuario, contrasena)
                if tablas:
                    st.session_state.tablas_pg = tablas
                    st.success(f"{len(tablas)} tablas encontradas")
                else:
                    _error_carga("No se pudo conectar o no hay tablas.", cargador)

            if st.session_state.tablas_pg:
                tabla = st.selectbox("Selecciona tabla", st.session_state.tablas_pg)
                if st.button("Cargar tabla", use_container_width=True) and tabla is not None:
                    cargador = Datos()
                    df = cargador.cargar_tabla_sql(host, int(puerto), base_datos, usuario, contrasena, tabla)
                    if df is not None:
                        _cargar_dataset(tabla, df)
                        st.success(f"Tabla '{tabla}' cargada")

    with tab_noest:
        archivo = st.file_uploader("Sube archivo JSON", type=["json"], key="JSON")
        # Solo re-parsear cuando el archivo es nuevo (ver nota en el cargador CSV).
        if archivo is not None and st.session_state.get("_json_file_id") != archivo.file_id:
            cargador = Datos()
            tablas = cargador.cargar_json(archivo)
            if tablas:
                st.session_state._json_file_id = archivo.file_id
                st.session_state.tablas_json = tablas
                st.success(f"{len(tablas)} tabla(s) encontrada(s)")
            else:
                _error_carga("No se pudo leer el archivo JSON.", cargador)

        if st.session_state.tablas_json:
            nombre = st.selectbox("Selecciona tabla", list(st.session_state.tablas_json.keys()), key="sel_json")
            if st.button("Cargar tabla", key="btn_json", use_container_width=True):
                df = st.session_state.tablas_json[nombre]
                _cargar_dataset(nombre, df)
                st.success(f"{df.shape[0]} filas × {df.shape[1]} columnas")

        st.divider()
        _section_label("Web scraping")
        url = st.text_input("URL", placeholder="https://es.wikipedia.org/wiki/...", key="url_input")
        if st.button("Buscar elementos", key="btn_buscar_url", use_container_width=True):
            if url:
                with st.spinner("Descargando página..."):
                    cargador = Datos()
                    elementos = cargador.cargar_url(url)
                if elementos:
                    st.session_state.elementos_url = elementos
                    total = sum(len(v) for v in elementos.values())
                    st.success(f"{total} elemento(s) en {len(elementos)} etiqueta(s)")
                elif cargador.ultimo_error:
                    st.session_state.elementos_url = {}
                    _error_carga("No se pudo descargar la página.", cargador)
                else:
                    st.session_state.elementos_url = {}
                    st.error("La página se descargó pero no contiene tablas ni listas.")
            else:
                st.warning("Ingresa una URL.")

        if st.session_state.elementos_url:
            etiquetas_disponibles = list(st.session_state.elementos_url.keys())
            etiquetas_sel = st.multiselect(
                "Etiquetas", etiquetas_disponibles, default=etiquetas_disponibles, key="sel_tags"
            )
            opciones = {
                nombre: df
                for tag in etiquetas_sel
                for nombre, df in st.session_state.elementos_url.get(tag, {}).items()
            }
            if opciones:
                nombre_el = st.selectbox("Elemento", list(opciones.keys()), key="sel_url")
                if st.button("Cargar elemento", key="btn_cargar_url", use_container_width=True):
                    df = opciones[nombre_el]
                    _cargar_dataset(nombre_el, df)
                    st.success(f"{df.shape[0]} filas × {df.shape[1]} columnas")
            else:
                st.info("Selecciona al menos una etiqueta.")

    if st.session_state.datasets:
        _section_label("Datasets cargados")
        nombres_ds = list(st.session_state.datasets.keys())
        idx_activo = nombres_ds.index(st.session_state.df_activo) if st.session_state.df_activo in nombres_ds else 0
        sel_ds = st.selectbox(
            "Dataset activo",
            nombres_ds,
            index=idx_activo,
            key="sel_dataset_activo",
        )
        if sel_ds != st.session_state.df_activo:
            st.session_state.df = st.session_state.datasets[sel_ds]
            st.session_state.df_activo = sel_ds
            st.session_state.df_original = None
            _limpiar_modelo()
            guardar_sesion()
            st.rerun()
        _df_sel = st.session_state.datasets[sel_ds]
        st.caption(f"{_df_sel.shape[0]} filas × {_df_sel.shape[1]} columnas")
        col_del, col_reset = st.columns(2)
        if col_del.button("Eliminar dataset", key="btn_del_ds", use_container_width=True):
            del st.session_state.datasets[sel_ds]
            st.session_state.historiales.pop(sel_ds, None)
            _limpiar_modelo()
            if st.session_state.datasets:
                nuevo_ds = list(st.session_state.datasets.keys())[0]
                st.session_state.df = st.session_state.datasets[nuevo_ds]
                st.session_state.df_activo = nuevo_ds
                st.session_state.df_original = None
                guardar_sesion()
            else:
                st.session_state.df = None
                st.session_state.df_activo = None
                st.session_state.df_original = None
                borrar_sesion()  # sin datasets no hay nada que conservar
            st.rerun()
        if col_reset.button("Nueva sesión", key="btn_nueva_sesion", use_container_width=True,
                            help="Borra todo lo cargado y el guardado en disco para empezar de cero."):
            borrar_sesion()
            st.rerun()

    # ── Sesiones con nombre (snapshots que se conservan al cambiar de trabajo) ──
    sesiones_guardadas = listar_sesiones()
    if st.session_state.datasets or sesiones_guardadas:
        _section_label("Sesiones")
        # Mensaje de confirmación que sobrevive al st.rerun() de guardar/cargar.
        msg_sesion = st.session_state.pop("_sesion_msg", None)
        if msg_sesion:
            st.success(msg_sesion)
        if st.session_state.datasets:
            nombre_sesion = st.text_input(
                "Guardar trabajo actual como…", key="nombre_sesion_guardar",
                placeholder="ej. experimento_1",
            )
            if st.button("Guardar sesión", key="btn_guardar_sesion", use_container_width=True,
                         disabled=not nombre_sesion.strip()):
                limpio = _nombre_sesion_seguro(nombre_sesion)
                if not limpio:
                    st.error("Ese nombre no es válido.")
                elif guardar_sesion_nombre(limpio):
                    st.session_state["_sesion_msg"] = f"Sesión '{limpio}' guardada."
                    st.rerun()
                else:
                    st.error("No se pudo guardar la sesión.")
        if sesiones_guardadas:
            sel_sesion = st.selectbox("Sesión guardada", sesiones_guardadas, key="sel_sesion_cargar")
            c_load, c_del = st.columns(2)
            if c_load.button("Cargar", key="btn_cargar_sesion", use_container_width=True):
                if cargar_sesion_nombre(sel_sesion):
                    st.session_state["_sesion_msg"] = f"Sesión '{sel_sesion}' cargada."
                    st.rerun()
                else:
                    st.error("No se pudo cargar la sesión (archivo dañado o ausente).")
            if c_del.button("Eliminar", key="btn_eliminar_sesion", use_container_width=True):
                eliminar_sesion_nombre(sel_sesion)
                st.rerun()
            st.caption("Cargar reemplaza tu trabajo actual; guárdalo antes si quieres conservarlo.")

    if st.session_state.df is not None:
        _section_label("Navegación")
        seccion = st.radio(
            "Seccion",
            [
                "① DataFrame",
                "② Conjunto",
                "③ Análisis exploratorio",
                "④ Preprocesamiento",
                "⑤ Visualización",
                "⑥ Modelos",
                "⑦ Evaluación",
                "⑧ Asistente IA",
            ],
            label_visibility="collapsed",
        )
    else:
        seccion = "inicio"

# --- MAIN AREA ---
if st.session_state.df is None:
    st.markdown(
        '<div class="daad-welcome">'
        '<div class="daad-welcome-title">DAAD</div>'
        '<div class="daad-welcome-accent"></div>'
        '<div class="daad-welcome-sub">'
        "Carga un conjunto de datos desde el panel lateral para comenzar el análisis."
        "</div>"
        '<div class="daad-source-grid">'
        '<div class="daad-source-card"><span class="s-icon">📄</span><span class="s-name">CSV / TSV</span><span class="s-fmt">.csv · .tsv · .txt</span></div>'
        '<div class="daad-source-card"><span class="s-icon">📊</span><span class="s-name">Excel</span><span class="s-fmt">.xlsx · .xls</span></div>'
        '<div class="daad-source-card"><span class="s-icon">🗄</span><span class="s-name">PostgreSQL</span><span class="s-fmt">host · puerto · db</span></div>'
        '<div class="daad-source-card"><span class="s-icon">{}</span><span class="s-name">JSON</span><span class="s-fmt">.json · anidado</span></div>'
        '<div class="daad-source-card"><span class="s-icon">🌐</span><span class="s-name">Web scraping</span><span class="s-fmt">wikipedia · tablas</span></div>'
        "</div>"
        '<span class="daad-hint">← abre el panel lateral para cargar datos</span>'
        "</div>",
        unsafe_allow_html=True,
    )

else:
    df = st.session_state.df

    if seccion == "① DataFrame":
        _page_header("DF", "DataFrame", "datos / vista")
        col1, col2, col3 = st.columns(3)
        col1.metric("Filas", df.shape[0])
        col2.metric("Columnas", df.shape[1])
        col3.metric("Valores nulos", int(df.isnull().sum().sum()))
        st.divider()
        mostrar_df(df)
        st.download_button(
            "Descargar CSV",
            _csv_bytes(df),
            file_name=_nombre_csv(st.session_state.df_activo),
            mime="text/csv",
            use_container_width=True,
        )

    elif seccion == "② Conjunto":
        _page_header("CNJ", "Análisis en conjunto", "datos / múltiples")
        nombres_ds = list(st.session_state.datasets.keys())
        if len(nombres_ds) < 2:
            st.info(
                "Necesitas al menos dos datasets cargados para combinarlos. "
                "Sube otro desde el panel lateral y vuelve aquí."
            )
        else:
            tab_comb, tab_comp = st.tabs(["Combinar", "Comparar"])

            with tab_comb:
                modo = st.radio(
                    "Modo de combinación",
                    ["Concatenar (apilar filas)", "Merge (cruzar por columna llave)"],
                    horizontal=True,
                    help=(
                        "Concatenar: apila datasets con columnas similares uno debajo de otro. "
                        "Merge: cruza dos datasets por una columna en común (como un JOIN de SQL)."
                    ),
                )

                if modo.startswith("Concatenar"):
                    sel = st.multiselect(
                        "Datasets a apilar (en orden)",
                        nombres_ds,
                        default=nombres_ds[:2],
                        help="Las columnas que no existan en todos se rellenan con nulos.",
                    )
                    marcar = st.checkbox("Añadir columna que identifique el origen", value=True)
                    col_origen = st.text_input("Nombre de la columna de origen", value="origen") if marcar else ""

                    if len(sel) >= 2:
                        frames = []
                        for nombre in sel:
                            parte = st.session_state.datasets[nombre].copy()
                            if marcar and col_origen:
                                parte[col_origen] = nombre
                            frames.append(parte)
                        resultado = pd.concat(frames, ignore_index=True)

                        conjuntos = [set(st.session_state.datasets[n].columns) for n in sel]
                        no_comunes = set.union(*conjuntos) - set.intersection(*conjuntos)
                        if no_comunes:
                            st.warning(
                                "Estas columnas no están en todos los datasets y quedarán con nulos "
                                "donde falten: " + ", ".join(sorted(no_comunes))
                            )

                        st.caption(f"Resultado: {resultado.shape[0]} filas × {resultado.shape[1]} columnas (vista previa de 50 filas)")
                        mostrar_df(resultado.head(50))
                        nombre_nuevo = st.text_input("Nombre del dataset combinado", value="+".join(sel))
                        if st.button("Crear dataset combinado", use_container_width=True, key="btn_concat"):
                            _crear_dataset_combinado(nombre_nuevo, resultado)
                    else:
                        st.info("Selecciona al menos dos datasets para apilar.")

                else:  # Merge
                    c1, c2 = st.columns(2)
                    izq = c1.selectbox("Dataset izquierdo", nombres_ds, key="merge_izq")
                    der = c2.selectbox(
                        "Dataset derecho",
                        [n for n in nombres_ds if n != izq],
                        key="merge_der",
                    )
                    df_izq = st.session_state.datasets[izq]
                    df_der = st.session_state.datasets[der]
                    comunes = [c for c in df_izq.columns if c in df_der.columns]

                    if not comunes:
                        st.warning(
                            "Estos datasets no tienen columnas en común para cruzarlos. "
                            "Usa Concatenar, o renombra una columna en ④ Preprocesamiento para que coincida."
                        )
                    else:
                        llaves = st.multiselect(
                            "Columna(s) llave (en común)",
                            comunes,
                            default=comunes[:1],
                            help="Las filas se emparejan cuando coinciden estos valores.",
                        )
                        tipo_join = st.selectbox(
                            "Tipo de cruce",
                            ["inner", "left", "right", "outer"],
                            help=(
                                "inner: solo filas que coinciden en ambos · "
                                "left/right: conserva todas las filas del lado izquierdo/derecho · "
                                "outer: todas las filas de ambos"
                            ),
                        )
                        if llaves:
                            try:
                                resultado = pd.merge(
                                    df_izq, df_der, on=llaves, how=tipo_join,  # type: ignore[arg-type]
                                    suffixes=(f"_{izq}", f"_{der}"),
                                )
                                st.caption(f"Resultado: {resultado.shape[0]} filas × {resultado.shape[1]} columnas (vista previa de 50 filas)")
                                mostrar_df(resultado.head(50))
                                nombre_nuevo = st.text_input("Nombre del dataset combinado", value=f"{izq}+{der}")
                                if st.button("Crear dataset combinado", use_container_width=True, key="btn_merge"):
                                    _crear_dataset_combinado(nombre_nuevo, resultado)
                            except Exception as e:
                                st.error(f"No se pudo cruzar: {e}")
                        else:
                            st.info("Selecciona al menos una columna llave.")

            with tab_comp:
                sel_c = st.multiselect(
                    "Datasets a comparar",
                    nombres_ds,
                    default=nombres_ds[:2],
                    key="comp_sel",
                )
                if len(sel_c) < 2:
                    st.info("Selecciona al menos dos datasets para compararlos.")
                else:
                    items = tuple((n, st.session_state.datasets[n]) for n in sel_c)

                    # Resumen general: una fila por dataset
                    _subheader("Resumen general")
                    mostrar_df(_comp_resumen(items))

                    # Comparación columna a columna sobre las columnas en común
                    comunes_c = sorted(set.intersection(*[set(d.columns) for _, d in items]))
                    st.divider()
                    _subheader("Comparar una columna")
                    if not comunes_c:
                        st.info("Los datasets seleccionados no comparten ninguna columna con el mismo nombre.")
                    else:
                        col_cmp = str(st.selectbox("Columna en común", comunes_c, key="comp_col"))
                        es_num = all(pd.api.types.is_numeric_dtype(d[col_cmp]) for _, d in items)

                        if es_num:
                            mostrar_df(_comp_stats(items, col_cmp))
                            st.plotly_chart(
                                Visualizacion(_comp_largo(items, col_cmp)).histograma_comparado(col_cmp, "origen"),  # type: ignore[arg-type]
                                use_container_width=True,
                            )
                        else:
                            st.caption("Columna no numérica: se comparan las frecuencias (% por dataset, top 20 categorías).")
                            mostrar_df(_comp_props(items, col_cmp))

    elif seccion == "③ Análisis exploratorio":
        _page_header("EDA", "Análisis exploratorio", "datos / análisis")
        eda = AnalisisExploratorio(df)

        opcion = st.selectbox("¿Qué deseas analizar?", [
            "Resumen general",
            "Estadísticas descriptivas",
            "Conteo de nulos",
            "Valores únicos por columna",
            "Distribución de una columna",
            "Matriz de correlación",
            "Detección de outliers",
        ])

        st.divider()

        if opcion == "Resumen general":
            resumen = _resumen_general(df)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Filas", resumen["filas"])
            col2.metric("Columnas", resumen["columnas"])
            col3.metric("Valores nulos", resumen["nulos_totales"])
            col4.metric("Duplicados", resumen["duplicados"])
            _subheader("Tipos de datos por columna")
            tipos_df = pd.DataFrame(resumen["tipos"].items(), columns=["Columna", "Tipo"])
            mostrar_df(tipos_df)

        elif opcion == "Estadísticas descriptivas":
            mostrar_df(_estadisticas_descriptivas(df))

        elif opcion == "Conteo de nulos":
            nulos = eda.conteo_nulos()
            nulos_df = pd.DataFrame(nulos.items(), columns=["Columna", "Nulos"])
            nulos_df = nulos_df.sort_values("Nulos", ascending=False)
            mostrar_df(nulos_df)

        elif opcion == "Valores únicos por columna":
            columna = st.selectbox("Selecciona una columna", df.columns)
            resultado = eda.valores_unicos(columna)
            st.metric("Total de valores únicos", resultado["total"])
            st.write("Valores:", resultado["valores"])

        elif opcion == "Distribución de una columna":
            columna = st.selectbox("Selecciona una columna", df.columns)
            dist = eda.distribucion_columna(columna)
            dist_df = pd.DataFrame(dist.items(), columns=[columna, "Frecuencia"])
            dist_df = dist_df.sort_values("Frecuencia", ascending=False)
            mostrar_df(dist_df)

        elif opcion == "Matriz de correlación":
            corr = _matriz_correlacion(df)
            if corr.empty:
                st.warning("No hay columnas numéricas para calcular correlación.")
            else:
                st.dataframe(corr.style.background_gradient(cmap="coolwarm"), width="stretch")

        elif opcion == "Detección de outliers":
            numericas = df.select_dtypes(include="number").columns.tolist()
            if not numericas:
                st.warning("No hay columnas numéricas en el dataset.")
            else:
                columna = st.selectbox("Selecciona una columna numérica", numericas)
                outliers = eda.detectar_outliers(columna)
                st.metric("Outliers detectados", len(outliers))
                if not outliers.empty:
                    mostrar_df(outliers)

    elif seccion == "④ Preprocesamiento":
        _page_header("PRE", "Preprocesamiento", "datos / transformación")

        # Guardar original la primera vez que se entra
        if st.session_state.df_original is None:
            st.session_state.df_original = df.copy()

        pre = Preprocesamiento(df)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Filas", df.shape[0])
        col2.metric("Columnas", df.shape[1])
        col3.metric("Nulos", int(df.isnull().sum().sum()))
        col4.metric("Duplicados", _contar_duplicados(df))
        st.divider()

        operaciones = ["Seleccionar columnas"]
        # Solo ofrecemos limpiar columnas vacías si existe alguna candidata
        # (≥ 50 % de nulos, el umbral mínimo que permite la operación).
        if _hay_columnas_vacias(df):
            operaciones.append("Eliminar columnas vacías")
        operaciones += [
            "Eliminar nulos",
            "Rellenar nulos",
            "Eliminar duplicados",
            "Filtrar filas",
            "Convertir tipo de columna",
            "Normalizar — Standard",
            "Normalizar — MinMax",
            "Codificar categóricas",
        ]
        operacion = st.selectbox("Operación", operaciones)
        st.divider()

        if operacion == "Seleccionar columnas":
            cols_sel = st.multiselect(
                "Columnas a conservar",
                df.columns.tolist(),
                default=df.columns.tolist(),
            )
            if st.button("Aplicar", key="btn_sel_cols", use_container_width=True):
                if cols_sel:
                    if _aplicar_transformacion(
                        f"Seleccionar columnas ({len(cols_sel)} conservadas)",
                        pre.seleccionar_columnas(cols_sel),
                    ):
                        st.success(f"Columnas reducidas a {len(cols_sel)}")
                        st.rerun()
                else:
                    st.warning("Selecciona al menos una columna.")

        elif operacion == "Eliminar columnas vacías":
            st.caption(
                "Elimina columnas completas según su porcentaje de valores nulos. "
                "Útil para descartar las columnas vacías que a veces deja el web scraping."
            )
            umbral_pct = st.slider(
                "Umbral de nulos (%)",
                min_value=50, max_value=100, value=100, step=5,
                help="Se eliminarán las columnas cuyo porcentaje de valores nulos sea mayor o igual a este umbral. "
                     "100 % = solo columnas completamente vacías.",
            )
            umbral = umbral_pct / 100
            cols_vacias = pre.columnas_vacias(umbral)
            if cols_vacias:
                st.warning(
                    f"{len(cols_vacias)} columna(s) con ≥ {umbral_pct} % de nulos: "
                    + ", ".join(f"`{c}`" for c in cols_vacias)
                )
            else:
                st.info(f"No hay columnas con ≥ {umbral_pct} % de valores nulos.")
            if st.button("Aplicar", key="btn_cols_vacias", use_container_width=True, disabled=not cols_vacias):
                resultado = pre.eliminar_columnas_vacias(umbral)
                if _aplicar_transformacion(
                    f"Eliminar columnas vacías (≥ {umbral_pct} % nulos, {len(cols_vacias)})",
                    resultado,
                ):
                    st.success(f"Columnas: {df.shape[1]} → {resultado.shape[1]}")
                    st.rerun()

        elif operacion == "Eliminar nulos":
            nulos = int(df.isnull().sum().sum())
            st.info(f"{nulos} valores nulos encontrados — se eliminarán todas las filas que contengan al menos uno.")
            if st.button("Aplicar", key="btn_elim_nulos", use_container_width=True):
                resultado = pre.eliminar_nulos()
                if _aplicar_transformacion("Eliminar filas con nulos", resultado):
                    st.success(f"Filas: {df.shape[0]} → {resultado.shape[0]}")
                    st.rerun()

        elif operacion == "Rellenar nulos":
            cols_con_nulos = [c for c in df.columns if df[c].isnull().any()]
            total_nulos = int(df.isnull().sum().sum())
            if total_nulos == 0:
                st.info("No hay valores nulos en el dataset.")
            else:
                st.info(f"{total_nulos} valores nulos en {len(cols_con_nulos)} columna(s).")
                estrategia = st.radio(
                    "Estrategia",
                    ["media", "mediana", "moda", "constante"],
                    horizontal=True,
                )

                # media/mediana solo aplican a numéricas; su default excluye binarias
                # (rellenar un 0/1 con la media lo convierte en una fracción sin sentido).
                if estrategia in ("media", "mediana"):
                    aplicables = [c for c in cols_con_nulos if pd.api.types.is_numeric_dtype(df[c])]
                    _, no_binarias = _columnas_escalables(df)
                    default_cols = [c for c in aplicables if c in no_binarias]
                    cats_con_nulos = [c for c in cols_con_nulos if c not in aplicables]
                    if cats_con_nulos:
                        st.warning(
                            f"'{estrategia}' solo rellena columnas numéricas; estas categóricas no "
                            f"aparecen abajo y quedarán con nulos: {', '.join(cats_con_nulos)}. "
                            "Usa 'moda' o 'constante' para ellas."
                        )
                else:
                    aplicables = cols_con_nulos
                    default_cols = cols_con_nulos

                if not aplicables:
                    st.warning("Ninguna columna con nulos es compatible con esta estrategia.")
                else:
                    cols_fill = st.multiselect(
                        "Columnas a rellenar",
                        aplicables,
                        default=default_cols,
                        help="Para columnas binarias (0/1) usa moda o constante: media/mediana las "
                             "convierten en fracciones. Quita el target o identificadores si no quieres imputarlos.",
                    )
                    excluidas = [c for c in aplicables if c not in default_cols]
                    if estrategia in ("media", "mediana") and excluidas:
                        st.caption(
                            "Excluidas del default por parecer binarias (0/1): "
                            + ", ".join(f"`{c}`" for c in excluidas)
                            + ". Para esas conviene moda o constante."
                        )

                    # El valor constante se pide según el tipo de las columnas elegidas:
                    # número si son numéricas, texto si son categóricas. Si se mezclan
                    # tipos no hay un único valor que encaje, así que se bloquea.
                    valor_cte = 0.0
                    bloquear = False
                    if estrategia == "constante" and cols_fill:
                        num_sel = [c for c in cols_fill if pd.api.types.is_numeric_dtype(df[c])]
                        cat_sel = [c for c in cols_fill if c not in num_sel]
                        if num_sel and cat_sel:
                            st.warning(
                                "Mezclaste columnas numéricas y de texto. Para rellenar con una constante, "
                                "selecciona solo columnas del mismo tipo (todas numéricas o todas de texto): "
                                "así el valor encaja sin cambiarles el tipo."
                            )
                            bloquear = True
                        elif cat_sel:  # todas de texto/categóricas
                            valor_cte = st.text_input("Valor constante (texto)", value="Sin dato")
                        else:  # todas numéricas
                            valor_cte = st.number_input("Valor constante", value=0.0)

                    if st.button("Aplicar", key="btn_rel_nulos", use_container_width=True,
                                 disabled=not cols_fill or bloquear):
                        resultado = pre.rellenar_nulos(estrategia, valor_cte, cols_fill)
                        if _aplicar_transformacion(f"Rellenar nulos ({estrategia}, {len(cols_fill)} columnas)", resultado):
                            st.success(f"Nulos rellenados. Restantes: {int(resultado.isnull().sum().sum())}")
                            st.rerun()

        elif operacion == "Eliminar duplicados":
            dups = _contar_duplicados(df)
            st.info(f"{dups} filas duplicadas encontradas.")
            if st.button("Aplicar", key="btn_elim_dups", use_container_width=True):
                resultado = pre.eliminar_duplicados()
                if _aplicar_transformacion("Eliminar duplicados", resultado):
                    st.success(f"Filas: {df.shape[0]} → {resultado.shape[0]}")
                    st.rerun()

        elif operacion == "Convertir tipo de columna":
            col_cast = st.selectbox("Columna", df.columns.tolist(), key="cast_col")
            tipo_actual = str(df[col_cast].dtype)
            st.caption(f"Tipo actual: `{tipo_actual}`")
            dtype_sel = st.selectbox("Convertir a", ["float64", "int64", "str", "datetime"], key="cast_dtype")
            if st.button("Aplicar", key="btn_cast", use_container_width=True):
                try:
                    resultado = pre.convertir_tipo(col_cast, dtype_sel)
                    if _aplicar_transformacion(f"Convertir '{col_cast}' a {dtype_sel}", resultado):
                        st.success(f"'{col_cast}' convertida a {dtype_sel}.")
                        st.rerun()
                except ValueError as e:
                    st.error(str(e))

        elif operacion == "Filtrar filas":
            col_filtro = st.selectbox("Columna", df.columns.tolist())
            operador = st.selectbox("Operador", ["==", "!=", ">", "<", ">=", "<="])
            valores_unicos = sorted(df[col_filtro].dropna().astype(str).unique().tolist())
            if len(valores_unicos) <= 200:
                valor_str = st.selectbox("Valor", valores_unicos)
            else:
                valor_str = st.text_input("Valor")
            if st.button("Aplicar", key="btn_filtrar", use_container_width=True):
                try:
                    vs = valor_str or ""
                    valor = float(vs) if pd.api.types.is_numeric_dtype(df[col_filtro]) else vs
                    resultado = pre.filtrar_filas(col_filtro, valor, operador)
                    if _aplicar_transformacion(f"Filtrar: {col_filtro} {operador} {vs}", resultado):
                        st.success(f"Filas: {df.shape[0]} → {resultado.shape[0]}")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error al filtrar: {e}")

        elif operacion in ("Normalizar — Standard", "Normalizar — MinMax"):
            es_standard = operacion == "Normalizar — Standard"
            numericas, recomendadas = _columnas_escalables(df)
            if not numericas:
                st.warning("No hay columnas numéricas.")
            else:
                cols_scale = st.multiselect(
                    "Columnas a escalar",
                    numericas,
                    default=recomendadas,
                    help="Por defecto solo se marcan las numéricas continuas. NO escales columnas "
                         "binarias o dummies (0/1), identificadores (id, año) ni tu target: "
                         "perderían su significado.",
                )
                excluidas = [c for c in numericas if c not in recomendadas]
                if excluidas:
                    st.caption(
                        "Se excluyeron del default por parecer binarias/0-1: "
                        + ", ".join(f"`{c}`" for c in excluidas)
                        + ". Quita también identificadores o el target si los ves marcados."
                    )
                rango = "media = 0, desviación = 1" if es_standard else "rango [0, 1]"
                st.info(f"{len(cols_scale)} columna(s) → {rango}.")
                st.caption(
                    "Nota: esto escala el dataset actual. Si vas a entrenar un modelo, usa mejor "
                    "la opción de escalado en ⑥ Modelos para evitar fuga de datos."
                )
                if st.button("Aplicar", key="btn_scale", use_container_width=True, disabled=not cols_scale):
                    resultado = (
                        pre.normalizar_standard(cols_scale) if es_standard
                        else pre.normalizar_minmax(cols_scale)
                    )
                    metodo = "Standard" if es_standard else "MinMax"
                    if _aplicar_transformacion(f"Normalizar {metodo} ({len(cols_scale)} columnas)", resultado):
                        st.success(f"Normalización {metodo} aplicada a {len(cols_scale)} columna(s).")
                        st.rerun()

        elif operacion == "Codificar categóricas":
            cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
            if not cats:
                st.warning("No hay columnas categóricas.")
            else:
                metodo = st.radio(
                    "Método",
                    ["One-Hot (columnas binarias)", "LabelEncoder (ordinal)"],
                    horizontal=True,
                    help="One-Hot no impone orden entre categorías (recomendado para nominales); "
                         "LabelEncoder asigna enteros 0..n y solo es adecuado si hay un orden real.",
                )
                cols_cod = st.multiselect("Columnas a codificar", cats, default=cats)
                es_onehot = metodo.startswith("One-Hot")
                if es_onehot:
                    altas = [c for c in cols_cod if df[c].nunique() > 30]
                    if altas:
                        st.warning(f"Columnas con más de 30 categorías: {altas} — One-Hot generará muchas columnas.")
                cols_con_nulos = [c for c in cols_cod if df[c].isnull().any()]
                if cols_con_nulos:
                    st.warning(
                        "Estas columnas tienen nulos: " + ", ".join(cols_con_nulos) + ". "
                        "Al codificar, los nulos se conservan como NaN (no se vuelven una categoría). "
                        "Trátalos en 'Eliminar nulos' o 'Rellenar nulos' antes de entrenar un modelo."
                    )
                if st.button("Aplicar", key="btn_enc", use_container_width=True, disabled=not cols_cod):
                    if es_onehot:
                        resultado = pre.codificar_onehot(cols_cod)
                        cambiado = _aplicar_transformacion(f"One-Hot ({len(cols_cod)} columnas)", resultado)
                        if cambiado:
                            # Recordar categoría→columna dummy para la predicción interactiva
                            st.session_state.onehot_mappings.update(pre.mapeos_onehot)
                            st.success(f"One-Hot aplicado. Columnas: {df.shape[1]} → {resultado.shape[1]}")
                    else:
                        cambiado = _aplicar_transformacion(
                            f"LabelEncoder ({len(cols_cod)} columnas)",
                            pre.codificar_categoricas(cols_cod),
                        )
                        if cambiado:
                            # Recordar categoría→código para la predicción interactiva
                            st.session_state.label_mappings.update(pre.mapeos_label)
                            st.success("Columnas codificadas con LabelEncoder.")
                    if cambiado:
                        st.rerun()

        st.divider()
        _subheader("Vista previa")
        mostrar_df(df)

        st.divider()
        historial = st.session_state.historiales.get(_clave_historial(), [])
        _subheader("Historial de transformaciones")
        if historial:
            for i, paso in enumerate(historial, 1):
                filas, cols = paso["df_antes"].shape
                st.caption(f"{i}. {paso['descripcion']}  ·  antes: {filas} × {cols}")
            if len(historial) == MAX_HISTORIAL:
                st.caption(f"(se conservan solo los últimos {MAX_HISTORIAL} pasos)")
        else:
            st.caption("Sin transformaciones en esta sesión.")

        c_undo, c_rest = st.columns(2)
        if c_undo.button(
            "Deshacer último paso",
            key="btn_undo",
            use_container_width=True,
            disabled=not historial,
        ):
            paso = historial.pop()
            _guardar_df(paso["df_antes"])
            _sincronizar_mapeos_codificacion()
            st.rerun()
        if c_rest.button(
            "Restaurar datos originales",
            key="btn_restaurar",
            use_container_width=True,
            disabled=st.session_state.df_original is None,
        ):
            _guardar_df(st.session_state.df_original.copy())
            st.session_state.historiales[_clave_historial()] = []
            _sincronizar_mapeos_codificacion()
            st.rerun()

    elif seccion == "⑤ Visualización":
        _page_header("VIZ", "Visualización", "datos / gráficas")
        viz = Visualizacion(df)
        numericas = df.select_dtypes(include="number").columns.tolist()
        todas = df.columns.tolist()

        tipo = st.selectbox("Tipo de gráfica", [
            "Histograma",
            "Boxplot",
            "Dispersión",
            "Barras",
            "Líneas",
            "Pastel",
            "Mapa de calor (correlación)",
        ])
        st.divider()

        if tipo == "Histograma":
            if not numericas:
                st.warning("No hay columnas numéricas.")
            else:
                col = st.selectbox("Columna", numericas)
                st.plotly_chart(viz.histograma(col), use_container_width=True)

        elif tipo == "Boxplot":
            if not numericas:
                st.warning("No hay columnas numéricas.")
            else:
                col = st.selectbox("Columna", numericas)
                st.plotly_chart(viz.boxplot(col), use_container_width=True)

        elif tipo == "Dispersión":
            if len(numericas) < 2:
                st.warning("Se necesitan al menos 2 columnas numéricas.")
            else:
                c1, c2, c3 = st.columns(3)
                col_x = c1.selectbox("Eje X", numericas, key="sc_x")
                col_y = c2.selectbox("Eje Y", numericas, index=1, key="sc_y")
                cats = [None] + [c for c in todas if c not in numericas]
                color = c3.selectbox("Color (opcional)", cats, key="sc_c")
                st.plotly_chart(viz.scatter(col_x, col_y, color), use_container_width=True)

        elif tipo == "Barras":
            c1, c2 = st.columns(2)
            col_x = c1.selectbox("Eje X (categoría)", todas, key="bar_x")
            usar_y = c2.checkbox("Usar columna numérica como Y", key="bar_use_y")
            col_y = None
            if usar_y and numericas:
                col_y = c2.selectbox("Eje Y", numericas, key="bar_y")
            st.plotly_chart(viz.grafica_barras(col_x, col_y), use_container_width=True)

        elif tipo == "Líneas":
            if len(todas) < 2:
                st.warning("Se necesitan al menos 2 columnas.")
            else:
                c1, c2 = st.columns(2)
                col_x = c1.selectbox("Eje X", todas, key="ln_x")
                col_y = c2.selectbox("Eje Y", numericas if numericas else todas, key="ln_y")
                st.plotly_chart(viz.grafica_lineas(col_x, col_y), use_container_width=True)

        elif tipo == "Pastel":
            col = st.selectbox("Columna", todas)
            st.plotly_chart(viz.grafica_pastel(col), use_container_width=True)

        elif tipo == "Mapa de calor (correlación)":
            if len(numericas) < 2:
                st.warning("Se necesitan al menos 2 columnas numéricas para la correlación.")
            else:
                st.plotly_chart(viz.heatmap_correlacion(), use_container_width=True)

    elif seccion == "⑥ Modelos":
        _page_header("MOD", "Modelos", "datos / machine learning")
        viz = Visualizacion(df)
        numericas = df.select_dtypes(include="number").columns.tolist()

        nulos = int(df.isnull().sum().sum())
        if nulos > 0:
            st.error(f"{nulos} valores nulos detectados. Ve a ④ Preprocesamiento y elimínalos o rellénalos primero.")
            st.stop()
        if not numericas:
            st.error("No hay columnas numéricas. Convierte o codifica tus columnas en ④ Preprocesamiento primero.")
            st.stop()

        tipo_modelo = st.selectbox("Tipo de modelo", [
            # Clasificación (predicen una categoría)
            "KNN — K-Nearest Neighbors",
            "Árbol de Decisión",
            "Random Forest",
            "Gradient Boosting",
            "Regresión Logística",
            "Naive Bayes",
            "Red Neuronal",
            # Regresión (predicen un número)
            "Regresión Lineal",
            "KNN (regresión)",
            "Árbol de Decisión (regresión)",
            "Random Forest (regresión)",
            "Gradient Boosting (regresión)",
            "Red Neuronal (regresión)",
            # Clustering (agrupan sin target)
            "K-Means (clustering)",
        ])
        st.divider()

        if tipo_modelo == "K-Means (clustering)":
            features = st.multiselect("Columnas (features)", numericas, default=numericas[:min(2, len(numericas))])
            n_clusters = int(st.number_input("Número de clusters (k)", min_value=2, max_value=20, value=3, step=1))
            escalar_km = st.checkbox(
                "Escalar features (StandardScaler)",
                value=True,
                help="K-Means usa distancias: sin escalar, las columnas de mayor magnitud "
                     "dominan los clusters. Recomendado salvo que las features ya sean comparables.",
            )

            mostrar_codo = st.checkbox("Mostrar curva del codo")
            if mostrar_codo and features:
                k_max = int(st.slider("k máximo", 2, 15, 10))
                tmp = ModeloKMeans(df)
                tmp.features = features
                with st.spinner("Calculando codo..."):
                    inercias = tmp.calcular_codo(k_max, escalar=escalar_km)
                st.plotly_chart(viz.grafica_codo(inercias), use_container_width=True)

            if st.button("Entrenar K-Means", use_container_width=True, disabled=not features):
                modelo = ModeloKMeans(df, n_clusters=n_clusters)
                modelo.features = features
                try:
                    with st.spinner("Entrenando..."):
                        modelo.entrenar(escalar=escalar_km)
                    st.session_state.modelo_entrenado = modelo
                    st.session_state.modelo_tipo = "clustering"
                    st.session_state.metricas = modelo.evaluar()
                    guardar_sesion()
                    st.success("Modelo entrenado.")
                except Exception as e:
                    st.error(f"Error al entrenar K-Means: {e}")

            # Resultados persistentes: sobreviven a los reruns de Streamlit
            if (
                st.session_state.modelo_tipo == "clustering"
                and st.session_state.modelo_entrenado is not None
                and st.session_state.metricas is not None
            ):
                modelo_e = st.session_state.modelo_entrenado
                m = st.session_state.metricas
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Clusters (k)", m["n_clusters"])
                c2.metric("Inercia", f"{m['inercia']:.2f}")
                c3.metric("Silhouette", f"{m['silhouette']:.4f}")
                datos_clusters = modelo_e.datos if isinstance(modelo_e.datos, pd.DataFrame) else pd.DataFrame(modelo_e.datos)
                if (
                    modelo_e.etiquetas is not None
                    and modelo_e.centroides is not None
                    and all(f in datos_clusters.columns for f in modelo_e.features)
                    and len(modelo_e.etiquetas) == datos_clusters.shape[0]
                ):
                    st.plotly_chart(
                        viz.grafica_clusters(datos_clusters[list(modelo_e.features)].copy(), modelo_e.etiquetas, modelo_e.centroides),
                        use_container_width=True,
                    )

        else:
            features = st.multiselect("Features (columnas X)", numericas)
            target_opts = [c for c in df.columns if c not in features]
            target = st.selectbox("Target (columna y)", target_opts if target_opts else df.columns.tolist())
            test_size = st.slider("Proporción de prueba", 0.1, 0.5, 0.2, 0.05)
            escalar = st.checkbox(
                "Escalar features (StandardScaler ajustado solo con train)",
                value=True,
                help="Evita fuga de datos: el scaler se ajusta con el conjunto de entrenamiento y solo transforma el de prueba.",
            )
            c_cv1, c_cv2 = st.columns([2, 1])
            hacer_cv = c_cv1.checkbox(
                "Validación cruzada (k-fold)",
                help="Entrena k veces sobre particiones distintas y reporta la métrica media ± desviación. "
                     "Si el escalado está activo, el scaler se ajusta dentro de cada fold.",
            )
            n_folds = int(c_cv2.number_input("Folds", min_value=3, max_value=10, value=5, step=1, disabled=not hacer_cv))

            tuning = st.checkbox(
                "Buscar mejores parámetros (GridSearchCV)",
                help="Prueba varias combinaciones de hiperparámetros por validación cruzada sobre el "
                     "entrenamiento y se queda con la mejor. Ignora los valores manuales de abajo y "
                     "puede tardar más.",
            )

            if tuning:
                st.caption("Tuning activado: los hiperparámetros marcados abajo se ignoran y se eligen por búsqueda automática.")

            hiperparams: dict = {}
            if tipo_modelo in ("KNN — K-Nearest Neighbors", "KNN (regresión)"):
                hiperparams["k"] = int(st.number_input("k (vecinos)", min_value=1, max_value=50, value=5, step=1))
                cw1, cw2 = st.columns(2)
                hiperparams["weights"] = cw1.selectbox("Peso de vecinos", ["uniform", "distance"],
                    help="'distance': los vecinos más cercanos pesan más (suele mejorar).")
                hiperparams["metric"] = cw2.selectbox("Métrica de distancia", ["minkowski", "manhattan"])
            elif tipo_modelo in ("Árbol de Decisión", "Árbol de Decisión (regresión)"):
                prof = int(st.number_input("Profundidad máxima (0 = sin límite)", min_value=0, max_value=50, value=0, step=1))
                hiperparams["profundidad_max"] = None if prof == 0 else prof
                ca1, ca2 = st.columns(2)
                hiperparams["min_samples_leaf"] = int(ca1.number_input("Mín. muestras por hoja", min_value=1, max_value=100, value=1, step=1,
                    help="Subirlo evita hojas con muy pocos casos (combate el sobreajuste)."))
                # El criterio gini/entropy es exclusivo de clasificación
                if tipo_modelo == "Árbol de Decisión":
                    hiperparams["criterion"] = ca2.selectbox("Criterio", ["gini", "entropy"])
            elif tipo_modelo in ("Random Forest", "Random Forest (regresión)"):
                cr1, cr2 = st.columns(2)
                hiperparams["n_estimators"] = int(cr1.number_input("Número de árboles", min_value=10, max_value=1000, value=100, step=10))
                profrf = int(cr2.number_input("Profundidad máxima (0 = sin límite)", min_value=0, max_value=50, value=0, step=1))
                hiperparams["profundidad_max"] = None if profrf == 0 else profrf
            elif tipo_modelo in ("Gradient Boosting", "Gradient Boosting (regresión)"):
                cg1, cg2, cg3 = st.columns(3)
                hiperparams["n_estimators"] = int(cg1.number_input("Número de árboles", min_value=10, max_value=1000, value=100, step=10))
                hiperparams["learning_rate"] = float(cg2.number_input("Tasa de aprendizaje", min_value=0.01, max_value=1.0, value=0.1, step=0.01))
                hiperparams["profundidad_max"] = int(cg3.number_input("Profundidad máxima", min_value=1, max_value=20, value=3, step=1))
            elif tipo_modelo == "Regresión Logística":
                hiperparams["max_iter"] = int(st.number_input("Max iteraciones", min_value=100, max_value=2000, value=200, step=100))
                cl1, cl2 = st.columns(2)
                hiperparams["C"] = float(cl1.number_input("C (regularización inversa)", min_value=0.001, max_value=1000.0, value=1.0, step=0.1,
                    help="Menor C = más regularización (modelo más simple)."))
                hiperparams["class_weight"] = cl2.selectbox("Peso de clases", [None, "balanced"],
                    format_func=lambda x: "ninguno" if x is None else x,
                    help="'balanced' compensa clases desbalanceadas.")
            elif tipo_modelo == "Regresión Lineal":
                reg = st.selectbox("Regularización", ["ninguna", "ridge", "lasso"],
                    help="Ridge/Lasso penalizan coeficientes grandes; Lasso además puede llevarlos a 0.")
                hiperparams["regularizacion"] = reg
                if reg in ("ridge", "lasso"):
                    hiperparams["alpha"] = float(st.number_input("alpha (fuerza de regularización)", min_value=0.0001, max_value=1000.0, value=1.0, step=0.1))
            elif tipo_modelo in ("Red Neuronal", "Red Neuronal (regresión)"):
                arquitecturas = {
                    "Pequeña — 1 capa de 50 neuronas": (50,),
                    "Mediana — 2 capas (100, 50)": (100, 50),
                    "Grande — 3 capas (100, 100, 50)": (100, 100, 50),
                }
                arq = st.selectbox("Tamaño de la red (capas ocultas)", list(arquitecturas.keys()),
                    help="Más capas/neuronas = más capacidad, pero necesita más datos y más iteraciones.")
                hiperparams["hidden_layer_sizes"] = arquitecturas[arq]
                hiperparams["max_iter"] = int(st.number_input("Máximo de iteraciones", min_value=100, max_value=2000, value=300, step=100))
                st.caption("La red neuronal necesita features escaladas: deja activado «Escalar features».")
            # Naive Bayes no tiene hiperparámetros que ajustar.

            invalidas = [f for f in features if f not in numericas]
            if invalidas:
                st.warning(f"Features no numéricas: {invalidas}. Codifícalas primero.")

            es_regresion = TAREA_MODELO.get(tipo_modelo) == "regresion"
            error_target = None
            if target:
                serie_target = df[target]
                if es_regresion and not pd.api.types.is_numeric_dtype(serie_target):
                    error_target = (
                        f"'{target}' no es numérica. Los modelos de regresión necesitan un target "
                        "numérico; conviértela o codifícala en ④ Preprocesamiento."
                    )
                elif not es_regresion and pd.api.types.is_float_dtype(serie_target) and serie_target.nunique() > 20:
                    error_target = (
                        f"'{target}' parece continua ({serie_target.nunique()} valores únicos). "
                        "Los clasificadores necesitan clases discretas; usa un modelo de regresión o discretiza el target."
                    )
            if error_target:
                st.error(error_target)

            puede_entrenar = bool(features) and target not in features and not invalidas and error_target is None

            if st.button("Entrenar modelo", use_container_width=True, disabled=not puede_entrenar):
                if tipo_modelo == "KNN — K-Nearest Neighbors":
                    modelo = ModeloKNN(df, **hiperparams)
                elif tipo_modelo == "KNN (regresión)":
                    modelo = ModeloKNNRegresion(df, **hiperparams)
                elif tipo_modelo == "Árbol de Decisión":
                    modelo = ModeloArbolDecision(df, **hiperparams)
                elif tipo_modelo == "Árbol de Decisión (regresión)":
                    modelo = ModeloArbolRegresion(df, **hiperparams)
                elif tipo_modelo == "Random Forest":
                    modelo = ModeloRandomForest(df, **hiperparams)
                elif tipo_modelo == "Random Forest (regresión)":
                    modelo = ModeloRandomForestRegresion(df, **hiperparams)
                elif tipo_modelo == "Gradient Boosting":
                    modelo = ModeloGradientBoosting(df, **hiperparams)
                elif tipo_modelo == "Gradient Boosting (regresión)":
                    modelo = ModeloGradientBoostingRegresion(df, **hiperparams)
                elif tipo_modelo == "Regresión Logística":
                    modelo = ModeloRegresionLogistica(df, **hiperparams)
                elif tipo_modelo == "Naive Bayes":
                    modelo = ModeloNaiveBayes(df, **hiperparams)
                elif tipo_modelo == "Red Neuronal":
                    modelo = ModeloRedNeuronal(df, **hiperparams)
                elif tipo_modelo == "Red Neuronal (regresión)":
                    modelo = ModeloRedNeuronalRegresion(df, **hiperparams)
                else:
                    modelo = ModeloRegresionLineal(df, **hiperparams)

                modelo.features = features
                modelo.target = target
                try:
                    with st.spinner("Buscando parámetros y entrenando..." if tuning else "Entrenando..."):
                        modelo.entrenar(test_size=test_size, escalar=escalar, tuning=tuning)
                    st.session_state.modelo_entrenado = modelo
                    st.session_state.modelo_tipo = "regresion" if es_regresion else "clasificacion"
                    st.session_state.metricas = modelo.evaluar()
                    st.session_state.cv_resultados = None
                    # La validación cruzada es un diagnóstico opcional: va en su propio
                    # try para que un fallo (p. ej. clases con menos muestras que folds)
                    # no descarte el modelo ya entrenado.
                    if hacer_cv:
                        try:
                            with st.spinner("Validación cruzada..."):
                                st.session_state.cv_resultados = modelo.validacion_cruzada(cv=n_folds, escalar=escalar)
                        except Exception as e:
                            st.warning(
                                f"El modelo se entrenó, pero no se pudo calcular la validación cruzada: {e} "
                                "(suele pasar cuando alguna clase tiene menos muestras que el número de folds; "
                                "reduce los folds o usa un target con clases más balanceadas)."
                            )
                    guardar_sesion()
                    st.success("Modelo entrenado.")
                    if getattr(modelo, "no_convergio", False):
                        st.warning(
                            "La red neuronal no terminó de converger en el máximo de iteraciones. "
                            "Sube el «Máximo de iteraciones», asegúrate de tener «Escalar features» "
                            "activado, o prueba Random Forest / Gradient Boosting con estos datos."
                        )
                    if modelo.mejores_hiperparametros:
                        st.info("Mejores parámetros encontrados: " + ", ".join(
                            f"{k} = {v}" for k, v in modelo.mejores_hiperparametros.items()
                        ))
                    elif getattr(modelo, "tuning_fallo", False):
                        st.warning(
                            "No se pudo completar la búsqueda de parámetros (suele ser por pocas "
                            "muestras o clases con menos casos que folds). Se entrenó con los valores por defecto."
                        )
                except Exception as e:
                    st.error(f"Error al entrenar el modelo: {e}")

            if not features:
                st.info("Selecciona al menos una feature para activar el entrenamiento.")

            # Resultados persistentes: sobreviven a los reruns de Streamlit
            if (
                st.session_state.modelo_tipo in ("clasificacion", "regresion")
                and st.session_state.modelo_entrenado is not None
                and st.session_state.metricas is not None
            ):
                modelo_e = st.session_state.modelo_entrenado
                metricas = st.session_state.metricas
                st.divider()

                if st.session_state.modelo_tipo == "clasificacion":
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy", f"{metricas['accuracy']:.3f}")
                    c2.metric("Precision", f"{metricas['precision']:.3f}")
                    c3.metric("Recall", f"{metricas['recall']:.3f}")
                    c4.metric("F1", f"{metricas['f1']:.3f}")
                    st.plotly_chart(viz.grafica_confusion(metricas["confusion_matrix"]), use_container_width=True)
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("R²", f"{metricas['r2']:.4f}")
                    c2.metric("MSE", f"{metricas['mse']:.4f}")
                    c3.metric("RMSE", f"{metricas['rmse']:.4f}")
                    if modelo_e.y_test is not None and modelo_e.y_pred is not None:
                        st.plotly_chart(viz.grafica_regresion(modelo_e.y_test, modelo_e.y_pred), use_container_width=True)

                cv_res = st.session_state.cv_resultados
                if cv_res:
                    st.divider()
                    _subheader("Validación cruzada")
                    st.metric(
                        f"{cv_res['metrica']} medio ({len(cv_res['scores'])} folds)",
                        f"{cv_res['media']:.3f} ± {cv_res['desviacion']:.3f}",
                    )
                    st.caption("Scores por fold: " + " · ".join(f"{s:.3f}" for s in cv_res["scores"]))
                    # Mucha dispersión entre folds = el rendimiento depende de qué
                    # datos toquen en cada partición → estimación poco fiable.
                    if cv_res["desviacion"] >= 0.10:
                        st.warning(
                            f"Alta variabilidad entre folds (± {cv_res['desviacion']:.3f}): el modelo es "
                            "inestable y su rendimiento real es incierto. Suele deberse a pocos datos o "
                            "clases desbalanceadas; toma la media con cautela."
                        )

    elif seccion == "⑦ Evaluación":
        _page_header("EVL", "Evaluación", "datos / métricas")

        if st.session_state.modelo_entrenado is None:
            st.info("Primero entrena un modelo en ⑥ Modelos.")
        else:
            modelo = st.session_state.modelo_entrenado
            tipo = st.session_state.modelo_tipo
            viz = Visualizacion(df)
            # Reusa las métricas calculadas al entrenar; evita recomputarlas en
            # cada render de esta sección (predicciones + métricas completas).
            metricas_eval = st.session_state.metricas if st.session_state.metricas is not None else modelo.evaluar()

            if tipo == "clustering":
                m = metricas_eval
                c1, c2, c3 = st.columns(3)
                c1.metric("Clusters (k)", m["n_clusters"])
                c2.metric("Inercia", f"{m['inercia']:.2f}")
                c3.metric("Silhouette", f"{m['silhouette']:.4f}")
                st.divider()
                _subheader("Visualización de clusters")
                # Graficar contra los datos con los que se entrenó (modelo.datos), no
                # contra el df activo: si este cambió en ④ Preprocesamiento (columnas
                # o filas), df[features] reventaría o desalinearía las etiquetas.
                datos_clusters = modelo.datos if isinstance(modelo.datos, pd.DataFrame) else pd.DataFrame(modelo.datos)
                if (
                    modelo.etiquetas is not None
                    and modelo.centroides is not None
                    and all(f in datos_clusters.columns for f in modelo.features)
                    and len(modelo.etiquetas) == datos_clusters.shape[0]
                ):
                    st.plotly_chart(
                        viz.grafica_clusters(datos_clusters[list(modelo.features)].copy(), modelo.etiquetas, modelo.centroides),
                        use_container_width=True,
                    )
                if modelo.inercias:
                    _subheader("Curva del codo")
                    st.plotly_chart(viz.grafica_codo(modelo.inercias), use_container_width=True)

            elif tipo == "regresion":
                if modelo.y_test is not None and modelo.y_pred is not None:
                    m = metricas_eval
                    c1, c2, c3 = st.columns(3)
                    c1.metric("R²", f"{m['r2']:.4f}")
                    c2.metric("R² ajustado", f"{m['r2_ajustado']:.4f}")
                    c3.metric("MAE", f"{m['mae']:.4f}")
                    c4, c5, c6 = st.columns(3)
                    c4.metric("MSE", f"{m['mse']:.4f}")
                    c5.metric("RMSE", f"{m['rmse']:.4f}")
                    # MAPE no se define si el target tiene ceros; se omite en ese caso.
                    c6.metric("MAPE", f"{m['mape']:.2f}%" if "mape" in m else "N/D",
                              help=None if "mape" in m else "No se calcula porque el target tiene valores 0.")
                    st.divider()
                    _subheader("Real vs Predicho")
                    st.plotly_chart(viz.grafica_regresion(modelo.y_test, modelo.y_pred), use_container_width=True)
                    _subheader("Residuos")
                    st.caption("Deben repartirse al azar alrededor de 0; un patrón (curva o embudo) indica que el modelo deja estructura sin capturar.")
                    st.plotly_chart(viz.grafica_residuos(modelo.y_test, modelo.y_pred), use_container_width=True)

            else:  # clasificacion
                if modelo.y_test is None or modelo.y_pred is None:
                    st.warning("El modelo no tiene predicciones almacenadas.")
                    st.stop()
                m = metricas_eval
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{m['accuracy']:.3f}")
                c2.metric("Precision", f"{m['precision']:.3f}")
                c3.metric("Recall", f"{m['recall']:.3f}")
                c4.metric("F1", f"{m['f1']:.3f}")
                if "roc_auc" in m:
                    st.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
                st.divider()
                _subheader("Matriz de confusión")
                # La matriz se dimensiona con la unión de clases reales y predichas;
                # las etiquetas deben cubrir esa misma unión para no desalinearse.
                clases = sorted(set(modelo.y_test.tolist()) | set(modelo.y_pred.tolist()))
                etiquetas_ev = [str(c) for c in clases]
                st.plotly_chart(viz.grafica_confusion(m["confusion_matrix"], etiquetas_ev), use_container_width=True)
                st.divider()
                _subheader("Reporte de clasificación")
                st.code(m["reporte"])

            # ── Confianza, coeficientes e importancias (modelos supervisados) ──
            if tipo in ("clasificacion", "regresion"):
                tt = metricas_eval.get("train_test")
                bl = metricas_eval.get("baseline")
                if tt or bl:
                    st.divider()
                    _subheader("Confianza del modelo")
                    if tt:
                        brecha = tt["train"] - tt["test"]
                        cc1, cc2 = st.columns(2)
                        cc1.metric(f"{tt['metrica']} entrenamiento", f"{tt['train']:.3f}")
                        cc2.metric(f"{tt['metrica']} prueba", f"{tt['test']:.3f}", delta=f"{-brecha:+.3f}")
                        if brecha > 0.15:
                            st.warning(
                                "Posible sobreajuste: el modelo rinde bastante mejor en entrenamiento que en "
                                "prueba, señal de que memoriza en vez de generalizar. Prueba a simplificarlo "
                                "(menos features, menor profundidad) o a conseguir más datos."
                            )
                    if bl:
                        linea = f"Línea base ({bl['estrategia']}): {bl['metrica']} = {bl['valor']:.3f}."
                        if tt:
                            linea += f" Tu modelo: {tt['test']:.3f}."
                        st.caption(linea)
                        if tt and tt["test"] <= bl["valor"]:
                            st.warning(
                                "Tu modelo no supera a la línea base: con estas features no está aportando "
                                "valor sobre adivinar a lo trivial. Revisa las features o prueba otro modelo."
                            )

                coef = metricas_eval.get("coeficientes")
                if coef:
                    st.divider()
                    _subheader("Coeficientes")
                    st.caption("Peso de cada feature en el modelo lineal (signo = dirección; magnitud = influencia).")
                    coef_df = pd.DataFrame(list(coef.items()), columns=["Feature", "Coeficiente"])
                    coef_df = coef_df.reindex(coef_df["Coeficiente"].abs().sort_values(ascending=False).index)
                    st.plotly_chart(
                        Visualizacion(coef_df).grafica_barras("Feature", "Coeficiente"),
                        use_container_width=True,
                    )

                imp = metricas_eval.get("importancias")
                if imp:
                    tipo_imp = metricas_eval.get("importancias_tipo", "")
                    st.divider()
                    _subheader("Importancia de features")
                    if tipo_imp:
                        st.caption(f"Método: {tipo_imp}.")
                    imp_df = pd.DataFrame(list(imp.items()), columns=["Feature", "Importancia"])
                    imp_df = imp_df.sort_values("Importancia", ascending=False)
                    st.plotly_chart(
                        Visualizacion(imp_df).grafica_barras("Feature", "Importancia"),
                        use_container_width=True,
                    )

            # ── Predicción interactiva y exportación (todos los modelos) ──
            if modelo.features:
                st.divider()
                _subheader("Predicción interactiva")
                mapeos = st.session_state.get("label_mappings", {})
                onehot = st.session_state.get("onehot_mappings", {})
                # dummy → (grupo One-Hot, categoría) para reconocer features dummy
                dummy_a_grupo = {
                    dummy: grupo
                    for grupo, cats in onehot.items()
                    for dummy in cats.values()
                }
                # Grupos One-Hot presentes entre las features del modelo: se piden
                # como un único selectbox de categoría en vez de un 0/1 por dummy.
                grupos_presentes = [
                    g for g in onehot
                    if any(d in modelo.features for d in onehot[g].values())
                ]
                campos = [("grupo", g) for g in grupos_presentes]
                campos += [("feat", f) for f in modelo.features if f not in dummy_a_grupo]

                n_cols = min(3, len(campos)) or 1
                cols_form = st.columns(n_cols)
                valores: dict = {f: 0 for f in modelo.features if f in dummy_a_grupo}  # dummies en 0 por defecto
                for i, (clase, nombre) in enumerate(campos):
                    widget = cols_form[i % n_cols]
                    if clase == "grupo":
                        # One-Hot: elegir la categoría pone su dummy en 1 y el resto en 0.
                        categoria = widget.selectbox(nombre, list(onehot[nombre].keys()), key=f"pred_oh_{nombre}")
                        dummy = onehot[nombre][categoria]
                        if dummy in valores:
                            valores[dummy] = 1
                    elif nombre in mapeos:
                        # Feature codificada con LabelEncoder: ofrecer las categorías
                        # por nombre y traducir a su código, en vez de pedir el entero.
                        categoria = widget.selectbox(nombre, list(mapeos[nombre].keys()), key=f"pred_{nombre}")
                        valores[nombre] = mapeos[nombre][categoria]
                    else:
                        if nombre in df.columns and pd.api.types.is_numeric_dtype(df[nombre]):
                            defecto = float(df[nombre].mean())
                        else:
                            defecto = 0.0
                        valores[nombre] = widget.number_input(nombre, value=defecto, key=f"pred_{nombre}")
                if st.button("Predecir", use_container_width=True, key="btn_predecir"):
                    try:
                        entrada = pd.DataFrame([valores])
                        pred = modelo.predecir(entrada)[0]
                        if tipo == "clustering":
                            st.metric("Cluster asignado", str(pred))
                        elif tipo == "regresion":
                            st.metric(f"Predicción de '{modelo.target}'", f"{float(pred):.4f}")
                        else:
                            # Probabilidades por clase si el modelo las soporta: convierte
                            # "clase 1" en "clase 1 (78% de confianza)", más honesto.
                            proba = modelo.predecir_proba(entrada) if hasattr(modelo, "predecir_proba") else None
                            if proba is not None:
                                clases, probs = proba
                                fila = probs[0]
                                idx = list(clases).index(pred)
                                st.metric(f"Clase predicha de '{modelo.target}'", str(pred), help=f"Confianza: {fila[idx]:.1%}")
                                prob_df = pd.DataFrame({"Clase": [str(c) for c in clases], "Probabilidad": fila})
                                prob_df = prob_df.sort_values("Probabilidad", ascending=False)
                                mostrar_df(prob_df.assign(Probabilidad=prob_df["Probabilidad"].map(lambda p: f"{p:.1%}")))
                            else:
                                st.metric(f"Clase predicha de '{modelo.target}'", str(pred))
                    except Exception as e:
                        st.error(f"No se pudo predecir: {e}")

            # ── Predicción por lotes (todos los modelos) ──
            if modelo.features:
                st.divider()
                _subheader("Predicción por lotes")
                st.caption(
                    "Sube un CSV con las columnas de features ya en el mismo formato del "
                    f"entrenamiento ({', '.join(modelo.features)}). Se predice cada fila y puedes descargar el resultado."
                )
                archivo_lote = st.file_uploader("CSV para predecir", type=["csv"], key="batch_pred")
                if archivo_lote is not None:
                    try:
                        df_lote = pd.read_csv(archivo_lote)
                        faltan = [f for f in modelo.features if f not in df_lote.columns]
                        if faltan:
                            st.error(f"Al CSV le faltan estas columnas de features: {', '.join(faltan)}")
                        else:
                            preds = modelo.predecir(df_lote)
                            col_pred = f"prediccion_{getattr(modelo, 'target', None) or 'cluster'}"
                            salida = df_lote.copy()
                            salida[col_pred] = preds
                            # Confianza por fila si el modelo da probabilidades
                            if tipo == "clasificacion":
                                proba_lote = modelo.predecir_proba(df_lote)
                                if proba_lote is not None:
                                    salida["confianza"] = proba_lote[1].max(axis=1)
                            st.success(f"{len(salida)} filas predichas.")
                            mostrar_df(salida.head(50))
                            st.download_button(
                                "Descargar predicciones (CSV)",
                                _csv_bytes(salida),
                                file_name="predicciones_daad.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
                    except Exception as e:
                        st.error(f"No se pudo predecir el lote: {e}")

            # ── Exportación del modelo entrenado ──
            st.divider()
            buffer_modelo = io.BytesIO()
            # Se empaqueta scaler + estimador en un Pipeline de sklearn: el archivo
            # exportado predice en un solo paso sobre un DataFrame con las features.
            estimador_export = (
                make_pipeline(modelo.scaler, modelo.modelo) if modelo.scaler is not None else modelo.modelo
            )
            joblib.dump(
                {
                    "pipeline": estimador_export,
                    "features": modelo.features,
                    "target": getattr(modelo, "target", None),
                    # Mapeos de codificación para reconstruir features categóricas a mano
                    # (la app codifica como paso manual, no van dentro del pipeline).
                    "label_mappings": st.session_state.get("label_mappings", {}),
                    "onehot_mappings": st.session_state.get("onehot_mappings", {}),
                },
                buffer_modelo,
            )
            st.download_button(
                "Descargar modelo entrenado (.joblib)",
                buffer_modelo.getvalue(),
                file_name="modelo_daad.joblib",
                use_container_width=True,
                help="Contiene un Pipeline (escalado + modelo). Cárgalo con joblib y usa "
                     "pipeline.predict(df[features]) sobre datos con las mismas features.",
            )

            # ── Explicación de resultados con IA local (Ollama) ──
            st.divider()
            _subheader("Explicación con IA")
            disponible_ia, modelos_ia = ollama_disponible()
            if not disponible_ia:
                st.caption("Inicia Ollama (`ollama run llama3.2`) para activar la explicación con IA.")
            elif st.button("Explicar resultados con IA", use_container_width=True, key="btn_explicar_ia"):
                metricas_ia = st.session_state.metricas if st.session_state.metricas else modelo.evaluar()
                importancias_ia = metricas_ia.get("importancias") if isinstance(metricas_ia, dict) else None
                prompt = prompt_explicacion(str(tipo), metricas_ia, importancias_ia, getattr(modelo, "target", None))
                mensajes = [
                    {"role": "system", "content": SISTEMA_DATASET},
                    {"role": "user", "content": prompt},
                ]
                with st.chat_message("assistant"):
                    st.write_stream(chat_stream(mensajes, modelos_ia[0] if modelos_ia else MODELO_POR_DEFECTO))

    elif seccion == "⑧ Asistente IA":
        _page_header("IA", "Asistente IA", "datos / asistente")
        disponible_ia, modelos_ia = ollama_disponible()
        if not disponible_ia:
            st.warning(
                "No detecté Ollama corriendo en http://localhost:11434.\n\n"
                "Para activarlo: instala Ollama desde ollama.com, abre una terminal y ejecuta "
                "una vez `ollama run llama3.2` (descarga el modelo, ~2 GB). Luego recarga esta página."
            )
        else:
            modelo_sel = st.selectbox("Modelo de IA", modelos_ia, index=0)
            st.caption(
                f"El asistente conoce tu dataset activo: {df.shape[0]} filas y {df.shape[1]} columnas. "
                "Pregúntale sobre nulos, correlaciones, qué modelo usar, etc."
            )

            if "chat_ia" not in st.session_state:
                st.session_state.chat_ia = []

            for msg in st.session_state.chat_ia:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            pregunta = st.chat_input("Pregunta sobre tus datos…")
            if pregunta:
                st.session_state.chat_ia.append({"role": "user", "content": pregunta})
                with st.chat_message("user"):
                    st.markdown(pregunta)
                with st.chat_message("assistant"):
                    respuesta = st.write_stream(
                        chat_stream(mensajes_chat(df, st.session_state.chat_ia), modelo_sel)
                    )
                st.session_state.chat_ia.append({"role": "assistant", "content": respuesta})

            if st.session_state.chat_ia and st.button("Limpiar conversación", key="btn_limpiar_chat"):
                st.session_state.chat_ia = []
                st.rerun()
