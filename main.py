import streamlit as st
import pandas as pd
from datos import Datos
from analisis_exploratorio import AnalisisExploratorio
from visualizacion import Visualizacion
from preprocesamiento import Preprocesamiento
from modelos import ModeloKNN, ModeloKMeans, ModeloRegresionLineal, ModeloRegresionLogistica, ModeloArbolDecision
from evaluacion import Evaluacion
from styles import apply_styles
import chart_theme  # noqa: F401 — registers "daad" Plotly template on import

st.set_page_config(page_title="DAAD", page_icon="", layout="wide")
apply_styles()


@st.cache_data
def _csv_bytes(data: pd.DataFrame) -> bytes:
    return data.to_csv(index=False).encode("utf-8")


@st.cache_data
def _contar_duplicados(data: pd.DataFrame) -> int:
    try:
        return int(data.duplicated().sum())
    except TypeError:  # celdas no hashables (listas/dicts provenientes de JSON)
        return int(data.astype(str).duplicated().sum())


@st.cache_data
def _estadisticas_descriptivas(data: pd.DataFrame) -> pd.DataFrame:
    return AnalisisExploratorio(data).estadisticas_descriptivas()


@st.cache_data
def _matriz_correlacion(data: pd.DataFrame) -> pd.DataFrame:
    return AnalisisExploratorio(data).matriz_correlacion()


@st.cache_data
def _resumen_general(data: pd.DataFrame) -> dict:
    return AnalisisExploratorio(data).resumen_general()


def _guardar_df(nuevo_df: pd.DataFrame) -> None:
    """Actualiza df activo y sincroniza con el dict de datasets."""
    st.session_state.df = nuevo_df
    if st.session_state.df_activo:
        st.session_state.datasets[st.session_state.df_activo] = nuevo_df


def mostrar_df(data: pd.DataFrame) -> None:
    cols_obj = data.select_dtypes(include=["object", "str"]).columns
    if len(cols_obj) > 0:
        data = data.copy()
        for col in cols_obj:
            data[col] = data[col].astype(str)
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
                else:
                    cargador = Datos()
                    df = cargador.cargar_csv(archivo) if tipo == "CSV" else cargador.cargar_tsv(archivo)
                    if df is not None:
                        nombre_ds = archivo.name
                        st.session_state.df = df
                        st.session_state.datasets[nombre_ds] = df
                        st.session_state.df_activo = nombre_ds
                        st.session_state.df_original = None
                        st.success(f"{df.shape[0]} filas × {df.shape[1]} columnas")
                    else:
                        detalle = f" Detalle: {cargador.ultimo_error}" if cargador.ultimo_error else ""
                        st.error(f"Error al leer el archivo.{detalle}")

        elif tipo == "Excel":
            archivo = st.file_uploader("Sube archivo Excel", type=["xlsx", "xls"], key="Excel")
            if archivo is not None:
                cargador = Datos()
                hojas = cargador.cargar_excel(archivo)
                if hojas:
                    st.session_state.hojas_excel = hojas
                    st.success(f"{len(hojas)} hoja(s) encontrada(s)")
                else:
                    detalle = f" Detalle: {cargador.ultimo_error}" if cargador.ultimo_error else ""
                    st.error(f"No se pudo leer el archivo Excel.{detalle}")

            if st.session_state.hojas_excel:
                nombre = st.selectbox("Selecciona hoja", list(st.session_state.hojas_excel.keys()), key="sel_excel")
                if st.button("Cargar hoja", key="btn_excel", use_container_width=True):
                    df = st.session_state.hojas_excel[nombre]
                    nombre_ds = f"{archivo.name if archivo else 'excel'}[{nombre}]"
                    st.session_state.df = df
                    st.session_state.datasets[nombre_ds] = df
                    st.session_state.df_activo = nombre_ds
                    st.session_state.df_original = None
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
                    detalle = f" Detalle: {cargador.ultimo_error}" if cargador.ultimo_error else ""
                    st.error(f"No se pudo conectar o no hay tablas.{detalle}")

            if st.session_state.tablas_pg:
                tabla = st.selectbox("Selecciona tabla", st.session_state.tablas_pg)
                if st.button("Cargar tabla", use_container_width=True) and tabla is not None:
                    cargador = Datos()
                    df = cargador.cargar_tabla_sql(host, int(puerto), base_datos, usuario, contrasena, tabla)
                    if df is not None:
                        st.session_state.df = df
                        st.session_state.datasets[tabla] = df
                        st.session_state.df_activo = tabla
                        st.session_state.df_original = None
                        st.success(f"Tabla '{tabla}' cargada")

    with tab_noest:
        archivo = st.file_uploader("Sube archivo JSON", type=["json"], key="JSON")
        if archivo is not None:
            cargador = Datos()
            tablas = cargador.cargar_json(archivo)
            if tablas:
                st.session_state.tablas_json = tablas
                st.success(f"{len(tablas)} tabla(s) encontrada(s)")
            else:
                detalle = f" Detalle: {cargador.ultimo_error}" if cargador.ultimo_error else ""
                st.error(f"No se pudo leer el archivo JSON.{detalle}")

        if st.session_state.tablas_json:
            nombre = st.selectbox("Selecciona tabla", list(st.session_state.tablas_json.keys()), key="sel_json")
            if st.button("Cargar tabla", key="btn_json", use_container_width=True):
                df = st.session_state.tablas_json[nombre]
                st.session_state.df = df
                st.session_state.datasets[nombre] = df
                st.session_state.df_activo = nombre
                st.session_state.df_original = None
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
                else:
                    st.session_state.elementos_url = {}
                    st.error("No se encontraron tablas ni listas en la página.")
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
                    st.session_state.df = df
                    st.session_state.datasets[nombre_el] = df
                    st.session_state.df_activo = nombre_el
                    st.session_state.df_original = None
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
            st.rerun()
        _df_sel = st.session_state.datasets[sel_ds]
        st.caption(f"{_df_sel.shape[0]} filas × {_df_sel.shape[1]} columnas")
        if st.button("Eliminar dataset", key="btn_del_ds"):
            del st.session_state.datasets[sel_ds]
            if st.session_state.datasets:
                nuevo_ds = list(st.session_state.datasets.keys())[0]
                st.session_state.df = st.session_state.datasets[nuevo_ds]
                st.session_state.df_activo = nuevo_ds
            else:
                st.session_state.df = None
                st.session_state.df_activo = None
            st.session_state.df_original = None
            st.rerun()

    if st.session_state.df is not None:
        _section_label("Navegación")
        seccion = st.radio(
            "Seccion",
            [
                "① DataFrame",
                "② Análisis exploratorio",
                "③ Preprocesamiento",
                "④ Visualización",
                "⑤ Modelos",
                "⑥ Evaluación",
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
            file_name="datos.csv",
            mime="text/csv",
            use_container_width=True,
        )

    elif seccion == "② Análisis exploratorio":
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
            st.markdown('<span class="daad-subheader">Tipos de datos por columna</span>', unsafe_allow_html=True)
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

    elif seccion == "③ Preprocesamiento":
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

        operacion = st.selectbox("Operación", [
            "Seleccionar columnas",
            "Eliminar nulos",
            "Rellenar nulos",
            "Eliminar duplicados",
            "Filtrar filas",
            "Convertir tipo de columna",
            "Normalizar — Standard",
            "Normalizar — MinMax",
            "Codificar categóricas",
        ])
        st.divider()

        if operacion == "Seleccionar columnas":
            cols_sel = st.multiselect(
                "Columnas a conservar",
                df.columns.tolist(),
                default=df.columns.tolist(),
            )
            if st.button("Aplicar", key="btn_sel_cols", use_container_width=True):
                if cols_sel:
                    _guardar_df(pre.seleccionar_columnas(cols_sel))
                    st.success(f"Columnas reducidas a {len(cols_sel)}")
                    st.rerun()
                else:
                    st.warning("Selecciona al menos una columna.")

        elif operacion == "Eliminar nulos":
            nulos = int(df.isnull().sum().sum())
            st.info(f"{nulos} valores nulos encontrados — se eliminarán todas las filas que contengan al menos uno.")
            if st.button("Aplicar", key="btn_elim_nulos", use_container_width=True):
                resultado = pre.eliminar_nulos()
                _guardar_df(resultado)
                st.success(f"Filas: {df.shape[0]} → {resultado.shape[0]}")
                st.rerun()

        elif operacion == "Rellenar nulos":
            nulos = int(df.isnull().sum().sum())
            st.info(f"{nulos} valores nulos en el dataset.")
            estrategia = st.radio(
                "Estrategia",
                ["media", "mediana", "moda", "constante"],
                horizontal=True,
            )
            valor_cte = 0.0
            if estrategia == "constante":
                valor_cte = st.number_input("Valor constante", value=0.0)
            if st.button("Aplicar", key="btn_rel_nulos", use_container_width=True):
                resultado = pre.rellenar_nulos(estrategia, valor_cte)
                _guardar_df(resultado)
                st.success(f"Nulos rellenados. Restantes: {int(resultado.isnull().sum().sum())}")
                st.rerun()

        elif operacion == "Eliminar duplicados":
            dups = _contar_duplicados(df)
            st.info(f"{dups} filas duplicadas encontradas.")
            if st.button("Aplicar", key="btn_elim_dups", use_container_width=True):
                resultado = pre.eliminar_duplicados()
                _guardar_df(resultado)
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
                    _guardar_df(resultado)
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
                    _guardar_df(resultado)
                    st.success(f"Filas: {df.shape[0]} → {resultado.shape[0]}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al filtrar: {e}")

        elif operacion == "Normalizar — Standard":
            numericas = df.select_dtypes(include="number").columns.tolist()
            if not numericas:
                st.warning("No hay columnas numéricas.")
            else:
                st.info(f"{len(numericas)} columnas numéricas → media = 0, desviación = 1.")
                st.caption(
                    "Nota: esto escala el dataset completo. Si vas a entrenar un modelo, "
                    "usa mejor la opción de escalado en ⑤ Modelos para evitar fuga de datos."
                )
                if st.button("Aplicar", key="btn_std", use_container_width=True):
                    _guardar_df(pre.normalizar_standard())
                    st.success("Normalización Standard aplicada.")
                    st.rerun()

        elif operacion == "Normalizar — MinMax":
            numericas = df.select_dtypes(include="number").columns.tolist()
            if not numericas:
                st.warning("No hay columnas numéricas.")
            else:
                st.info(f"{len(numericas)} columnas numéricas → rango [0, 1].")
                st.caption(
                    "Nota: esto escala el dataset completo. Si vas a entrenar un modelo, "
                    "usa mejor la opción de escalado en ⑤ Modelos para evitar fuga de datos."
                )
                if st.button("Aplicar", key="btn_mm", use_container_width=True):
                    _guardar_df(pre.normalizar_minmax())
                    st.success("Normalización MinMax aplicada.")
                    st.rerun()

        elif operacion == "Codificar categóricas":
            cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
            if not cats:
                st.warning("No hay columnas categóricas.")
            else:
                st.info(f"Columnas a codificar: {', '.join(cats)}")
                if st.button("Aplicar", key="btn_enc", use_container_width=True):
                    _guardar_df(pre.codificar_categoricas())
                    st.success("Columnas categóricas codificadas con LabelEncoder.")
                    st.rerun()

        st.divider()
        st.markdown('<span class="daad-subheader">Vista previa</span>', unsafe_allow_html=True)
        mostrar_df(df)

        if st.session_state.df_original is not None:
            if st.button("Restaurar datos originales", key="btn_restaurar"):
                _guardar_df(st.session_state.df_original.copy())
                st.success("Datos restaurados al estado original.")
                st.rerun()

    elif seccion == "④ Visualización":
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

    elif seccion == "⑤ Modelos":
        _page_header("MOD", "Modelos", "datos / machine learning")
        viz = Visualizacion(df)
        numericas = df.select_dtypes(include="number").columns.tolist()

        nulos = int(df.isnull().sum().sum())
        if nulos > 0:
            st.error(f"{nulos} valores nulos detectados. Ve a ③ Preprocesamiento y elimínalos o rellénalos primero.")
            st.stop()
        if not numericas:
            st.error("No hay columnas numéricas. Convierte o codifica tus columnas en ③ Preprocesamiento primero.")
            st.stop()

        tipo_modelo = st.selectbox("Tipo de modelo", [
            "KNN — K-Nearest Neighbors",
            "Árbol de Decisión",
            "Regresión Logística",
            "Regresión Lineal",
            "K-Means (clustering)",
        ])
        st.divider()

        if tipo_modelo == "K-Means (clustering)":
            features = st.multiselect("Columnas (features)", numericas, default=numericas[:min(2, len(numericas))])
            n_clusters = int(st.number_input("Número de clusters (k)", min_value=2, max_value=20, value=3, step=1))

            mostrar_codo = st.checkbox("Mostrar curva del codo")
            if mostrar_codo and features:
                k_max = int(st.slider("k máximo", 2, 15, 10))
                tmp = ModeloKMeans(df)
                tmp.features = features
                with st.spinner("Calculando codo..."):
                    inercias = tmp.calcular_codo(k_max)
                st.plotly_chart(viz.grafica_codo(inercias), use_container_width=True)

            if st.button("Entrenar K-Means", use_container_width=True, disabled=not features):
                modelo = ModeloKMeans(df, n_clusters=n_clusters)
                modelo.features = features
                try:
                    with st.spinner("Entrenando..."):
                        modelo.entrenar()
                    st.session_state.modelo_entrenado = modelo
                    st.session_state.modelo_tipo = "clustering"
                    st.session_state.metricas = modelo.evaluar()
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
                if (
                    modelo_e.etiquetas is not None
                    and modelo_e.centroides is not None
                    and all(f in df.columns for f in modelo_e.features)
                    and len(modelo_e.etiquetas) == df.shape[0]
                ):
                    st.plotly_chart(
                        viz.grafica_clusters(df[modelo_e.features].copy(), modelo_e.etiquetas, modelo_e.centroides),
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

            hiperparams: dict = {}
            if tipo_modelo == "KNN — K-Nearest Neighbors":
                hiperparams["k"] = int(st.number_input("k (vecinos)", min_value=1, max_value=50, value=5, step=1))
            elif tipo_modelo == "Árbol de Decisión":
                prof = int(st.number_input("Profundidad máxima (0 = sin límite)", min_value=0, max_value=50, value=0, step=1))
                hiperparams["profundidad_max"] = None if prof == 0 else prof
            elif tipo_modelo == "Regresión Logística":
                hiperparams["max_iter"] = int(st.number_input("Max iteraciones", min_value=100, max_value=2000, value=200, step=100))

            invalidas = [f for f in features if f not in numericas]
            if invalidas:
                st.warning(f"Features no numéricas: {invalidas}. Codifícalas primero.")

            es_regresion = tipo_modelo == "Regresión Lineal"
            error_target = None
            if target:
                serie_target = df[target]
                if es_regresion and not pd.api.types.is_numeric_dtype(serie_target):
                    error_target = (
                        f"'{target}' no es numérica. La Regresión Lineal necesita un target numérico; "
                        "conviértela o codifícala en ③ Preprocesamiento."
                    )
                elif not es_regresion and pd.api.types.is_float_dtype(serie_target) and serie_target.nunique() > 20:
                    error_target = (
                        f"'{target}' parece continua ({serie_target.nunique()} valores únicos). "
                        "Los clasificadores necesitan clases discretas; usa Regresión Lineal o discretiza el target."
                    )
            if error_target:
                st.error(error_target)

            puede_entrenar = bool(features) and target not in features and not invalidas and error_target is None

            if st.button("Entrenar modelo", use_container_width=True, disabled=not puede_entrenar):
                if tipo_modelo == "KNN — K-Nearest Neighbors":
                    modelo = ModeloKNN(df, k=hiperparams["k"])
                elif tipo_modelo == "Árbol de Decisión":
                    modelo = ModeloArbolDecision(df, profundidad_max=hiperparams["profundidad_max"])
                elif tipo_modelo == "Regresión Logística":
                    modelo = ModeloRegresionLogistica(df, max_iter=hiperparams["max_iter"])
                else:
                    modelo = ModeloRegresionLineal(df)

                modelo.features = features
                modelo.target = target
                try:
                    with st.spinner("Entrenando..."):
                        modelo.entrenar(test_size=test_size, escalar=escalar)
                    st.session_state.modelo_entrenado = modelo
                    st.session_state.modelo_tipo = "regresion" if es_regresion else "clasificacion"
                    st.session_state.metricas = modelo.evaluar()
                    st.success("Modelo entrenado.")
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

    elif seccion == "⑥ Evaluación":
        _page_header("EVL", "Evaluación", "datos / métricas")

        if st.session_state.modelo_entrenado is None:
            st.info("Primero entrena un modelo en ⑤ Modelos.")
        else:
            modelo = st.session_state.modelo_entrenado
            tipo = st.session_state.modelo_tipo
            viz = Visualizacion(df)

            if tipo == "clustering":
                m = modelo.evaluar()
                c1, c2, c3 = st.columns(3)
                c1.metric("Clusters (k)", m["n_clusters"])
                c2.metric("Inercia", f"{m['inercia']:.2f}")
                c3.metric("Silhouette", f"{m['silhouette']:.4f}")
                st.divider()
                st.markdown('<span class="daad-subheader">Visualización de clusters</span>', unsafe_allow_html=True)
                if modelo.etiquetas is not None and modelo.centroides is not None:
                    st.plotly_chart(
                        viz.grafica_clusters(df[modelo.features].copy(), modelo.etiquetas, modelo.centroides),
                        use_container_width=True,
                    )
                if modelo.inercias:
                    st.markdown('<span class="daad-subheader">Curva del codo</span>', unsafe_allow_html=True)
                    st.plotly_chart(viz.grafica_codo(modelo.inercias), use_container_width=True)

            elif tipo == "regresion":
                if modelo.y_test is not None and modelo.y_pred is not None:
                    ev = Evaluacion(modelo, modelo.y_test, modelo.y_pred)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("R²", f"{ev.r2():.4f}")
                    c2.metric("MSE", f"{ev.error_cuadratico_medio():.4f}")
                    c3.metric("RMSE", f"{ev.error_cuadratico_medio() ** 0.5:.4f}")
                    st.divider()
                    st.markdown('<span class="daad-subheader">Real vs Predicho</span>', unsafe_allow_html=True)
                    st.plotly_chart(viz.grafica_regresion(modelo.y_test, modelo.y_pred), use_container_width=True)

            else:  # clasificacion
                if modelo.y_test is None or modelo.y_pred is None:
                    st.warning("El modelo no tiene predicciones almacenadas.")
                    st.stop()
                ev = Evaluacion(modelo, modelo.y_test, modelo.y_pred)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{ev.accuracy():.3f}")
                c2.metric("Precision", f"{ev.precision():.3f}")
                c3.metric("Recall", f"{ev.recall():.3f}")
                c4.metric("F1", f"{ev.f1_score():.3f}")
                st.divider()
                st.markdown('<span class="daad-subheader">Matriz de confusión</span>', unsafe_allow_html=True)
                etiquetas_ev = [str(c) for c in sorted(set(modelo.y_test.tolist()))]
                st.plotly_chart(viz.grafica_confusion(ev.matriz_confusion(), etiquetas_ev), use_container_width=True)
                st.divider()
                st.markdown('<span class="daad-subheader">Reporte de clasificación</span>', unsafe_allow_html=True)
                st.code(ev.reporte_clasificacion())
                metricas_eval = st.session_state.metricas if st.session_state.metricas is not None else modelo.evaluar()
                imp = metricas_eval.get("importancias")
                if imp:
                    imp_df = pd.DataFrame(list(imp.items()), columns=["Feature", "Importancia"])
                    imp_df = imp_df.sort_values("Importancia", ascending=False)
                    st.divider()
                    st.markdown('<span class="daad-subheader">Importancia de features (Árbol)</span>', unsafe_allow_html=True)
                    st.plotly_chart(
                        Visualizacion(imp_df).grafica_barras("Feature", "Importancia"),
                        use_container_width=True,
                    )
