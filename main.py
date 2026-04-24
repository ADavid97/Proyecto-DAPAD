import streamlit as st
import pandas as pd
from datos import Datos
from analisis_exploratorio import AnalisisExploratorio

st.set_page_config(page_title="DAAD", page_icon="", layout="wide")

st.markdown("<style>*, *::before, *::after { border-radius: 0 !important; }</style>", unsafe_allow_html=True)


def mostrar_df(data: pd.DataFrame) -> None:
    df_safe = data.copy()
    for col in df_safe.select_dtypes(include="object").columns:
        df_safe[col] = df_safe[col].astype(str)
    st.dataframe(df_safe, width="stretch")


# --- SESSION STATE ---
if "df" not in st.session_state:
    st.session_state.df = None
if "tablas_pg" not in st.session_state:
    st.session_state.tablas_pg = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("APLICAIÓN PARA ANÁLISIS DE DATOS")
    st.divider()

    st.subheader("Desarrollado por:")
    st.caption("Reyes Calva Angel David")
    st.caption("Ortiz Juarez Emiliano")
    st.caption("Hernandez Gaspar Andrei")

    st.subheader("Cargar datos")
    tipo = st.radio("Tipo de fuente", ["CSV", "TSV", "PostgreSQL"])

    if tipo in ("CSV", "TSV"):
        extensiones = ["csv"] if tipo == "CSV" else ["tsv", "txt"]
        archivo = st.file_uploader(f"Sube archivo {tipo}", type=extensiones, key=tipo)
        if archivo is not None:
            extension = archivo.name.rsplit(".", 1)[-1].lower()
            esperadas = {"CSV": ["csv"], "TSV": ["tsv", "txt"]}
            if extension not in esperadas[tipo]:
                st.error(f"El archivo '{archivo.name}' no es un {tipo} válido. Sube un archivo con extensión {extensiones}.")
            else:
                sep = "," if tipo == "CSV" else "\t"
                try:
                    df = pd.read_csv(archivo, sep=sep)
                    st.session_state.df = df
                    st.success(f"{df.shape[0]} filas × {df.shape[1]} columnas")
                except Exception as e:
                    st.error(f"Error al leer el archivo: {e}")

    else:
        host = st.text_input("Host", "localhost")
        puerto = st.number_input("Puerto", value=5432, step=1)
        base_datos = st.text_input("Base de datos")
        usuario = st.text_input("Usuario")
        contrasena = st.text_input("Contrasena", type="password")

        if st.button("Conectar", use_container_width=True):
            cargador = Datos()
            tablas = cargador.listar_tablas(host, int(puerto), base_datos, usuario, contrasena)
            if tablas:
                st.session_state.tablas_pg = tablas
                st.success(f"{len(tablas)} tablas encontradas")
            else:
                st.error("No se pudo conectar o no hay tablas.")

        if st.session_state.tablas_pg:
            tabla = st.selectbox("Selecciona tabla", st.session_state.tablas_pg)
            if st.button("Cargar tabla", use_container_width=True) and tabla is not None:
                cargador = Datos()
                df = cargador.cargar_tabla_sql(host, int(puerto), base_datos, usuario, contrasena, tabla)
                if df is not None:
                    st.session_state.df = df
                    st.success(f"Tabla '{tabla}' cargada")

    if st.session_state.df is not None:
        st.divider()
        st.subheader("Navegacion")
        seccion = st.radio(
            "Seccion",
            ["DataFrame", "Analisis exploratorio", "Preprocesamiento",
             "Visualizacion", "Modelos", "Evaluacion"],
            label_visibility="collapsed"
        )
    else:
        seccion = "inicio"

# --- AREA PRINCIPAL ---
if st.session_state.df is None:
    st.title("Bienvenido")
    st.info("Carga un archivo CSV, TSV o conecta a PostgreSQL desde el panel izquierdo para comenzar.")

else:
    df = st.session_state.df

    if seccion == "DataFrame":
        st.title("DataFrame")
        col1, col2, col3 = st.columns(3)
        col1.metric("Filas", df.shape[0])
        col2.metric("Columnas", df.shape[1])
        col3.metric("Valores nulos", int(df.isnull().sum().sum()))
        st.divider()
        mostrar_df(df)

    elif seccion == "Analisis exploratorio":
        st.title("Analisis exploratorio")
        eda = AnalisisExploratorio(df)

        opcion = st.selectbox("¿Qué deseas analizar?", [
            "Resumen general",
            "Estadísticas descriptivas",
            "Conteo de nulos",
            "Valores únicos por columna",
            "Distribución de una columna",
            "Matriz de correlación",
            "Detección de outliers"
        ])

        st.divider()

        if opcion == "Resumen general":
            resumen = eda.resumen_general()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Filas", resumen["filas"])
            col2.metric("Columnas", resumen["columnas"])
            col3.metric("Valores nulos", resumen["nulos_totales"])
            col4.metric("Duplicados", resumen["duplicados"])
            st.subheader("Tipos de datos por columna")
            tipos_df = pd.DataFrame(resumen["tipos"].items(), columns=["Columna", "Tipo"])
            mostrar_df(tipos_df)

        elif opcion == "Estadísticas descriptivas":
            mostrar_df(eda.estadisticas_descriptivas())

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
            corr = eda.matriz_correlacion()
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

    elif seccion == "Preprocesamiento":
        st.title("Preprocesamiento")
        st.info("Modulo en construccion.")

    elif seccion == "Visualizacion":
        st.title("Visualizacion")
        st.info("Modulo en construccion.")

    elif seccion == "Modelos":
        st.title("Modelos")
        st.info("Modulo en construccion.")

    elif seccion == "Evaluacion":
        st.title("Evaluacion")
        st.info("Modulo en construccion.")
