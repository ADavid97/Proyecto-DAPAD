import streamlit as st
import pandas as pd
from datos import Datos

st.set_page_config(page_title="DAAD", page_icon="", layout="wide")

# --- SESSION STATE ---
if "df" not in st.session_state:
    st.session_state.df = None
if "tablas_pg" not in st.session_state:
    st.session_state.tablas_pg = []

# --- SIDEBAR ---
with st.sidebar:
    st.title(" DAAD Creado por Lupita Tiktok")
    st.divider()

    st.subheader("Cargar datos")
    tipo = st.radio("Tipo de fuente", ["CSV", "TSV", "PostgreSQL"])

    if tipo in ("CSV", "TSV"):
        archivo = st.file_uploader(f"Sube archivo {tipo}", type=["csv", "tsv", "txt"], key=tipo)
        if archivo is not None:
            sep = "," if tipo == "CSV" else "\t"
            try:
                df = pd.read_csv(archivo, sep=sep)
                st.session_state.df = df
                st.success(f"✓ {df.shape[0]} filas × {df.shape[1]} columnas")
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
                st.success(f"✓ {len(tablas)} tablas encontradas")
            else:
                st.error("No se pudo conectar o no hay tablas.")

        if st.session_state.tablas_pg:
            tabla = st.selectbox("Selecciona tabla", st.session_state.tablas_pg)
            if st.button("Cargar tabla", use_container_width=True):
                cargador = Datos()
                df = cargador.cargar_tabla_sql(host, int(puerto), base_datos, usuario, contrasena, tabla)
                if df is not None:
                    st.session_state.df = df
                    st.success(f"✓ Tabla '{tabla}' cargada")

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
    st.title("Bienvenido a DAAD ")
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
        st.dataframe(df, use_container_width=True)

    elif seccion == "Analisis exploratorio":
        st.title("Analisis exploratorio")
        st.info("Modulo en construccion.")

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
