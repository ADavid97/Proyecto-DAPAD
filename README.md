# DAAD — Data Analysis Application with Data

Aplicación web en Streamlit para cargar, explorar, limpiar, visualizar y modelar conjuntos de datos sin escribir código. Proyecto de la Licenciatura en Ciencia de Datos (ESCOM-IPN).

## Funcionalidades

| Sección | Qué hace |
|---|---|
| Carga de datos | CSV, TSV, Excel (multi-hoja), PostgreSQL, JSON anidado y web scraping (tablas, listas y definiciones) |
| ① DataFrame | Vista del dataset activo, métricas básicas y descarga en CSV |
| ② Análisis exploratorio | Resumen general, estadísticas descriptivas, nulos, valores únicos, distribuciones, correlación y outliers (IQR) |
| ③ Preprocesamiento | Selección de columnas, nulos (eliminar/rellenar), duplicados, filtros, conversión de tipos, escalado y codificación (One-Hot o LabelEncoder), con historial de transformaciones y deshacer paso a paso |
| ④ Visualización | Histograma, boxplot, dispersión, barras, líneas, pastel y mapa de calor (Plotly) |
| ⑤ Modelos | KNN, Árbol de Decisión, Regresión Logística, Regresión Lineal y K-Means, con escalado sin fuga de datos, split estratificado y validación cruzada k-fold opcional |
| ⑥ Evaluación | Métricas de clasificación/regresión/clustering, matriz de confusión, reporte, importancia de features, predicción interactiva y descarga del modelo (.joblib) |

Soporta varios datasets cargados a la vez (selector de dataset activo en el panel lateral).

## Requisitos

- Python 3.11+
- Las dependencias de [requirements.txt](requirements.txt) (versiones pinneadas)

## Instalación y ejecución

```powershell
# 1. Clonar y entrar al proyecto
git clone <url-del-repo>
cd DAAD

# 2. Crear y activar entorno virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la app
streamlit run main.py
```

La app abre en `http://localhost:8501`. Carga un dataset desde el panel lateral para habilitar las secciones.

## Estructura del proyecto

```
DAAD/
├── main.py                     # UI de Streamlit (sidebar + 6 secciones)
├── styles.py                   # CSS del design system
├── chart_theme.py              # Template "daad" de Plotly
├── datos/                      # Carga: CSV/TSV, Excel, PostgreSQL, JSON, scraping
├── analisis_exploratorio/      # EDA: resúmenes, outliers, correlación
├── preprocesamiento/           # Limpieza, escalado, codificación
├── modelos/                    # Modelo (ABC) + KNN, Árbol, Log., Lineal, K-Means
├── evaluacion/                 # Métricas de modelos entrenados
├── visualizacion/              # Gráficas Plotly
└── .streamlit/config.toml      # Tema de Streamlit
```

## Notas de diseño

- Los modelos hacen el `train_test_split` **antes** de escalar y ajustan el `StandardScaler` solo con train (sin fuga de datos hacia test); los clasificadores estratifican el split.
- Las operaciones costosas de EDA (describe, correlación, duplicados, CSV de descarga) usan `@st.cache_data`.
- El modelo entrenado y sus métricas persisten en `st.session_state` mientras dura la sesión.

## Equipo

- Reyes Calva Angel David
- Ortiz Juarez Emiliano
- Hernandez Gaspar Andrei
