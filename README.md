# DAAD — Data Analysis Application with Data

Aplicación web en Streamlit para cargar, explorar, limpiar, visualizar y modelar conjuntos de datos sin escribir código. Proyecto de la Licenciatura en Ciencia de Datos (ESCOM-IPN).

## Funcionalidades

| Sección | Qué hace |
|---|---|
| Carga de datos | CSV, TSV, Excel (multi-hoja), PostgreSQL, JSON anidado y web scraping (tablas con `pd.read_html`, conversión numérica con convención es/en, listas y definiciones). Varios datasets a la vez con selector de dataset activo. |
| ① DataFrame | Vista del dataset activo, métricas básicas y descarga en CSV |
| ② Conjunto | Combinar datasets (concatenar/apilar o merge por columna llave) y comparar varios datasets lado a lado (estadísticas e histograma superpuesto) |
| ③ Análisis exploratorio | Resumen general, estadísticas descriptivas, nulos, valores únicos, distribuciones, matriz de correlación y detección de outliers (IQR) |
| ④ Preprocesamiento | Selección de columnas, nulos (eliminar/rellenar por media, mediana, moda o constante), columnas vacías, duplicados, filtros, conversión de tipos, escalado (Standard/MinMax) y codificación One-Hot o LabelEncoder (preservan NaN), con historial de transformaciones y deshacer paso a paso |
| ⑤ Visualización | Histograma, boxplot, dispersión, barras, líneas, pastel y mapa de calor (Plotly) |
| ⑥ Modelos | KNN, Árbol de Decisión, Random Forest, Gradient Boosting, Regresión Logística, Regresión Lineal y K-Means; con escalado sin fuga de datos, split estratificado, hiperparámetros por modelo, validación cruzada k-fold opcional y búsqueda automática de hiperparámetros (GridSearchCV) |
| ⑦ Evaluación | Métricas de clasificación/regresión/clustering; diagnóstico de confianza (train vs test + línea base); matriz de confusión y reporte; coeficientes e importancia de features (nativa o por permutación); residuos en regresión; predicción interactiva con probabilidades; predicción por lotes desde CSV; y export del modelo como Pipeline (.joblib) |
| ⑧ Asistente IA | Chat local (vía Ollama) que conoce el dataset activo y guía el uso de la app |

## Modelos disponibles (⑥)

- **Clasificación**: KNN, Árbol de Decisión, Random Forest, Gradient Boosting, Regresión Logística.
- **Regresión**: Regresión Lineal (con regularización opcional Ridge/Lasso).
- **Clustering** (no supervisado): K-Means, con curva del codo y silhouette.

Cada modelo supervisado ofrece sus hiperparámetros relevantes y una casilla opcional de
búsqueda automática de los mejores valores por validación cruzada.

## Requisitos

- Python 3.11+
- Las dependencias de [requirements.txt](requirements.txt) (versiones pinneadas)
- Opcional: [Ollama](https://ollama.com) corriendo en local para el Asistente IA (⑧) y la
  explicación de resultados con IA (⑦). Sin Ollama, el resto de la app funciona igual.

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

Para el Asistente IA (opcional), instala Ollama y descarga un modelo una vez:

```powershell
ollama run llama3.2
```

## Estructura del proyecto

```
DAAD/
├── main.py                     # UI de Streamlit (sidebar + 8 secciones)
├── styles.py                   # CSS del design system
├── chart_theme.py              # Template "daad" de Plotly
├── datos/                      # Carga: CSV/TSV, Excel, PostgreSQL, JSON, scraping
├── analisis_exploratorio/      # EDA: resúmenes, outliers, correlación
├── preprocesamiento/           # Limpieza, escalado, codificación
├── modelos/                    # Modelo (ABC) + ModeloSupervisado + 7 modelos
├── evaluacion/                 # Métricas de modelos entrenados
├── visualizacion/              # Gráficas Plotly
├── asistente/                  # Asistente IA local (Ollama)
└── .streamlit/config.toml      # Tema de Streamlit
```

## Notas de diseño

- **Sin fuga de datos**: los modelos hacen el `train_test_split` **antes** de escalar y ajustan
  el `StandardScaler` solo con train; en la validación cruzada el scaler entra al pipeline (se
  ajusta dentro de cada fold) y la búsqueda de hiperparámetros se hace solo sobre train. Los
  clasificadores estratifican el split.
- **Modelos supervisados**: comparten la lógica común en `ModeloSupervisado`; cada modelo solo
  declara su estimador y si es clasificación o regresión, de modo que añadir uno nuevo es trivial.
- **Confianza e interpretabilidad**: cada evaluación reporta train vs test (sobreajuste), una
  línea base (Dummy), importancia de features (nativa o por permutación) y, en modelos lineales,
  los coeficientes.
- **Rendimiento**: las operaciones costosas de EDA (describe, correlación, duplicados, CSV de
  descarga) usan `@st.cache_data`.
- **Persistencia**: el trabajo (datasets, transformaciones, modelo y métricas) se guarda de forma
  asíncrona y atómica en `resultados/_sesion.pkl` (local, no versionado) y se restaura al abrir.

## Equipo

- Reyes Calva Angel David
- Ortiz Juarez Emiliano
- Hernandez Gaspar Andrei
