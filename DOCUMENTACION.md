# DAAD — Documentación técnica completa

> **DAAD** (*Data Analysis Application with Data*) es una aplicación web hecha en **Streamlit**
> para **cargar, explorar, limpiar, visualizar y modelar datos sin escribir una sola línea de
> código**. Todo se opera con botones y menús. Proyecto de la Licenciatura en Ciencia de Datos
> (ESCOM-IPN).

Este documento describe **todo** el proyecto: cómo se usa, dónde queda cada archivo, el flujo
completo, cómo están implementados los modelos, qué bibliotecas se usan y en qué parte del
código vive cada cosa. Para una visión rápida, ver [README.md](README.md).

---

## Índice

1. [Visión general y filosofía](#1-visión-general-y-filosofía)
2. [Cómo ejecutarla](#2-cómo-ejecutarla)
3. [Stack y bibliotecas (qué y para qué)](#3-stack-y-bibliotecas-qué-y-para-qué)
4. [Arquitectura y mapa de archivos](#4-arquitectura-y-mapa-de-archivos)
5. [Dónde se guardan los archivos y los datos (persistencia)](#5-dónde-se-guardan-los-archivos-y-los-datos-persistencia)
6. [El estado de la sesión (`session_state`)](#6-el-estado-de-la-sesión-session_state)
7. [Flujo completo de uso, paso a paso](#7-flujo-completo-de-uso-paso-a-paso)
8. [Carga de datos en detalle (capa `datos`)](#8-carga-de-datos-en-detalle-capa-datos)
9. [Las 8 secciones de la app en detalle](#9-las-8-secciones-de-la-app-en-detalle)
10. [Cómo se implementan los modelos](#10-cómo-se-implementan-los-modelos)
11. [Evaluación de modelos](#11-evaluación-de-modelos)
12. [Codificación de categóricas y predicción interactiva](#12-codificación-de-categóricas-y-predicción-interactiva)
13. [Asistente IA (Ollama)](#13-asistente-ia-ollama)
14. [Tema visual (gráficas y CSS)](#14-tema-visual-gráficas-y-css)
15. [Decisiones de diseño importantes](#15-decisiones-de-diseño-importantes)
16. [Cómo extender la app](#16-cómo-extender-la-app)

---

## 1. Visión general y filosofía

- **Sin código**: el usuario nunca programa, ni abre una terminal, ni instala nada. Todo es UI.
- **Público**: estudiantes / personas no técnicas que quieren entender el flujo de un proyecto
  de datos (cargar → explorar → limpiar → visualizar → modelar → evaluar).
- **Separación por capas**: la UI (`main.py`) solo orquesta; toda la lógica vive en clases dentro
  de paquetes (`datos`, `preprocesamiento`, `modelos`, etc.). Cada capa es independiente y
  testeable por separado.
- **El panel se adapta**: en ⑥ Modelos, al elegir un modelo el formulario se redibuja y muestra
  solo los parámetros de ese modelo.

---

## 2. Cómo ejecutarla

Requisitos: **Python 3.11+** y las dependencias pinneadas en [requirements.txt](requirements.txt).

```powershell
# 1. Entrar al proyecto
cd DAAD

# 2. Crear y activar el entorno virtual
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar
streamlit run main.py
```

La app abre en `http://localhost:8501`. **Hay que cargar un dataset desde el panel lateral**
para que se habiliten las secciones (sin datos solo se ve la pantalla de bienvenida).

**Opcional — Asistente IA**: instala [Ollama](https://ollama.com) y descarga un modelo una vez:

```powershell
ollama run llama3.2
```

DAAD habla con Ollama en `http://localhost:11434`. Si Ollama no está corriendo, **el resto de la
app funciona igual**; solo se desactivan ⑧ Asistente IA y la explicación con IA de ⑦.

---

## 3. Stack y bibliotecas (qué y para qué)

| Biblioteca | Para qué se usa | Dónde |
|---|---|---|
| **streamlit** | Framework web: widgets, layout, estado de sesión (`st.session_state`), caché (`@st.cache_data`), gráficas. Es el corazón de la UI. | `main.py` |
| **pandas** | Estructura central de datos (`DataFrame`). Lectura de CSV/TSV/Excel/JSON/SQL/HTML, transformaciones. | en todo el proyecto |
| **numpy** | Operaciones numéricas y arrays (sklearn los exige; `to_numpy()` evita el `ArrowStringArray` de pandas 3). | modelos, evaluación |
| **scikit-learn** | Todos los modelos, los escaladores (`StandardScaler`, `MinMaxScaler`), el `LabelEncoder`, las métricas, `train_test_split`, validación cruzada, `GridSearchCV`, `Pipeline`, `permutation_importance`. | `modelos/`, `evaluacion/`, `preprocesamiento/` |
| **joblib** | Serializar el modelo entrenado (Pipeline escalado+estimador) para descargarlo como `.joblib`. | `main.py` (⑦) |
| **plotly** (express + graph_objects) | Todas las gráficas, con un template propio "daad". | `visualizacion/`, `chart_theme.py` |
| **psycopg2** (`psycopg2-binary`) | Conexión a **PostgreSQL**: listar tablas y cargarlas como DataFrame. | `datos/datos.py` |
| **openpyxl** | Motor que usa pandas para leer archivos **Excel** (`.xlsx`). | indirecto vía `pd.read_excel` |
| **cloudscraper** | Descargar páginas web para el **scraping** (sortea protecciones tipo Cloudflare). | `datos/datos.py` |
| **beautifulsoup4** + **lxml** | Parsear el HTML descargado: tablas, listas (`ul`/`ol`) y definiciones (`dl`). | `datos/datos.py` |
| **requests** | Hablar con la API REST local de **Ollama** (chat en streaming). | `asistente/asistente.py` |
| **pickle** (stdlib) | Serializar/restaurar el estado de trabajo en `resultados/_sesion.pkl`. | `main.py` |
| **threading** (stdlib) | Guardar la sesión en un hilo aparte para no bloquear la UI. | `main.py` |
| **Ollama** (servicio externo, opcional) | Modelo de lenguaje local que potencia el asistente. No es una librería de Python: es un programa que corre aparte. | externo |

---

## 4. Arquitectura y mapa de archivos

```
DAAD/
├── main.py                       # UI Streamlit: sidebar + las 8 secciones. Orquesta TODO.
├── styles.py                     # CSS del "design system" (apply_styles()).
├── chart_theme.py                # Registra el template "daad" de Plotly + la PALETA de colores.
├── requirements.txt              # Dependencias pinneadas.
├── README.md                     # Resumen breve.
├── DOCUMENTACION.md              # (este archivo) documentación técnica completa.
│
├── datos/
│   └── datos.py                  # Clase Datos: carga CSV/TSV, Excel, PostgreSQL, JSON, scraping.
│
├── analisis_exploratorio/
│   └── analisis_exploratorio.py  # Clase AnalisisExploratorio: resúmenes, outliers, correlación.
│
├── preprocesamiento/
│   └── preprocesamiento.py       # Clase Preprocesamiento: limpieza, escalado, codificación.
│
├── visualizacion/
│   └── visualizacion.py          # Clase Visualizacion: todas las figuras Plotly.
│
├── evaluacion/
│   └── evaluacion.py             # Clase Evaluacion: métricas de un modelo ya entrenado.
│
├── modelos/
│   ├── modelo.py                 # Modelo (ABC) + ModeloSupervisado (base común).
│   ├── modelo_kmeans.py          # K-Means (hereda de Modelo).
│   ├── modelo_knn.py             # … los 13 modelos supervisados (heredan ModeloSupervisado).
│   └── …                         # (ver la tabla de la sección 10)
│
├── asistente/
│   └── asistente.py              # Asistente IA local vía Ollama (prompt + streaming).
│
├── .streamlit/config.toml        # Tema de Streamlit.
└── resultados/                   # (se crea en runtime) NO versionada
    ├── _sesion.pkl               #   autoguardado del trabajo actual
    └── sesiones/<nombre>.pkl     #   sesiones con nombre (guardar/cargar manual)
```

**Regla de dependencia**: `main.py` importa de las capas; las capas **no** importan de `main.py`.
Cada paquete expone su clase principal en su `__init__.py`.

---

## 5. Dónde se guardan los archivos y los datos (persistencia)

Esta es una de las dudas más comunes. **Importante: DAAD casi nunca escribe los archivos que
subes al disco.** El flujo real es:

### 5.1. Archivos que subes (CSV, Excel, JSON…)
- `st.file_uploader` entrega el archivo **en memoria** (un objeto tipo `BytesIO`, NO una ruta en
  disco). DAAD lo lee directo con pandas (`pd.read_csv(fuente)`, `pd.read_excel(fuente)`, …) en
  [datos/datos.py](datos/datos.py).
- El resultado es un `DataFrame` que se guarda en **RAM**, dentro de
  `st.session_state.datasets[nombre]` (un diccionario `nombre → DataFrame`).
- **El archivo original nunca se copia a una carpeta del proyecto.** Si cierras la app sin
  persistir, ese archivo no queda en ningún lado del servidor.

### 5.2. PostgreSQL y web scraping
- **PostgreSQL**: las credenciales se escriben en los campos del panel, se usan para una consulta
  puntual y **no se guardan**. La tabla se trae como DataFrame a `datasets`.
- **Scraping**: la página se descarga, se parsea a DataFrames en memoria y, al "Cargar elemento",
  ese DataFrame entra a `datasets`. La página en sí no se guarda.

### 5.3. Lo único que SÍ se escribe en disco: `resultados/_sesion.pkl`
Para que no pierdas tu trabajo al recargar, DAAD serializa el **estado de trabajo** con `pickle`:
- **Archivo**: `resultados/_sesion.pkl` (se crea solo; la carpeta `resultados/` también).
- **Qué contiene** (las claves de `_CLAVES_SESION` en [main.py](main.py)):
  `datasets`, `df`, `df_original`, `df_activo`, `historiales`, `modelo_entrenado`, `modelo_tipo`,
  `metricas`, `cv_resultados`, `label_mappings`, `onehot_mappings`.
- **Cuándo**: tras cada cambio real (cargar, transformar, entrenar…), **no** en cada *rerun*.
- **Cómo** (función `guardar_sesion`):
  - **Asíncrono**: el volcado corre en un **hilo daemon** para no congelar la UI.
  - **Atómico**: escribe a `_sesion.pkl.tmp` y luego hace `os.replace` → nunca queda un `.pkl` a
    medio escribir aunque el proceso muera.
  - **Versionado**: un contador descarta una instantánea vieja que llegue tarde y pisaría a una
    más reciente.
- **Restauración** (`cargar_sesion`): al abrir la app, si existe el `.pkl`, se restaura. Si está
  corrupto, se borra y arranca limpio.
- **"Nueva sesión"** (botón en el sidebar, `borrar_sesion`): borra el `.pkl` y limpia el estado.

> ⚠️ **Nota de seguridad**: `pickle` ejecuta código arbitrario al deserializar. Es aceptable
> porque el archivo lo escribe solo esta app en local. **No cargues un `_sesion.pkl` de origen
> ajeno.** Si la app se expusiera en red, habría que migrar a un formato de datos (parquet).

### 5.4. Lo que NO se persiste
Resultados intermedios de carga (hojas de Excel, tablas JSON, elementos scrapeados) viven en
`session_state` pero **no** están en `_CLAVES_SESION`, así que se descartan al recargar.

### 5.5. Lo que se descarga (sale del servidor al navegador)
- **CSV del dataset** (① y ⑦): se genera en memoria y el navegador lo descarga a tu carpeta de
  descargas. Nombre de archivo = nombre del dataset activo (helper `_nombre_csv`).
- **Modelo entrenado** (⑦): `.joblib` con un `Pipeline` (escalado + estimador) + las features,
  el target y los mapeos de codificación.
- **Predicciones por lotes** (⑦): CSV con una columna de predicción añadida.

Estas descargas las maneja el navegador; **no** quedan en el servidor.

### 5.6. Sesiones con nombre (guardar/cargar manual)
Además del autoguardado (un único slot), puedes guardar **varios snapshots con nombre** desde la
sección **"Sesiones"** del panel lateral:
- **Guardar**: escribe un nombre y pulsa "Guardar sesión" → se crea `resultados/sesiones/<nombre>.pkl`
  con la misma instantánea que el autoguardado (`_snapshot_sesion`). Escritura atómica.
- **Cargar**: elige una sesión guardada y pulsa "Cargar" → restaura ese estado como tu trabajo
  actual (y se refleja también en el autoguardado). **Reemplaza** lo que tengas abierto.
- **Eliminar**: borra del disco la sesión seleccionada.
- El nombre se sanea (`_nombre_sesion_seguro`): se quitan caracteres inválidos y se evita la
  travesía de rutas, así que el archivo siempre queda dentro de `resultados/sesiones/`.

Funciones en `main.py`: `guardar_sesion_nombre`, `cargar_sesion_nombre`, `listar_sesiones`,
`eliminar_sesion_nombre`. Aplica la misma nota de seguridad de pickle del punto 5.3.

---

## 6. El estado de la sesión (`session_state`)

Streamlit re-ejecuta `main.py` completo en **cada** interacción. Para que nada se pierda entre
*reruns*, todo el estado vive en `st.session_state`. Claves principales:

| Clave | Contenido |
|---|---|
| `datasets` | `dict` `nombre → DataFrame` con todos los datasets cargados. |
| `df` | El DataFrame **activo** (sobre el que operan ②–⑦). |
| `df_activo` | Nombre del dataset activo (clave de `datasets`). |
| `df_original` | Copia del df al entrar a ④, para "Restaurar datos originales". |
| `historiales` | `dict` por dataset con la pila de transformaciones (para deshacer). |
| `modelo_entrenado` | El objeto modelo entrenado (instancia de una clase de `modelos/`). |
| `modelo_tipo` | `"clasificacion"`, `"regresion"` o `"clustering"`. |
| `metricas` | Dict de métricas calculadas al entrenar (se reusa en ⑦). |
| `cv_resultados` | Resultado de la validación cruzada, si se pidió. |
| `label_mappings` | `{columna: {categoría: código}}` de LabelEncoder. |
| `onehot_mappings` | `{columna: {categoría: nombre_dummy}}` de One-Hot. |
| `tablas_pg`, `tablas_json`, `hojas_excel`, `elementos_url` | Resultados intermedios de carga. |
| `_ultimo_archivo_id`, `_excel_file_id`, `_json_file_id` | IDs para no recargar el mismo archivo en cada *rerun*. |
| `chat_ia` | Historial del chat con el asistente. |

Helpers clave en `main.py`:
- `_cargar_dataset(nombre, df)` — registra un dataset, lo vuelve activo y persiste.
- `_guardar_df(nuevo)` — actualiza el df activo + el dict + persiste.
- `_aplicar_transformacion(desc, nuevo)` — aplica un cambio guardando el estado previo para
  deshacer; avisa si la operación no tuvo efecto.
- `_limpiar_modelo()` — olvida el modelo y los mapeos al cambiar de dataset (evita que ⑦ intente
  graficar features que ya no existen).
- `_sincronizar_mapeos_codificacion()` — poda mapeos que dejaron de reflejar el df (tras deshacer).

---

## 7. Flujo completo de uso, paso a paso

1. **Abrir la app** → si hay un `resultados/_sesion.pkl`, se restaura el trabajo previo; si no, se
   ve la pantalla de bienvenida.
2. **Cargar datos** (panel lateral): elegir formato (CSV/TSV/Excel/PostgreSQL en la pestaña
   *Estructurados*; JSON o scraping en *No estructurados*) y subir/conectar. El dataset entra a
   `datasets` y se vuelve el **activo**.
3. **Elegir dataset activo** (si hay varios): selector "Dataset activo" en el sidebar. Aquí también
   se elimina un dataset o se empieza una "Nueva sesión".
4. **① DataFrame**: ver la tabla, métricas (filas, columnas, nulos) y descargar CSV.
5. **② Conjunto** (si hay ≥2 datasets): combinar (concatenar o merge) o comparar lado a lado.
6. **③ Análisis exploratorio**: resumen, estadísticas, nulos, distribuciones, correlación, outliers.
7. **④ Preprocesamiento**: limpiar y transformar (nulos, duplicados, filtros, tipos, escalado,
   codificación). Cada paso se registra en el historial y se puede **deshacer**.
8. **⑤ Visualización**: graficar columnas.
9. **⑥ Modelos**: elegir modelo, features, target e hiperparámetros, y entrenar.
10. **⑦ Evaluación**: ver métricas, diagnóstico de confianza, importancias, predicción interactiva
    y por lotes, y descargar el modelo.
11. **⑧ Asistente IA**: preguntar al chat local sobre el dataset y cómo usar la app.

**Requisitos para entrenar (⑥)**: el dataset **no** debe tener nulos (trátalos en ④) y las features
deben ser **numéricas** (codifica las categóricas en ④). La app bloquea el botón si no se cumplen.

---

## 8. Carga de datos en detalle (capa `datos`)

Todo en [datos/datos.py](datos/datos.py), clase **`Datos`**. Cada método de carga devuelve el
resultado o `None` si falla, dejando la causa en `self.ultimo_error`.

| Fuente | Método | Notas |
|---|---|---|
| CSV | `cargar_csv(fuente)` | `pd.read_csv(sep=",")`. |
| TSV | `cargar_tsv(fuente)` | `pd.read_csv(sep="\t")`. |
| Excel | `cargar_excel(fuente)` | `pd.read_excel(sheet_name=None)` → `{hoja: DataFrame}`. El usuario elige hoja. |
| PostgreSQL | `listar_tablas(...)` + `cargar_tabla_sql(...)` | `psycopg2`; el nombre de tabla se escapa con `sql.Identifier` (anti-inyección). |
| JSON | `cargar_json(fuente)` | Recorre el JSON (aun anidado) y extrae cada lista de registros como tabla (`_extraer_tablas`, `pd.json_normalize`). |
| Web (scraping) | `cargar_url(url)` | Descarga con `cloudscraper`, parsea con BeautifulSoup. |

**Detalle del scraping** (`cargar_url`): extrae tres tipos de contenido y devuelve
`{etiqueta: {sección: DataFrame}}`:
- **Tablas** (`<table>`): se parsean con `pd.read_html`; se nombran con el encabezado (h2/h3/h4)
  que las precede; las columnas mayormente numéricas se convierten a número con `_a_numero`, que
  entiende convención española e inglesa (miles con espacio/punto, decimales con coma, porcentajes).
- **Listas** (`<ul>`/`<ol>`): se ignoran las de navegación (menús, header, footer).
- **Definiciones** (`<dl>`): pares término/definición.

Utilidades internas: `_limpiar` (quita referencias `[1]` y colapsa espacios), `_aplanar_columnas`
(encabezados multinivel y nombres repetidos), `_limpiar_celdas` (marcadores de dato faltante →
NaN), `_convertir_numericas` (convierte columnas de texto a número si ≥ 80 % parsea).

---

## 9. Las 8 secciones de la app en detalle

Todas viven en el bloque `# --- MAIN AREA ---` de [main.py](main.py), despachadas por
`if seccion == …`. Cada sección instancia la clase de la capa correspondiente.

### ① DataFrame
Vista de la tabla, métricas (filas/columnas/nulos) y descarga CSV (nombre = dataset activo).

### ② Conjunto (multi-dataset)
Dos pestañas:
- **Combinar**: *concatenar* (apilar filas; opción de columna "origen") o *merge* (cruzar dos
  datasets por una columna llave; tipo inner/left/right/outer). El resultado se registra como un
  **dataset nuevo y activo**, listo para analizar.
- **Comparar**: resumen lado a lado (filas, columnas, nulos por dataset) y, para una columna en
  común, estadísticas comparadas + histograma superpuesto (si es numérica) o tabla de frecuencias
  (si es categórica). Cacheado con `_comp_*`.

### ③ Análisis exploratorio
Clase **`AnalisisExploratorio`** ([analisis_exploratorio/](analisis_exploratorio/analisis_exploratorio.py)):
`resumen_general`, `estadisticas_descriptivas`, `conteo_nulos`, `valores_unicos`,
`distribucion_columna`, `matriz_correlacion` (Pearson), `detectar_outliers` (regla IQR).
Las costosas se cachean en `main.py` (`_estadisticas_descriptivas`, `_matriz_correlacion`, …).

### ④ Preprocesamiento
Clase **`Preprocesamiento`** ([preprocesamiento/](preprocesamiento/preprocesamiento.py)). Las
operaciones son **encadenables**: cada una parte del último resultado. Operaciones:
- `seleccionar_columnas`, `eliminar_nulos`, `eliminar_columnas_vacias` (por umbral de % nulos),
  `rellenar_nulos` (media/mediana/moda/constante, **con selección de columnas** para no imputar
  binarias/target/IDs; en *constante* el widget se adapta al tipo de las columnas elegidas —
  número para numéricas, **texto** para categóricas, p. ej. "No tiene"— y bloquea si se mezclan
  tipos), `eliminar_duplicados`, `filtrar_filas`, `convertir_tipo`,
  `normalizar_standard` / `normalizar_minmax` (**con selección de columnas**; el default excluye
  binarias 0/1), `codificar_categoricas` (LabelEncoder) y `codificar_onehot` (One-Hot).
- **Historial + deshacer**: cada cambio guarda el estado previo (`_aplicar_transformacion`); se
  puede deshacer paso a paso (hasta `MAX_HISTORIAL`) o restaurar el original.
- Codificaciones **preservan NaN**: no convierten un nulo en una categoría más.

### ⑤ Visualización
Clase **`Visualizacion`** ([visualizacion/](visualizacion/visualizacion.py)), figuras Plotly con el
template "daad": `histograma`, `boxplot`, `scatter`, `grafica_barras`, `grafica_lineas` (ordena por
X), `grafica_pastel`, `heatmap_correlacion`. (Métodos extra para modelos: ver §11.)

### ⑥ Modelos
Ver §10.

### ⑦ Evaluación
Ver §11.

### ⑧ Asistente IA
Ver §13.

---

## 10. Cómo se implementan los modelos

### 10.1. La arquitectura: una base, muchos modelos

En [modelos/modelo.py](modelos/modelo.py):

- **`Modelo(ABC)`** — contrato mínimo: `entrenar`, `predecir`, `evaluar`. Del que cuelga K-Means
  (no supervisado, sin target).
- **`ModeloSupervisado(Modelo)`** — concentra **toda** la lógica común de los modelos con target:
  - `entrenar(test_size, escalar, tuning)`: hace `train_test_split` (estratificado si es
    clasificación), escala **solo con train** si `escalar=True`, y entrena. Si `tuning=True`,
    corre `GridSearchCV` sobre train (con la rejilla del modelo) y se queda con el mejor.
  - `predecir` / `predecir_proba`: aplican el scaler si existe.
  - `evaluar`: delega en la clase `Evaluacion` y añade diagnóstico de sobreajuste
    (`_train_vs_test`), línea base (`_baseline` con `DummyClassifier`/`DummyRegressor`),
    importancia de features (`_importancias`: nativa si el modelo la expone, si no por
    permutación) y coeficientes (`_coeficientes`, solo modelos lineales).
  - `validacion_cruzada`: k-fold (estratificado en clasificación), con el scaler dentro del
    pipeline para no filtrar datos entre folds.

  **Cada subclase solo declara dos cosas:**
  ```python
  class ModeloX(ModeloSupervisado):
      es_clasificacion = True          # o False para regresión
      scoring_cv = "accuracy"          # o "r2"
      def _crear_estimador(self):      # devuelve el estimador de scikit-learn
          return EstimadorDeSklearn(...)
      def _rejilla_busqueda(self):     # (opcional) hiperparámetros para GridSearch
          return {...}
  ```

### 10.2. Los 14 modelos disponibles

| Modelo (en la UI) | Archivo | Clase | Estimador sklearn | Tarea | Hiperparámetros en el panel |
|---|---|---|---|---|---|
| KNN — K-Nearest Neighbors | `modelo_knn.py` | `ModeloKNN` | `KNeighborsClassifier` | Clasificación | k, peso, métrica |
| Árbol de Decisión | `modelo_arbol_decision.py` | `ModeloArbolDecision` | `DecisionTreeClassifier` | Clasificación | profundidad, min. hoja, criterio |
| Random Forest | `modelo_random_forest.py` | `ModeloRandomForest` | `RandomForestClassifier` | Clasificación | nº árboles, profundidad |
| Gradient Boosting | `modelo_gradient_boosting.py` | `ModeloGradientBoosting` | `GradientBoostingClassifier` | Clasificación | nº árboles, learning rate, profundidad |
| Regresión Logística | `modelo_regresion_logistica.py` | `ModeloRegresionLogistica` | `LogisticRegression` | Clasificación | max iter, C, balanceo |
| Naive Bayes | `modelo_naive_bayes.py` | `ModeloNaiveBayes` | `GaussianNB` | Clasificación | (ninguno) |
| Red Neuronal | `modelo_red_neuronal.py` | `ModeloRedNeuronal` | `MLPClassifier` | Clasificación | tamaño de red, max iter |
| Regresión Lineal | `modelo_regresion_lineal.py` | `ModeloRegresionLineal` | `LinearRegression`/`Ridge`/`Lasso` | Regresión | regularización, alpha |
| KNN (regresión) | `modelo_knn_regresion.py` | `ModeloKNNRegresion` | `KNeighborsRegressor` | Regresión | k, peso, métrica |
| Árbol de Decisión (regresión) | `modelo_arbol_regresion.py` | `ModeloArbolRegresion` | `DecisionTreeRegressor` | Regresión | profundidad, min. hoja |
| Random Forest (regresión) | `modelo_random_forest_regresion.py` | `ModeloRandomForestRegresion` | `RandomForestRegressor` | Regresión | nº árboles, profundidad |
| Gradient Boosting (regresión) | `modelo_gradient_boosting_regresion.py` | `ModeloGradientBoostingRegresion` | `GradientBoostingRegressor` | Regresión | nº árboles, learning rate, profundidad |
| Red Neuronal (regresión) | `modelo_red_neuronal_regresion.py` | `ModeloRedNeuronalRegresion` | `MLPRegressor` | Regresión | tamaño de red, max iter |
| K-Means (clustering) | `modelo_kmeans.py` | `ModeloKMeans` | `KMeans` | Clustering | nº clusters k, curva del codo |

### 10.3. El "cerrojo" de tarea (`TAREA_MODELO`)

En [main.py](main.py), el diccionario **`TAREA_MODELO`** mapea cada etiqueta del selector a su
tarea (`clasificacion`/`regresion`/`clustering`). Es la **única fuente de verdad** del tipo de
problema: gobierna la validación del target y la vista de resultados en ⑦. Sin él, un regresor
nuevo se trataría como clasificación e intentaría dibujar una matriz de confusión sobre una salida
continua. **Para añadir un modelo hay que registrarlo aquí.**

### 10.4. Sin fuga de datos (data leakage)

- El `train_test_split` se hace **antes** de escalar; el `StandardScaler` se ajusta **solo con
  train** y solo transforma test.
- En validación cruzada y en `GridSearchCV`, el scaler entra al `Pipeline` → se reajusta dentro
  de cada fold. La búsqueda de hiperparámetros usa solo train; el test queda intacto para una
  evaluación honesta.
- Los clasificadores **estratifican** el split (conservan la proporción de clases).

---

## 11. Evaluación de modelos

Clase **`Evaluacion`** ([evaluacion/evaluacion.py](evaluacion/evaluacion.py)): recibe el modelo y
`y_real` vs `y_predicho`, y calcula el conjunto de métricas según la tarea.

- **Clasificación**: accuracy, precision, recall, F1 (todas *weighted*), matriz de confusión,
  reporte por clase y **ROC-AUC** (si el modelo da `predict_proba`; binario o one-vs-rest).
- **Regresión**: R², R² ajustado, MAE, MSE, RMSE y **MAPE** (se omite si el target tiene ceros,
  para no dividir entre cero).
- **Clustering** (en el propio `ModeloKMeans`): inercia, silhouette, nº de clusters.

En **⑦ Evaluación** ([main.py](main.py)) se muestra, además de las métricas:
- **Confianza del modelo**: métrica en train vs test (avisa de sobreajuste si la brecha es grande)
  y comparación contra una **línea base** trivial (Dummy); avisa si el modelo no la supera.
- **Coeficientes** (modelos lineales) e **importancia de features** (nativa o por permutación).
- **Gráficas**: matriz de confusión (clasificación); real-vs-predicho y **residuos** (regresión);
  clusters y curva del codo (clustering). Métodos en `Visualizacion`: `grafica_confusion`,
  `grafica_regresion`, `grafica_residuos`, `grafica_clusters`, `grafica_codo`.
- **Predicción interactiva** (ver §12), **predicción por lotes** (sube un CSV con las features) y
  **descarga del modelo** como `Pipeline` `.joblib`.
- **Explicación con IA** (si hay Ollama): manda las métricas al asistente para una lectura en
  lenguaje sencillo.

---

## 12. Codificación de categóricas y predicción interactiva

Para entrenar, las features deben ser numéricas, así que en ④ se codifican las categóricas:
- **LabelEncoder** (`codificar_categoricas`): cada categoría → un entero. Guarda
  `mapeos_label = {columna: {categoría: código}}`.
- **One-Hot** (`codificar_onehot`): una columna binaria por categoría. Guarda
  `mapeos_onehot = {columna: {categoría: nombre_dummy}}`.

Estos mapeos se copian a `session_state` (`label_mappings`, `onehot_mappings`) y permiten que la
**predicción interactiva** de ⑦ sea usable: en vez de pedir el entero o el 0/1 de cada dummy,
muestra un **selector con los nombres de categoría** y traduce por dentro al código/dummy correcto.
Los mapeos se podan al deshacer/restaurar (`_sincronizar_mapeos_codificacion`) y se descartan al
cambiar de dataset (`_limpiar_modelo`), para no ofrecer categorías de columnas que ya no existen.

---

## 13. Asistente IA (Ollama)

En [asistente/asistente.py](asistente/asistente.py):
- **`ollama_disponible()`** — comprueba si Ollama corre en `http://localhost:11434` (no falla si no).
- **`SISTEMA_DATASET` / `CAPACIDADES_APP`** — un *system prompt* extenso que le enseña al modelo
  **exactamente** qué puede y qué no puede hacer la app (las 8 secciones, los modelos, los
  requisitos), con reglas estrictas para que no invente funciones ni modelos inexistentes.
- **`resumen_dataset(df)`** — arma un resumen compacto del dataset activo (dimensiones, columnas,
  tipos, nulos, estadísticas) que se inyecta como contexto.
- **`chat_stream(mensajes, modelo)`** — habla con la API REST de Ollama en *streaming* (token a
  token), con `temperatura` baja para que se ciña a las funciones reales.
- **`prompt_explicacion(...)`** — arma el mensaje para que la IA explique las métricas de un modelo
  recién entrenado.

El chat conoce el dataset activo, pero **no** ejecuta acciones: solo guía al usuario.

---

## 14. Tema visual (gráficas y CSS)

- **[chart_theme.py](chart_theme.py)**: registra el template **"daad"** de Plotly (colores, fuentes
  IBM Plex, fondos, leyenda) y la `PALETTE`. Se importa una vez en `main.py` y todas las gráficas
  lo usan con `template="daad"`.
- **[styles.py](styles.py)**: `apply_styles()` inyecta el CSS del *design system* (la estética de
  "navegador retro", encabezados, tarjetas, etc.).
- **`.streamlit/config.toml`**: tema base de Streamlit.

---

## 15. Decisiones de diseño importantes

- **Capas independientes**: la lógica vive en clases; `main.py` solo orquesta. Facilita probar y
  extender.
- **Estado en `session_state` + persistencia atómica/asíncrona**: el trabajo sobrevive a recargas
  sin bloquear la UI.
- **Sin fuga de datos** en el escalado, la CV y la búsqueda de hiperparámetros (ver §10.4).
- **Caché** (`@st.cache_data`) en las operaciones costosas de EDA, comparación y descarga, con
  `hash_funcs` para DataFrames.
- **Robustez**: cada carga devuelve `None` + `ultimo_error` en vez de reventar; el guardado nunca
  tumba la app; los mapeos se podan al cambiar el dataset.
- **Escalado y relleno selectivos**: ni `normalizar_*` ni `rellenar_nulos` se aplican a ciegas a
  todas las columnas; el usuario elige cuáles, y el default ya excluye las binarias.

---

## 16. Cómo extender la app

**Añadir un modelo supervisado nuevo** (lo más común):
1. Crear `modelos/modelo_xxx.py` con una subclase de `ModeloSupervisado`: define
   `es_clasificacion`, `scoring_cv`, `_crear_estimador()` y, opcional, `_rejilla_busqueda()`.
2. Exportarla en `modelos/__init__.py`.
3. Importarla en `main.py` y registrarla en: el `selectbox` de ⑥, el diccionario **`TAREA_MODELO`**,
   el panel de hiperparámetros (si tiene) y el *dispatch* del botón "Entrenar".
4. (Recomendado) Documentarla en `asistente/asistente.py` para que el chat la conozca.

Toda la infraestructura (split, escalado, métricas, evaluación, predicción, export, CV, tuning) se
hereda; no hay que tocarla.

**Añadir una fuente de datos**: un método nuevo en `Datos` que devuelva un `DataFrame` (o `None` +
`ultimo_error`) y un bloque de UI en el sidebar que llame a `_cargar_dataset`.

**Añadir una gráfica**: un método en `Visualizacion` que devuelva una `go.Figure` con
`template="daad"`, y una opción en ⑤.

---

*Documento generado para la versión final del proyecto. Para el resumen breve, ver
[README.md](README.md).*
