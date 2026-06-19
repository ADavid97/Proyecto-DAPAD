"""Asistente de IA local vía Ollama (gratis, sin cuenta, sin internet).

Requiere tener Ollama corriendo en http://localhost:11434 y al menos un modelo
descargado (p.ej. `ollama run llama3.2`). Se comunica por la API REST de Ollama
con `requests`, que ya está disponible en el entorno.
"""

import json

import requests
import pandas as pd

OLLAMA_HOST = "http://localhost:11434"
MODELO_POR_DEFECTO = "llama3.2"

CAPACIDADES_APP = """\
DAAD es una aplicación web (hecha en Streamlit) para analizar datos SIN escribir código.
Todo se hace con botones y menús. El usuario NO programa, NO usa la terminal y NO instala
nada. Estas son las únicas funciones que existen en la aplicación:

- Carga de datos (panel lateral izquierdo): cargar CSV, TSV, Excel (varias hojas),
  PostgreSQL, JSON anidado o web scraping de tablas. Se puede tener varios datasets a la
  vez y elegir el dataset activo con un selector en el panel lateral.
- Sección "① DataFrame": ver la tabla del dataset activo, métricas básicas (filas, columnas,
  valores nulos) y descargar el dataset en CSV.
- Sección "② Conjunto": combinar varios datasets cargados en uno solo. Tiene dos pestañas:
  "Combinar" (concatenar/apilar datasets con columnas similares, o hacer un merge/cruce de dos
  datasets por una columna llave en común) y "Comparar" (ver lado a lado estadísticas de varios
  datasets y un histograma superpuesto de una columna en común). El resultado de "Combinar" se
  registra como un dataset nuevo y activo, listo para analizar en las demás secciones.
- Sección "③ Análisis exploratorio": resumen general, estadísticas descriptivas, conteo de
  nulos, valores únicos, distribuciones, matriz de correlación y detección de outliers (IQR).
- Sección "④ Preprocesamiento": seleccionar columnas, tratar nulos (eliminar o rellenar),
  quitar duplicados, aplicar filtros, convertir tipos de datos, escalar y codificar variables
  (One-Hot o LabelEncoder). Tiene historial con opción de deshacer paso a paso.
- Sección "⑤ Visualización": SOLO sirve para graficar columnas: histograma, boxplot,
  dispersión (scatter), barras, líneas, pastel y mapa de calor de correlación. AQUÍ NO se
  entrenan modelos ni existe ninguna "gráfica de regresión" ni "línea de tendencia".
- Sección "⑥ Modelos": AQUÍ se entrena CUALQUIER modelo, incluida la Regresión Lineal.
  Modelos disponibles, agrupados por tarea: para CLASIFICAR (predecir una categoría) → KNN,
  Árbol de Decisión, Random Forest, Gradient Boosting, Regresión Logística, Naive Bayes y Red
  Neuronal. Para REGRESIÓN (predecir un número) → Regresión Lineal, KNN (regresión), Árbol de
  Decisión (regresión), Random Forest (regresión), Gradient Boosting (regresión) y Red Neuronal
  (regresión). Para AGRUPAR sin target → K-Means.
  EL PANEL SE ADAPTA AL MODELO ELEGIDO: en cuanto el usuario elige el tipo de modelo
  en el selector, el panel se redibuja y muestra ÚNICAMENTE los parámetros relevantes para ese
  modelo (los que no aplican simplemente no aparecen). Por modelo, esto es lo que se ve:
    * Para todos menos K-Means: Features (columnas X, numéricas), Target (columna y),
      Proporción de prueba, casilla "Escalar features" y, opcional, "Validación cruzada
      (k-fold)" con número de folds.
    * Hiperparámetros extra que aparecen SOLO para ese modelo: KNN → k (vecinos), tipo de peso
      y métrica de distancia; Árbol de Decisión → profundidad máxima, mínimo de muestras por
      hoja y criterio; Random Forest → número de árboles y profundidad máxima; Gradient Boosting
      → número de árboles, tasa de aprendizaje y profundidad máxima; Regresión Logística →
      máximo de iteraciones, C (regularización) y balanceo de clases; Regresión Lineal →
      regularización (ninguna/Ridge/Lasso) y, si es Ridge/Lasso, alpha; Naive Bayes → NINGUNO
      (no se muestra ningún campo extra); Red Neuronal (y su versión de regresión) → tamaño de
      la red (capas ocultas) y máximo de iteraciones; K-Means → número de clusters k y, opcional,
      curva del codo. Con K-Means el panel OCULTA Target, Proporción de prueba, Escalar features
      y Validación cruzada, porque usa solo Features y no hay objetivo.
    * Todos los modelos supervisados ofrecen además una casilla opcional "Buscar mejores
      parámetros" (búsqueda automática por validación cruzada).
  Las variantes "(regresión)" de KNN, Árbol, Random Forest y Gradient Boosting usan los MISMOS
  hiperparámetros que su versión de clasificación (el Árbol de regresión no tiene "criterio").
  Después se pulsa el botón de entrenar. Los modelos de regresión requieren un Target numérico.
- Sección "⑦ Evaluación": DESPUÉS de entrenar en ⑥, aquí se ven los resultados del modelo:
  métricas (clasificación, regresión o clustering), matriz de confusión, reporte, importancia
  de features, predicciones interactivas y descarga del modelo entrenado (.joblib).
- Sección "⑧ Asistente IA": este chat de ayuda.

REQUISITOS Y ELECCIÓN DE CADA MODELO (todo en ⑥ Modelos):
Antes de entrenar CUALQUIER modelo, dos requisitos obligatorios: (a) el dataset NO debe tener
valores nulos —si los hay, trátalos en ④ Preprocesamiento— y (b) las Features deben ser
numéricas —si son texto o categorías, codifícalas en ④ (One-Hot o LabelEncoder)—.

- Regresión Lineal → PREDICE UN NÚMERO CONTINUO (ej.: ingreso, precio, un puntaje).
  Requiere: Features numéricas y un Target NUMÉRICO. No sirve si el Target es una categoría.
  Sin hiperparámetros extra.
- Regresión Logística → CLASIFICA en categorías (ej.: sí/no, aprobado/reprobado).
  Requiere: Features numéricas y un Target categórico/discreto; ideal binario (2 clases),
  también admite varias clases. Hiperparámetro: máximo de iteraciones.
- KNN → CLASIFICA en categorías por cercanía a los vecinos más parecidos.
  Requiere: Features numéricas y un Target categórico/discreto. Es sensible a la escala, así
  que conviene activar "Escalar features". Hiperparámetro: k (número de vecinos).
- Árbol de Decisión → CLASIFICA en categorías y es fácil de interpretar (da importancia de
  cada feature). Requiere: Features numéricas y un Target categórico/discreto. No necesita
  escalado. Hiperparámetros: profundidad máxima, mínimo de muestras por hoja y criterio.
- Random Forest → CLASIFICA con un ensamble de muchos árboles; suele ser bastante más preciso
  y estable que un árbol único y da importancia de features. Requiere: Features numéricas y un
  Target categórico/discreto. No necesita escalado. Hiperparámetros: número de árboles y
  profundidad máxima.
- Gradient Boosting → CLASIFICA encadenando árboles que corrigen el error del anterior; suele
  dar la mejor exactitud en datos tabulares, a cambio de más cómputo. Requiere: Features
  numéricas y un Target categórico/discreto. No necesita escalado. Hiperparámetros: número de
  árboles, tasa de aprendizaje y profundidad máxima.
- Naive Bayes → CLASIFICA con probabilidades, asumiendo que las features son independientes.
  Muy rápido y sin hiperparámetros; buena línea de base. Requiere: Features numéricas y un Target
  categórico/discreto.
- Red Neuronal (perceptrón multicapa) → CLASIFICA aprendiendo relaciones no lineales con capas
  ocultas. Requiere: Features numéricas y un Target categórico/discreto. Es MUY sensible a la
  escala: hay que activar "Escalar features". En datasets pequeños suele perder contra los
  árboles y puede avisar de que "no convergió" (subir el máximo de iteraciones). Hiperparámetros:
  tamaño de la red y máximo de iteraciones.
- KNN / Árbol / Random Forest / Gradient Boosting / Red Neuronal (regresión) → las MISMAS técnicas
  pero para PREDECIR UN NÚMERO en vez de una categoría. Requieren Features numéricas y un Target
  NUMÉRICO. Útiles cuando la relación no es lineal (a diferencia de la Regresión Lineal); Random
  Forest y Gradient Boosting de regresión suelen ser los más precisos. No necesitan escalado,
  salvo KNN y Red Neuronal, que sí lo requieren.
- K-Means → AGRUPA en clusters (no supervisado); descubre grupos por similitud.
  Requiere: solo Features numéricas; NO se elige Target. Hiperparámetro: número de clusters k.

Cómo orientar la elección según lo que quiera el usuario:
  * Predecir un NÚMERO continuo → un modelo de regresión: Regresión Lineal si la relación es
    lineal; KNN, Árbol, Random Forest o Gradient Boosting (regresión) si no lo es (RF/GB suelen
    dar más precisión).
  * Predecir una CATEGORÍA/clase → Regresión Logística (sobre todo si son 2 clases), KNN,
    Árbol de Decisión, Naive Bayes (rápido, buena línea de base), o para más precisión Random
    Forest, Gradient Boosting o Red Neuronal.
  * NO tiene variable objetivo y quiere segmentar/agrupar → K-Means.
Importante: los clasificadores (KNN, Árbol, Regresión Logística) necesitan un Target con clases
discretas; si el Target es continuo, usa un modelo de regresión o discretízalo antes en ④.
"""

SISTEMA_DATASET = (
    "Eres el asistente de ayuda integrado en DAAD. Tu único propósito es GUIAR al usuario "
    "para que use la aplicación. Respondes siempre en español, de forma clara y concisa.\n\n"
    + CAPACIDADES_APP
    + "\nReglas estrictas que debes cumplir SIEMPRE:\n"
    "1. Solo recomienda acciones que existan en la lista de funciones de arriba, indicando "
    "en qué sección se hacen (p. ej. 've a ④ Preprocesamiento y usa la opción de rellenar "
    "nulos'). Describe la sección y la acción de forma general; NO inventes nombres exactos "
    "de botones, submenús, pestañas ni pasos numerados que no aparezcan en la lista.\n"
    "2. NUNCA sugieras escribir código, usar pandas, abrir una terminal, ejecutar comandos, "
    "instalar librerías ni editar archivos. El usuario solo usa la interfaz con botones.\n"
    "3. Los ÚNICOS modelos que existen en la app son: KNN, Árbol de Decisión, Random Forest, "
    "Gradient Boosting, Regresión Logística, Naive Bayes, Red Neuronal (perceptrón multicapa), "
    "Regresión Lineal, las variantes de regresión de KNN/Árbol/Random Forest/Gradient Boosting/"
    "Red Neuronal, y K-Means. La Red Neuronal es un perceptrón multicapa (MLP) sencillo: NO hay "
    "redes convolucionales, recurrentes ni deep learning con GPU. Cualquier otro modelo (SVM, "
    "XGBoost, etc.) NO existe en DAAD: dilo claramente y sugiere el más parecido de la lista "
    "(por ejemplo, Gradient Boosting en lugar de XGBoost, o Random Forest en lugar de un "
    "ensamble que no esté disponible).\n"
    "4. Si el usuario pide cualquier otra cosa que la app NO puede hacer, dilo con claridad "
    "y ofrece la alternativa más cercana dentro de DAAD. Nunca inventes funciones.\n"
    "5. Cuando se te dé un resumen del dataset cargado, básate solo en esa información y no "
    "inventes valores ni columnas que no aparezcan en el resumen.\n"
    "6. Sé breve y directo; ve al grano con los pasos que el usuario debe seguir.\n"
    "7. Para entrenar o aplicar CUALQUIER modelo (incluida la Regresión Lineal) el usuario "
    "SIEMPRE va a ⑥ Modelos, y ve los resultados en ⑦ Evaluación. ⑤ Visualización es SOLO "
    "para graficar columnas: NUNCA mandes ahí a entrenar un modelo ni hables de una 'gráfica "
    "de regresión' porque no existe.\n"
    "8. Al explicar cómo configurar un modelo, menciona ÚNICAMENTE los parámetros que aparecen "
    "en la descripción de ⑥ Modelos. NO inventes opciones como 'intercepto', 'escala lineal o "
    "logística', 'menú del gráfico' ni pasos numerados que no existan.\n"
    "9. Cuando el usuario ya haya elegido (o mencione) un modelo concreto, NO des una respuesta "
    "genérica: enumera de forma explícita SOLO los parámetros que ese modelo necesita y que el "
    "panel mostrará para él, y deja claro que el panel se adapta —es decir, que solo verá esos "
    "campos y no los de otros modelos—. Por ejemplo, para K-Means aclara que NO verá Target, "
    "Proporción de prueba, Escalar features ni Validación cruzada, solo Features, k y la curva "
    "del codo; para Regresión Lineal aclara que no hay ningún hiperparámetro extra que ajustar. "
    "Si el usuario aún no ha dicho qué modelo quiere, pregúntale o ayúdale a elegirlo antes de "
    "listar parámetros, en lugar de describir todos los modelos a la vez."
)


def ollama_disponible() -> tuple[bool, list[str]]:
    """Devuelve (disponible, lista_de_modelos). No lanza excepción si Ollama no corre."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        r.raise_for_status()
        modelos = [m["name"] for m in r.json().get("models", [])]
        return True, modelos
    except Exception:
        return False, []


def resumen_dataset(df: pd.DataFrame, max_cols: int = 40) -> str:
    """Construye un resumen compacto del DataFrame para dar contexto al modelo."""
    lineas = [f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas.", "", "Columnas (nombre · tipo · nulos):"]
    for col in df.columns[:max_cols]:
        nulos = int(df[col].isnull().sum())
        lineas.append(f"- {col} · {df[col].dtype} · {nulos} nulos")
    if df.shape[1] > max_cols:
        lineas.append(f"… y {df.shape[1] - max_cols} columnas más.")

    numericas = df.select_dtypes(include="number")
    if not numericas.empty:
        lineas += ["", "Estadísticas de columnas numéricas:", numericas.describe().round(3).T.to_string()]
    return "\n".join(lineas)


def mensajes_chat(df: pd.DataFrame, historial: list[dict]) -> list[dict]:
    """Arma la lista de mensajes (sistema + contexto del dataset + historial) para el chat."""
    sistema = f"{SISTEMA_DATASET}\n\nResumen del dataset cargado:\n{resumen_dataset(df)}"
    return [{"role": "system", "content": sistema}, *historial]


def _veredicto_sobreajuste(tt: dict) -> str:
    """Decide en Python si hay sobreajuste comparando train vs test (mayor = mejor).

    No lo dejamos al criterio del modelo de lenguaje: modelos pequeños suelen
    invertir la regla (afirmar sobreajuste cuando train≈test, que es justo lo
    contrario). Aquí se calcula el veredicto y se le entrega ya cerrado.
    """
    metrica, train, test = tt["metrica"], tt["train"], tt["test"]
    gap = train - test
    if gap >= 0.15:
        estado = "HAY SOBREAJUSTE CLARO"
        detalle = ("el rendimiento en entrenamiento es bastante mejor que en prueba, "
                   "así que el modelo memoriza los datos de entrenamiento y generaliza peor")
    elif gap >= 0.05:
        estado = "HAY ALGO DE SOBREAJUSTE"
        detalle = ("el rendimiento baja algo de entrenamiento a prueba; conviene vigilarlo, "
                   "pero no es grave")
    else:
        estado = "NO HAY SOBREAJUSTE"
        detalle = ("el rendimiento en entrenamiento y en prueba es casi igual, "
                   "así que el modelo generaliza bien (que se parezcan es BUENO, no malo)")
    return (
        f"VEREDICTO SOBREAJUSTE (ya calculado; explícalo, NO lo contradigas ni lo recalcules): "
        f"{estado}. {metrica} train={train:.4f} vs test={test:.4f} (diferencia {gap:+.4f}); {detalle}."
    )


def _veredicto_baseline(bl: dict, modelo_test: float | None) -> str:
    """Decide en Python si el modelo supera la línea base (referencia trivial)."""
    metrica, base = bl["metrica"], bl["valor"]
    if modelo_test is None:
        return (f"Línea base ({bl['estrategia']}): {metrica} = {base:.4f}. "
                "El modelo solo aporta valor si supera claramente esta referencia.")
    diff = modelo_test - base
    if diff <= 0:
        estado = "NO SUPERA la línea base"
        detalle = "el modelo no aporta valor: la referencia trivial es igual o mejor"
    elif diff < 0.05:
        estado = "apenas SUPERA la línea base"
        detalle = "el modelo aporta poco por encima de la referencia trivial"
    else:
        estado = "SUPERA la línea base"
        detalle = "el modelo aporta valor real por encima de la referencia trivial"
    return (
        f"VEREDICTO LÍNEA BASE (ya calculado; explícalo, NO lo contradigas): {estado}. "
        f"{metrica} del modelo={modelo_test:.4f} vs línea base ({bl['estrategia']})={base:.4f} "
        f"(diferencia {diff:+.4f}); {detalle}."
    )


def _veredicto_fuerza_regresion(test_r2: float) -> str:
    """Interpreta el R² de prueba en términos absolutos (solo regresión)."""
    if test_r2 >= 0.75:
        nivel = "FUERTE: explica la mayor parte de la variabilidad del objetivo"
    elif test_r2 >= 0.5:
        nivel = "MODERADO: explica una parte razonable de la variabilidad"
    elif test_r2 >= 0.25:
        nivel = "DÉBIL: explica poca variabilidad (subajuste); faltan features útiles"
    else:
        nivel = ("MUY DÉBIL: explica muy poca variabilidad (subajuste claro); "
                 "el problema NO es sobreajuste sino que el modelo se queda corto")
    return (
        f"VEREDICTO FUERZA (ya calculado; el R² mide la proporción de la VARIANZA explicada, "
        f"no el ruido): R² de prueba={test_r2:.4f} → modelo {nivel}."
    )


def prompt_explicacion(tipo: str, metricas: dict, importancias: dict | None = None,
                       target: str | None = None) -> str:
    """Genera el mensaje del usuario para que la IA explique los resultados del modelo.

    Los juicios de fondo (sobreajuste, línea base y fuerza del modelo) se calculan
    aquí en Python y se inyectan ya cerrados, para que la IA solo los redacte y no
    los razone (evita que un modelo pequeño los invierta o alucine).
    """
    # Claves con estructura propia: se presentan aparte, no en la lista de métricas sueltas.
    aparte = {"train_test", "baseline", "importancias", "importancias_tipo",
              "coeficientes", "confusion_matrix", "reporte"}
    es_regresion = tipo.lower().startswith("regres")
    partes = [f"Acabo de entrenar un modelo de {tipo}."]
    if target:
        partes.append(f"La variable objetivo es '{target}'.")

    partes.append("Métricas en el conjunto de prueba:")
    for clave, valor in metricas.items():
        if clave not in aparte and isinstance(valor, (int, float)):
            partes.append(f"- {clave}: {valor:.4f}")

    tt = metricas.get("train_test")
    tt = tt if isinstance(tt, dict) else None
    if tt is not None:
        partes.append(
            f"Rendimiento en entrenamiento vs prueba ({tt['metrica']}): "
            f"train = {tt['train']:.4f}, test = {tt['test']:.4f}."
        )
        partes.append(_veredicto_sobreajuste(tt))
        if es_regresion and tt["metrica"] == "R²":
            partes.append(_veredicto_fuerza_regresion(tt["test"]))

    bl = metricas.get("baseline")
    if isinstance(bl, dict):
        modelo_test = tt["test"] if tt and tt.get("metrica") == bl["metrica"] else None
        partes.append(_veredicto_baseline(bl, modelo_test))

    if importancias:
        partes.append("Importancia de las features (mayor = más influyente):")
        for feat, peso in sorted(importancias.items(), key=lambda x: x[1], reverse=True):
            partes.append(f"- {feat}: {float(peso):.4f}")

    if tt is not None or isinstance(bl, dict):
        cierre = (
            "\nRedacta en lenguaje sencillo (máximo ~150 palabras) qué significan estos resultados. "
            "USA el contenido de los VEREDICTOS de arriba sobre sobreajuste, línea base y fuerza del "
            "modelo: tu tarea es explicarlos, NO recalcularlos ni cuestionarlos ni invertirlos. PERO "
            "escribe en prosa natural y fluida para el usuario: NO copies la palabra 'VEREDICTO' ni "
            "esos rótulos en mayúsculas, no los uses como títulos ni los menciones como tales; "
            "intégralos como parte de la explicación. No inventes valores ni features que no aparezcan "
            "arriba. Cierra con 1-2 ideas de mejora coherentes con lo anterior."
        )
    else:
        cierre = (
            "\nExplica en lenguaje sencillo qué significan estos resultados, si el modelo "
            "es bueno o no y qué se podría mejorar. Máximo ~150 palabras."
        )
    partes.append(cierre)
    return "\n".join(partes)


def chat_stream(mensajes: list[dict], modelo: str = MODELO_POR_DEFECTO, temperatura: float = 0.2):
    """Generador que produce la respuesta del modelo token a token (para st.write_stream).

    `temperatura` baja (0.2) reduce la creatividad para que el asistente se ciña a las
    funciones reales de la app y no invente pasos ni modelos inexistentes.
    """
    try:
        respuesta = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": modelo,
                "messages": mensajes,
                "stream": True,
                "options": {"temperature": temperatura},
            },
            stream=True,
            timeout=120,
        )
        respuesta.raise_for_status()
    except Exception as e:
        yield (
            f"No pude conectar con Ollama ({e}).\n\n"
            "Verifica que Ollama esté abierto y que hayas descargado un modelo "
            "con `ollama run llama3.2`."
        )
        return

    for linea in respuesta.iter_lines():
        if not linea:
            continue
        try:
            dato = json.loads(linea)
        except json.JSONDecodeError:
            continue
        contenido = dato.get("message", {}).get("content", "")
        if contenido:
            yield contenido
        if dato.get("done"):
            break
