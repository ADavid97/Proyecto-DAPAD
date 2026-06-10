import io
import json
import re
import pandas as pd
import psycopg2
from psycopg2 import sql
import cloudscraper
from bs4 import BeautifulSoup


class Datos:
    """Capa de carga de datos: CSV/TSV, Excel, PostgreSQL, JSON anidado y web scraping.

    Cuando un método de carga falla devuelve None y deja la causa en `ultimo_error`.
    """

    def __init__(self, ruta_archivo: str = ""):
        self.ruta_archivo = ruta_archivo
        self.datos_crudos = None
        self.separador = ","
        self.ultimo_error: str | None = None

    def cargar_csv(self, fuente) -> pd.DataFrame | None:
        """Lee un CSV desde una ruta o un file-like (st.file_uploader)."""
        try:
            self.datos_crudos = pd.read_csv(fuente, sep=",")
            self.ruta_archivo = getattr(fuente, "name", str(fuente))
            self.ultimo_error = None
            return self.datos_crudos
        except Exception as e:
            self.ultimo_error = str(e)
            return None

    def cargar_tsv(self, fuente) -> pd.DataFrame | None:
        """Lee un TSV (separado por tabuladores) desde una ruta o un file-like."""
        try:
            self.datos_crudos = pd.read_csv(fuente, sep="\t")
            self.ruta_archivo = getattr(fuente, "name", str(fuente))
            self.ultimo_error = None
            return self.datos_crudos
        except Exception as e:
            self.ultimo_error = str(e)
            return None

    def listar_tablas(self, host: str, puerto: int, base_datos: str, usuario: str, contrasena: str) -> list | None:
        """Devuelve los nombres de las tablas del esquema public, o None si falla la conexión."""
        try:
            conexion = psycopg2.connect(
                host=host,
                port=puerto,
                dbname=base_datos,
                user=usuario,
                password=contrasena
            )
            cursor = conexion.cursor()
            cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """)
            tablas = [fila[0] for fila in cursor.fetchall()]
            cursor.close()
            conexion.close()
            self.ultimo_error = None
            return tablas
        except Exception as e:
            self.ultimo_error = str(e)
            return None

    def cargar_tabla_sql(self, host: str, puerto: int, base_datos: str, usuario: str, contrasena: str, tabla: str) -> pd.DataFrame | None:
        """Carga una tabla completa de PostgreSQL como DataFrame."""
        try:
            conexion = psycopg2.connect(
                host=host,
                port=puerto,
                dbname=base_datos,
                user=usuario,
                password=contrasena
            )
            # Identifier escapa el nombre de la tabla para evitar inyección SQL
            query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(tabla))
            self.datos_crudos = pd.read_sql_query(query.as_string(conexion), conexion)
            conexion.close()
            self.ultimo_error = None
            return self.datos_crudos
        except Exception as e:
            self.ultimo_error = str(e)
            return None

    def cargar_excel(self, fuente) -> dict | None:
        """Lee todas las hojas de un Excel; devuelve {nombre_hoja: DataFrame}."""
        try:
            hojas = pd.read_excel(fuente, sheet_name=None)
            self.ruta_archivo = getattr(fuente, "name", str(fuente))
            self.ultimo_error = None
            return hojas if hojas else None
        except Exception as e:
            self.ultimo_error = str(e)
            return None

    def cargar_json(self, fuente) -> dict | None:
        """Extrae tablas de un JSON (incluso anidado); devuelve {ruta: DataFrame}."""
        try:
            data = json.load(fuente)
            tablas = {}
            self._extraer_tablas(data, "", tablas)
            self.ruta_archivo = getattr(fuente, "name", str(fuente))
            self.ultimo_error = None
            return tablas if tablas else None
        except Exception as e:
            self.ultimo_error = str(e)
            return None

    def _extraer_tablas(self, obj: object, prefijo: str, tablas: dict) -> None:
        """Recorre el JSON recursivamente acumulando en `tablas` cada lista de registros encontrada."""
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            sublist_keys = list(dict.fromkeys(
                k for item in obj
                for k, v in item.items()
                if isinstance(v, list) and v and isinstance(v[0], dict)
            ))
            if sublist_keys:
                record_key = sublist_keys[0]
                meta_keys = list(dict.fromkeys(
                    k for item in obj
                    for k, v in item.items()
                    if not isinstance(v, list)
                ))
                try:
                    df = pd.json_normalize(obj, record_path=record_key, meta=meta_keys, errors='ignore')
                    tablas[prefijo or "datos"] = df
                    return
                except Exception:
                    pass
                for item in obj:
                    for clave, valor in item.items():
                        nuevo = f"{prefijo}/{clave}" if prefijo else clave
                        self._extraer_tablas(valor, nuevo, tablas)
            else:
                tablas[prefijo or "datos"] = pd.json_normalize(obj)
        elif isinstance(obj, dict):
            for clave, valor in obj.items():
                nuevo = f"{prefijo}/{clave}" if prefijo else clave
                self._extraer_tablas(valor, nuevo, tablas)

    @staticmethod
    def _limpiar(texto: str) -> str:
        """Quita referencias tipo [1] de Wikipedia y colapsa espacios."""
        texto = re.sub(r"\[\d+\]", "", texto)
        return re.sub(r"\s+", " ", texto).strip()

    @staticmethod
    def _seccion(elemento) -> str:
        """Devuelve el encabezado h2/h3/h4 más cercano que precede al elemento."""
        nodo = elemento.find_previous(["h2", "h3", "h4"])
        if nodo:
            span = nodo.find("span", class_="mw-headline")
            return Datos._limpiar(span.get_text() if span else nodo.get_text())
        return "Sin sección"

    @staticmethod
    def _es_navegacion(elemento) -> bool:
        """True si el elemento vive dentro de menús/encabezados/pies de página."""
        if elemento.find_parent(["nav", "header", "footer", "aside"]) is not None:
            return True
        return elemento.find_parent(attrs={"role": "navigation"}) is not None

    @classmethod
    def _aplanar_columnas(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Aplana encabezados multinivel y desambigua nombres repetidos."""
        if isinstance(df.columns, pd.MultiIndex):
            nombres = []
            for tupla in df.columns:
                partes: list[str] = []
                for parte in map(str, tupla):
                    if parte.startswith("Unnamed") or parte in partes:
                        continue
                    partes.append(parte)
                nombres.append(" ".join(partes))
            df.columns = nombres
        df.columns = [cls._limpiar(str(c)) or f"col_{i}" for i, c in enumerate(df.columns)]
        vistos: dict[str, int] = {}
        unicas = []
        for col in df.columns:
            if col in vistos:
                vistos[col] += 1
                unicas.append(f"{col} ({vistos[col]})")
            else:
                vistos[col] = 0
                unicas.append(col)
        df.columns = unicas
        return df

    @classmethod
    def _limpiar_celdas(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Quita referencias [n] y espacios redundantes de las celdas de texto."""
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].map(lambda v: cls._limpiar(v) if isinstance(v, str) else v)
        return df

    _VALORES_FALTANTES = {"", "-", "—", "–", "n/d", "n/a", "s/d"}

    @staticmethod
    def _a_numero(texto: str) -> float | None:
        """Interpreta un string como número con convención española o inglesa.

        Maneja miles con espacio/punto/coma ("1 417 492 000", "1.234.567"),
        decimales con coma ("17,57") y porcentajes ("3,2%" -> 0.032).
        """
        s = re.sub(r"[\s  ]", "", str(texto))
        if not re.fullmatch(r"[+-]?\d[\d.,]*%?", s):
            return None
        es_pct = s.endswith("%")
        if es_pct:
            s = s[:-1]
        if "," in s and "." in s:
            # el separador que aparece más a la derecha es el decimal
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        elif "," in s:
            # una sola coma con grupo distinto de 3 dígitos -> decimal; si no, miles
            if s.count(",") == 1 and len(s.split(",")[1]) != 3:
                s = s.replace(",", ".")
            else:
                s = s.replace(",", "")
        elif s.count(".") > 1 or (s.count(".") == 1 and len(s.split(".")[1]) == 3):
            s = s.replace(".", "")
        try:
            valor = float(s)
        except ValueError:
            return None
        return valor / 100 if es_pct else valor

    @classmethod
    def _convertir_numericas(cls, df: pd.DataFrame, umbral: float = 0.8) -> pd.DataFrame:
        """Convierte a número las columnas de texto donde ≥ umbral de los datos parsean.

        Los marcadores de dato faltante ("-", "n/d", vacío) se vuelven NaN y no
        cuentan contra el umbral.
        """
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            serie = df[col]
            mask = serie.notna()
            if not mask.any():
                continue
            textos = serie[mask].astype(str).map(cls._limpiar)
            es_dato = ~textos.str.lower().isin(cls._VALORES_FALTANTES)
            convertida = textos.map(cls._a_numero)
            n_datos = int(es_dato.sum())
            if n_datos and int((convertida.notna() & es_dato).sum()) / n_datos >= umbral:
                df[col] = convertida.where(es_dato).reindex(df.index)
        return df

    def cargar_url(self, url: str) -> dict | None:
        """Scrapea tablas, listas (ul/ol) y listas de definición (dl) de una página.

        Las tablas se parsean con pd.read_html (maneja thead, th de fila y
        colspan/rowspan) y se nombran con el encabezado que las precede; las
        columnas mayormente numéricas se convierten a número. Devuelve
        {tag: {sección: DataFrame}}, o None si no se encontró nada.
        """
        try:
            scraper = cloudscraper.create_scraper()
            resp = scraper.get(url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            resultado = {}

            tablas = {}
            for tabla in soup.find_all("table"):
                if tabla.find("table") is not None:
                    continue  # contenedora de layout: los datos están en sus tablas hijas
                try:
                    df = pd.read_html(io.StringIO(str(tabla)), thousands=None)[0]
                except Exception:
                    continue  # tabla de layout sin datos parseables
                if df.empty:
                    continue
                df = self._aplanar_columnas(df)
                df = self._limpiar_celdas(df)
                df = self._convertir_numericas(df)
                seccion = self._seccion(tabla)
                clave = seccion if seccion not in tablas else f"{seccion} ({len(tablas)})"
                tablas[clave] = df
            if tablas:
                resultado["table"] = tablas

            for tag in ("ul", "ol"):
                listas = {}
                for i, lst in enumerate(soup.find_all(tag), 1):
                    if self._es_navegacion(lst):
                        continue  # menús, header, footer: no son contenido
                    items = [self._limpiar(li.get_text()) for li in lst.find_all("li", recursive=False)]
                    if items:
                        seccion = self._seccion(lst)
                        clave = seccion if seccion not in listas else f"{seccion} ({i})"
                        listas[clave] = pd.DataFrame(items, columns=["Elemento"])
                if listas:
                    resultado[tag] = listas

            dls = {}
            for i, dl in enumerate(soup.find_all("dl"), 1):
                terms = [self._limpiar(dt.get_text()) for dt in dl.find_all("dt")]
                defs = [self._limpiar(dd.get_text()) for dd in dl.find_all("dd")]
                if terms:
                    n = max(len(terms), len(defs))
                    terms += [""] * (n - len(terms))
                    defs += [""] * (n - len(defs))
                    seccion = self._seccion(dl)
                    clave = seccion if seccion not in dls else f"{seccion} ({i})"
                    dls[clave] = pd.DataFrame({"Termino": terms, "Definicion": defs})
            if dls:
                resultado["dl"] = dls

            self.ruta_archivo = url
            self.ultimo_error = None
            return resultado if resultado else None
        except Exception as e:
            self.ultimo_error = str(e)
            return None
