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

    def cargar_url(self, url: str, clase_tabla: str | None = None) -> dict | None:
        """Scrapea tablas, listas (ul/ol) y listas de definición (dl) de una página.

        Devuelve {tag: {sección: DataFrame}} agrupando cada elemento por el
        encabezado que lo precede, o None si no se encontró nada.
        """
        try:
            scraper = cloudscraper.create_scraper()
            resp = scraper.get(url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            resultado = {}

            # Tablas: agrupar filas por el encabezado h2/h3 que las precede
            filas_por_seccion: dict[str, list] = {}
            tablas_html = soup.find_all("table", class_=clase_tabla) if clase_tabla else soup.find_all("table")
            for table in tablas_html:
                seccion = self._seccion(table)
                encabezados = [self._limpiar(th.get_text()) for th in table.find_all("th")]
                for fila in table.find_all("tr"):
                    celdas = fila.find_all("td")
                    if not celdas:
                        continue
                    valores = [self._limpiar(c.get_text()) for c in celdas]
                    doc = {
                        (encabezados[i] if i < len(encabezados) else f"col_{i}"): v
                        for i, v in enumerate(valores)
                    }
                    filas_por_seccion.setdefault(seccion, []).append(doc)

            if filas_por_seccion:
                resultado["table"] = {
                    sec: pd.DataFrame(filas)
                    for sec, filas in filas_por_seccion.items()
                }

            if clase_tabla:
                self.ruta_archivo = url
                return resultado if resultado else None

            for tag in ("ul", "ol"):
                listas = {}
                for i, lst in enumerate(soup.find_all(tag), 1):
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
