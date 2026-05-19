import json
import pandas as pd
import psycopg2


class Datos:
    def __init__(self, ruta_archivo: str = ""):
        self.ruta_archivo = ruta_archivo
        self.datos_crudos = None
        self.separador = ","

    def cargar_csv(self, fuente) -> pd.DataFrame:
        try:
            self.datos_crudos = pd.read_csv(fuente, sep=",")
            self.ruta_archivo = getattr(fuente, "name", str(fuente))
            return self.datos_crudos
        except Exception:
            return None

    def cargar_tsv(self, fuente) -> pd.DataFrame:
        try:
            self.datos_crudos = pd.read_csv(fuente, sep="\t")
            self.ruta_archivo = getattr(fuente, "name", str(fuente))
            return self.datos_crudos
        except Exception:
            return None

    def listar_tablas(self, host: str, puerto: int, base_datos: str, usuario: str, contrasena: str) -> list | None:
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
            return tablas
        except psycopg2.OperationalError as e:
            print(f"Error de conexion a PostgreSQL: {e}")
            return None
        except Exception as e:
            print(f"Error al obtener tablas: {e}")
            return None

    def cargar_tabla_sql(self, host: str, puerto: int, base_datos: str, usuario: str, contrasena: str, tabla: str) -> pd.DataFrame | None:
        try:
            conexion = psycopg2.connect(
                host=host,
                port=puerto,
                dbname=base_datos,
                user=usuario,
                password=contrasena
            )
            self.datos_crudos = pd.read_sql_query(f"SELECT * FROM {tabla};", conexion)
            conexion.close()
            print(f"\nTabla '{tabla}' cargada exitosamente.")
            print(f"Filas: {self.datos_crudos.shape[0]} | Columnas: {self.datos_crudos.shape[1]}")
            print(f"Columnas: {list(self.datos_crudos.columns)}\n")
            return self.datos_crudos
        except psycopg2.OperationalError as e:
            print(f"Error de conexion a PostgreSQL: {e}")
            return None
        except Exception as e:
            print(f"Error al cargar la tabla: {e}")
            return None

    def cargar_excel(self, fuente) -> dict | None:
        try:
            hojas = pd.read_excel(fuente, sheet_name=None)
            self.ruta_archivo = getattr(fuente, "name", str(fuente))
            return hojas if hojas else None
        except Exception:
            return None

    def cargar_json(self, fuente) -> dict | None:
        try:
            data = json.load(fuente)
            tablas = {}
            self._extraer_tablas(data, "", tablas)
            self.ruta_archivo = getattr(fuente, "name", str(fuente))
            return tablas if tablas else None
        except Exception:
            return None

    def _extraer_tablas(self, obj: object, prefijo: str, tablas: dict) -> None:
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

    def obtener_resumen(self) -> dict:
        pass

    def obtener_tipos(self) -> dict:
        pass

    def obtener_nulos(self) -> dict:
        pass

    def obtener_dimensiones(self) -> tuple:
        pass

    def obtener_columnas(self) -> list:
        pass

    def mostrar_primeros(self, n: int = 5) -> pd.DataFrame:
        pass
