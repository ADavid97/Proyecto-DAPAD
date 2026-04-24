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

    def cargar_excel(self, ruta: str) -> pd.DataFrame:
        pass

    def cargar_json(self, ruta: str) -> pd.DataFrame:
        pass

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
