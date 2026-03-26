import pandas as pd


class Datos:
    def __init__(self, ruta_archivo: str = ""):
        self.ruta_archivo = ruta_archivo
        self.datos_crudos = None
        self.separador = ","

    def cargar_csv(self, ruta: str) -> pd.DataFrame:
        try:
            self.datos_crudos = pd.read_csv(ruta, sep=self.separador)
            self.ruta_archivo = ruta
            print(f"\nArchivo '{ruta}' cargado exitosamente.")
            print(f"Filas: {self.datos_crudos.shape[0]} | Columnas: {self.datos_crudos.shape[1]}")
            print(f"Columnas: {list(self.datos_crudos.columns)}\n")
            return self.datos_crudos
        except FileNotFoundError:
            print(f"Error: No se encontro el archivo '{ruta}'.")
            return None
        except Exception as e:
            print(f"Error al cargar '{ruta}': {e}")
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
