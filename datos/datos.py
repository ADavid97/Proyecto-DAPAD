import pandas as pd


class Datos:
    def __init__(self, ruta_archivo: str):
        self.ruta_archivo = ruta_archivo
        self.datos_crudos = None
        self.separador = ","

    def cargar_csv(self, ruta: str) -> pd.DataFrame:
        pass

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
