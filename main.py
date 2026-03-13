from datos import Datos
from analisis_exploratorio import AnalisisExploratorio
from preprocesamiento import Preprocesamiento
from modelos import Modelo, ModeloKMeans, ModeloRegresionLineal, ModeloRegresionLogistica, ModeloArbolDecision, ModeloKNN
from evaluacion import Evaluacion
from visualizacion import Visualizacion


class Main:
    def __init__(self):
        self.datos = None
        self.eda = None
        self.preprocesamiento = None
        self.modelo = None
        self.evaluacion = None
        self.visualizacion = None

    def ejecutar(self) -> None:
        pass

    def menu(self) -> None:
        pass


if __name__ == "__main__":
    app = Main()
    app.menu()
