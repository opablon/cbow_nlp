"""
Modulo de configuracion centralizada del modelo CBOW.
Permite cargar hiperparametros desde un archivo YAML o valores por defecto.
"""

from pathlib import Path
import yaml

class Configuracion:
    """Clase para administrar la configuracion del modelo CBOW y el entorno de entrenamiento."""

    def __init__(self, ruta_configuracion: str | None = "configuracion.yaml"):
        """
        Inicializa la configuracion cargando parametros por defecto y opcionalmente desde un YAML.
        :param ruta_configuracion: Ruta al archivo YAML (o None para omitir carga externa).
        """
        # Valores por defecto en Python
        self.ruta_corpus = "datos/corpus.txt"
        self.directorio_respaldos = "respaldos"
        self.motor_computo = "cupy"  # "cupy" o "pytorch"
        self.estrategia_tokenizacion = "palabra"
        self.token_desconocido = "<UNK>"
        self.criterio_seleccion_vocabulario = "cantidad"
        self.cantidad_palabras_unicas = 15000
        self.porcentaje_palabras_unicas = 0.80
        self.frecuencia_minima = 1
        self.muestreo_negativo = False
        self.cantidad_muestras_negativas = 5
        self.tamanio_ventana = 4
        self.dimension_embedding = 100
        self.tasa_aprendizaje = 0.025
        self.cantidad_epocas = 10
        self.hacer_respaldo = True
        self.frecuencia_respaldo = 2
        self.tamanio_lote = 2048
        self.tipo_similaridad = "producto_interno"
        self.semilla_aleatoria = 26

        if ruta_configuracion is not None:
            self.ruta_configuracion = Path(ruta_configuracion)
            if self.ruta_configuracion.exists():
                self.cargar_desde_yaml(self.ruta_configuracion)

    def cargar_desde_yaml(self, ruta_yaml: Path) -> None:
        """Carga y actualiza los parametros desde un archivo YAML usando bucles imperativos."""
        with open(ruta_yaml, "r", encoding="utf-8") as archivo:
            datos = yaml.safe_load(archivo)
            if datos is not None:
                for clave, valor in datos.items():
                    if hasattr(self, clave):
                        setattr(self, clave, valor)

    def a_diccionario(self) -> dict:
        """Retorna todos los parametros como un diccionario utilizando construccion imperativa."""
        diccionario_parametros = {}
        diccionario_parametros["ruta_corpus"] = self.ruta_corpus
        diccionario_parametros["directorio_respaldos"] = self.directorio_respaldos
        diccionario_parametros["motor_computo"] = self.motor_computo
        diccionario_parametros["estrategia_tokenizacion"] = self.estrategia_tokenizacion
        diccionario_parametros["token_desconocido"] = self.token_desconocido
        diccionario_parametros["criterio_seleccion_vocabulario"] = self.criterio_seleccion_vocabulario
        diccionario_parametros["cantidad_palabras_unicas"] = self.cantidad_palabras_unicas
        diccionario_parametros["porcentaje_palabras_unicas"] = self.porcentaje_palabras_unicas
        diccionario_parametros["frecuencia_minima"] = self.frecuencia_minima
        diccionario_parametros["muestreo_negativo"] = self.muestreo_negativo
        diccionario_parametros["cantidad_muestras_negativas"] = self.cantidad_muestras_negativas
        diccionario_parametros["tamanio_ventana"] = self.tamanio_ventana
        diccionario_parametros["dimension_embedding"] = self.dimension_embedding
        diccionario_parametros["tasa_aprendizaje"] = self.tasa_aprendizaje
        diccionario_parametros["cantidad_epocas"] = self.cantidad_epocas
        diccionario_parametros["hacer_respaldo"] = self.hacer_respaldo
        diccionario_parametros["frecuencia_respaldo"] = self.frecuencia_respaldo
        diccionario_parametros["tamanio_lote"] = self.tamanio_lote
        diccionario_parametros["tipo_similaridad"] = self.tipo_similaridad
        diccionario_parametros["semilla_aleatoria"] = self.semilla_aleatoria
        return diccionario_parametros

    def guardar_en_yaml(self, ruta_yaml: Path | str | None = None) -> None:
        """
        Exporta los parametros actuales de la instancia a un archivo YAML de forma segura.
        Si no se especifica ruta_yaml, utiliza self.ruta_configuracion o 'configuracion.yaml'.
        """
        if ruta_yaml is not None:
            destino = Path(ruta_yaml)
        elif hasattr(self, "ruta_configuracion") and self.ruta_configuracion is not None:
            destino = Path(self.ruta_configuracion)
        else:
            destino = Path("configuracion.yaml")

        datos = self.a_diccionario()
        with open(destino, "w", encoding="utf-8") as archivo:
            yaml.safe_dump(datos, archivo, default_flow_style=False, sort_keys=False, allow_unicode=True)

