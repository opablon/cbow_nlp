"""
Modulo de configuracion del modelo CBOW mediante funciones puras.
Permite cargar y guardar hiperparametros desde y hacia archivos YAML.
"""

from pathlib import Path
import yaml


def cargar_configuracion(ruta_yaml: str | Path = "configuracion.yaml") -> dict:
    """
    Carga los parametros de configuracion desde un archivo YAML y devuelve un diccionario.
    Establece valores por defecto en español para cualquier parametro omitido.

    :param ruta_yaml: Ruta al archivo YAML de configuracion (por defecto 'configuracion.yaml').
    :return: Diccionario con la totalidad de los hiperparametros de entrenamiento y modelo.
    """
    configuracion_por_defecto = {
        "ruta_corpus": "datos/corpus.txt",
        "directorio_respaldos": "respaldos",
        "estrategia_tokenizacion": "palabra",
        "incluir_puntuacion_y_numeros": True,
        "criterio_seleccion_vocabulario": "porcentaje_palabras",
        "cantidad_palabras_unicas": 15000,
        "porcentaje_palabras_unicas": 1.0,
        "bpe_tamanio_vocabulario": 15000,
        "bpe_frecuencia_minima": 2,
        "bpe_cantidad_fusiones": 1000,
        "muestreo_negativo": False,
        "cantidad_muestras_negativas": 5,
        "tamanio_ventana": 5,
        "dimension_embedding": 100,
        "tasa_aprendizaje": 0.2,
        "cantidad_epocas": 10,
        "hacer_respaldo": True,
        "frecuencia_respaldo": 2,
        "tamanio_lote": 2048,
        "semilla_aleatoria": 26,
        "reanudar_entrenamiento": False,
        "ruta_checkpoint": "respaldos/modelo_cbow_w5_epoca_10.npz",
    }

    path_yaml = Path(ruta_yaml)
    if path_yaml.exists():
        with open(path_yaml, "r", encoding="utf-8") as archivo:
            datos_cargados = yaml.safe_load(archivo)
            if datos_cargados is not None and isinstance(datos_cargados, dict):
                for clave, valor in datos_cargados.items():
                    configuracion_por_defecto[clave] = valor

    return configuracion_por_defecto


def guardar_configuracion(
    configuracion_dict: dict, ruta_yaml: str | Path = "configuracion.yaml"
) -> None:
    """
    Guarda un diccionario de configuracion en un archivo YAML.

    :param configuracion_dict: Diccionario que contiene los parametros a exportar.
    :param ruta_yaml: Ruta del archivo YAML de destino (por defecto 'configuracion.yaml').
    :return: None.
    """
    destino = Path(ruta_yaml)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8") as archivo:
        yaml.safe_dump(
            configuracion_dict,
            archivo,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
