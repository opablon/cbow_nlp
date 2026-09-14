"""
Modulo de construccion de vocabulario.
"""

from collections import Counter
from pathlib import Path
import numpy as np

from .tokenizador import tokenizar_corpus


def construir_vocabulario(
    ruta_corpus: str | Path,
    incluir_puntuacion_y_numeros: bool = True,
    token_desconocido: str = "<UNK>",
) -> np.ndarray:
    """
    Construye y devuelve el arreglo unidimensional vocabulario_palabras de tamaño |V|.
    La posicion del indice en el arreglo corresponde a la dimension activa del vector One-Hot.

    :param ruta_corpus: Ruta obligatoria al archivo del corpus en disco.
    :param incluir_puntuacion_y_numeros: Incluir signos y numeros en tokenizacion por palabra.
    :param token_desconocido: Token reservado para elementos fuera de vocabulario (por defecto '<UNK>').
    :return: Arreglo de NumPy unidimensional con las |V| cadenas de texto del vocabulario.
    """
    tokens_corpus = tokenizar_corpus(
        ruta_corpus=ruta_corpus,
        incluir_puntuacion_y_numeros=incluir_puntuacion_y_numeros,
    )
    frecuencias = Counter(tokens_corpus)
    palabras_unicas = list(frecuencias.keys())

    # Reservar la primera posicion (indice 0) para token_desconocido
    if token_desconocido in palabras_unicas:
        palabras_unicas.remove(token_desconocido)

    vocabulario_lista = [token_desconocido] + palabras_unicas
    return np.array(vocabulario_lista, dtype=object)


