"""
Modulo de construccion de vocabulario.
"""

from collections import Counter
import numpy as np


def construir_vocabulario(tokens_corpus: list[str]) -> np.ndarray:
    """
    Construye y devuelve el arreglo unidimensional vocabulario_palabras de tamaño |V|
    a partir de la lista de tokens del corpus.
    La posicion del indice en el arreglo corresponde a la dimension activa del vector One-Hot.

    :param tokens_corpus: Lista completa de cadenas de texto extraidas del corpus.
    :return: Arreglo de NumPy unidimensional con las |V| cadenas de texto del vocabulario.
    """
    frecuencias = Counter(tokens_corpus)
    palabras_unicas = list(frecuencias.keys())
    return np.array(palabras_unicas, dtype=object)



