"""
Modulo de tokenizacion para el preprocesamiento de textos y corpora mediante funciones puras.
"""

import re
from pathlib import Path


def tokenizar_corpus(
    ruta_corpus: str | Path, incluir_puntuacion_y_numeros: bool = True
) -> list[str]:
    """
    Lee un archivo de texto desde el disco y extrae la secuencia completa de tokens en minusculas.

    :param ruta_corpus: Ruta al archivo de texto del corpus de entrenamiento.
    :param incluir_puntuacion_y_numeros: Si es True, incluye signos de puntuacion y numeros como tokens independientes;
                                          si es False, extrae unicamente palabras alfabeticas.
    :return: Lista de cadenas de texto con los tokens extraidos.
    """
    path_corpus = Path(ruta_corpus)
    if not path_corpus.exists():
        raise FileNotFoundError(f"No se encontro el archivo de corpus en la ruta: {path_corpus}")

    with open(path_corpus, "r", encoding="utf-8", errors="ignore") as archivo:
        contenido = archivo.read().lower()

    if incluir_puntuacion_y_numeros:
        # Regex que captura secuencias alfanumericas o cualquier caracter de puntuacion no blanco
        patron = re.compile(r"\w+|[^\w\s]", re.IGNORECASE)
    else:
        # Regex que captura exclusivamente palabras alfabeticas con tildes y ñ
        patron = re.compile(r"[a-záéíóúüñ]+", re.IGNORECASE)

    return patron.findall(contenido)
