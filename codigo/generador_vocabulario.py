"""
Modulo de construccion de vocabulario y distribucion de frecuencia unigrama.
"""

from collections import Counter
from pathlib import Path
import numpy as np

from .tokenizador import tokenizar_corpus

try:
    import cupy as cp

    _ = cp.zeros((1,), dtype=cp.float32)
    USAR_CUPY = True
    xp = cp
except Exception:
    USAR_CUPY = False
    xp = np


def construir_vocabulario(
    ruta_corpus: str | Path,
    estrategia_tokenizacion: str = "palabra",
    incluir_puntuacion_y_numeros: bool = True,
    bpe_tamanio_vocabulario: int = 15000,
    bpe_frecuencia_minima: int = 2,
    token_desconocido: str = "<UNK>",
) -> np.ndarray:
    """
    Construye y devuelve el arreglo unidimensional vocabulario_palabras de tamaño |V|.
    La posicion del indice en el arreglo corresponde a la dimension activa del vector One-Hot.

    :param ruta_corpus: Ruta obligatoria al archivo del corpus en disco.
    :param estrategia_tokenizacion: 'palabra' (palabras completas) o 'bpe' (Byte Pair Encoding).
    :param incluir_puntuacion_y_numeros: Incluir signos y numeros en tokenizacion por palabra.
    :param bpe_tamanio_vocabulario: Tamaño final del vocabulario BPE (solo para 'bpe').
    :param bpe_frecuencia_minima: Frecuencia minima de pares de tokens para BPE (solo para 'bpe').
    :param token_desconocido: Token reservado para elementos fuera de vocabulario (por defecto '<UNK>').
    :return: Arreglo de NumPy unidimensional con las |V| cadenas de texto del vocabulario.
    """
    path_corpus = Path(ruta_corpus)
    if not path_corpus.exists():
        raise FileNotFoundError(f"No se encontro el archivo del corpus en: '{path_corpus}'")

    if estrategia_tokenizacion == "bpe":
        return _construir_vocabulario_bpe(
            ruta_corpus=path_corpus,
            tamanio_objetivo=bpe_tamanio_vocabulario,
            frecuencia_minima=bpe_frecuencia_minima,
            token_desconocido=token_desconocido,
            incluir_puntuacion_y_numeros=incluir_puntuacion_y_numeros,
        )

    # --- Estrategia 'palabra' (100% de las palabras por orden de aparicion) ---
    tokens_corpus = tokenizar_corpus(
        ruta_corpus=path_corpus,
        incluir_puntuacion_y_numeros=incluir_puntuacion_y_numeros,
    )
    frecuencias = Counter(tokens_corpus)
    palabras_unicas = list(frecuencias.keys())

    # Reservar la primera posicion (indice 0) para token_desconocido
    if token_desconocido in palabras_unicas:
        palabras_unicas.remove(token_desconocido)

    vocabulario_lista = [token_desconocido] + palabras_unicas
    return np.array(vocabulario_lista, dtype=object)


def _construir_vocabulario_bpe(
    ruta_corpus: Path,
    tamanio_objetivo: int,
    frecuencia_minima: int,
    token_desconocido: str,
    incluir_puntuacion_y_numeros: bool = True,
) -> np.ndarray:
    """
    Funcion auxiliar para entrenar y construir el vocabulario BPE utilizando la libreria tokenizers.
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=tamanio_objetivo,
        min_frequency=frecuencia_minima,
    )

    tokenizer.train([str(ruta_corpus)], trainer=trainer)

    vocab_dict = tokenizer.get_vocab()
    vocab_ordenado = sorted(vocab_dict.items(), key=lambda item: item[1])
    return np.array([token for token, indice in vocab_ordenado], dtype=object)


def construir_distribucion_unigrama(
    tokens_corpus: list[str],
    vocabulario_palabras: np.ndarray,
) -> xp.ndarray:
    """
    Calcula la distribucion de frecuencia unigrama elevada a la potencia 0.75 a partir del
    conteo de ocurrencias de cada palabra del vocabulario en el corpus.

    :param tokens_corpus: Lista completa de tokens de texto del corpus.
    :param vocabulario_palabras: Vocabulario de palabras |V|.
    :return: Arreglo de probabilidades unigrama de forma (|V|,).
    """
    tamanio_vocabulario = len(vocabulario_palabras)
    mapeo_vocabulario = {palabra: i for i, palabra in enumerate(vocabulario_palabras)}

    frecuencias = np.zeros(tamanio_vocabulario, dtype=np.float32)
    conteo_corpus = Counter(tokens_corpus)

    for token, freq in conteo_corpus.items():
        idx = mapeo_vocabulario.get(token, 0)
        frecuencias[idx] += freq

    frecuencias_modificadas = frecuencias ** 0.75
    suma_total = np.sum(frecuencias_modificadas)

    if float(suma_total) > 0.0:
        probabilidades = frecuencias_modificadas / suma_total
    else:
        probabilidades = np.ones(tamanio_vocabulario, dtype=np.float32) / tamanio_vocabulario

    return xp.asarray(probabilidades, dtype=xp.float32)
