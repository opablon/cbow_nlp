"""
Modulo de construccion de vocabulario y generacion de matrices de vectores One-Hot mediante funciones puras.
"""

from collections import Counter
import random
from pathlib import Path
import numpy as np

try:
    import cupy as cp

    _ = cp.zeros((1,), dtype=cp.float32)
    USAR_CUPY = True
    xp = cp
except Exception:
    USAR_CUPY = False
    xp = np


def construir_vocabulario(
    lista_tokens: list[str],
    ruta_corpus: str | Path | None = None,
    estrategia_tokenizacion: str = "palabra",
    bpe_tamanio_vocabulario: int = 15000,
    bpe_frecuencia_minima: int = 2,
    bpe_cantidad_fusiones: int = 1000,
    criterio_seleccion_vocabulario: str = "porcentaje_palabras",
    cantidad_palabras_unicas: int = 15000,
    porcentaje_palabras_unicas: float = 1.0,
    token_desconocido: str = "<UNK>",
    semilla_aleatoria: int = 26,
) -> np.ndarray:
    """
    Construye y devuelve el arreglo unidimensional vocabulario_palabras de tamaño |V|.
    La posicion del indice en el arreglo corresponde a la dimension activa del vector One-Hot.

    :param lista_tokens: Lista de tokens extraidos del corpus.
    :param ruta_corpus: Ruta al archivo del corpus (necesario si estrategia_tokenizacion es 'bpe').
    :param estrategia_tokenizacion: 'palabra' (palabras completas) o 'bpe' (Byte Pair Encoding).
    :param bpe_tamanio_vocabulario: Tamaño final del vocabulario BPE (solo para 'bpe').
    :param bpe_frecuencia_minima: Frecuencia minima de pares de tokens para BPE (solo para 'bpe').
    :param bpe_cantidad_fusiones: Numero maximo de fusiones BPE (solo para 'bpe').
    :param criterio_seleccion_vocabulario: 'porcentaje_palabras' o 'cantidad_palabras' (solo para 'palabra').
    :param cantidad_palabras_unicas: Cantidad fija de palabras si criterio es 'cantidad_palabras'.
    :param porcentaje_palabras_unicas: Porcentaje entre 0.0 y 1.0 si criterio es 'porcentaje_palabras'.
    :param token_desconocido: Token reservado para elementos fuera de vocabulario (por defecto '<UNK>').
    :param semilla_aleatoria: Semilla para garatizar la reproducibilidad.
    :return: Arreglo de NumPy unidimensional con las |V| cadenas de texto del vocabulario.
    """
    if estrategia_tokenizacion == "bpe":
        return _construir_vocabulario_bpe(
            ruta_corpus=ruta_corpus,
            lista_tokens=lista_tokens,
            tamanio_objetivo=bpe_tamanio_vocabulario,
            frecuencia_minima=bpe_frecuencia_minima,
            cantidad_fusiones=bpe_cantidad_fusiones,
            token_desconocido=token_desconocido,
        )

    # --- Estrategia 'palabra' (Palabras completas) ---
    frecuencias = Counter(lista_tokens)
    palabras_unicas = list(frecuencias.keys())
    total_unicas = len(palabras_unicas)

    if criterio_seleccion_vocabulario == "cantidad_palabras":
        limite_efectivo = min(cantidad_palabras_unicas, total_unicas)
    elif criterio_seleccion_vocabulario == "porcentaje_palabras":
        porcentaje_validado = max(0.001, min(1.0, porcentaje_palabras_unicas))
        limite_efectivo = max(1, int(total_unicas * porcentaje_validado))
    else:
        raise ValueError(f"Criterio no soportado: {criterio_seleccion_vocabulario}")

    random.seed(semilla_aleatoria)
    random.shuffle(palabras_unicas)

    palabras_seleccionadas = palabras_unicas[:limite_efectivo]

    # Reservar la primera posicion (indice 0) para token_desconocido
    if token_desconocido in palabras_seleccionadas:
        palabras_seleccionadas.remove(token_desconocido)

    vocabulario_lista = [token_desconocido] + palabras_seleccionadas
    return np.array(vocabulario_lista, dtype=object)


def _construir_vocabulario_bpe(
    ruta_corpus: str | Path | None,
    lista_tokens: list[str],
    tamanio_objetivo: int,
    frecuencia_minima: int,
    cantidad_fusiones: int,
    token_desconocido: str,
) -> np.ndarray:
    """
    Funcion auxiliar para entrenar y construir el vocabulario BPE.
    """
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers

        tokenizer = Tokenizer(models.BPE(unk_token=token_desconocido))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size=tamanio_objetivo,
            min_frequency=frecuencia_minima,
            special_tokens=[token_desconocido],
        )

        if ruta_corpus is not None and Path(ruta_corpus).exists():
            tokenizer.train([str(ruta_corpus)], trainer=trainer)
        else:
            tokenizer.train_from_iterator(lista_tokens, trainer=trainer)

        vocab_dict = tokenizer.get_vocab()
        vocab_ordenado = sorted(vocab_dict.items(), key=lambda item: item[1])
        return np.array([token for token, idx in vocab_ordenado], dtype=object)
    except ImportError:
        pass

    import re

    vocab_bpe = Counter()
    for token in lista_tokens:
        palabra_espaciada = " ".join(list(token)) + " </w>"
        vocab_bpe[palabra_espaciada] += 1

    def obtener_estadisticas_pares(vocab):
        pares = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pares[symbols[i], symbols[i + 1]] += freq
        return pares

    def fusionar_vocabulario(par, v_in):
        v_out = {}
        bigram = re.escape(" ".join(par))
        patron = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        remplazo = "".join(par)
        for word in v_in:
            w_out = patron.sub(remplazo, word)
            v_out[w_out] = v_in[word]
        return v_out

    simbolos_unicos = set()
    for palabra_str in vocab_bpe.keys():
        simbolos_unicos.update(palabra_str.split())

    subpalabras = [token_desconocido] + list(simbolos_unicos)

    num_merges = min(cantidad_fusiones, max(0, tamanio_objetivo - len(subpalabras)))
    for _ in range(num_merges):
        pares = obtener_estadisticas_pares(vocab_bpe)
        if not pares:
            break
        mejor_par = max(pares, key=pares.get)
        if pares[mejor_par] < frecuencia_minima:
            break
        vocab_bpe = fusionar_vocabulario(mejor_par, vocab_bpe)
        nueva_subpalabra = "".join(mejor_par)
        if nueva_subpalabra not in subpalabras:
            subpalabras.append(nueva_subpalabra)
        if len(subpalabras) >= tamanio_objetivo:
            break

    return np.array(subpalabras, dtype=object)


def generar_vectores_one_hot(
    indices_tokens: list[int],
    tamanio_vocabulario: int,
    tamanio_ventana: int,
    tamanio_lote: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae la matriz completa de indices de contexto y palabra objetivo para el corpus.

    :param indices_tokens: Lista de indices enteros correspondientes a las palabras del corpus.
    :param tamanio_vocabulario: Cardinalidad |V| del vocabulario.
    :param tamanio_ventana: Tamaño de ventana C/2 a izquierda y a derecha.
    :param tamanio_lote: Cantidad de muestras por mini-lote B (por defecto 2048).
    :return: Tupla (matriz_contextos_idx, matriz_objetivos_idx).
    """
    longitud_total = len(indices_tokens)
    limite_inicio = tamanio_ventana
    limite_fin = longitud_total - tamanio_ventana

    if limite_fin <= limite_inicio:
        raise ValueError("El corpus es demasiado corto para el tamaño de ventana especificado.")

    muestras_contexto_indices = []
    muestras_objetivo_indices = []

    for i in range(limite_inicio, limite_fin):
        contexto = []
        for idx_izq in range(i - tamanio_ventana, i):
            contexto.append(indices_tokens[idx_izq])
        for idx_der in range(i + 1, i + tamanio_ventana + 1):
            contexto.append(indices_tokens[idx_der])

        muestras_contexto_indices.append(contexto)
        muestras_objetivo_indices.append(indices_tokens[i])

    matriz_contextos_idx = np.array(muestras_contexto_indices, dtype=np.int32)
    matriz_objetivos_idx = np.array(muestras_objetivo_indices, dtype=np.int32)

    return matriz_contextos_idx, matriz_objetivos_idx


def crear_matriz_one_hot_lote(
    sub_contextos_idx: np.ndarray,
    sub_objetivos_idx: np.ndarray,
    tamanio_vocabulario: int,
) -> tuple[xp.ndarray, xp.ndarray]:
    """
    Construye bajo demanda en GPU/CPU las matrices densas de codificación One-Hot x y t para un mini-lote.

    :param sub_contextos_idx: Matriz de indices de contexto para el lote (B_lote x C).
    :param sub_objetivos_idx: Arreglo de indices objetivo para el lote (B_lote,).
    :param tamanio_vocabulario: Cardinalidad |V| del vocabulario.
    :return: Tupla (x_lote, t_lote) con las matrices One-Hot (B_lote x |V|).
    """
    B_lote = sub_contextos_idx.shape[0]

    # x_lote (B_lote x |V|): Matriz One-Hot con la suma de las palabras del contexto
    x_lote = xp.zeros((B_lote, tamanio_vocabulario), dtype=xp.float32)
    for c in range(sub_contextos_idx.shape[1]):
        indices_col = xp.asarray(sub_contextos_idx[:, c])
        xp.add.at(x_lote, (xp.arange(B_lote), indices_col), 1.0)

    # t_lote (B_lote x |V|): Matriz One-Hot con 1 en el indice objetivo deseado
    t_lote = xp.zeros((B_lote, tamanio_vocabulario), dtype=xp.float32)
    indices_obj = xp.asarray(sub_objetivos_idx)
    t_lote[xp.arange(B_lote), indices_obj] = 1.0

    return x_lote, t_lote
