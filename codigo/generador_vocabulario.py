"""
Modulo de construccion de vocabulario y generacion de matrices de vectores One-Hot.
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
    bpe_cantidad_fusiones: int = 1000,
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
    :param bpe_cantidad_fusiones: Numero maximo de fusiones BPE (solo para fallback en Python puro).
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
            cantidad_fusiones=bpe_cantidad_fusiones,
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
    cantidad_fusiones: int,
    token_desconocido: str,
    incluir_puntuacion_y_numeros: bool = True,
) -> np.ndarray:
    """
    Funcion auxiliar para entrenar y construir el vocabulario BPE siguiendo la diapositiva 39 de Fragmentacion.pdf.
    """
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers

        # Inicializar el fragmentador BPE estandar de la catedra
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
    except ImportError:
        pass

    import re

    tokens_corpus = tokenizar_corpus(
        ruta_corpus=ruta_corpus,
        incluir_puntuacion_y_numeros=incluir_puntuacion_y_numeros,
    )

    vocab_bpe = Counter()
    for token in tokens_corpus:
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
    :return: Tupla (matriz_contextos_indice, matriz_objetivos_indice).
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
        for indice_izq in range(i - tamanio_ventana, i):
            contexto.append(indices_tokens[indice_izq])
        for indice_der in range(i + 1, i + tamanio_ventana + 1):
            contexto.append(indices_tokens[indice_der])

        muestras_contexto_indices.append(contexto)
        muestras_objetivo_indices.append(indices_tokens[i])

    matriz_contextos_indice = np.array(muestras_contexto_indices, dtype=np.int32)
    matriz_objetivos_indice = np.array(muestras_objetivo_indices, dtype=np.int32)

    return matriz_contextos_indice, matriz_objetivos_indice


def crear_matriz_one_hot_lote(
    sub_contextos_indice: np.ndarray,
    sub_objetivos_indice: np.ndarray,
    tamanio_vocabulario: int,
) -> tuple[xp.ndarray, xp.ndarray]:
    """
    Construye las matrices densas de codificacion One-Hot x y t para un mini-lote.

    :param sub_contextos_indice: Matriz de indices de contexto para el lote (L x C).
    :param sub_objetivos_indice: Arreglo de indices objetivo para el lote (L,).
    :param tamanio_vocabulario: Cardinalidad |V| del vocabulario.
    :return: Tupla (x_lote, t_lote) con las matrices One-Hot (L x |V|).
    """
    L = sub_contextos_indice.shape[0]

    # x_lote (L x |V|): Matriz One-Hot con la suma de las palabras del contexto
    x_lote = xp.zeros((L, tamanio_vocabulario), dtype=xp.float32)
    for c in range(sub_contextos_indice.shape[1]):
        indices_col = xp.asarray(sub_contextos_indice[:, c])
        xp.add.at(x_lote, (xp.arange(L), indices_col), 1.0)

    # t_lote (L x |V|): Matriz One-Hot con 1 en el indice objetivo deseado
    t_lote = xp.zeros((L, tamanio_vocabulario), dtype=xp.float32)
    indices_obj = xp.asarray(sub_objetivos_indice)
    t_lote[xp.arange(L), indices_obj] = 1.0

    return x_lote, t_lote


def construir_distribucion_unigrama(
    indices_tokens: list[int],
    tamanio_vocabulario: int,
    potencia: float = 0.75,
) -> xp.ndarray:
    """
    Calcula la distribucion de frecuencia unigrama elevada a la potencia de 0.75 para rebalancear
    las probabilidades de seleccion en el muestreo negativo.

    :param indices_tokens: Lista de indices numéricos de las palabras del corpus.
    :param tamanio_vocabulario: Cantidad total de palabras en el vocabulario |V|.
    :param potencia: Exponente para modificar la distribucion unigrama (por defecto 0.75).
    :return: Arreglo de probabilidades de forma (|V|,).
    """
    conteo = Counter(indices_tokens)
    frecuencias = np.zeros(tamanio_vocabulario, dtype=np.float32)
    for indice, freq in conteo.items():
        if 0 <= indice < tamanio_vocabulario:
            frecuencias[indice] = freq

    frecuencias_modificadas = frecuencias ** potencia
    probabilidades = frecuencias_modificadas / np.sum(frecuencias_modificadas)

    return xp.asarray(probabilidades, dtype=xp.float32)


def generar_muestras_negativas(
    matriz_objetivo_t: xp.ndarray | np.ndarray,
    distribucion_unigrama: xp.ndarray,
    cantidad_negativos: int = 5,
) -> xp.ndarray:
    """
    Genera K muestras negativas por cada ejemplo del lote utilizando la distribucion unigrama,
    garantizando que la palabra objetivo p_O quede excluida del conjunto de muestra negativa.

    :param matriz_objetivo_t: Matriz One-Hot de la palabra objetivo deseada (L x |V|).
    :param distribucion_unigrama: Distribucion de probabilidad unigrama P(w)^0.75 de forma (|V|,).
    :param cantidad_negativos: Cantidad K de palabras negativas por cada ejemplo.
    :return: Matriz de indices de palabras negativas de forma (L x K).
    """
    L = matriz_objetivo_t.shape[0]
    V = len(distribucion_unigrama)

    # Extraer el indice de la palabra objetivo p_O a partir de la matriz One-Hot t -> Forma (L, 1)
    indices_objetivo = xp.argmax(matriz_objetivo_t, axis=1)[:, None]

    p_vec = xp.asarray(distribucion_unigrama)

    # Muestreo vectorizado directo de tamaño (L, K)
    indices_flat = xp.random.choice(V, size=L * cantidad_negativos, p=p_vec)
    negativos = indices_flat.reshape(L, cantidad_negativos).astype(xp.int32)

    # Reemplazar vectorialmente cualquier coincidencia accidental con la palabra objetivo
    coincidencias = (negativos == indices_objetivo)
    intentos = 0
    while xp.any(coincidencias) and intentos < 5:
        cantidad_faltantes = int(xp.sum(coincidencias))
        reemplazos = xp.random.choice(V, size=cantidad_faltantes, p=p_vec).astype(xp.int32)
        negativos[coincidencias] = reemplazos
        coincidencias = (negativos == indices_objetivo)
        intentos += 1

    if xp.any(coincidencias):
        negativos[coincidencias] = (negativos[coincidencias] + 1) % V

    return negativos



