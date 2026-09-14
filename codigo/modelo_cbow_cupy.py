"""
Modulo de calculo matricial para la red neuronal CBOW en CuPy / NumPy.
"""

import os
import sys
import ctypes
import ast
from pathlib import Path
import numpy as np

# Cargar automaticamente bibliotecas dinamicas CUDA instaladas en el .venv
_ruta_venv = Path(__file__).resolve().parent.parent / ".venv"
_carpeta_nvidia = (
    _ruta_venv
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
    / "nvidia"
)

if _carpeta_nvidia.exists():
    for _lib_dir in _carpeta_nvidia.glob("*/lib"):
        for _so_file in _lib_dir.glob("*.so*"):
            try:
                ctypes.CDLL(str(_so_file))
            except Exception:
                pass

try:
    import cupy as cp

    _ = cp.zeros((1,), dtype=cp.float32)
    USAR_CUPY = True
    xp = cp
except Exception:
    USAR_CUPY = False
    xp = np


def crear_matrices_lote(
    subconjunto_tokens_lote: list[str],
    mapeo_vocabulario: dict[str, int],
    tamanio_ventana: int,
) -> tuple[xp.ndarray, xp.ndarray]:
    """
    Construye las matrices de contexto (x) y palabra objetivo (t) para un mini-lote de muestras
    a partir del subconjunto de tokens correspondientes.

    :param subconjunto_tokens_lote: Subconjunto de tokens de texto del mini-lote (incluyendo margenes de ventana).
    :param mapeo_vocabulario: Diccionario de mapeo palabra -> indice.
    :param tamanio_ventana: Tamaño de ventana C/2 a izquierda y a derecha.
    :return: Tupla (matriz_contexto_x, matriz_objetivo_t) de forma (L, |V|).
    """
    tamanio_vocabulario = len(mapeo_vocabulario)

    limite_inicio = tamanio_ventana
    limite_fin = len(subconjunto_tokens_lote) - tamanio_ventana

    L = max(0, limite_fin - limite_inicio)

    matriz_contexto_x_cpu = np.zeros((L, tamanio_vocabulario), dtype=np.float32)
    matriz_objetivo_t_cpu = np.zeros((L, tamanio_vocabulario), dtype=np.float32)

    for indice_lote, i in enumerate(range(limite_inicio, limite_fin)):
        for offset in range(-tamanio_ventana, tamanio_ventana + 1):
            if offset != 0:
                token_contexto = subconjunto_tokens_lote[i + offset]
                indice_activo = mapeo_vocabulario.get(token_contexto, 0)
                matriz_contexto_x_cpu[indice_lote, indice_activo] += 1.0

        token_objetivo = subconjunto_tokens_lote[i]
        indice_objetivo = mapeo_vocabulario.get(token_objetivo, 0)
        matriz_objetivo_t_cpu[indice_lote, indice_objetivo] = 1.0

    return xp.asarray(matriz_contexto_x_cpu), xp.asarray(matriz_objetivo_t_cpu)


def inicializar_pesos(
    tamanio_vocabulario: int, dimension_embedding: int, semilla_aleatoria: int = 26
) -> tuple[xp.ndarray, xp.ndarray]:
    """
    Inicializa las matrices de pesos W (entrada) y W' (salida) de la red neuronal CBOW.
    Matriz W se inicializa con distribucion uniforme en [-0.1, 0.1].
    Matriz W' se inicializa en ceros (W' = 0).

    :param tamanio_vocabulario: Cardinalidad del vocabulario |V|.
    :param dimension_embedding: Dimension de la capa oculta o vector de embedding N.
    :param semilla_aleatoria: Semilla para reproducibilidad aleatoria (por defecto 26).
    :return: Tupla con las matrices de pesos (W, W_prima).
    """
    xp.random.seed(semilla_aleatoria)
    limite = 0.1

    # W (|V| x N): Matriz de pesos entre entrada y capa oculta
    W = xp.random.uniform(
        -limite, limite, (tamanio_vocabulario, dimension_embedding)
    ).astype(xp.float32)

    # W' (N x |V|): Matriz de pesos entre capa oculta y capa de salida (inicializada en CEROS)
    W_prima = xp.zeros(
        (dimension_embedding, tamanio_vocabulario), dtype=xp.float32
    )

    return W, W_prima


def propagar_hacia_adelante(
    x: xp.ndarray, W: xp.ndarray, W_prima: xp.ndarray, C: float = 1.0
) -> tuple[xp.ndarray, xp.ndarray]:
    """
    Propagacion hacia adelante matricial para el lote de muestras.
    Calcula el vector de la capa oculta h y las probabilidades Softmax y.

    Formulacion matricial:
    h = (1 / C) * x * W
    u = h * W'
    y = softmax(u)

    :param x: Matriz de contexto (L x |V|).
    :param W: Matriz de pesos de entrada (|V| x N).
    :param W_prima: Matriz de pesos de salida (N x |V|).
    :param C: Numero de palabras en la ventana de contexto (2 * tamanio_ventana).
    :return: Tupla (h, y) con las activaciones y probabilidades.
    """
    # h = (1 / C) * x * W  ->  Forma: (L, N)
    h = (1.0 / C) * xp.dot(x, W)

    # u = h * W'  ->  Forma: (L, |V|)
    u = xp.dot(h, W_prima)

    # y = softmax(u) estabilizada numericamente sobre todo el vocabulario |V|
    u -= xp.max(u, axis=1, keepdims=True)
    exponenciales = xp.exp(u)
    del u
    y = exponenciales / xp.sum(exponenciales, axis=1, keepdims=True)
    del exponenciales

    return h, y


def retropropagar_y_actualizar(
    x: xp.ndarray,
    t: xp.ndarray,
    h: xp.ndarray,
    y: xp.ndarray,
    W: xp.ndarray,
    W_prima: xp.ndarray,
    eta: float,
    C: float = 1.0,
) -> tuple[xp.ndarray, xp.ndarray, float]:
    """
    Retropropagacion de errores y actualizacion de gradientes para las matrices W y W'.

    Formulacion matricial Softmax Completo:
    e = y - t
    EH = e * W'^T
    W' = W' - eta * (h^T * e) / L
    W = W - eta * (1 / C) * (x^T * EH) / L

    :param x: Matriz de contexto (L x |V|).
    :param t: Matriz de la palabra objetivo deseada (L x |V|).
    :param h: Vector/matriz de activacion de la capa oculta (L x N).
    :param y: Vector/matriz de probabilidades de salida Softmax (L x |V|).
    :param W: Matriz de pesos de entrada (|V| x N).
    :param W_prima: Matriz de pesos de salida (N x |V|).
    :param eta: Tasa de aprendizaje.
    :param C: Cantidad de palabras en el contexto.
    :return: Tupla (W, W_prima, perdida_promedio).
    """
    L = x.shape[0]

    # Vector de error en la capa de salida: e = y - t  ->  Forma: (L, |V|)
    e = y - t

    # Calculo de perdida Cross-Entropy promedio del lote
    probabilidades_objetivo = xp.sum(y * t, axis=1)
    probabilidades_estables = xp.maximum(probabilidades_objetivo, 1e-12)
    perdida_promedio = float(-xp.mean(xp.log(probabilidades_estables)))

    # Error retropropagado a la capa oculta: EH = e * W'^T  ->  Forma: (L, N)
    EH = xp.dot(e, W_prima.T)

    # Actualizacion de W': W' = W' - eta * (h^T * e) / L  ->  Forma: (N, |V|)
    delta_W_prima = xp.dot(h.T, e) / L
    del e
    W_prima -= eta * delta_W_prima
    del delta_W_prima

    # Actualizacion de W: W = W - eta * (1 / C) * (x^T * EH) / L  ->  Forma: (|V|, N)
    delta_W = xp.dot(x.T, EH) / (L * C)
    del EH
    W -= eta * delta_W
    del delta_W

    return W, W_prima, perdida_promedio


def guardar_modelo(modelo: dict, ruta_archivo: str | Path) -> None:
    """
    Almacena el estado completo del modelo en un archivo binario comprimido .npz.

    :param modelo: Diccionario que contiene W, W_prima, vocabulario_palabras, historial_perdida, etc.
    :param ruta_archivo: Ruta del archivo de destino (.npz).
    :return: None.
    """
    destino = Path(ruta_archivo)
    destino.parent.mkdir(parents=True, exist_ok=True)

    W = modelo["W"]
    W_prima = modelo["W_prima"]
    vocabulario_palabras = modelo["vocabulario_palabras"]

    W_np = cp.asnumpy(W) if USAR_CUPY else np.asarray(W)
    W_prima_np = cp.asnumpy(W_prima) if USAR_CUPY else np.asarray(W_prima)

    datos_guardar = {
        "W": W_np,
        "W_prima": W_prima_np,
        "vocabulario_palabras": np.asarray(vocabulario_palabras, dtype=object),
        "epoca_actual": int(modelo.get("epoca_actual", 0)),
        "historial_perdida": np.asarray(modelo.get("historial_perdida", []), dtype=np.float32),
        "tamanio_vocabulario": int(len(vocabulario_palabras)),
        "dimension_embedding": int(W_np.shape[1]),
    }

    if "configuracion" in modelo:
        datos_guardar["configuracion_json"] = str(modelo["configuracion"])

    np.savez_compressed(destino, **datos_guardar)
    print(f"Modelo guardado exitosamente en: '{destino}'")


def cargar_modelo(ruta_archivo: str | Path) -> dict:
    """
    Carga y restaura el estado del modelo desde un archivo comprimido .npz.

    :param ruta_archivo: Ruta al archivo .npz de respaldo.
    :return: Diccionario con W, W_prima, vocabulario_palabras, epoca_actual e historial_perdida.
    """
    origen = Path(ruta_archivo)
    if not origen.exists():
        raise FileNotFoundError(f"No se encontro el archivo de modelo en: '{origen}'")

    datos = np.load(origen, allow_pickle=True)
    W_np = datos["W"]
    W_prima_np = datos["W_prima"]

    W = xp.asarray(W_np, dtype=xp.float32)
    W_prima = xp.asarray(W_prima_np, dtype=xp.float32)
    vocabulario_palabras = datos["vocabulario_palabras"]
    epoca_actual = int(datos.get("epoca_actual", 0))
    historial_perdida = [float(v) for v in datos.get("historial_perdida", [])]

    configuracion = None
    if "configuracion_json" in datos:
        configuracion_str = str(datos["configuracion_json"])
        configuracion = ast.literal_eval(configuracion_str)

    modelo = {
        "W": W,
        "W_prima": W_prima,
        "vocabulario_palabras": vocabulario_palabras,
        "epoca_actual": epoca_actual,
        "historial_perdida": historial_perdida,
        "tamanio_vocabulario": len(vocabulario_palabras),
        "dimension_embedding": W.shape[1],
    }

    if configuracion is not None:
        modelo["configuracion"] = configuracion

    print(f"Modelo cargado exitosamente desde: '{origen}' (Epoca cargada: {epoca_actual})")
    return modelo
