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
) -> tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
    """
    Propagacion hacia adelante matricial One-Hot para el lote de muestras.
    Calcula el vector de la capa oculta h, las excitaciones de salida u y las probabilidades Softmax y.

    Formulacion matricial:
    h = (1 / C) * x * W
    u = h * W'
    y = softmax(u)

    :param x: Matriz One-Hot de contexto (L x |V|).
    :param W: Matriz de pesos de entrada (|V| x N).
    :param W_prima: Matriz de pesos de salida (N x |V|).
    :param C: Numero de palabras en la ventana de contexto (2 * tamanio_ventana).
    :return: Tupla (h, u, y) con las activaciones, excitaciones y probabilidades.
    """
    # h = (1 / C) * x * W  ->  Forma: (L, N)
    h = (1.0 / C) * xp.dot(x, W)

    # u = h * W'  ->  Forma: (L, |V|)
    u = xp.dot(h, W_prima)

    # y = softmax(u) estabilizada numericamente sobre todo el vocabulario |V|
    u_estabilizada = u - xp.max(u, axis=1, keepdims=True)
    exponenciales = xp.exp(u_estabilizada)
    y = exponenciales / xp.sum(exponenciales, axis=1, keepdims=True)

    return h, u, y


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

    :param x: Matriz One-Hot de contexto (L x |V|).
    :param t: Matriz One-Hot de la palabra objetivo deseada (L x |V|).
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
    gradiente_W_prima = xp.dot(h.T, e) / L
    W_prima -= eta * gradiente_W_prima

    # Actualizacion de W: W = W - eta * (1 / C) * (x^T * EH) / L  ->  Forma: (|V|, N)
    gradiente_W = xp.dot(x.T, EH) / (L * C)
    W -= eta * gradiente_W

    return W, W_prima, perdida_promedio


def propagar_y_actualizar_muestreo_negativo(
    x: xp.ndarray,
    t: xp.ndarray,
    indices_negativos: xp.ndarray | np.ndarray,
    W: xp.ndarray,
    W_prima: xp.ndarray,
    eta: float,
    C: float = 1.0,
) -> tuple[xp.ndarray, xp.ndarray, float]:
    """
    Propagacion hacia adelante y retropropagacion para el modelo CBOW utilizando Muestreo Negativo.
    Evalua exclusivamente las palabras seleccionadas (la palabra objetivo deseada y el conjunto de ejemplos negativos P_sel).

    :param x: Matriz One-Hot de contexto (L x |V|).
    :param t: Matriz One-Hot de la palabra objetivo deseada (L x |V|).
    :param indices_negativos: Matriz de indices de palabras negativas seleccionadas de forma excluyente (L x K).
    :param W: Matriz de pesos de entrada (|V| x N).
    :param W_prima: Matriz de pesos de salida (N x |V|).
    :param eta: Tasa de aprendizaje.
    :param C: Cantidad de palabras en el contexto de entrada (2 * tamanio_ventana).
    :return: Tupla (W, W_prima, perdida_promedio).
    """
    L = x.shape[0]

    # h = (1 / C) * x * W  -> Forma: (L, N)
    h = (1.0 / C) * xp.dot(x, W)

    # Extraer el indice de la palabra objetivo deseada p_O a partir de la matriz One-Hot t  -> Forma: (L, 1)
    indices_objetivo = xp.argmax(t, axis=1)[:, None]

    # Matriz de indices de las palabras procesadas: P_sel U {p_O}  -> Forma: (L, 1+K)
    neg_arr = xp.asarray(indices_negativos)
    indices_procesados = xp.concatenate((indices_objetivo, neg_arr), axis=1)

    K_total = indices_procesados.shape[1]

    # Salida deseada de referencia: 1.0 para la palabra positiva (columna 0), 0.0 para las negativas  -> Forma: (L, 1+K)
    salida_deseada = xp.zeros((L, K_total), dtype=xp.float32)
    salida_deseada[:, 0] = 1.0

    # Extraer los vectores de salida v'_j = W'[:, j] para las palabras procesadas  -> Forma: (L, 1+K, N)
    v_prima_procesados = W_prima.T[indices_procesados]

    # Estado de excitacion u_j = (v'_j)^T * h  -> Forma: (L, 1+K)
    excitacion_procesada = xp.sum(h[:, None, :] * v_prima_procesados, axis=2)

    # Funcion de activacion sigmoide logistica sigma(u_j)  -> Forma: (L, 1+K)
    sigmoide_u = 1.0 / (1.0 + xp.exp(-excitacion_procesada))

    # Calculo de la funcion de perdida para Muestreo Negativo
    perdida_positiva = -xp.log(sigmoide_u[:, 0] + 1e-12)
    perdida_negativa = -xp.sum(xp.log(1.0 - sigmoide_u[:, 1:] + 1e-12), axis=1)
    perdida_promedio = float(xp.mean(perdida_positiva + perdida_negativa))

    # Vector de error de prediccion e_j = sigma(u_j) - t_j  -> Forma: (L, 1+K)
    error_salida_procesado = sigmoide_u - salida_deseada

    # EH = sum_{j in P_sel} (sigma(u_j) - t_j) * v'_j  -> Forma: (L, N)
    EH = xp.sum(error_salida_procesado[:, :, None] * v_prima_procesados, axis=1)

    # Actualizacion de los vectores de salida v'_j de las palabras procesadas en W'
    # v'_j (nuevo) = v'_j (anterior) - (eta / L) * (sigma(u_j) - t_j) * h
    gradiente_salida_procesado = h[:, None, :] * error_salida_procesado[:, :, None]  # Forma: (L, 1+K, N)
    indices_flat = indices_procesados.flatten()
    gradiente_flat = (gradiente_salida_procesado / L).reshape(-1, W.shape[1]).T  # Forma: (N, L*(1+K))

    xp.add.at(W_prima, (slice(None), indices_flat), -eta * gradiente_flat)

    # Actualizacion de la matriz de entrada W: W = W - eta * (1 / C) * (x^T * EH) / L  -> Forma: (|V|, N)
    gradiente_W = xp.dot(x.T, EH) / (L * C)
    W -= eta * gradiente_W

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
