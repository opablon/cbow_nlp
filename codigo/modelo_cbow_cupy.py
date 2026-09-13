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
    vocabulario_palabras: np.ndarray,
    tamanio_ventana: int,
) -> tuple[xp.ndarray, xp.ndarray]:
    """
    Construye las matrices de contexto (x) y palabra objetivo (t) para un mini-lote de muestras
    a partir del subconjunto de tokens correspondientes.

    :param subconjunto_tokens_lote: Subconjunto de tokens de texto del mini-lote (incluyendo margenes de ventana).
    :param vocabulario_palabras: Arreglo del vocabulario |V|.
    :param tamanio_ventana: Tamaño de ventana C/2 a izquierda y a derecha.
    :return: Tupla (matriz_contexto_x, matriz_objetivo_t) de forma (L, |V|).
    """
    tamanio_vocabulario = len(vocabulario_palabras)
    mapeo_vocabulario = {palabra: indice for indice, palabra in enumerate(vocabulario_palabras)}

    limite_inicio = tamanio_ventana
    limite_fin = len(subconjunto_tokens_lote) - tamanio_ventana

    L = max(0, limite_fin - limite_inicio)

    matriz_contexto_x = xp.zeros((L, tamanio_vocabulario), dtype=xp.float32)
    matriz_objetivo_t = xp.zeros((L, tamanio_vocabulario), dtype=xp.float32)

    for indice_lote, i in enumerate(range(limite_inicio, limite_fin)):
        suma_contexto = xp.zeros(tamanio_vocabulario, dtype=xp.float32)
        for offset in range(-tamanio_ventana, tamanio_ventana + 1):
            if offset != 0:
                token_contexto = subconjunto_tokens_lote[i + offset]
                indice_activo = mapeo_vocabulario.get(token_contexto, 0)
                suma_contexto[indice_activo] += 1.0

        matriz_contexto_x[indice_lote] = suma_contexto

        token_objetivo = subconjunto_tokens_lote[i]
        indice_objetivo = mapeo_vocabulario.get(token_objetivo, 0)
        matriz_objetivo_t[indice_lote, indice_objetivo] = 1.0

    return matriz_contexto_x, matriz_objetivo_t


def generar_matriz_muestra_negativa(
    matriz_objetivo_t: xp.ndarray,
    distribucion_unigrama: xp.ndarray,
) -> xp.ndarray:
    """
    Genera una matriz de muestras negativas de forma (L, |V|) mediante muestreo vectorizado.

    :param matriz_objetivo_t: Matriz de la palabra objetivo deseada (L x |V|).
    :param distribucion_unigrama: Distribucion de probabilidad unigrama P(w)^0.75 de forma (|V|,).
    :return: Matriz de muestras negativas de forma (L, |V|).
    """
    L, V = matriz_objetivo_t.shape
    vector_probabilidades = xp.asarray(distribucion_unigrama)

    indices_muestras_negativas = xp.random.choice(V, size=L, p=vector_probabilidades).astype(xp.int32)
    matriz_muestra_negativa = xp.zeros((L, V), dtype=xp.float32)
    matriz_muestra_negativa[xp.arange(L), indices_muestras_negativas] = 1.0

    coincidencias = (xp.sum(matriz_muestra_negativa * matriz_objetivo_t, axis=1) > 0.0)
    intentos = 0
    while xp.any(coincidencias) and intentos < 5:
        cantidad_faltantes = int(xp.sum(coincidencias))
        reemplazos = xp.random.choice(V, size=cantidad_faltantes, p=vector_probabilidades).astype(xp.int32)
        indices_coincidentes = xp.where(coincidencias)[0]

        matriz_muestra_negativa[indices_coincidentes] = 0.0
        matriz_muestra_negativa[indices_coincidentes, reemplazos] = 1.0

        coincidencias = (xp.sum(matriz_muestra_negativa * matriz_objetivo_t, axis=1) > 0.0)
        intentos += 1

    return matriz_muestra_negativa


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
    Propagacion hacia adelante matricial para el lote de muestras.
    Calcula el vector de la capa oculta h, las excitaciones de salida u y las probabilidades Softmax y.

    Formulacion matricial:
    h = (1 / C) * x * W
    u = h * W'
    y = softmax(u)

    :param x: Matriz de contexto (L x |V|).
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
    gradiente_W_prima = xp.dot(h.T, e) / L
    W_prima -= eta * gradiente_W_prima

    # Actualizacion de W: W = W - eta * (1 / C) * (x^T * EH) / L  ->  Forma: (|V|, N)
    gradiente_W = xp.dot(x.T, EH) / (L * C)
    W -= eta * gradiente_W

    return W, W_prima, perdida_promedio


def propagar_y_actualizar_muestreo_negativo(
    x: xp.ndarray,
    t: xp.ndarray,
    distribucion_unigrama: xp.ndarray,
    cantidad_negativos: int,
    W: xp.ndarray,
    W_prima: xp.ndarray,
    eta: float,
    C: float = 1.0,
) -> tuple[xp.ndarray, xp.ndarray, float]:
    """
    Propagacion hacia adelante y retropropagacion para el modelo CBOW utilizando Muestreo Negativo.

    :param x: Matriz de contexto (L x |V|).
    :param t: Matriz de la palabra objetivo deseada (L x |V|).
    :param distribucion_unigrama: Distribucion de probabilidad unigrama P(w)^0.75 de forma (|V|,).
    :param cantidad_negativos: Cantidad de palabras de ruido muestreadas por ejemplo.
    :param W: Matriz de pesos de entrada (|V| x N).
    :param W_prima: Matriz de pesos de salida (N x |V|).
    :param eta: Tasa de aprendizaje.
    :param C: Cantidad de palabras en el contexto de entrada (2 * tamanio_ventana).
    :return: Tupla (W, W_prima, perdida_promedio).
    """
    L = x.shape[0]

    # h = (1 / C) * x * W  -> Forma: (L, N)
    h = (1.0 / C) * xp.dot(x, W)

    # Vector de salida de la palabra objetivo deseada obtenido via multiplicacion matricial -> Forma: (L, N)
    vector_salida_palabra_objetivo = xp.dot(t, W_prima.T)

    # Excitacion para la palabra objetivo deseada u_objetivo = sum(h * vector_salida_palabra_objetivo) -> Forma: (L, 1)
    excitacion_palabra_objetivo = xp.sum(h * vector_salida_palabra_objetivo, axis=1, keepdims=True)
    activacion_sigmoide_palabra_objetivo = 1.0 / (1.0 + xp.exp(-excitacion_palabra_objetivo))

    # Error de salida para la palabra objetivo deseada (t_deseado = 1.0) -> Forma: (L, 1)
    error_palabra_objetivo = activacion_sigmoide_palabra_objetivo - 1.0

    # Acumuladores matriciales de error retropropagado (EH) y gradiente (W')
    EH = error_palabra_objetivo * vector_salida_palabra_objetivo  # Forma: (L, N)
    gradiente_W_prima = xp.dot(h.T, error_palabra_objetivo * t)  # Forma: (N, |V|)

    perdida_palabra_objetivo = -xp.log(activacion_sigmoide_palabra_objetivo + 1e-12)
    perdida_muestras_negativas = 0.0

    # Procesar cada una de las palabras de ruido/muestras negativas
    for _ in range(cantidad_negativos):
        matriz_muestra_negativa = generar_matriz_muestra_negativa(
            matriz_objetivo_t=t,
            distribucion_unigrama=distribucion_unigrama,
        )  # Forma: (L, |V|)

        vector_salida_muestra_negativa = xp.dot(matriz_muestra_negativa, W_prima.T)  # Forma: (L, N)
        excitacion_muestra_negativa = xp.sum(h * vector_salida_muestra_negativa, axis=1, keepdims=True)  # Forma: (L, 1)
        activacion_sigmoide_muestra_negativa = 1.0 / (1.0 + xp.exp(-excitacion_muestra_negativa))

        # Error de salida para la palabra de ruido/muestra negativa (t_deseado = 0.0) -> Forma: (L, 1)
        error_muestra_negativa = activacion_sigmoide_muestra_negativa - 0.0

        EH += error_muestra_negativa * vector_salida_muestra_negativa
        gradiente_W_prima += xp.dot(h.T, error_muestra_negativa * matriz_muestra_negativa)

        perdida_muestras_negativas += -xp.log(1.0 - activacion_sigmoide_muestra_negativa + 1e-12)

        del matriz_muestra_negativa

    perdida_promedio = float(xp.mean(perdida_palabra_objetivo + perdida_muestras_negativas))

    # Actualizacion de W' mediante suma de gradientes matriciales
    W_prima -= (eta / L) * gradiente_W_prima

    # Actualizacion de W: W = W - eta * (1 / C) * (x^T * EH) / L -> Forma: (|V|, N)
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
