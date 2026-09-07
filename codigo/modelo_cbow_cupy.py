"""
Modulo de la red neuronal CBOW implementada en CuPy GPU / NumPy.
Optimizado para maxima velocidad de entrenamiento mediante procesamiento por lotes vectorizado en GPU.
"""

import os
import sys
from pathlib import Path
import numpy as np

# Configurar rutas de librerias CUDA instaladas en .venv para CuPy
ruta_base_venv = Path(__file__).resolve().parent.parent / ".venv"
carpeta_nvidia = (
    ruta_base_venv
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
    / "nvidia"
)

if carpeta_nvidia.exists():
    rutas_lib = []
    for subcarpeta in carpeta_nvidia.glob("*"):
        carpeta_lib = subcarpeta / "lib"
        if carpeta_lib.exists():
            rutas_lib.append(str(carpeta_lib))

    if len(rutas_lib) > 0:
        ld_actual = os.environ.get("LD_LIBRARY_PATH", "")
        if ld_actual != "":
            rutas_lib.append(ld_actual)
        os.environ["LD_LIBRARY_PATH"] = ":".join(rutas_lib)

try:
    import cupy as cp

    # Verificar ejecucion real en GPU de CuPy
    _ = cp.zeros((1,), dtype=cp.float32)
    _ = cp.random.seed(26)
    USAR_CUPY = True
    xp = cp
except Exception as error_cupy:
    print(
        f"Advertencia: CuPy GPU no disponible en este entorno ({error_cupy}). Ejecutando con NumPy en CPU."
    )
    USAR_CUPY = False
    xp = np


class ModeloCbowCuPy:
    """Red Neuronal CBOW implementada con procesamiento por lotes vectorizado para ejecucion acelerada en GPU NVIDIA o CPU."""

    def __init__(
        self,
        tamanio_vocabulario: int,
        dimension_embedding: int = 100,
        semilla_aleatoria: int = 26,
    ):
        """
        Inicializa las matrices de pesos W y W' en la GPU o CPU.
        :param tamanio_vocabulario: Cardinalidad del vocabulario |V|.
        :param dimension_embedding: Numero de neuronas en la capa oculta N.
        :param semilla_aleatoria: Semilla para reproducibilidad (por defecto 26).
        """
        self.tamanio_vocabulario = tamanio_vocabulario
        self.dimension_embedding = dimension_embedding

        xp.random.seed(semilla_aleatoria)

        # Matriz W de entrada (|V| x N): cada fila corresponde a la representacion de una palabra
        # Forma: (|V|, N)
        limite = 0.5 / dimension_embedding
        self.matriz_pesos_entrada = xp.random.uniform(
            -limite, limite, (tamanio_vocabulario, dimension_embedding)
        ).astype(xp.float32)

        # Matriz W' de salida (N x |V|): cada columna corresponde a los pesos hacia la palabra predicha
        # Forma: (N, |V|)
        self.matriz_pesos_salida = xp.zeros(
            (dimension_embedding, tamanio_vocabulario), dtype=xp.float32
        )

    def propagar_hacia_adelante(self, indices_contexto_batch: xp.ndarray) -> tuple:
        """
        Propagacion hacia adelante vectorizada por lotes para B muestras.
        Calcula el vector promedio de la capa oculta h, las activaciones lineales u y la salida Softmax y.
        :param indices_contexto_batch: Arreglo de indices de contexto de forma (B, C) donde C = 2 * W.
        :return: Tupla (vector_oculto_h, activacion_lineal_u, probabilidades_y).
        """
        # B = cantidad de muestras en el lote, C = cantidad total de palabras de contexto
        B, C = indices_contexto_batch.shape

        # h es el promedio de las filas de W correspondientes a los indices del contexto
        # h = (1 / C) * sum_{c=1}^C W[I_c, :]^T
        # Forma de vectores_contexto: (B, C, N)
        vectores_contexto = self.matriz_pesos_entrada[indices_contexto_batch]

        # Forma de vector_oculto_h: (B, N)
        vector_oculto_h = xp.mean(vectores_contexto, axis=1)

        # u = h * W' (Activacion lineal no normalizada de la capa de salida)
        # Forma de activacion_lineal_u: (B, |V|)
        activacion_lineal_u = xp.dot(vector_oculto_h, self.matriz_pesos_salida)

        # Estabilizacion numerica para el calculo de Softmax por lote:
        # Se resta el valor maximo de cada muestra max(activacion_lineal_u) antes de exponencializar
        # para evitar el desbordamiento numerico (overflow) al calcular e^x en coma flotante.
        # Forma de activacion_estabilizada: (B, |V|)
        activacion_estabilizada = activacion_lineal_u - xp.max(
            activacion_lineal_u, axis=1, keepdims=True
        )
        exponenciales = xp.exp(activacion_estabilizada)

        # Forma de probabilidades_y: (B, |V|)
        probabilidades_y = exponenciales / xp.sum(exponenciales, axis=1, keepdims=True)

        return vector_oculto_h, activacion_lineal_u, probabilidades_y

    def retropropagar_y_actualizar(
        self,
        indices_contexto_batch: xp.ndarray,
        indices_palabra_objetivo_batch: xp.ndarray,
        vector_oculto_h: xp.ndarray,
        probabilidades_y: xp.ndarray,
        tasa_aprendizaje: float,
    ) -> float:
        """
        Calcula los gradientes vectoriales por lote y actualiza las matrices W' y W a máxima velocidad en la GPU.
        :param indices_contexto_batch: Arreglo de indices de contexto de forma (B, C).
        :param indices_palabra_objetivo_batch: Arreglo de indices objetivo de forma (B,).
        :param vector_oculto_h: Vector promedio de la capa oculta h de forma (B, N).
        :param probabilidades_y: Vector de probabilidades Softmax y de forma (B, |V|).
        :param tasa_aprendizaje: Tasa de aprendizaje eta.
        :return: Valor de la perdida promedio del lote.
        """
        B, C = indices_contexto_batch.shape

        # Calculo de la perdida promedio del lote: E = -log(y_O)
        # Se utiliza el valor 1e-12 para evitar indeterminacion matematica por logaritmo de cero (-infinito)
        # Forma de probabilidades_objetivo: (B,)
        probabilidades_objetivo = probabilidades_y[
            xp.arange(B), indices_palabra_objetivo_batch
        ]
        probabilidades_estables = xp.maximum(probabilidades_objetivo, 1e-12)
        perdida_promedio = float(-xp.mean(xp.log(probabilidades_estables)))

        # Vector de error de salida por lote: e_{b, j} = y_{b, j} - t_{b, j}
        # Se utiliza .copy() para crear una copia independiente en memoria y no mutar por referencia
        # el arreglo original probabilidades_y que podria requerirse posteriormente sin modificaciones.
        # Forma de vector_error_salida: (B, |V|)
        vector_error_salida = probabilidades_y.copy()
        vector_error_salida[xp.arange(B), indices_palabra_objetivo_batch] -= 1.0

        # Actualizacion matricial vectorizada para W' (N x |V|):
        # Gradiente promedio sobre el lote: (1 / B) * h^T * e
        # Forma de gradiente_W_prima: (N, |V|)
        gradiente_W_prima = xp.dot(vector_oculto_h.T, vector_error_salida) / B
        self.matriz_pesos_salida -= tasa_aprendizaje * gradiente_W_prima

        # Error retropropagado a la capa oculta: E_H = e * W'^T
        # Forma de vector_error_oculto_EH: (B, N)
        vector_error_oculto_EH = xp.dot(vector_error_salida, self.matriz_pesos_salida.T)

        # Actualizacion matricial vectorizada para W (|V| x N):
        # Forma de delta_entrada: (B, N)
        delta_entrada = (tasa_aprendizaje / (B * C)) * vector_error_oculto_EH

        # Aplicar actualizacion sobre las filas correspondientes al contexto de cada muestra
        for col_c in range(C):
            indices_columna = indices_contexto_batch[:, col_c]  # Forma: (B,)
            xp.add.at(self.matriz_pesos_entrada, indices_columna, -delta_entrada)

        return perdida_promedio

    def guardar_modelo(
        self, ruta_archivo: Path, epoca_actual: int, historial_perdida: list
    ) -> None:
        """
        Guarda el estado completo del modelo y su entrenamiento en un archivo comprimido .npz.
        :param ruta_archivo: Ruta donde almacenar el archivo .npz.
        :param epoca_actual: Numero de la epoca completada.
        :param historial_perdida: Lista de valores de perdida por epoca.
        """
        ruta_archivo = Path(ruta_archivo)
        ruta_archivo.parent.mkdir(parents=True, exist_ok=True)

        # Convertir arreglos de CuPy a NumPy antes de guardar
        if USAR_CUPY:
            pesos_entrada_np = cp.asnumpy(self.matriz_pesos_entrada)
            pesos_salida_np = cp.asnumpy(self.matriz_pesos_salida)
        else:
            pesos_entrada_np = self.matriz_pesos_entrada
            pesos_salida_np = self.matriz_pesos_salida

        np.savez_compressed(
            ruta_archivo,
            matriz_pesos_entrada=pesos_entrada_np,
            matriz_pesos_salida=pesos_salida_np,
            epoca_actual=epoca_actual,
            historial_perdida=np.array(historial_perdida),
            tamanio_vocabulario=self.tamanio_vocabulario,
            dimension_embedding=self.dimension_embedding,
        )
        print(f"Backup guardado exitosamente en: {ruta_archivo}")

    def cargar_modelo(self, ruta_archivo: Path) -> tuple:
        """
        Carga el estado del modelo desde un archivo .npz.
        :param ruta_archivo: Ruta al archivo de backup.
        :return: Tupla (epoca_actual, historial_perdida).
        """
        datos = np.load(ruta_archivo)
        pesos_entrada_np = datos["matriz_pesos_entrada"]
        pesos_salida_np = datos["matriz_pesos_salida"]

        self.matriz_pesos_entrada = xp.asarray(pesos_entrada_np, dtype=xp.float32)
        self.matriz_pesos_salida = xp.asarray(pesos_salida_np, dtype=xp.float32)

        epoca_actual = int(datos["epoca_actual"])
        historial_perdida = []
        for valor in datos["historial_perdida"]:
            historial_perdida.append(float(valor))

        print(f"Modelo cargado desde {ruta_archivo}. Epoca reanudada: {epoca_actual}")
        return epoca_actual, historial_perdida
