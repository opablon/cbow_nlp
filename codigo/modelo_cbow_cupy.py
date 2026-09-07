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
        muestreo_negativo: bool = False,
        cantidad_muestras_negativas: int = 5,
        distribucion_unigrama: xp.ndarray | None = None,
    ):
        """
        Inicializa las matrices de pesos W y W' en la GPU o CPU.
        :param tamanio_vocabulario: Cardinalidad del vocabulario |V|.
        :param dimension_embedding: Numero de neuronas en la capa oculta N.
        :param semilla_aleatoria: Semilla para reproducibilidad (por defecto 26).
        :param muestreo_negativo: Si es True, activa el Muestreo Negativo (Negative Sampling).
        :param cantidad_muestras_negativas: Cantidad de muestras negativas K por elemento.
        :param distribucion_unigrama: Arreglo de probabilidad unigrama P(w)^0.75 para muestras negativas.
        """
        self.tamanio_vocabulario = tamanio_vocabulario
        self.dimension_embedding = dimension_embedding
        self.muestreo_negativo = muestreo_negativo
        self.cantidad_muestras_negativas = cantidad_muestras_negativas

        xp.random.seed(semilla_aleatoria)

        # Matriz W de entrada (|V| x N): cada fila corresponde a la representacion de una palabra
        limite = 0.5 / dimension_embedding
        self.matriz_pesos_entrada = xp.random.uniform(
            -limite, limite, (tamanio_vocabulario, dimension_embedding)
        ).astype(xp.float32)

        # Matriz W' de salida (N x |V|):
        # Softmax: Inicializado en cero absoluto (especificacion clasica).
        # Negative Sampling: Inicializado aleatorio uniforme en [-0.5/N, 0.5/N] para romper simetria.
        if self.muestreo_negativo:
            self.matriz_pesos_salida = xp.random.uniform(
                -limite, limite, (dimension_embedding, tamanio_vocabulario)
            ).astype(xp.float32)
        else:
            self.matriz_pesos_salida = xp.zeros(
                (dimension_embedding, tamanio_vocabulario), dtype=xp.float32
            )

        # Preparar distribucion de probabilidad unigrama en GPU/CPU
        if distribucion_unigrama is not None:
            self.distribucion_unigrama = xp.asarray(distribucion_unigrama, dtype=xp.float32)
        else:
            prob_uniforme = np.ones(tamanio_vocabulario, dtype=np.float32) / tamanio_vocabulario
            self.distribucion_unigrama = xp.asarray(prob_uniforme, dtype=xp.float32)

    def propagar_hacia_adelante(self, indices_contexto_batch: xp.ndarray) -> tuple:
        """
        Propagacion hacia adelante vectorizada por lotes para B muestras.
        Calcula el vector promedio de la capa oculta h, las activaciones lineales u y la salida Softmax y (si aplica).
        :param indices_contexto_batch: Arreglo de indices de contexto de forma (B, C) donde C = 2 * W.
        :return: Tupla (vector_oculto_h, activacion_lineal_u, probabilidades_y).
        """
        # B = cantidad de muestras en el lote, C = cantidad total de palabras de contexto
        B, C = indices_contexto_batch.shape

        # h es el promedio de las filas de W correspondientes a los indices del contexto
        # h = (1 / C) * sum_{c=1}^C W[I_c, :]^T
        vectores_contexto = self.matriz_pesos_entrada[indices_contexto_batch]
        vector_oculto_h = xp.mean(vectores_contexto, axis=1)

        if self.muestreo_negativo:
            # En Muestreo Negativo se omite la Softmax global masiva O(|V|)
            return vector_oculto_h, None, None

        # u = h * W' (Activacion lineal no normalizada de la capa de salida)
        activacion_lineal_u = xp.dot(vector_oculto_h, self.matriz_pesos_salida)

        # Softmax estabilizada numericante
        activacion_estabilizada = activacion_lineal_u - xp.max(
            activacion_lineal_u, axis=1, keepdims=True
        )
        exponenciales = xp.exp(activacion_estabilizada)
        probabilidades_y = exponenciales / xp.sum(exponenciales, axis=1, keepdims=True)

        return vector_oculto_h, activacion_lineal_u, probabilidades_y

    def retropropagar_y_actualizar(
        self,
        indices_contexto_batch: xp.ndarray,
        indices_palabra_objetivo_batch: xp.ndarray,
        vector_oculto_h: xp.ndarray,
        probabilidades_y: xp.ndarray | None,
        tasa_aprendizaje: float,
    ) -> float:
        """
        Calcula los gradientes vectoriales por lote y actualiza W' y W sin bucles 'for' en Python.
        """
        B, C = indices_contexto_batch.shape

        if self.muestreo_negativo:
            K = self.cantidad_muestras_negativas
            
            # Muestreo de K palabras negativas por cada muestra del lote
            indices_negativos = xp.random.choice(
                self.tamanio_vocabulario,
                size=(B, K),
                p=self.distribucion_unigrama,
            )

            # Salvaguarda Teorica: Excluir activamente el objetivo positivo del conjunto de negativos
            mascara_colision = (indices_negativos == indices_palabra_objetivo_batch[:, None])
            while float(xp.sum(mascara_colision)) > 0:
                num_colisiones = int(xp.sum(mascara_colision))
                nuevos_negativos = xp.random.choice(
                    self.tamanio_vocabulario,
                    size=num_colisiones,
                    p=self.distribucion_unigrama,
                )
                indices_negativos[mascara_colision] = nuevos_negativos
                mascara_colision = (indices_negativos == indices_palabra_objetivo_batch[:, None])

            # W'_salida transpuesto es (|V|, N)
            pesos_salida_T = self.matriz_pesos_salida.T
            
            # Pesos de palabras positivas: (B, N)
            W_pos = pesos_salida_T[indices_palabra_objetivo_batch]
            
            # Pesos de palabras negativas: (B, K, N)
            W_neg = pesos_salida_T[indices_negativos]

            # u_pos = h * W'_pos: (B,)
            u_pos = xp.sum(vector_oculto_h * W_pos, axis=1)
            sigma_pos = 1.0 / (1.0 + xp.exp(-u_pos))
            loss_pos = -xp.log(xp.maximum(sigma_pos, 1e-12))

            # u_neg = h * W'_neg: (B, K)
            u_neg = xp.sum(vector_oculto_h[:, None, :] * W_neg, axis=2)
            sigma_neg_inv = 1.0 / (1.0 + xp.exp(u_neg)) # sigma(-u)
            loss_neg = -xp.log(xp.maximum(sigma_neg_inv, 1e-12))

            perdida_promedio = float(xp.mean(loss_pos + xp.sum(loss_neg, axis=1)))

            # Gradientes respecto a W':
            g_pos = (sigma_pos - 1.0)[:, None]  # (B, 1)
            grad_W_pos = (tasa_aprendizaje / B) * (g_pos * vector_oculto_h)  # (B, N)
            xp.add.at(pesos_salida_T, indices_palabra_objetivo_batch, -grad_W_pos)

            g_neg = (1.0 - sigma_neg_inv)[:, :, None]  # (B, K, 1) -> sigma(u)
            grad_W_neg = (tasa_aprendizaje / B) * (g_neg * vector_oculto_h[:, None, :]).reshape(-1, self.dimension_embedding)
            indices_neg_flat = indices_negativos.ravel()
            xp.add.at(pesos_salida_T, indices_neg_flat, -grad_W_neg)

            # Error retropropagado a la capa oculta EH: (B, N)
            vector_error_oculto_EH = g_pos * W_pos + xp.sum(g_neg * W_neg, axis=1)

            # Actualizacion vectorizada de W (entrada) sin bucles Python
            delta_entrada = (tasa_aprendizaje / (B * C)) * vector_error_oculto_EH
            indices_flat = indices_contexto_batch.ravel()
            delta_repetida = xp.repeat(delta_entrada, C, axis=0)
            xp.add.at(self.matriz_pesos_entrada, indices_flat, -delta_repetida)

            return perdida_promedio

        # --- Softmax Completa ---
        probabilidades_objetivo = probabilidades_y[
            xp.arange(B), indices_palabra_objetivo_batch
        ]
        probabilidades_estables = xp.maximum(probabilidades_objetivo, 1e-12)
        perdida_promedio = float(-xp.mean(xp.log(probabilidades_estables)))

        vector_error_salida = probabilidades_y.copy()
        vector_error_salida[xp.arange(B), indices_palabra_objetivo_batch] -= 1.0

        # Actualizacion W'
        gradiente_W_prima = xp.dot(vector_oculto_h.T, vector_error_salida) / B
        self.matriz_pesos_salida -= tasa_aprendizaje * gradiente_W_prima

        # Error en capa oculta EH: (B, N)
        vector_error_oculto_EH = xp.dot(vector_error_salida, self.matriz_pesos_salida.T)

        # Actualizacion vectorizada de W (entrada) sin bucles Python
        delta_entrada = (tasa_aprendizaje / (B * C)) * vector_error_oculto_EH
        indices_flat = indices_contexto_batch.ravel()
        delta_repetida = xp.repeat(delta_entrada, C, axis=0)
        xp.add.at(self.matriz_pesos_entrada, indices_flat, -delta_repetida)

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
            dist_unigrama_np = (
                cp.asnumpy(self.distribucion_unigrama)
                if hasattr(self, "distribucion_unigrama") and self.distribucion_unigrama is not None
                else None
            )
        else:
            pesos_entrada_np = self.matriz_pesos_entrada
            pesos_salida_np = self.matriz_pesos_salida
            dist_unigrama_np = getattr(self, "distribucion_unigrama", None)

        datos_guardar = {
            "matriz_pesos_entrada": pesos_entrada_np,
            "matriz_pesos_salida": pesos_salida_np,
            "epoca_actual": epoca_actual,
            "historial_perdida": np.array(historial_perdida),
            "tamanio_vocabulario": self.tamanio_vocabulario,
            "dimension_embedding": self.dimension_embedding,
            "muestreo_negativo": self.muestreo_negativo,
            "cantidad_muestras_negativas": self.cantidad_muestras_negativas,
        }
        if dist_unigrama_np is not None:
            datos_guardar["distribucion_unigrama"] = dist_unigrama_np

        np.savez_compressed(
            ruta_archivo,
            **datos_guardar
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

        if "tamanio_vocabulario" in datos:
            self.tamanio_vocabulario = int(datos["tamanio_vocabulario"])
        if "dimension_embedding" in datos:
            self.dimension_embedding = int(datos["dimension_embedding"])
        if "muestreo_negativo" in datos:
            self.muestreo_negativo = bool(datos["muestreo_negativo"])
        if "cantidad_muestras_negativas" in datos:
            self.cantidad_muestras_negativas = int(datos["cantidad_muestras_negativas"])
        if "distribucion_unigrama" in datos:
            self.distribucion_unigrama = xp.asarray(datos["distribucion_unigrama"], dtype=xp.float32)

        print(f"Modelo CuPy/NumPy cargado desde {ruta_archivo}. Epoca reanudada: {epoca_actual} (Muestreo Negativo: {self.muestreo_negativo})")
        return epoca_actual, historial_perdida

