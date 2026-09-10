"""
Modulo de entrenamiento para la red neuronal CBOW optimizado para GPU de alta velocidad.
Administra la iteracion por lotes, medicion de tiempos de ejecucion, reanudacion por epocas adicionales y autoguardado de respaldos.
"""

import time
from pathlib import Path
import numpy as np
from tqdm import tqdm

try:
    import torch
    USAR_TORCH = True
except ImportError:
    USAR_TORCH = False

from .modelo_cbow_cupy import ModeloCbowCuPy, USAR_CUPY, xp

class EntrenadorCbow:
    """Clase encargada de ejecutar el bucle de entrenamiento por lotes vectorizado del modelo CBOW."""

    def __init__(
        self,
        modelo,
        tasa_aprendizaje: float = 0.025,
        directorio_respaldos: str = "respaldos",
        hacer_respaldo: bool = True,
        frecuencia_respaldo: int = 2,
    ):
        """
        Inicializa el entrenador.
        :param modelo: Instancia del modelo ModeloCbowCuPy o ModeloCbowPyTorch.
        :param tasa_aprendizaje: Tasa de aprendizaje eta para la actualizacion de pesos.
        :param directorio_respaldos: Directorio donde almacenar los checkpoints .npz.
        :param hacer_respaldo: Booleano para activar o desactivar el guardado de checkpoints.
        :param frecuencia_respaldo: Cantidad de epocas entre cada backup.
        """
        self.modelo = modelo
        self.tasa_aprendizaje = tasa_aprendizaje
        self.directorio_respaldos = Path(directorio_respaldos)
        self.hacer_respaldo = hacer_respaldo
        self.frecuencia_respaldo = frecuencia_respaldo
        self.historial_perdida = []
        self.epoca_actual_cargada = 0

        if self.hacer_respaldo:
            if not isinstance(self.frecuencia_respaldo, int) or self.frecuencia_respaldo <= 0:
                raise ValueError(
                    f"El parametro 'frecuencia_respaldo' debe ser un entero mayor que 0 cuando 'hacer_respaldo' es True. Valor recibido: {self.frecuencia_respaldo}"
                )

    def generar_lotes_cbow(
        self, indices_tokens: list[int], tamanio_ventana: int, tamanio_lote: int = 2048
    ) -> tuple[list, list, int]:
        """
        Genera los lotes vectorizados (contextos_batch, objetivos_batch) para GPU.
        :param indices_tokens: Lista completa de indices del corpus.
        :param tamanio_ventana: Tamanio de ventana W (a izquierda y a derecha).
        :param tamanio_lote: Cantidad de muestras por lote B (por defecto 2048).
        :return: Tupla (lotes_contexto, lotes_objetivo, total_muestras).
        """
        muestras_contexto = []
        muestras_objetivo = []
        longitud_total = len(indices_tokens)
        
        limite_inicio = tamanio_ventana
        limite_fin = longitud_total - tamanio_ventana

        for i in range(limite_inicio, limite_fin):
            contexto = []
            for idx_izq in range(i - tamanio_ventana, i):
                contexto.append(indices_tokens[idx_izq])
            for idx_der in range(i + 1, i + tamanio_ventana + 1):
                contexto.append(indices_tokens[idx_der])

            muestras_contexto.append(contexto)
            muestras_objetivo.append(indices_tokens[i])

        total_muestras = len(muestras_objetivo)
        
        # Convertir a arreglos NumPy e inmediatamente pre-cargar a VRAM de la GPU
        matriz_contextos = np.array(muestras_contexto, dtype=np.int64 if USAR_TORCH else np.int32)
        matriz_objetivos = np.array(muestras_objetivo, dtype=np.int64 if USAR_TORCH else np.int32)

        es_pytorch = hasattr(self.modelo, "capa_embedding")

        if es_pytorch and USAR_TORCH:
            dispositivo = next(self.modelo.parameters()).device
            vram_contextos = torch.from_numpy(matriz_contextos).to(dispositivo)
            vram_objetivos = torch.from_numpy(matriz_objetivos).to(dispositivo)
        else:
            vram_contextos = xp.asarray(matriz_contextos)
            vram_objetivos = xp.asarray(matriz_objetivos)

        lotes_contexto = []
        lotes_objetivo = []
        
        # Slicing directo en VRAM para eliminar el cuello de botella PCIe
        for i in range(0, total_muestras, tamanio_lote):
            lotes_contexto.append(vram_contextos[i : i + tamanio_lote])
            lotes_objetivo.append(vram_objetivos[i : i + tamanio_lote])

        return lotes_contexto, lotes_objetivo, total_muestras

    def entrenar(
        self,
        indices_tokens: list[int],
        tamanio_ventana: int = 4,
        tamanio_lote: int = 2048,
        cantidad_epocas: int = 10,
        epoca_inicial: int = 0,
    ) -> list[float]:
        """
        Ejecuta el entrenamiento vectorizado acelerado en GPU por el numero de epocas indicado.
        :param indices_tokens: Secuencia entera de tokens del corpus.
        :param tamanio_ventana: Ventana W a izquierda y a derecha.
        :param tamanio_lote: Cantidad de muestras por lote B (por defecto 2048).
        :param cantidad_epocas: Cantidad total de epocas a entrenar.
        :param epoca_inicial: Epoca desde la cual comenzar (0 para entrenamiento nuevo).
        :return: Lista con la perdida promedio por epoca.
        """
        print(f"\n--- Iniciando Entrenamiento CBOW de Alta Velocidad (GPU) ---")
        print(f"Ventana (W): {tamanio_ventana} | Batch: {tamanio_lote} | Epocas Objetivo: {cantidad_epocas} | Tasa: {self.tasa_aprendizaje}")
        
        # Generar lotes vectorizados
        lotes_contexto, lotes_objetivo, total_muestras = self.generar_lotes_cbow(
            indices_tokens=indices_tokens,
            tamanio_ventana=tamanio_ventana,
            tamanio_lote=tamanio_lote,
        )
        cantidad_lotes = len(lotes_contexto)
        print(f"Total de muestras: {total_muestras:,} distribuidas en {cantidad_lotes:,} lotes pre-cargados en VRAM.")

        for epoca in range(epoca_inicial + 1, cantidad_epocas + 1):
            tiempo_inicio = time.time()
            perdida_acumulada = 0.0
            
            # Barra de progreso por lote
            barra_progreso = tqdm(
                range(cantidad_lotes),
                desc=f"Epoca {epoca}/{cantidad_epocas}",
                unit="lote",
                leave=True,
            )

            for idx_lote in barra_progreso:
                ctx_batch = lotes_contexto[idx_lote]
                obj_batch = lotes_objetivo[idx_lote]
                
                # 1. Propagacion hacia adelante vectorizada por lote
                vector_oculto_h, activacion_u, probabilidades_y = self.modelo.propagar_hacia_adelante(ctx_batch)
                
                # 2. Retropropagacion y actualizacion de gradientes por lote
                perdida_lote = self.modelo.retropropagar_y_actualizar(
                    indices_contexto_batch=ctx_batch,
                    indices_palabra_objetivo_batch=obj_batch,
                    vector_oculto_h=vector_oculto_h,
                    probabilidades_y=probabilidades_y,
                    tasa_aprendizaje=self.tasa_aprendizaje,
                    activacion_u=activacion_u,
                )
                
                perdida_acumulada += perdida_lote * len(obj_batch)
                barra_progreso.set_postfix({"Perdida": f"{perdida_lote:.4f}"})

            tiempo_fin = time.time()
            duracion_epoca = tiempo_fin - tiempo_inicio
            perdida_promedio = perdida_acumulada / total_muestras
            self.historial_perdida.append(perdida_promedio)

            velocidad = total_muestras / duracion_epoca
            print(
                f"Epoca {epoca} finalizada. Perdida Promedio: {perdida_promedio:.4f} | "
                f"Tiempo: {duracion_epoca:.2f}s ({velocidad:,.1f} muestras/s)"
            )

            # Autoguardado de respaldos periodicos (gobernado por hacer_respaldo=True)
            if self.hacer_respaldo and (epoca % self.frecuencia_respaldo == 0 or epoca == cantidad_epocas):
                ruta_backup = (
                    self.directorio_respaldos
                    / f"modelo_cbow_w{tamanio_ventana}_epoca_{epoca}.npz"
                )
                self.modelo.guardar_modelo(
                    ruta_archivo=ruta_backup,
                    epoca_actual=epoca,
                    historial_perdida=self.historial_perdida,
                )

        return self.historial_perdida

    def entrenar_adicional(
        self,
        indices_tokens: list[int],
        epocas_adicionales: int = 10,
        tamanio_ventana: int = 4,
        tamanio_lote: int = 2048,
    ) -> list[float]:
        """
        Reanuda el entrenamiento de forma automatica entrenando N epocas adicionales desde el ultimo estado guardado.
        :param indices_tokens: Secuencia entera de tokens del corpus.
        :param epocas_adicionales: Numero de epocas adicionales a entrenar.
        :param tamanio_ventana: Ventana W a izquierda y a derecha.
        :param tamanio_lote: Cantidad de muestras por lote B.
        :return: Lista actualizada con las perdidas acumuladas.
        """
        epoca_inicial = len(self.historial_perdida)
        if self.epoca_actual_cargada > epoca_inicial:
            epoca_inicial = self.epoca_actual_cargada

        cantidad_epocas_total = epoca_inicial + epocas_adicionales
        print(f"\n--- Reanudando Entrenamiento: +{epocas_adicionales} epocas adicionales (de {epoca_inicial} a {cantidad_epocas_total}) ---")

        return self.entrenar(
            indices_tokens=indices_tokens,
            tamanio_ventana=tamanio_ventana,
            tamanio_lote=tamanio_lote,
            cantidad_epocas=cantidad_epocas_total,
            epoca_inicial=epoca_inicial,
        )

    def cargar_respaldo(self, ruta_backup: Path | str) -> tuple[int, list[float]]:
        """
        Carga el estado del modelo y sincroniza la epoca_actual_cargada e historial_perdida del entrenador.
        :param ruta_backup: Ruta al archivo .npz de respaldo.
        :return: Tupla (epoca_cargada, historial_perdida).
        """
        ruta = Path(ruta_backup)
        epoca_cargada, historial = self.modelo.cargar_modelo(ruta)
        self.epoca_actual_cargada = epoca_cargada
        self.historial_perdida = list(historial)
        return epoca_cargada, historial

