"""
Modulo de entrenamiento matricial para la red neuronal CBOW mediante funciones puras.
Coordina la tokenizacion, construccion de vocabulario, lotes One-Hot y retropropagacion en GPU/CPU.
"""

import time
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .tokenizador import tokenizar_corpus
from .generador_vocabulario import (
    construir_vocabulario,
    generar_vectores_one_hot,
    crear_matriz_one_hot_lote,
)
from .modelo_cbow_cupy import (
    inicializar_pesos,
    propagar_hacia_adelante,
    retropropagar_y_actualizar,
    guardar_modelo,
    cargar_modelo,
    xp,
    USAR_CUPY,
)


def entrenar(configuracion_dict: dict) -> dict:
    """
    Ejecuta el pipeline completo de entrenamiento matricial One-Hot para el modelo CBOW.
    Permite iniciar un entrenamiento limpio o reanudar desde una copia de respaldo (.npz).

    :param configuracion_dict: Diccionario que contiene todos los hiperparametros y rutas.
    :return: Diccionario 'modelo' con las matrices W, W_prima, vocabulario_palabras, historial_perdida y metadatos.
    """
    ruta_corpus = configuracion_dict.get("ruta_corpus", "datos/corpus.txt")
    directorio_respaldos = Path(configuracion_dict.get("directorio_respaldos", "respaldos"))
    estrategia_tokenizacion = configuracion_dict.get("estrategia_tokenizacion", "palabra")
    incluir_puntuacion_y_numeros = configuracion_dict.get("incluir_puntuacion_y_numeros", True)
    criterio_seleccion_vocabulario = configuracion_dict.get("criterio_seleccion_vocabulario", "porcentaje_palabras")
    cantidad_palabras_unicas = configuracion_dict.get("cantidad_palabras_unicas", 15000)
    porcentaje_palabras_unicas = configuracion_dict.get("porcentaje_palabras_unicas", 1.0)
    bpe_tamanio_vocabulario = configuracion_dict.get("bpe_tamanio_vocabulario", 15000)
    bpe_frecuencia_minima = configuracion_dict.get("bpe_frecuencia_minima", 2)
    bpe_cantidad_fusiones = configuracion_dict.get("bpe_cantidad_fusiones", 1000)
    tamanio_ventana = configuracion_dict.get("tamanio_ventana", 5)
    dimension_embedding = configuracion_dict.get("dimension_embedding", 100)
    tasa_aprendizaje = configuracion_dict.get("tasa_aprendizaje", 0.2)
    cantidad_epocas = configuracion_dict.get("cantidad_epocas", 10)
    hacer_respaldo = configuracion_dict.get("hacer_respaldo", True)
    frecuencia_respaldo = configuracion_dict.get("frecuencia_respaldo", 2)
    tamanio_lote = configuracion_dict.get("tamanio_lote", 2048)
    semilla_aleatoria = configuracion_dict.get("semilla_aleatoria", 26)
    reanudar_entrenamiento = configuracion_dict.get("reanudar_entrenamiento", False)
    ruta_checkpoint = configuracion_dict.get("ruta_checkpoint", "respaldos/modelo_cbow_w5_epoca_10.npz")
    token_desconocido = "<UNK>"

    print("\n--- Iniciando Pipeline de Entrenamiento CBOW Matricial (One-Hot) ---")
    print(f"Corpus: '{ruta_corpus}' | Ventana: {tamanio_ventana} | Batch: {tamanio_lote} | Epocas: {cantidad_epocas}")

    # 1. Tokenizacion del Corpus
    print("\n[1/4] Tokenizando corpus de texto...")
    tokens_corpus = tokenizar_corpus(
        ruta_corpus=ruta_corpus,
        incluir_puntuacion_y_numeros=incluir_puntuacion_y_numeros,
    )
    print(f"Tokens totales extraidos del corpus: {len(tokens_corpus):,}")

    # 2. Construccion del Vocabulario de tamaño |V|
    print("\n[2/4] Construyendo vocabulario de palabras...")
    vocabulario_palabras = construir_vocabulario(
        lista_tokens=tokens_corpus,
        ruta_corpus=ruta_corpus,
        estrategia_tokenizacion=estrategia_tokenizacion,
        bpe_tamanio_vocabulario=bpe_tamanio_vocabulario,
        bpe_frecuencia_minima=bpe_frecuencia_minima,
        bpe_cantidad_fusiones=bpe_cantidad_fusiones,
        criterio_seleccion_vocabulario=criterio_seleccion_vocabulario,
        cantidad_palabras_unicas=cantidad_palabras_unicas,
        porcentaje_palabras_unicas=porcentaje_palabras_unicas,
        token_desconocido=token_desconocido,
        semilla_aleatoria=semilla_aleatoria,
    )
    tamanio_vocabulario = len(vocabulario_palabras)
    print(f"Tamaño del vocabulario |V|: {tamanio_vocabulario:,} elementos")

    # Mapeo de tokens a sus indices enteros basados en la posicion en vocabulario_palabras
    mapeo_vocabulario = {palabra: idx for idx, palabra in enumerate(vocabulario_palabras)}
    idx_unk = 0

    indices_tokens = []
    for token in tokens_corpus:
        if token in mapeo_vocabulario:
            indices_tokens.append(mapeo_vocabulario[token])
        else:
            indices_tokens.append(idx_unk)

    # 3. Extraer indices de contexto y objetivo
    print("\n[3/4] Indexando contextos y palabras objetivo...")
    matriz_contextos_idx, matriz_objetivos_idx = generar_vectores_one_hot(
        indices_tokens=indices_tokens,
        tamanio_vocabulario=tamanio_vocabulario,
        tamanio_ventana=tamanio_ventana,
        tamanio_lote=tamanio_lote,
    )
    total_muestras = len(matriz_objetivos_idx)
    cantidad_lotes = int(np.ceil(total_muestras / tamanio_lote))
    print(f"Total de muestras: {total_muestras:,} distribuidas en {cantidad_lotes:,} lotes.")

    # 4. Inicializacion de Pesos o Reanudacion
    print("\n[4/4] Inicializando matrices de pesos o cargando checkpoint...")
    historial_perdida = []
    epoca_inicial = 0

    if reanudar_entrenamiento and Path(ruta_checkpoint).exists():
        modelo_cargado = cargar_modelo(ruta_checkpoint)
        W = modelo_cargado["W"]
        W_prima = modelo_cargado["W_prima"]
        epoca_inicial = modelo_cargado["epoca_actual"]
        historial_perdida = modelo_cargado["historial_perdida"]
        print(f"Reanudando entrenamiento desde la epoca: {epoca_inicial}")
    else:
        W, W_prima = inicializar_pesos(
            tamanio_vocabulario=tamanio_vocabulario,
            dimension_embedding=dimension_embedding,
            semilla_aleatoria=semilla_aleatoria,
        )

    C_contexto = float(2 * tamanio_ventana)

    # Bucle Principal de Entrenamiento por Epocas
    for epoca in range(epoca_inicial + 1, epoca_inicial + cantidad_epocas + 1):
        tiempo_inicio = time.time()
        perdida_acumulada = 0.0

        barra_progreso = tqdm(
            range(cantidad_lotes),
            desc=f"Epoca {epoca}/{epoca_inicial + cantidad_epocas}",
            unit="lote",
            leave=True,
        )

        for idx_lote in barra_progreso:
            inicio = idx_lote * tamanio_lote
            fin = min(inicio + tamanio_lote, total_muestras)

            sub_ctx_idx = matriz_contextos_idx[inicio:fin]
            sub_obj_idx = matriz_objetivos_idx[inicio:fin]

            # Generar matrices One-Hot x_lote y t_lote bajo demanda en GPU/CPU
            x_lote, t_lote = crear_matriz_one_hot_lote(
                sub_contextos_idx=sub_ctx_idx,
                sub_objetivos_idx=sub_obj_idx,
                tamanio_vocabulario=tamanio_vocabulario,
            )

            # 1. Propagacion hacia adelante matricial One-Hot
            h, u, y = propagar_hacia_adelante(
                x=x_lote, W=W, W_prima=W_prima, C=C_contexto
            )

            # 2. Retropropagacion y actualizacion de pesos
            W, W_prima, perdida_lote = retropropagar_y_actualizar(
                x=x_lote,
                t=t_lote,
                h=h,
                y=y,
                W=W,
                W_prima=W_prima,
                eta=tasa_aprendizaje,
                C=C_contexto,
            )

            # Liberar matrices temporales del lote en GPU
            del x_lote, t_lote, h, u, y
            if hasattr(xp, "get_default_memory_pool"):
                xp.get_default_memory_pool().free_all_blocks()

            perdida_acumulada += perdida_lote
            barra_progreso.set_postfix({"Perdida": f"{perdida_lote:.4f}"})

        tiempo_fin = time.time()
        duracion_epoca = tiempo_fin - tiempo_inicio
        perdida_promedio = perdida_acumulada / cantidad_lotes
        historial_perdida.append(perdida_promedio)

        print(
            f"Epoca {epoca} finalizada. Perdida Promedio: {perdida_promedio:.4f} | "
            f"Tiempo: {duracion_epoca:.2f}s"
        )

        # Autoguardado periodico de respaldos si hacer_respaldo es True
        if hacer_respaldo and (
            epoca % frecuencia_respaldo == 0 or epoca == (epoca_inicial + cantidad_epocas)
        ):
            modelo_temp = {
                "W": W,
                "W_prima": W_prima,
                "vocabulario_palabras": vocabulario_palabras,
                "epoca_actual": epoca,
                "historial_perdida": historial_perdida,
                "configuracion": configuracion_dict,
            }
            ruta_backup = (
                directorio_respaldos
                / f"modelo_cbow_w{tamanio_ventana}_epoca_{epoca}.npz"
            )
            guardar_modelo(modelo_temp, ruta_backup)

    # Estructura del modelo retornado
    modelo = {
        "W": W,
        "W_prima": W_prima,
        "vocabulario_palabras": vocabulario_palabras,
        "epoca_actual": epoca_inicial + cantidad_epocas,
        "historial_perdida": historial_perdida,
        "configuracion": configuracion_dict,
    }

    return modelo
