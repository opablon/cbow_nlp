"""
Modulo de entrenamiento matricial para la red neuronal CBOW.
Coordina la tokenizacion, construccion de vocabulario, matrices x y t, y retropropagacion en GPU/CPU.
"""

import time
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .tokenizador import tokenizar_corpus
from .generador_vocabulario import (
    construir_vocabulario,
    construir_distribucion_unigrama,
)
from .modelo_cbow_cupy import (
    crear_matrices_lote,
    inicializar_pesos,
    propagar_hacia_adelante,
    retropropagar_y_actualizar,
    propagar_y_actualizar_muestreo_negativo,
    guardar_modelo,
    cargar_modelo,
    USAR_CUPY,
    xp,
)


def imprimir_configuracion_entrenamiento(
    estado_reanudacion: str,
    ruta_corpus: str,
    estrategia_tokenizacion: str,
    incluir_puntuacion_y_numeros: bool,
    bpe_tamanio_vocabulario: int,
    tamanio_diccionario_v: int,
    dimension_embedding: int,
    tamanio_ventana: int,
    tasa_aprendizaje: float,
    muestreo_negativo: bool,
    cantidad_muestras_negativas: int,
    tamanio_lote: int,
    cantidad_epocas: int,
    hacer_respaldo: bool,
    frecuencia_respaldo: int,
    directorio_respaldos: str | Path,
    semilla_aleatoria: int,
    duracion_total_segundos: float | None = None,
) -> None:
    """
    Imprime en formato estructurado la configuracion completa del entrenamiento (al inicio o al finalizar).
    """
    titulo = "RESUMEN FINAL DE CONFIGURACIÓN Y RESULTADOS" if duracion_total_segundos is not None else "CONFIGURACIÓN DEL ENTRENAMIENTO"
    print(f"\n================ {titulo} ================")
    print(f"Estado: {estado_reanudacion}")
    print(f"Ruta del corpus: {ruta_corpus}")
    print(f"Estrategia de tokenización: {estrategia_tokenizacion}")
    print(f"Incluir puntuación y números: {incluir_puntuacion_y_numeros}")
    if estrategia_tokenizacion == "bpe":
        print(f"Tamaño vocabulario BPE objetivo: {bpe_tamanio_vocabulario}")
    print(f"Tamaño final del vocabulario (|V|): {tamanio_diccionario_v:,}")
    print(f"Dimensión del embedding (N): {dimension_embedding}")
    C_total = 2 * tamanio_ventana
    print(f"Tamaño de ventana: {tamanio_ventana} palabras a cada lado (C = {C_total} palabras totales)")
    print(f"Tasa de aprendizaje (eta): {tasa_aprendizaje}")
    if muestreo_negativo:
        print(f"Muestreo negativo: Activado (K = {cantidad_muestras_negativas} muestras por ejemplo)")
    else:
        print("Muestreo negativo: Desactivado (Softmax Completa sobre todo |V|)")
    print(f"Tamaño del mini-lote (L): {tamanio_lote}")
    print(f"Cantidad total de épocas: {cantidad_epocas}")
    print(f"Semilla aleatoria: {semilla_aleatoria}")
    if hacer_respaldo:
        print(f"Copias de respaldo: Activadas (Cada {frecuencia_respaldo} épocas en '{directorio_respaldos}')")
    else:
        print("Copias de respaldo: Desactivadas")
    if duracion_total_segundos is not None:
        minutos = int(duracion_total_segundos // 60)
        segundos_resto = duracion_total_segundos % 60
        if minutos > 0:
            print(f"Tiempo total de entrenamiento: {duracion_total_segundos:.2f} s ({minutos} m {segundos_resto:.2f} s)")
        else:
            print(f"Tiempo total de entrenamiento: {duracion_total_segundos:.2f} s")
    print("=================================================================\n")


def entrenar(configuracion_dict: dict) -> dict:
    """
    Ejecuta el pipeline completo de entrenamiento matricial para el modelo CBOW.
    Permite iniciar un entrenamiento limpio o reanudar desde una copia de respaldo (.npz).

    :param configuracion_dict: Diccionario que contiene todos los hiperparametros y rutas.
    :return: Diccionario 'modelo' con las matrices W, W_prima, vocabulario_palabras, historial_perdida y metadatos.
    """
    tiempo_inicio_total = time.time()

    ruta_corpus = configuracion_dict.get("ruta_corpus", "datos/corpus.txt")
    directorio_respaldos = Path(configuracion_dict.get("directorio_respaldos", "respaldos"))
    estrategia_tokenizacion = configuracion_dict.get("estrategia_tokenizacion", "palabra")
    incluir_puntuacion_y_numeros = configuracion_dict.get("incluir_puntuacion_y_numeros", True)
    bpe_tamanio_vocabulario = configuracion_dict.get("bpe_tamanio_vocabulario", 15000)
    bpe_frecuencia_minima = configuracion_dict.get("bpe_frecuencia_minima", 2)
    muestreo_negativo = configuracion_dict.get("muestreo_negativo", False)
    cantidad_muestras_negativas = configuracion_dict.get("cantidad_muestras_negativas", 5)
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

    print("\n--- Iniciando Pipeline de Entrenamiento CBOW Matricial ---")

    # 1. Reanudacion directa desde Checkpoint o Inicialización Completa desde Corpus
    historial_perdida = []
    epoca_inicial = 0
    estado_reanudacion = "Iniciando entrenamiento limpio desde corpus"

    if reanudar_entrenamiento and Path(ruta_checkpoint).exists():
        print("\n[1/3] Cargando modelo y vocabulario desde archivo de respaldo checkpoint...")
        modelo_cargado = cargar_modelo(ruta_checkpoint)
        W = modelo_cargado["W"]
        W_prima = modelo_cargado["W_prima"]
        vocabulario_palabras = modelo_cargado["vocabulario_palabras"]
        epoca_inicial = modelo_cargado["epoca_actual"]
        historial_perdida = modelo_cargado["historial_perdida"]
        tamanio_vocabulario = len(vocabulario_palabras)
        estado_reanudacion = f"Reanudando desde checkpoint '{ruta_checkpoint}' (Época {epoca_inicial})"

        print("\n[2/3] Tokenizando corpus de texto...")
        tokens_corpus = tokenizar_corpus(
            ruta_corpus=ruta_corpus,
            incluir_puntuacion_y_numeros=incluir_puntuacion_y_numeros,
        )
    else:
        print("\n[1/4] Tokenizando corpus de texto...")
        tokens_corpus = tokenizar_corpus(
            ruta_corpus=ruta_corpus,
            incluir_puntuacion_y_numeros=incluir_puntuacion_y_numeros,
        )
        print(f"Tokens totales extraidos del corpus: {len(tokens_corpus):,}")

        print("\n[2/4] Construyendo vocabulario de palabras...")
        vocabulario_palabras = construir_vocabulario(
            ruta_corpus=ruta_corpus,
            estrategia_tokenizacion=estrategia_tokenizacion,
            incluir_puntuacion_y_numeros=incluir_puntuacion_y_numeros,
            bpe_tamanio_vocabulario=bpe_tamanio_vocabulario,
            bpe_frecuencia_minima=bpe_frecuencia_minima,
            token_desconocido=token_desconocido,
        )
        tamanio_vocabulario = len(vocabulario_palabras)

        W, W_prima = inicializar_pesos(
            tamanio_vocabulario=tamanio_vocabulario,
            dimension_embedding=dimension_embedding,
            semilla_aleatoria=semilla_aleatoria,
        )
        if muestreo_negativo:
            xp.random.seed(semilla_aleatoria)
            W_prima = xp.random.uniform(
                -0.1, 0.1, (dimension_embedding, tamanio_vocabulario)
            ).astype(xp.float32)

    imprimir_configuracion_entrenamiento(
        estado_reanudacion=estado_reanudacion,
        ruta_corpus=ruta_corpus,
        estrategia_tokenizacion=estrategia_tokenizacion,
        incluir_puntuacion_y_numeros=incluir_puntuacion_y_numeros,
        bpe_tamanio_vocabulario=bpe_tamanio_vocabulario,
        tamanio_diccionario_v=tamanio_vocabulario,
        dimension_embedding=dimension_embedding,
        tamanio_ventana=tamanio_ventana,
        tasa_aprendizaje=tasa_aprendizaje,
        muestreo_negativo=muestreo_negativo,
        cantidad_muestras_negativas=cantidad_muestras_negativas,
        tamanio_lote=tamanio_lote,
        cantidad_epocas=cantidad_epocas,
        hacer_respaldo=hacer_respaldo,
        frecuencia_respaldo=frecuencia_respaldo,
        directorio_respaldos=directorio_respaldos,
        semilla_aleatoria=semilla_aleatoria,
    )

    # 3. Preparación de la distribución unigrama matricial
    print("\n[3/4] Preparando representaciones matriciales del corpus...")

    distribucion_unigrama = None
    if muestreo_negativo:
        distribucion_unigrama = construir_distribucion_unigrama(
            tokens_corpus=tokens_corpus,
            vocabulario_palabras=vocabulario_palabras,
        )

    total_muestras = max(0, len(tokens_corpus) - (2 * tamanio_ventana))
    cantidad_lotes = int(np.ceil(total_muestras / tamanio_lote))
    print(f"Total de muestras: {total_muestras:,} distribuidas en {cantidad_lotes:,} lotes.")

    cantidad_palabras_contexto = float(2 * tamanio_ventana)

    # 4. Bucle Principal de Entrenamiento por Epocas
    print("\n[4/4] Ejecutando bucle de entrenamiento matricial...")
    for epoca in range(epoca_inicial + 1, epoca_inicial + cantidad_epocas + 1):
        tiempo_inicio = time.time()
        perdida_acumulada = 0.0

        barra_progreso = tqdm(
            range(cantidad_lotes),
            desc=f"Epoca {epoca}/{epoca_inicial + cantidad_epocas}",
            unit="lote",
            leave=True,
        )

        for indice_lote in barra_progreso:
            posicion_inicio = indice_lote * tamanio_lote
            posicion_fin = min(posicion_inicio + tamanio_lote + 2 * tamanio_ventana, len(tokens_corpus))
            subconjunto_tokens_lote = tokens_corpus[posicion_inicio:posicion_fin]

            matriz_contexto_x, matriz_objetivo_t = crear_matrices_lote(
                subconjunto_tokens_lote=subconjunto_tokens_lote,
                vocabulario_palabras=vocabulario_palabras,
                tamanio_ventana=tamanio_ventana,
            )

            if muestreo_negativo:
                W, W_prima, perdida_lote = propagar_y_actualizar_muestreo_negativo(
                    x=matriz_contexto_x,
                    t=matriz_objetivo_t,
                    distribucion_unigrama=distribucion_unigrama,
                    cantidad_negativos=cantidad_muestras_negativas,
                    W=W,
                    W_prima=W_prima,
                    eta=tasa_aprendizaje,
                    C=cantidad_palabras_contexto,
                )
                del matriz_contexto_x, matriz_objetivo_t
            else:
                matriz_oculta_h, matriz_excitacion_u, matriz_probabilidades_y = propagar_hacia_adelante(
                    x=matriz_contexto_x, W=W, W_prima=W_prima, C=cantidad_palabras_contexto
                )

                W, W_prima, perdida_lote = retropropagar_y_actualizar(
                    x=matriz_contexto_x,
                    t=matriz_objetivo_t,
                    h=matriz_oculta_h,
                    y=matriz_probabilidades_y,
                    W=W,
                    W_prima=W_prima,
                    eta=tasa_aprendizaje,
                    C=cantidad_palabras_contexto,
                )
                del matriz_contexto_x, matriz_objetivo_t, matriz_oculta_h, matriz_excitacion_u, matriz_probabilidades_y

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

    duracion_total_segundos = time.time() - tiempo_inicio_total

    imprimir_configuracion_entrenamiento(
        estado_reanudacion=estado_reanudacion,
        ruta_corpus=ruta_corpus,
        estrategia_tokenizacion=estrategia_tokenizacion,
        incluir_puntuacion_y_numeros=incluir_puntuacion_y_numeros,
        bpe_tamanio_vocabulario=bpe_tamanio_vocabulario,
        tamanio_diccionario_v=tamanio_vocabulario,
        dimension_embedding=dimension_embedding,
        tamanio_ventana=tamanio_ventana,
        tasa_aprendizaje=tasa_aprendizaje,
        muestreo_negativo=muestreo_negativo,
        cantidad_muestras_negativas=cantidad_muestras_negativas,
        tamanio_lote=tamanio_lote,
        cantidad_epocas=cantidad_epocas,
        hacer_respaldo=hacer_respaldo,
        frecuencia_respaldo=frecuencia_respaldo,
        directorio_respaldos=directorio_respaldos,
        semilla_aleatoria=semilla_aleatoria,
        duracion_total_segundos=duracion_total_segundos,
    )

    modelo = {
        "W": W,
        "W_prima": W_prima,
        "vocabulario_palabras": vocabulario_palabras,
        "epoca_actual": epoca_inicial + cantidad_epocas,
        "historial_perdida": historial_perdida,
        "configuracion": configuracion_dict,
        "duracion_total_segundos": duracion_total_segundos,
    }

    return modelo
