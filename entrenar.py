#!/usr/bin/env python3
"""
Script autónomo de entrenamiento para la red neuronal CBOW desde la terminal.
Permite ejecutar el entrenamiento completo o reanudar desde un checkpoint (.npz),
reutilizando de forma estricta los módulos nativos del paquete `codigo/`.
"""

import argparse
import sys
from pathlib import Path

# Asegurar acceso a los módulos del directorio codigo/ si se ejecuta desde cualquier directorio
ruta_raiz = Path(__file__).resolve().parent
if str(ruta_raiz) not in sys.path:
    sys.path.append(str(ruta_raiz))

from codigo.configuracion import Configuracion
from codigo.tokenizador import Tokenizador
from codigo.generador_vocabulario import GeneradorVocabulario
from codigo.modelo_cbow_cupy import ModeloCbowCuPy, USAR_CUPY
from codigo.modelo_cbow_pytorch import ModeloCbowPyTorch
from codigo.entrenador_cbow import EntrenadorCbow


def parsear_argumentos() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos para el entrenamiento por consola."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento independiente por consola de redes neuronales CBOW (Local / IDE)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configuracion.yaml",
        help="Ruta al archivo YAML de configuración (por defecto: configuracion.yaml).",
    )
    parser.add_argument(
        "--motor",
        type=str,
        choices=["pytorch", "cupy"],
        default=None,
        help="Sobrescribe el motor de cómputo GPU ('pytorch' o 'cupy').",
    )
    parser.add_argument(
        "--epocas",
        type=int,
        default=None,
        help="Sobrescribe la cantidad de épocas a entrenar.",
    )
    parser.add_argument(
        "--lote",
        type=int,
        default=None,
        help="Sobrescribe el tamaño de mini-lote (ej. 2048, 4096).",
    )
    parser.add_argument(
        "--tasa",
        type=float,
        default=None,
        help="Sobrescribe la tasa de aprendizaje eta (ej. 0.025, 0.2).",
    )
    parser.add_argument(
        "--reanudar",
        action="store_true",
        help="Activa el modo de reanudación desde un checkpoint .npz.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Ruta al archivo .npz desde el cual reanudar (requiere --reanudar).",
    )
    return parser.parse_args()


def main():
    args = parsear_argumentos()

    print("==========================================================================")
    print("      ENTRENAMIENTO CBOW LOCAL - LÍNEA DE COMANDOS (TERMINAL IDE)        ")
    print("==========================================================================")

    # 1. Carga de Configuración Centralizada
    ruta_yaml = Path(args.config)
    if not ruta_yaml.exists():
        print(f"Error: No se encontró el archivo de configuración '{ruta_yaml}'.")
        sys.exit(1)

    config = Configuracion(str(ruta_yaml))

    # Sobrescrituras desde CLI si fueron especificadas
    if args.motor is not None:
        config.motor_computo = args.motor
    if args.epocas is not None:
        config.cantidad_epocas = args.epocas
    if args.lote is not None:
        config.tamanio_lote = args.lote
    if args.tasa is not None:
        config.tasa_aprendizaje = args.tasa
    if args.reanudar:
        config.reanudar_entrenamiento = True
    if args.checkpoint is not None:
        config.ruta_checkpoint = args.checkpoint

    print(f"Configuración cargada desde: {ruta_yaml}")
    print(f" - Modo Reanudación: {config.reanudar_entrenamiento}")
    if config.reanudar_entrenamiento:
        print(f" - Ruta Checkpoint: {config.ruta_checkpoint}")
    print(f" - Motor de Cómputo: {config.motor_computo}")
    print(f" - Corpus: {config.ruta_corpus}")
    print(f" - Dimensión Embedding (N): {config.dimension_embedding}")
    print(f" - Ventana Contexto (W): {config.tamanio_ventana}")
    print(f" - Tamaño de Lote (B): {config.tamanio_lote}")
    print(f" - Tasa Aprendizaje (eta): {config.tasa_aprendizaje}")
    print(f" - Épocas a Procesar: {config.cantidad_epocas}")
    print(f" - Muestreo Negativo: {config.muestreo_negativo}")
    print(f" - Resguardos: {'Activados (cada ' + str(config.frecuencia_respaldo) + ' épocas)' if config.hacer_respaldo else 'Desactivados'}")

    # 2. Carga y Tokenización del Corpus
    ruta_corpus = Path(config.ruta_corpus)
    if not ruta_corpus.exists():
        print(f"Error: El archivo de corpus '{ruta_corpus}' no existe.")
        sys.exit(1)

    print(f"\n[1/4] Tokenizando corpus desde '{ruta_corpus}'...")
    tokenizador = Tokenizador(estrategia=config.estrategia_tokenizacion)
    tokens_corpus = tokenizador.tokenizar_archivo(ruta_corpus)
    print(f"Tokens totales en el corpus: {len(tokens_corpus):,}")

    # 3. Vocabulario y Persistencia
    print("\n[2/4] Procesando vocabulario y mapeos de índices...")
    vocabulario = GeneradorVocabulario(token_desconocido=config.token_desconocido)
    ruta_vocab_json = Path(config.directorio_respaldos) / "vocabulario.json"

    if config.reanudar_entrenamiento:
        print(f"Modo Reanudación: Cargando vocabulario preexistente desde '{ruta_vocab_json}'...")
        if not ruta_vocab_json.exists():
            print(f"Error crítico: Se solicitó reanudar entrenamiento, pero no existe el archivo '{ruta_vocab_json}'.")
            sys.exit(1)
        vocabulario.cargar_vocabulario(ruta_vocab_json)
        print(f"Vocabulario cargado exitosamente ({vocabulario.tamanio_vocabulario:,} palabras).")
    else:
        print("Modo Entrenamiento Limpio: Construyendo vocabulario...")
        vocabulario.construir_vocabulario(
            lista_tokens=tokens_corpus,
            criterio_seleccion=config.criterio_seleccion_vocabulario,
            cantidad_palabras_unicas=config.cantidad_palabras_unicas,
            porcentaje_palabras_unicas=config.porcentaje_palabras_unicas,
            frecuencia_minima=config.frecuencia_minima,
            semilla_aleatoria=config.semilla_aleatoria,
        )
        vocabulario.guardar_vocabulario(ruta_vocab_json)
        print(f"Vocabulario guardado automáticamente en '{ruta_vocab_json}'.")

    indices_corpus = vocabulario.convertir_tokens_a_indices(tokens_corpus)

    distribucion_unigrama = (
        vocabulario.obtener_distribucion_unigrama(exponente=0.75)
        if config.muestreo_negativo
        else None
    )

    # 4. Instanciación del Modelo de Cómputo
    print(f"\n[3/4] Inicializando motor de cómputo '{config.motor_computo}'...")
    if config.motor_computo == "pytorch":
        modelo_cbow = ModeloCbowPyTorch(
            tamanio_vocabulario=vocabulario.tamanio_vocabulario,
            dimension_embedding=config.dimension_embedding,
            muestreo_negativo=config.muestreo_negativo,
            cantidad_muestras_negativas=config.cantidad_muestras_negativas,
            distribucion_unigrama=distribucion_unigrama,
            semilla_aleatoria=config.semilla_aleatoria,
        )
    elif config.motor_computo == "cupy":
        if not USAR_CUPY:
            print("Advertencia: CuPy no se encuentra instalado o disponible en este sistema.")
            print("Cambiando automáticamente a motor PyTorch...")
            config.motor_computo = "pytorch"
            modelo_cbow = ModeloCbowPyTorch(
                tamanio_vocabulario=vocabulario.tamanio_vocabulario,
                dimension_embedding=config.dimension_embedding,
                muestreo_negativo=config.muestreo_negativo,
                cantidad_muestras_negativas=config.cantidad_muestras_negativas,
                distribucion_unigrama=distribucion_unigrama,
                semilla_aleatoria=config.semilla_aleatoria,
            )
        else:
            modelo_cbow = ModeloCbowCuPy(
                tamanio_vocabulario=vocabulario.tamanio_vocabulario,
                dimension_embedding=config.dimension_embedding,
                muestreo_negativo=config.muestreo_negativo,
                cantidad_muestras_negativas=config.cantidad_muestras_negativas,
                distribucion_unigrama=distribucion_unigrama,
                semilla_aleatoria=config.semilla_aleatoria,
            )
    else:
        print(f"Error: Motor de cómputo '{config.motor_computo}' no reconocido.")
        sys.exit(1)

    # 5. Instanciación del Entrenador
    entrenador = EntrenadorCbow(
        modelo=modelo_cbow,
        tasa_aprendizaje=config.tasa_aprendizaje,
        directorio_respaldos=config.directorio_respaldos,
        hacer_respaldo=config.hacer_respaldo,
        frecuencia_respaldo=config.frecuencia_respaldo,
    )

    # 6. Bucle de Entrenamiento (Limpio o Reanudación)
    print("\n[4/4] Ejecutando bucle de entrenamiento...")

    if config.reanudar_entrenamiento:
        ruta_checkpoint = Path(config.ruta_checkpoint)
        if not ruta_checkpoint.exists():
            print(f"Error: El archivo de checkpoint '{ruta_checkpoint}' no existe.")
            sys.exit(1)

        epoca_cargada, _ = entrenador.cargar_respaldo(ruta_checkpoint)
        print(f"Checkpoint cargado: '{ruta_checkpoint.name}' (Época base: {epoca_cargada}).")
        print(f"Reanudando por +{config.cantidad_epocas} épocas adicionales...")

        historial = entrenador.entrenar_adicional(
            indices_tokens=indices_corpus,
            epocas_adicionales=config.cantidad_epocas,
            tamanio_ventana=config.tamanio_ventana,
            tamanio_lote=config.tamanio_lote,
        )
    else:
        historial = entrenador.entrenar(
            indices_tokens=indices_corpus,
            tamanio_ventana=config.tamanio_ventana,
            tamanio_lote=config.tamanio_lote,
            cantidad_epocas=config.cantidad_epocas,
            epoca_inicial=0,
        )

    print("\n==========================================================================")
    print("                     ENTRENAMIENTO FINALIZADO CON ÉXITO                  ")
    print("==========================================================================")
    print(f"Pérdida final alcanzada: {historial[-1]:.4f}")
    if config.hacer_respaldo:
        print(f"Resguardos guardados en el directorio: '{config.directorio_respaldos}/'")
        print(f"Vocabulario guardado en: '{ruta_vocab_json}'")


if __name__ == "__main__":
    main()
