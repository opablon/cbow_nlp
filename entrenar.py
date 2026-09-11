#!/usr/bin/env python3
"""
Script de ejecucion autonoma para el entrenamiento de la red neuronal CBOW mediante funciones puras.
Permite iniciar un entrenamiento desde cero o reanudar un checkpoint (.npz) desde la terminal.
"""

import argparse
import sys
from pathlib import Path

# Asegurar acceso a los modulos del directorio codigo/
ruta_raiz = Path(__file__).resolve().parent
if str(ruta_raiz) not in sys.path:
    sys.path.append(str(ruta_raiz))

from codigo.configuracion import cargar_configuracion
from codigo.entrenador_cbow import entrenar
from codigo.modelo_cbow_cupy import guardar_modelo


def parsear_argumentos() -> argparse.Namespace:
    """
    Parsea los argumentos de linea de comandos para el entrenamiento por consola.

    :return: Objeto Namespace con los argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description="Entrenamiento independiente por consola de redes neuronales CBOW (CuPy / NumPy)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configuracion.yaml",
        help="Ruta al archivo YAML de configuracion (por defecto: configuracion.yaml).",
    )
    parser.add_argument(
        "--epocas",
        type=int,
        default=None,
        help="Sobrescribe la cantidad de epocas a entrenar.",
    )
    parser.add_argument(
        "--lote",
        type=int,
        default=None,
        help="Sobrescribe el tamaño de mini-lote (ejemplo: 2048, 4096).",
    )
    parser.add_argument(
        "--tasa",
        type=float,
        default=None,
        help="Sobrescribe la tasa de aprendizaje eta (ejemplo: 0.2).",
    )
    parser.add_argument(
        "--reanudar",
        action="store_true",
        help="Activa el modo de reanudacion desde un checkpoint .npz.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Ruta al archivo .npz desde el cual reanudar (requiere --reanudar).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Funcion principal de punto de entrada para ejecutar el pipeline de entrenamiento CBOW.

    :return: None.
    """
    args = parsear_argumentos()

    print("==========================================================================")
    print("      ENTRENAMIENTO CBOW LOCAL - LINEA DE COMANDOS (TERMINAL IDE)        ")
    print("==========================================================================")

    # 1. Carga de configuracion desde YAML
    configuracion = cargar_configuracion(args.config)

    # Sobrescrituras opcionales desde CLI
    if args.epocas is not None:
        configuracion["cantidad_epocas"] = args.epocas
    if args.lote is not None:
        configuracion["tamanio_lote"] = args.lote
    if args.tasa is not None:
        configuracion["tasa_aprendizaje"] = args.tasa
    if args.reanudar:
        configuracion["reanudar_entrenamiento"] = True
    if args.checkpoint is not None:
        configuracion["ruta_checkpoint"] = args.checkpoint

    # 2. Ejecucion del entrenamiento matricial
    modelo = entrenar(configuracion)

    # 3. Guardado atomico final del modelo entrenado
    directorio_respaldos = Path(configuracion.get("directorio_respaldos", "respaldos"))
    tamanio_ventana = configuracion.get("tamanio_ventana", 5)
    epoca_final = modelo.get("epoca_actual", configuracion.get("cantidad_epocas", 10))

    ruta_guardado_final = (
        directorio_respaldos / f"modelo_cbow_w{tamanio_ventana}_epoca_{epoca_final}.npz"
    )
    guardar_modelo(modelo, ruta_guardado_final)

    print("\n==========================================================================")
    print("                     ENTRENAMIENTO FINALIZADO CON EXITO                  ")
    print("==========================================================================")
    print(f"Perdida final alcanzada: {modelo['historial_perdida'][-1]:.4f}")
    print(f"Modelo resguardado exitosamente en: '{ruta_guardado_final}'")


if __name__ == "__main__":
    main()
