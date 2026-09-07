"""
Modulo para la evaluacion de similaridad entre palabras del vocabulario.
Implementa producto interno por defecto y similaridad de coseno como parametro opcional.
"""

import numpy as np
from .modelo_cbow_cupy import ModeloCbowCuPy, USAR_CUPY, xp
from .generador_vocabulario import GeneradorVocabulario


class EvaluadorSimilaridad:
    """Clase encargada de evaluar la similaridad de palabras a partir de la matriz de embeddings W en estilo imperativo."""

    def __init__(
        self, modelo: ModeloCbowCuPy, generador_vocabulario: GeneradorVocabulario
    ):
        """
        Inicializa el evaluador de similaridad.
        :param modelo: Instancia del modelo ModeloCbowCuPy entrenado.
        :param generador_vocabulario: Instancia de GeneradorVocabulario con las palabras cargadas.
        """
        self.modelo = modelo
        self.vocabulario = generador_vocabulario

    def buscar_palabras_similares(
        self,
        palabra_buscada: str,
        top_k: int = 10,
        tipo_similaridad: str = "producto_interno",
    ) -> list[tuple[str, float]]:
        """
        Busca las palabras mas similares a una palabra objetivo dada utilizando bucles imperativos.
        :param palabra_buscada: Palabra objetivo ingresada por el usuario.
        :param top_k: Cantidad de palabras mas similares a retornar.
        :param tipo_similaridad: "producto_interno" (default) o "coseno".
        :return: Lista de tuplas (palabra, puntaje_similaridad).
        """
        palabra_normalizada = palabra_buscada.lower().strip()

        if palabra_normalizada not in self.vocabulario.palabra_a_indice:
            print(
                f"Advertencia: La palabra '{palabra_buscada}' no se encuentra en el vocabulario."
            )
            palabra_normalizada = self.vocabulario.token_desconocido

        indice_objetivo = self.vocabulario.palabra_a_indice[palabra_normalizada]

        # Matriz completa de embeddings W (|V| x N)
        matriz_W = self.modelo.matriz_pesos_entrada

        # Conversión adaptativa si matriz_W es un Tensor de PyTorch
        if hasattr(matriz_W, "detach"):
            matriz_W = xp.asarray(matriz_W.detach().cpu().numpy())

        vector_palabra = matriz_W[indice_objetivo, :]

        if tipo_similaridad == "producto_interno":
            # Producto interno (producto punto) entre el vector objetivo y todos los vectores de W
            puntajes = xp.dot(matriz_W, vector_palabra)
        elif tipo_similaridad == "coseno":
            # Similaridad de coseno: (v_w . v_i) / (||v_w|| * ||v_i||)
            norma_vector = xp.linalg.norm(vector_palabra)
            normas_matriz = xp.linalg.norm(matriz_W, axis=1)
            producto_punto = xp.dot(matriz_W, vector_palabra)
            puntajes = producto_punto / (normas_matriz * norma_vector + 1e-12)
        else:
            raise ValueError(
                f"Tipo de similaridad no soportado: '{tipo_similaridad}'. Use 'producto_interno' o 'coseno'."
            )

        # Convertir arreglos a NumPy para ordenamiento
        if USAR_CUPY and hasattr(xp, "asnumpy"):
            puntajes_np = xp.asnumpy(puntajes)
        else:
            puntajes_np = np.asarray(puntajes)

        # Obtener los indices de mayor a menor puntaje mediante ordenamiento
        indices_ordenados = np.argsort(puntajes_np)[::-1]

        resultados = []
        for indice in indices_ordenados:
            # Excluir la propia palabra buscada del ranking de resultados
            if indice == indice_objetivo:
                continue

            if indice in self.vocabulario.indice_a_palabra:
                palabra_similar = self.vocabulario.indice_a_palabra[indice]
            else:
                palabra_similar = self.vocabulario.token_desconocido

            puntaje = float(puntajes_np[indice])
            resultados.append((palabra_similar, puntaje))

            if len(resultados) >= top_k:
                break

        return resultados
