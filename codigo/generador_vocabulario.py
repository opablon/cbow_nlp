"""
Modulo para la construccion del vocabulario y mapeos de palabras a indices.
Soporta seleccion aleatoria directa con semilla reproducible y seleccion mutuamente excluyente de tamaño de diccionario.
"""

import random
from collections import Counter


class GeneradorVocabulario:
    """Clase encargada de la generacion del diccionario y conversion de tokens a indices en estilo imperativo puro."""

    def __init__(self, token_desconocido: str = "<UNK>"):
        """
        Inicializa el generador de vocabulario.
        :param token_desconocido: Token especial para palabras fuera de vocabulario (por defecto "<UNK>").
        """
        self.token_desconocido = token_desconocido
        self.palabra_a_indice = {}
        self.indice_a_palabra = {}
        self.frecuencias_palabras = Counter()
        self.tamanio_vocabulario = 0

    def construir_vocabulario(
        self,
        lista_tokens: list[str],
        criterio_seleccion: str = "cantidad",
        cantidad_palabras_unicas: int | None = 15000,
        porcentaje_palabras_unicas: float | None = None,
        frecuencia_minima: int = 1,
        semilla_aleatoria: int = 26,
    ) -> None:
        """
        Construye los diccionarios palabra_a_indice e indice_a_palabra.
        Selecciona las palabras de manera aleatoria directa utilizando una semilla reproducible en estilo imperativo.

        :param lista_tokens: Lista completa de tokens extraidos del corpus.
        :param criterio_seleccion: "cantidad" para usar cantidad_palabras_unicas o "porcentaje" para usar porcentaje_palabras_unicas.
        :param cantidad_palabras_unicas: Cantidad fija de palabras unicas a seleccionar al azar.
        :param porcentaje_palabras_unicas: Porcentaje entre 0.0 y 1.0 de palabras unicas a seleccionar al azar.
        :param frecuencia_minima: Frecuencia minima para incluir una palabra (por defecto 1).
        :param semilla_aleatoria: Semilla para garantizar la reproducibilidad del muestreo aleatorio (por defecto 26).
        """
        self.frecuencias_palabras = Counter()
        for token in lista_tokens:
            self.frecuencias_palabras[token] += 1

        # Filtrado imperativo por frecuencia minima con bucle for
        palabras_filtradas = []
        for palabra, frec in self.frecuencias_palabras.items():
            if frec >= frecuencia_minima:
                palabras_filtradas.append(palabra)

        total_unicas_filtradas = len(palabras_filtradas)

        # Seleccion de criterio mutuamente excluyente
        if criterio_seleccion == "cantidad":
            if cantidad_palabras_unicas is None:
                raise ValueError(
                    "Se selecciono el criterio 'cantidad' pero no se especifico 'cantidad_palabras_unicas'."
                )
            limite_efectivo = min(cantidad_palabras_unicas, total_unicas_filtradas)
            print(
                f"Criterio de Vocabulario: CANTIDAD de palabras unicas = {limite_efectivo} (de {total_unicas_filtradas} disponibles)."
            )
        elif criterio_seleccion == "porcentaje":
            if porcentaje_palabras_unicas is None:
                raise ValueError(
                    "Se selecciono el criterio 'porcentaje' pero no se especifico 'porcentaje_palabras_unicas'."
                )
            if not (0.0 < porcentaje_palabras_unicas <= 1.0):
                raise ValueError(
                    "El parametro 'porcentaje_palabras_unicas' debe estar en el rango (0.0, 1.0]."
                )
            limite_efectivo = max(
                1, int(total_unicas_filtradas * porcentaje_palabras_unicas)
            )
            print(
                f"Criterio de Vocabulario: PORCENTAJE de palabras unicas = {porcentaje_palabras_unicas:.1%} ({limite_efectivo} palabras)."
            )
        else:
            raise ValueError(
                f"Criterio de seleccion desconocido: '{criterio_seleccion}'. Use 'cantidad' o 'porcentaje'."
            )

        # Seleccion aleatoria directa sin ordenar previamente para mayor velocidad
        random.seed(semilla_aleatoria)
        random.shuffle(palabras_filtradas)

        palabras_seleccionadas = []
        contador = 0
        for palabra in palabras_filtradas:
            if contador < limite_efectivo:
                palabras_seleccionadas.append(palabra)
                contador += 1

        # Inicializar los diccionarios reservando el indice 0 para <UNK>
        self.palabra_a_indice = {}
        self.palabra_a_indice[self.token_desconocido] = 0

        self.indice_a_palabra = {}
        self.indice_a_palabra[0] = self.token_desconocido

        indice_actual = 1
        for palabra in palabras_seleccionadas:
            self.palabra_a_indice[palabra] = indice_actual
            self.indice_a_palabra[indice_actual] = palabra
            indice_actual += 1

        self.tamanio_vocabulario = len(self.palabra_a_indice)
        print(
            f"Vocabulario construido exitosamente al azar (semilla={semilla_aleatoria}). Tamanio total: {self.tamanio_vocabulario} tokens."
        )

    def convertir_tokens_a_indices(self, lista_tokens: list[str]) -> list[int]:
        """
        Convierte una lista de tokens de texto a sus correspondientes indices enteros utilizando un bucle for imperativo.
        Cualquier palabra fuera del vocabulario se reemplaza por el indice de <UNK>.
        :param lista_tokens: Lista de palabras en texto.
        :return: Lista de indices enteros.
        """
        indice_unk = self.palabra_a_indice[self.token_desconocido]
        lista_indices = []
        for token in lista_tokens:
            if token in self.palabra_a_indice:
                lista_indices.append(self.palabra_a_indice[token])
            else:
                lista_indices.append(indice_unk)
        return lista_indices

    def convertir_indices_a_tokens(self, lista_indices: list[int]) -> list[str]:
        """
        Convierte una lista de indices enteros a sus palabras originales utilizando un bucle for imperativo.
        :param lista_indices: Lista de indices enteros.
        :return: Lista de cadenas de texto.
        """
        lista_tokens = []
        for idx_entero in lista_indices:
            if idx_entero in self.indice_a_palabra:
                lista_tokens.append(self.indice_a_palabra[idx_entero])
            else:
                lista_tokens.append(self.token_desconocido)
        return lista_tokens
