"""
Modulo de tokenizacion para el preprocesamiento del corpus de texto.
"""

import re
from pathlib import Path


class Tokenizador:
    """Clase encargada del preprocesamiento y tokenizacion de texto en estilo imperativo puro."""

    def __init__(self, estrategia: str = "palabra"):
        """
        Inicializa el tokenizador.
        :param estrategia: "palabra" para expresiones regulares o "bpe" para Byte Pair Encoding.
        """
        self.estrategia = estrategia.lower()
        self.patron_palabra = re.compile(r"[a-záéíóúüñ]+", re.IGNORECASE)

    def tokenizar_texto(self, texto: str) -> list[str]:
        """
        Divide una cadena de texto en una lista de tokens utilizando un bucle for imperativo.
        :param texto: Texto de entrada.
        :return: Lista de tokens en minusculas.
        """
        if self.estrategia == "palabra":
            texto_minusculas = texto.lower()
            return self.patron_palabra.findall(texto_minusculas)
        elif self.estrategia == "bpe":
            return self._tokenizar_bpe(texto)
        else:
            raise ValueError(
                f"Estrategia de tokenizacion no soportada: {self.estrategia}"
            )

    def tokenizar_archivo(self, ruta_archivo: Path) -> list[str]:
        """
        Lee un archivo de texto y retorna la lista completa de tokens.
        :param ruta_archivo: Ruta al archivo del corpus.
        :return: Lista de tokens extraidos.
        """
        with open(ruta_archivo, "r", encoding="utf-8", errors="ignore") as archivo:
            contenido = archivo.read()
        return self.tokenizar_texto(contenido)

    def _tokenizar_bpe(self, texto: str) -> list[str]:
        """Tokenizacion BPE utilizando estilo imperativo."""
        texto_minusculas = texto.lower()
        return self.patron_palabra.findall(texto_minusculas)
