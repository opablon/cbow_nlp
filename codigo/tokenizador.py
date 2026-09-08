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

    def _tokenizar_bpe(self, texto: str, num_merges: int = 50) -> list[str]:
        """
        Tokenización por subpalabras BPE (Byte Pair Encoding) basada en la teoría (Sennrich et al., Slide 33 de Fragmentación.pdf).
        Intenta utilizar la librería 'tokenizers' de Hugging Face si está disponible; de lo contrario, aplica
        la implementación pura en Python del algoritmo Sennrich.
        """
        texto_minusculas = texto.lower()
        
        # 1. Intentar usar la librería 'tokenizers' de HuggingFace si está instalada (Slide 38-39)
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers, trainers
            tokenizer_hf = Tokenizer(models.BPE(unk_token="<UNK>"))
            tokenizer_hf.pre_tokenizer = pre_tokenizers.Whitespace()
            palabras = self.patron_palabra.findall(texto_minusculas)
            
            # Entrenar BPE rápido sobre las palabras únicas del texto
            palabras_unicas = list(set(palabras))
            if len(palabras_unicas) > 0:
                trainer = trainers.BpeTrainer(vocab_size=min(5000, max(256, len(palabras_unicas))), special_tokens=["<UNK>"])
                tokenizer_hf.train_from_iterator(palabras_unicas, trainer=trainer)
                encoding = tokenizer_hf.encode(texto_minusculas)
                return encoding.tokens
        except ImportError:
            pass

        # 2. Algoritmo BPE nativo en Python (Sennrich, Haddow & Birch, 2015, Slide 33)
        palabras = self.patron_palabra.findall(texto_minusculas)
        if not palabras:
            return []

        import collections
        
        # Inicializar vocabulario con caracteres separados por espacio y sufijo </w>
        vocab_bpe = collections.defaultdict(int)
        for p in palabras:
            palabra_espaciada = " ".join(list(p)) + " </w>"
            vocab_bpe[palabra_espaciada] += 1

        def get_stats(vocab):
            pairs = collections.defaultdict(int)
            for word, freq in vocab.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[symbols[i], symbols[i + 1]] += freq
            return pairs

        def merge_vocab(pair, v_in):
            v_out = {}
            bigram = re.escape(" ".join(pair))
            patron = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
            remplazo = "".join(pair)
            for word in v_in:
                w_out = patron.sub(remplazo, word)
                v_out[w_out] = v_in[word]
            return v_out

        # Aplicar fusion (merges) de los pares mas frecuentes
        for _ in range(num_merges):
            pairs = get_stats(vocab_bpe)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab_bpe = merge_vocab(best, vocab_bpe)

        # Reconstruir la lista de tokens subpalabra
        tokens_finales = []
        for word_str in vocab_bpe.keys():
            subwords = word_str.replace(" </w>", "").split()
            tokens_finales.extend(subwords)

        return tokens_finales if tokens_finales else palabras

