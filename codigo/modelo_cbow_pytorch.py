"""
Modulo de la red neuronal CBOW implementada en PyTorch GPU (torch.cuda).
Provee una alternativa optimizada en PyTorch que mantiene compatibilidad de interfaz con el modelo CuPy.
"""

from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    USAR_TORCH = True
    dispositivo_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clase_base = nn.Module
except Exception:
    USAR_TORCH = False
    torch = None
    nn = None
    dispositivo_torch = None
    clase_base = object

class ModeloCbowPyTorch(clase_base):
    """Red Neuronal CBOW implementada en PyTorch para aceleracion GPU o CPU."""

    def __init__(
        self,
        tamanio_vocabulario: int,
        dimension_embedding: int = 100,
        semilla_aleatoria: int = 26,
        muestreo_negativo: bool = False,
        cantidad_muestras_negativas: int = 5,
        distribucion_unigrama: np.ndarray | None = None,
    ):
        """
        Inicializa las capas Embedding y Linear de PyTorch.
        :param tamanio_vocabulario: Cardinalidad del vocabulario |V|.
        :param dimension_embedding: Numero de neuronas en la capa oculta N.
        :param semilla_aleatoria: Semilla para reproducibilidad (por defecto 26).
        :param muestreo_negativo: Activa la optimizacion por Muestreo Negativo.
        :param cantidad_muestras_negativas: Numero de muestras negativas K por elemento.
        :param distribucion_unigrama: Probabilidades P(w)^0.75 para muestras negativas.
        """
        if not USAR_TORCH:
            raise ImportError("PyTorch no se encuentra instalado o disponible en este entorno.")
            
        super().__init__()
        self.tamanio_vocabulario = tamanio_vocabulario
        self.dimension_embedding = dimension_embedding
        self.muestreo_negativo = muestreo_negativo
        self.cantidad_muestras_negativas = cantidad_muestras_negativas

        torch.manual_seed(semilla_aleatoria)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(semilla_aleatoria)

        # Capa Embedding de entrada W (|V| x N)
        self.capa_embedding = nn.Embedding(tamanio_vocabulario, dimension_embedding)
        limite = 0.5 / dimension_embedding
        nn.init.uniform_(self.capa_embedding.weight, -limite, limite)

        # Capa Lineal de salida W' (|V| x N)
        # Softmax: Inicializado en cero absoluto (especificacion clasica).
        # Negative Sampling: Inicializado aleatorio uniforme en [-0.5/N, 0.5/N] para romper simetria.
        self.capa_salida = nn.Linear(dimension_embedding, tamanio_vocabulario, bias=False)
        if self.muestreo_negativo:
            nn.init.uniform_(self.capa_salida.weight, -limite, limite)
        else:
            nn.init.zeros_(self.capa_salida.weight)

        # Mover modelo al dispositivo adecuado (GPU o CPU)
        self.to(dispositivo_torch)

        # Preparar distribucion de probabilidad unigrama en PyTorch
        if distribucion_unigrama is not None:
            self.distribucion_unigrama = torch.from_numpy(distribucion_unigrama).float().to(dispositivo_torch)
        else:
            prob_uniforme = np.ones(tamanio_vocabulario, dtype=np.float32) / tamanio_vocabulario
            self.distribucion_unigrama = torch.from_numpy(prob_uniforme).float().to(dispositivo_torch)

        # Atributos de compatibilidad con la interfaz de CuPy
        self.matriz_pesos_entrada = self.capa_embedding.weight
        self.matriz_pesos_salida = self.capa_salida.weight
        
        # Optimizador restrictivo SGD puro (momentum=0, dampening=0, weight_decay=0)
        self.optimizador = optim.SGD(
            self.parameters(),
            lr=0.025,
            momentum=0,
            dampening=0,
            weight_decay=0,
        )
        self.funcion_perdida = nn.CrossEntropyLoss()

    def propagar_hacia_adelante(self, indices_contexto_batch) -> tuple:
        """
        Propagacion hacia adelante vectorizada por lotes.
        :param indices_contexto_batch: Tensor de indices de contexto de forma (B, C).
        :return: Tupla (vector_oculto_h, activacion_lineal_u, probabilidades_y).
        """
        vectores_contexto = self.capa_embedding(indices_contexto_batch)
        vector_oculto_h = torch.mean(vectores_contexto, dim=1)

        if self.muestreo_negativo:
            return vector_oculto_h, None, None

        activacion_lineal_u = self.capa_salida(vector_oculto_h)
        probabilidades_y = torch.softmax(activacion_lineal_u, dim=1)

        return vector_oculto_h, activacion_lineal_u, probabilidades_y

    def retropropagar_y_actualizar(
        self,
        indices_contexto_batch,
        indices_palabra_objetivo_batch,
        vector_oculto_h,
        probabilidades_y,
        tasa_aprendizaje: float,
    ) -> float:
        """
        Calcula gradientes y actualiza los pesos de PyTorch.
        """
        for grupo in self.optimizador.param_groups:
            grupo["lr"] = tasa_aprendizaje

        self.optimizador.zero_grad()

        if self.muestreo_negativo:
            B = indices_contexto_batch.size(0)
            K = self.cantidad_muestras_negativas

            # Muestreo de K negativos por cada muestra del lote
            indices_negativos = torch.multinomial(
                self.distribucion_unigrama,
                num_samples=B * K,
                replacement=True,
            ).view(B, K)

            # Salvaguarda Teorica: Excluir activamente el objetivo positivo de los negativos
            mascara = (indices_negativos == indices_palabra_objetivo_batch.unsqueeze(1))
            while mascara.any():
                num_col = int(mascara.sum().item())
                nuevos = torch.multinomial(self.distribucion_unigrama, num_samples=num_col, replacement=True)
                indices_negativos[mascara] = nuevos
                mascara = (indices_negativos == indices_palabra_objetivo_batch.unsqueeze(1))

            vectores_ctx = self.capa_embedding(indices_contexto_batch)
            h = torch.mean(vectores_ctx, dim=1) # (B, N)

            # Pesos de palabras positivas: (B, N)
            w_pos = self.capa_salida.weight[indices_palabra_objetivo_batch]
            u_pos = torch.sum(h * w_pos, dim=1) # (B,)
            target_pos = torch.ones_like(u_pos)
            loss_pos = nn.functional.binary_cross_entropy_with_logits(u_pos, target_pos, reduction='none')

            # Pesos de palabras negativas: (B, K, N)
            w_neg = self.capa_salida.weight[indices_negativos]
            u_neg = torch.sum(h.unsqueeze(1) * w_neg, dim=2) # (B, K)
            target_neg = torch.zeros_like(u_neg)
            loss_neg = nn.functional.binary_cross_entropy_with_logits(u_neg, target_neg, reduction='none')

            loss_total = torch.mean(loss_pos + torch.sum(loss_neg, dim=1))
            loss_total.backward()
            self.optimizador.step()

            return float(loss_total.item())

        # --- Softmax Completa ---
        vectores_ctx = self.capa_embedding(indices_contexto_batch)
        h = torch.mean(vectores_ctx, dim=1)
        logits_u = self.capa_salida(h)

        perdida_tensor = self.funcion_perdida(logits_u, indices_palabra_objetivo_batch)
        perdida_tensor.backward()
        self.optimizador.step()

        return float(perdida_tensor.item())

    def guardar_modelo(self, ruta_archivo: Path, epoca_actual: int, historial_perdida: list) -> None:
        """
        Guarda el estado del modelo PyTorch en formato comprimido .npz para mantener intercompatibilidad.
        :param ruta_archivo: Ruta del archivo .npz.
        :param epoca_actual: Numero de la epoca completada.
        :param historial_perdida: Lista de perdidas registradas.
        """
        ruta_archivo = Path(ruta_archivo)
        ruta_archivo.parent.mkdir(parents=True, exist_ok=True)

        pesos_entrada_np = self.capa_embedding.weight.detach().cpu().numpy()
        pesos_salida_np = self.capa_salida.weight.detach().cpu().numpy().T # Transpuesto para coincidir con CuPy (N x |V|)
        dist_unigrama_np = (
            self.distribucion_unigrama.detach().cpu().numpy()
            if hasattr(self, "distribucion_unigrama") and self.distribucion_unigrama is not None
            else None
        )

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
        print(f"Backup PyTorch guardado exitosamente en: {ruta_archivo}")

    def cargar_modelo(self, ruta_archivo: Path) -> tuple:
        """
        Carga el estado del modelo desde un archivo .npz.
        :param ruta_archivo: Ruta al archivo de backup.
        :return: Tupla (epoca_actual, historial_perdida).
        """
        datos = np.load(ruta_archivo)
        pesos_entrada_np = datos["matriz_pesos_entrada"]
        pesos_salida_np = datos["matriz_pesos_salida"]

        with torch.no_grad():
            self.capa_embedding.weight.copy_(torch.from_numpy(pesos_entrada_np).to(dispositivo_torch))
            if pesos_salida_np.shape == (self.dimension_embedding, self.tamanio_vocabulario):
                pesos_salida_np = pesos_salida_np.T
            self.capa_salida.weight.copy_(torch.from_numpy(pesos_salida_np).to(dispositivo_torch))

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
            self.distribucion_unigrama = torch.from_numpy(datos["distribucion_unigrama"]).float().to(dispositivo_torch)

        print(f"Modelo PyTorch cargado desde {ruta_archivo}. Epoca reanudada: {epoca_actual} (Muestreo Negativo: {self.muestreo_negativo})")
        return epoca_actual, historial_perdida

