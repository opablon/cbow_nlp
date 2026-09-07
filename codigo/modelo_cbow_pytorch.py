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
    ):
        """
        Inicializa las capas Embedding y Linear de PyTorch.
        :param tamanio_vocabulario: Cardinalidad del vocabulario |V|.
        :param dimension_embedding: Numero de neuronas en la capa oculta N.
        :param semilla_aleatoria: Semilla para reproducibilidad (por defecto 26).
        """
        if not USAR_TORCH:
            raise ImportError("PyTorch no se encuentra instalado o disponible en este entorno.")
            
        super().__init__()
        self.tamanio_vocabulario = tamanio_vocabulario
        self.dimension_embedding = dimension_embedding

        torch.manual_seed(semilla_aleatoria)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(semilla_aleatoria)

        # Capa Embedding de entrada W (|V| x N)
        # Forma de W: (|V|, N)
        self.capa_embedding = nn.Embedding(tamanio_vocabulario, dimension_embedding)
        
        # Inicialización uniforme en [-0.5/N, 0.5/N] siguiendo la especificacion teórica
        limite = 0.5 / dimension_embedding
        nn.init.uniform_(self.capa_embedding.weight, -limite, limite)

        # Capa Lineal de salida W' (N x |V|) sin sesgo (bias=False)
        # Forma de W': (N, |V|)
        self.capa_salida = nn.Linear(dimension_embedding, tamanio_vocabulario, bias=False)
        nn.init.zeros_(self.capa_salida.weight)

        # Mover modelo al dispositivo adecuado (GPU o CPU)
        self.to(dispositivo_torch)

        # Atributos de compatibilidad con la interfaz de CuPy
        self.matriz_pesos_entrada = self.capa_embedding.weight
        self.matriz_pesos_salida = self.capa_salida.weight
        
        # Optimizador para actualizar pesos
        self.optimizador = optim.SGD(self.parameters(), lr=0.025)
        self.funcion_perdida = nn.CrossEntropyLoss()

    def propagar_hacia_adelante(self, indices_contexto_batch) -> tuple:
        """
        Propagacion hacia adelante vectorizada por lotes.
        :param indices_contexto_batch: Tensor de indices de contexto de forma (B, C).
        :return: Tupla (vector_oculto_h, activacion_lineal_u, probabilidades_y).
        """
        # B = cantidad de muestras en el lote, C = cantidad de palabras de contexto
        # vectores_contexto de forma: (B, C, N)
        vectores_contexto = self.capa_embedding(indices_contexto_batch)
        
        # Vector promedio h de forma: (B, N)
        vector_oculto_h = torch.mean(vectores_contexto, dim=1)

        # Activacion lineal u = h * W' de forma: (B, |V|)
        activacion_lineal_u = self.capa_salida(vector_oculto_h)

        # Softmax de forma: (B, |V|)
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
        :param indices_contexto_batch: Tensor de contexto (B, C).
        :param indices_palabra_objetivo_batch: Tensor objetivo (B,).
        :param vector_oculto_h: Tensor promedio oculto (B, N).
        :param probabilidades_y: Tensor Softmax (B, |V|).
        :param tasa_aprendizaje: Tasa de aprendizaje eta.
        :return: Valor de la perdida promedio.
        """
        # Asegurar tasa de aprendizaje actualizada en el optimizador
        for grupo in self.optimizador.param_groups:
            grupo["lr"] = tasa_aprendizaje

        self.optimizador.zero_grad()

        # Re-calcular activacion lineal para la gráfica de autograd
        vectores_ctx = self.capa_embedding(indices_contexto_batch)
        h = torch.mean(vectores_ctx, dim=1) # (B, N)
        logits_u = self.capa_salida(h)      # (B, |V|)

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

        np.savez_compressed(
            ruta_archivo,
            matriz_pesos_entrada=pesos_entrada_np,
            matriz_pesos_salida=pesos_salida_np,
            epoca_actual=epoca_actual,
            historial_perdida=np.array(historial_perdida),
            tamanio_vocabulario=self.tamanio_vocabulario,
            dimension_embedding=self.dimension_embedding,
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

        print(f"Modelo PyTorch cargado desde {ruta_archivo}. Epoca reanudada: {epoca_actual}")
        return epoca_actual, historial_perdida
