# TP1: Representación Contextual de Palabras con CBOW

**Asignatura**: Aprendizaje Automático Avanzado  
**Carrera**: Tecnicatura Universitaria en Inteligencia Artificial (UNAHUR)  
**Profesor**: Juan Miguel Santos (LIDEC)  

Este repositorio contiene la implementación del modelo de redes neuronales **CBOW (Continuous Bag-of-Words)** para Procesamiento de Lenguaje Natural (PLN), acelerado por GPU NVIDIA mediante **Python, CuPy y NumPy**.

---

## 🚀 Instalación y Configuración del Entorno con `uv`

El proyecto utiliza el gestor de paquetes de alto rendimiento **`uv`**.

### 1. Clonar el Repositorio
```bash
git clone https://github.com/opablon/cbow_nlp.git
cd cbow_nlp
```

### 2. Crear e Instalar el Entorno Virtual
```bash
# Crear el entorno virtual con uv
uv venv .venv

# Activar el entorno virtual (Linux/macOS)
source .venv/bin/activate

# Activar el entorno virtual (Windows PowerShell)
# .venv\Scripts\activate

# Instalar las dependencias declaradas en pyproject.toml
uv pip install -e .
```

> **Detección Automática de Hardware**: El módulo `codigo/modelo_cbow_cupy.py` detecta automáticamente la disponibilidad de GPU NVIDIA con CuPy / CUDA 12. Si no se detecta una GPU compatible, conmuta de forma transparente al motor de cómputo NumPy en CPU.

---

## 📁 Organización del Proyecto

```text
cbow_nlp/
├── README.md                               # Instrucciones de instalacion, configuracion y uso
├── pyproject.toml                          # Dependencias del proyecto para uv
├── configuracion.yaml                      # Hiperparametros centralizados por defecto
├── entrenar.py                             # Script ejecutable por consola para entrenamiento local
├── TP1_CBOW_Procesamiento_Lenguaje_Natural.ipynb  # Notebook interactivo para analisis y evaluacion
├── datos/                                  # Directorio del corpus de texto (ej. corpus.txt)
├── respaldos/                              # Directorio de copias de seguridad de modelos (.npz)
└── codigo/                                 # Modulos Python reutilizables
    ├── __init__.py                         # Identificador del paquete de Python
    ├── configuracion.py                    # Carga y guardado de hiperparametros YAML
    ├── tokenizador.py                      # Tokenizacion por Palabra (Regex)
    ├── generador_vocabulario.py            # Construccion del vocabulario y mapeos
    ├── modelo_cbow_cupy.py                 # Red CBOW matricial acelerada por GPU (CuPy / NumPy)
    └── entrenador_cbow.py                  # Bucle de entrenamiento por lotes y resguardos
```

> **Nota**: El archivo del corpus de entrenamiento (`datos/corpus.txt`) se ubica en el directorio `datos/` y está excluido del control de versiones mediante `.gitignore`.

---

## ⚙️ Configuración del Corpus y Parámetros del Proyecto

Todos los parámetros del proyecto (ruta del corpus, tamaño de lote, semilla aleatoria, tamaño de ventana, etc.) están centralizados en `configuracion.yaml`.

### 1. Selección de Fuente del Corpus de Texto
Para entrenar el modelo con tu propio corpus de texto o con el archivo provisto por la cátedra:
1. Coloca tu archivo de texto dentro de la carpeta `datos/` (por ejemplo `datos/corpus.txt`).
2. Modifica el atributo `ruta_corpus` en `configuracion.yaml`:
   ```yaml
   ruta_corpus: datos/corpus.txt
   ```

### 2. Parámetros Principales de Entrenamiento
- **`incluir_puntuacion_y_numeros`**: Incluir signos de puntuación y números en la tokenización (`true` o `false`, por defecto: `true`).
- **`tamanio_ventana`**: Tamaños de ventana de contexto $m$ a cada lado de la palabra objetivo (por defecto: `4` para $m = 4$ con $C = 8$ palabras contextuales totales; se evalúa también $m = 5$ para $C = 10$).
- **`dimension_embedding`**: Dimensión de la capa oculta o vector de embedding $N$ (por defecto: `100`).
- **`tasa_aprendizaje`**: Tasa de aprendizaje $\eta$ para la actualización de gradientes (por defecto: `0.3`).
- **`cantidad_epocas`**: Número total de épocas de entrenamiento a ejecutar (por defecto: `500`).
- **`frecuencia_respaldo`**: Frecuencia en épocas para guardar respaldos automáticos `.npz` (por defecto: `100`).
- **`semilla_aleatoria`**: Semilla para garantizar reproducibilidad en la inicialización de pesos (por defecto: `26`).

### 3. Modelo de Salida: Softmax Completo
El modelo calcula las probabilidades Softmax sobre la totalidad del vocabulario $|V|$ para cada mini-lote de tamaño $L$:
- Activación de la capa oculta: $h = \frac{1}{C} x W \in \mathbb{R}^{L \times N}$
- Puntuación de salida: $u = h W' \in \mathbb{R}^{L \times |V|}$
- Probabilidades de salida: $y = \text{softmax}(u) \in \mathbb{R}^{L \times |V|}$
- Vector de error: $e = y - t \in \mathbb{R}^{L \times |V|}$
- Pérdida Cross-Entropy promedio: $E = -\frac{1}{L} \sum_{l=1}^L \log(y_{l, \text{objetivo}} + 1e-12)$

donde $W \in \mathbb{R}^{|V| \times N}$ es la matriz de pesos de entrada y $W' \in \mathbb{R}^{N \times |V|}$ es la matriz de pesos de salida.

### 4. Tamaño de Lote (Batch Size $L$)
El parámetro `tamanio_lote` ($L$) se configura preferentemente en potencias de 2 (ej. `512`, `1024`, `2048`, `4096`) para maximizar el aprovechamiento del paralelismo matricial en la GPU.

#### Sugerencias de `tamanio_lote` ($L$) según la VRAM de la GPU:

| VRAM de la GPU | Tamaño de Lote Recomendado (`tamanio_lote`) | Consideraciones |
| :--- | :---: | :--- |
| **< 4 GB** (GPUs de entrada / integradas) | `256` - `512` | Previene errores de memoria excesiva (*Out of Memory* - OOM). |
| **4 GB – 8 GB** (GPUs gama media) | `1024` - `2048` | Ofrece un equilibrio óptimo entre velocidad de cómputo y consumo de memoria. |
| **8 GB – 16 GB+** (GPUs gama alta) | `4096` - `8192` | Maximiza el rendimiento de CuPy y reduce drásticamente el tiempo de entrenamiento por época. |

---

## 💻 Flujo de Entrenamiento y Análisis

### 1. Entrenamiento Local por Consola
Para ejecutar el pipeline de entrenamiento desde la terminal:
```bash
# Ejecutar entrenamiento con la configuracion por defecto (configuracion.yaml)
uv run python entrenar.py

# Especificando parametros y sobrescrituras por consola
uv run python entrenar.py --config configuracion.yaml --epocas 500 --lote 2048 --tasa 0.3
```

#### Opciones de Línea de Comandos (CLI):
- `--config`: Ruta al archivo YAML de configuración (por defecto: `configuracion.yaml`).
- `--epocas`: Cantidad de épocas de entrenamiento a ejecutar.
- `--lote`: Tamaño del mini-lote $L$.
- `--tasa`: Tasa de aprendizaje $\eta$.
- `--reanudar`: Activa el modo de reanudación desde un checkpoint `.npz`.
- `--checkpoint`: Ruta al archivo `.npz` desde el cual reanudar.

Al finalizar, se guardará automáticamente la copia de respaldo `.npz` en la carpeta `respaldos/`.

### 2. Análisis y Evaluación en Jupyter Notebook
Abre el notebook `TP1_CBOW_Procesamiento_Lenguaje_Natural.ipynb` para:
1. **Distribución del Corpus**: Extraer el vocabulario y analizar la frecuencia de tokens.
2. **Carga e Inspección de Checkpoints**: Cargar modelos resguardados `.npz` e inspeccionar su configuración.
3. **Gráficos de Pérdida**: Visualizar las curvas de pérdida por época individual o comparar múltiples experimentos ($m=4$ vs $m=5$, con y sin puntuación).
4. **Búsqueda de Similaridad**: Encontrar las palabras más similares mediante producto interno o similaridad de coseno sobre la matriz de embeddings $W$.
5. **Evaluación de Analogías**: Evaluar relaciones semánticas vectoriales mediante la operación $v_A - v_B + v_C \approx v_D$ (ejemplo: `evaluar_analogia("padre", "hombre", "mujer", ...)`).

---

## 🔄 Reanudación del Entrenamiento desde un Checkpoint (.npz)

Para continuar entrenando un modelo existente a partir de una copia de seguridad en formato `.npz`:

```bash
uv run python entrenar.py --reanudar --checkpoint respaldos/modelo_cbow_m4_epoca_500_con_puntuacion.npz --epocas 100
```

Al reanudar, se restaurarán las matrices de pesos $W$ y $W'$, el vocabulario de palabras, la época actual y el historial de pérdida acumulado.
