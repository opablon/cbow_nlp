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

---

## 📁 Organización del Proyecto

```text
cbow_nlp/
├── README.md                               # Instrucciones de instalacion, configuracion y uso
├── pyproject.toml                          # Dependencias del proyecto para uv
├── configuracion.yaml                      # Hiperparametros centralizados por defecto
├── entrenar.py                             # Script ejecutable por consola para entrenamiento local
├── TP1_CBOW_Procesamiento_Lenguaje_Natural.ipynb  # Notebook interactivo para analisis y evaluacion
└── codigo/                                 # Modulos Python reutilizables
    ├── __init__.py                         # Identificador del paquete de Python
    ├── configuracion.py                    # Carga y guardado de hiperparametros YAML
    ├── tokenizador.py                      # Tokenizacion por Palabra (Regex) y BPE
    ├── generador_vocabulario.py            # Construccion del vocabulario y matrices One-Hot (x y t)
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
- **`estrategia_tokenizacion`**: `'palabra'` (utiliza la totalidad de palabras por orden de aparición) o `'bpe'` (Byte Pair Encoding).
- **`tamanio_ventana`**: Tamaño de la ventana de contexto $C/2$ a izquierda y a derecha (ej. `5` para un contexto total $C = 10$).
- **`dimension_embedding`**: Dimensión de la capa oculta o vector de embedding $N$ (ej. `100`).
- **`tasa_aprendizaje`**: Tasa de aprendizaje $\eta$ para la actualización de gradientes (ej. `0.2`).
- **`cantidad_epocas`**: Número total de épocas de entrenamiento (ej. `10`).

### 3. Muestreo Negativo (Negative Sampling) vs Softmax Completo
- **Softmax Completo (Predeterminado)**:
  - Salida: $y = \text{softmax}(u) \in \mathbb{R}^{B \times |V|}$.
  - Error: $e = y - t \in \mathbb{R}^{B \times |V|}$.
  - Pérdida: $E = -\log(y_{\text{objetivo}})$.
- **Muestreo Negativo (`muestreo_negativo: true`)**:
  - Salida: Función logística $\sigma(u_j) = \frac{1}{1 + e^{-u_j}}$ sobre $P_{\text{sel}} = \{p_{\text{objetivo}}\} \cup P_{\text{negativos}}$.
  - Error: $e_j = \sigma(u_j) - t_j$ para $j \in P_{\text{sel}}$ ($t_{\text{objetivo}} = 1$, $t_{\text{negativo}} = 0$), y $e_j = 0$ para $j \notin P_{\text{sel}}$.
  - Pérdida: $E = -\log(\sigma(u_{\text{objetivo}})) - \sum_{p_n \in P_{\text{negativos}}} \log(\sigma(-u_n))$.

### 4. Tamaño de Lote (Batch Size)
El parámetro `tamanio_lote` se configura preferentemente en potencias de 2 (ej. `512`, `1024`, `2048`, `4096`) para maximizar el aprovechamiento del paralelismo matricial en la GPU.

#### Sugerencias de `tamanio_lote` según la VRAM de la GPU:

| VRAM de la GPU | Tamaño de Lote Recomendado (`tamanio_lote`) | Consideraciones |
| :--- | :---: | :--- |
| **< 4 GB** (GPUs de entrada / integradas) | `256` - `512` | Previene errores de memoria excesiva (*Out of Memory* - OOM), especialmente usando Softmax completo. |
| **4 GB – 8 GB** (GPUs gama media) | `1024` - `2048` | Ofrece un equilibrio óptimo entre velocidad de cómputo y consumo de memoria. |
| **8 GB – 16 GB+** (GPUs gama alta) | `4096` - `8192` | Maximiza el rendimiento de CuPy y reduce drásticamente el tiempo de entrenamiento por época. |

> **Nota sobre Muestreo Negativo (`muestreo_negativo: true`)**: El uso de Muestreo Negativo reduce drásticamente la huella de memoria VRAM frente a Softmax completo, lo que permite utilizar lotes más grandes aun en placas con menor VRAM disponible.

---

## 💻 Flujo de Entrenamiento y Análisis

### 1. Entrenamiento Local por Consola
Para ejecutar el pipeline de entrenamiento desde la terminal:
```bash
# Ejecutar entrenamiento con la configuracion por defecto
uv run python entrenar.py

# O bien especificando parametros de consola
uv run python entrenar.py --epocas 10 --lote 2048 --tasa 0.2
```
Al finalizar, se guardará automáticamente la copia de respaldo `.npz` en la carpeta `respaldos/`.

### 2. Análisis y Evaluación en Jupyter Notebook
Abre el notebook `TP1_CBOW_Procesamiento_Lenguaje_Natural.ipynb` para:
1. Cargar modelos resguardados `.npz`.
2. Graficar las curvas de pérdida por época.
3. Buscar palabras más similares mediante producto interno o similaridad de coseno sobre la matriz $W$.
4. Comparar curvas de entrenamiento entre distintos experimentos.

---

## 🔄 Reanudación del Entrenamiento desde un Checkpoint (.npz)

Para continuar entrenando un modelo existente a partir de una copia de seguridad en formato `.npz`:

```bash
uv run python entrenar.py --reanudar --checkpoint respaldos/modelo_cbow_w5_epoca_10.npz --epocas 5
```

Al reanudar, se restaurarán las matrices de pesos $W$ y $W'$, el vocabulario de palabras, la época actual y el historial de pérdida acumulado.
