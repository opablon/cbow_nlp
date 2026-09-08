# TP1: Representación Contextual de Palabras con CBOW

**Asignatura**: Aprendizaje Automático Avanzado  
**Carrera**: Tecnicatura Universitaria en Inteligencia Artificial (UNAHUR)  
**Profesor**: Juan Miguel Santos (LIDEC)  

Este repositorio contiene la implementación completa del modelo de redes neuronales **CBOW (Continuous Bag-of-Words)** para Procesamiento de Lenguaje Natural (PLN), acelerado por GPU NVIDIA mediante **Python, CuPy y PyTorch**.

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

## 📁 Estructura del Repositorio Sincronizado

```text
cbow_nlp/
├── README.md                               # Instrucciones de clonacion, instalacion y uso
├── pyproject.toml                          # Dependencias del proyecto para uv
├── configuracion.yaml                      # Hiperparametros centralizados por defecto
├── entrenar.py                             # Script ejecutable de consola para entrenamiento local
├── TP1_CBOW_Procesamiento_Lenguaje_Natural.ipynb  # Notebook Master Interactivo
└── codigo/                                 # Modulos Python reutilizables (snake_case)
    ├── __init__.py
    ├── configuracion.py                    # Carga de parametros desde configuracion.yaml
    ├── tokenizador.py                      # Tokenizacion por Palabra (Regex) y BPE
    ├── generador_vocabulario.py            # Construccion del vocabulario (Criterio Excluyente)
    ├── modelo_cbow_cupy.py                 # Red CBOW acelerada por GPU (CuPy)
    ├── modelo_cbow_pytorch.py              # Red CBOW acelerada por GPU (PyTorch)
    ├── entrenador_cbow.py                  # Bucle de entrenamiento por lotes y respaldos
    └── evaluador_similaridad.py            # Producto Interno (default) y Similaridad de Coseno
```

> **Nota**: Los datos del corpus (`datos/`), los archivos de respaldos `.npz` (`respaldos/`), el entorno virtual (`.venv/`), las reglas (`rules.md`) y la bitácora (`bitacora.md`) están excluidos del control de versiones mediante `.gitignore` para mantener el repositorio liviano y privado.

---

## ⚙️ Configuración del Corpus y Parámetros del Proyecto

Todos los parámetros del proyecto (ruta del corpus, motor de cómputo, tamaño de lote, semilla aleatoria, tamaño de ventana, etc.) pueden configurarse directamente en **dos lugares opcionales**:

1. **Directamente desde el Jupyter Notebook (Recomendado)**: En la **Sección 1** del notebook [TP1_CBOW_Procesamiento_Lenguaje_Natural.ipynb](TP1_CBOW_Procesamiento_Lenguaje_Natural.ipynb), todos los atributos de `config` están visibles y comentados en una celda de código para que los modifiques directamente en Python sin abrir archivos externos.
2. **Desde `configuracion.yaml`**: Si prefieres centralizar la configuración en un archivo externo, puedes editar `configuracion.yaml`. Tanto el notebook como el script de consola `entrenar.py` lo leerán automáticamente.

### 1. Selección de Fuente del Corpus de Texto
Para entrenar el modelo con tu propio corpus de texto o con el archivo provisto por la cátedra:
1. Coloca tu archivo de texto (por ejemplo `corpus.txt`) dentro de la carpeta `datos/` (o en la ubicación deseada).
2. Modifica la variable `config.ruta_corpus = "datos/corpus.txt"` en la **Sección 1** del Notebook (o en `configuracion.yaml`).

### 2. Motores de Cómputo Disponibles (`motor_computo`)
Puedes alternar entre dos motores de ejecución acelerada por GPU NVIDIA modificando el parámetro `motor_computo` en `configuracion.yaml` o directamente en el notebook:
- `motor_computo: "cupy"`: Implementación directa mediante operaciones matriciales con CuPy (ideal para seguir paso a paso la matemática de la cátedra).
- `motor_computo: "pytorch"`: Implementación utilizando PyTorch GPU (`torch.cuda` / `nn.Module`), optimizada para alto rendimiento y compatibilidad.

Ambos motores de cómputo están 100% implementados y permiten entrenar el modelo CBOW a máxima velocidad en GPU NVIDIA.

### 3. Optimización por Muestreo Negativo (Negative Sampling)
Puedes activar la optimización mediante Muestreo Negativo modificando los parámetros en `configuracion.yaml` o en el Notebook:
- `muestreo_negativo: true` (por defecto `false` para Softmax Completa).
- `cantidad_muestras_negativas: 5` ($K=5$ muestras negativas extraídas sobre la distribución $U^{3/4} \propto f(w)^{0.75}$).
- **Ventaja de Rendimiento**: Reduce el costo computacional de $O(|V|)$ a $O(K)$, permitiendo acelerar drásticamente el tiempo de convergencia.
- **Inicialización de Pesos**: Con Softmax se inicializa $W'$ en ceros; con Muestreo Negativo se inicializa $W'$ de forma aleatoria uniforme en `[-0.5/N, 0.5/N]` para romper la simetría de las sigmoides logísticas.

### 4. Regla General para Seleccionar `tamanio_lote` (Batch Size)
El tamaño de mini-lote $B$ debe seleccionarse en potencias de 2 ($2^k$, ej. 1024, 2048, 4096) para máxima velocidad en GPU NVIDIA, asegurando la alineación óptima con las unidades de cómputo (*Warp* / *Tensor Cores*):
- **Regla de Selección según VRAM**:
  - **GPUs con 4 GB - 6 GB VRAM**: `tamanio_lote = 2048` (valor recomendado para máxima velocidad sin saturar memoria).
  - **GPUs con 8 GB - 16 GB VRAM**: `tamanio_lote = 4096` a `8192`.
  - **GPUs de Servidor / Colab (T4, A100)**: `tamanio_lote = 8192` a `16384`.
- **Generalización**: Aumentar el tamaño de lote incrementa la cantidad de muestras procesadas en paralelo por segundo. Si se produce un error de memoria insuficiente (*Out of Memory*), simplemente se reduce $B$ a la potencia de 2 inmediatamente anterior (ej. de 4096 a 2048).

---

## 💻-☁️ Flujo de Entrenamiento Híbrido (Terminal Local <-> Google Colab)

El proyecto soporta una integración bidireccional entre la terminal de tu máquina local y Google Colab:

### 1. Entrenar en la PC Local desde la Terminal (IDE)
Para entrenar sin sobrecarga de memoria del navegador y garantizando estabilidad en ejecuciones largas:
```bash
# Ejecutar entrenamiento completo con hiperparámetros de configuracion.yaml
python entrenar.py

# O bien especificando parámetros desde la línea de comandos:
python entrenar.py --epocas 10 --motor pytorch --lote 2048
```
El script procesará el corpus, autoguardará el diccionario en `respaldos/vocabulario.json` y generará los checkpoints periódicos `.npz` en la carpeta `respaldos/`.

### 2. Subir Resguardos a Google Colab / Drive
Sube la carpeta `respaldos/` completa (con `vocabulario.json` y el checkpoint `.npz` generado, ej. `modelo_cbow_w4_epoca_10.npz`) a tu entorno de Google Colab o carpeta vinculada de Google Drive.

### 3. Reanudar o Evaluar en la GPU de la Nube (Google Colab)
En la **Sección 4** del notebook [TP1_CBOW_Procesamiento_Lenguaje_Natural.ipynb](TP1_CBOW_Procesamiento_Lenguaje_Natural.ipynb), ajusta los controles interactivos de entrenamiento:
```python
REANUDAR_ENTRENAMIENTO = True
RUTA_CHECKPOINT = "respaldos/modelo_cbow_w4_epoca_10.npz"
EPOCAS_A_ENTRENAR = 10  # Épocas adicionales a entrenar en la nube
```
Al ejecutar la celda, se cargará el estado exacto del modelo local y se continuará el entrenamiento acelerado en la GPU de Colab. Del mismo modo, puedes utilizar la **Sección 5** para evaluar interactivamente similaridades de palabras sobre los modelos entrenados.

---

## 🔄 Reanudar o Continuar el Entrenamiento por Épocas Adicionales

Para continuar entrenando un modelo existente desde un archivo de resguardo `.npz` guardado previamente (ya sea localmente o por script):

1. Abre el notebook [TP1_CBOW_Procesamiento_Lenguaje_Natural.ipynb](TP1_CBOW_Procesamiento_Lenguaje_Natural.ipynb) (o tu script en Python).
2. Carga cualquier archivo de backup `.npz` existente (ej. `modelo.cargar_modelo("respaldos/modelo_cbow_w4_epoca_10.npz")`).
3. Ejecuta `entrenador.entrenar_adicional(epocas_adicionales=N)` indicando únicamente cuántas **épocas adicionales ($N$)** deseas entrenar (por ejemplo `epocas_adicionales=10`).
4. El entrenador continuará automáticamente el entrenamiento a partir del estado guardado por las $N$ épocas indicadas.




