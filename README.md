# Evaluación Experimental de Precisión Mixta con Tensor Cores en GPUs NVIDIA

## Descripción General

Este proyecto investiga el impacto numérico y energético de la computación en **precisión mixta** utilizando **Tensor Cores** en GPUs NVIDIA para kernels representativos de HPC.

**Objetivo Principal**: Determinar empíricamente las configuraciones de precisión mixta que ofrezcan el mejor compromiso entre **rendimiento computacional**, **consumo energético** y **exactitud numérica**.

## Kernels Evaluados

- **GEMM**: Multiplicación de matrices densas (512×512 a 4096×4096)
- **Convolución 2D**: Operaciones de convolución con diversos tamaños de filtro
- **Stencil 2D**: Operadores de diferencias finitas (512² a 2048² elementos)

## Formatos de Precisión

- **FP64**: Doble precisión (línea base de referencia)
- **FP32**: Precisión simple
- **FP16**: Media precisión (con Tensor Cores)
- **BF16**: Brain Floating Point (con Tensor Cores)

## Fases del Proyecto

1. **Fase 1**: Construcción de línea base analítica (FP64 y FP32)
2. **Fase 2**: Integración de precisión mixta y activación de Tensor Cores (throughput, sin encadenar iteraciones)
3. **Fase 3**: Encadenamiento genuino de iteraciones en el Stencil 2D (salida(i) → entrada(i+1)) para cuantificar drift numérico acumulado, horizonte de overflow por formato y consumo energético (NVML), comparando suma compensada Kahan local frente a compensación espacial
4. **Fase 4**: Campañas de variabilidad estadística y análisis del Frente de Pareto 3D (rendimiento-energía-error) sobre el operador de estrés difusivo

## Herramientas Utilizadas

- **Compilador**: NVIDIA nvcc (CUDA)
- **Bibliotecas**: cuBLAS, cuDNN, CUTLASS
- **Profiling**: NVIDIA Nsight Compute
- **Telemetría**: NVML (GPU, por contador de energía de 2 lecturas), RAPL (CPU)
- **Post-procesamiento**: Python 3 (biblioteca estándar) para extracción y resumen de CSV
- **Métricas**: Normas L₂ y L∞, horizonte de overflow (n*), Energy-Delay Product (EDP)
- **Ejecución**: SLURM (sbatch) sobre el clúster PACCA — la compilación CUDA no se realiza en local

## Estructura del Repositorio

```
mixed-precision-hpc-tensor-cores-project/
├── Fase_1/                        # Línea base analítica (FP64 y FP32)
│   ├── GEMM/                      # gemm_compare_balanced.cu + run_gemm_fase1.sbatch
│   ├── Convolution/                # cudnn_conv_balanced.cu + run_conv_fase1.sbatch
│   └── Stencil2D/                  # stencil2d_baseline.cu + run_stencil_fase1.sbatch
├── Fase_2/                        # Precisión mixta y activación de Tensor Cores
│   ├── GEMM/
│   ├── Convolution/
│   ├── Stencil/
│   ├── common.cuh                  # Utilidades compartidas (CHECK_CUDA, CudaEventTimer, Metrics, ErrorMetrics, compare_*)
│   └── telemetry.cuh
├── Fase_3/                        # Encadenamiento genuino: drift, horizonte de overflow y energía
│   └── Stencil/
│       ├── stencil_tensor_activation.cu  # rutas CPU_FP32/FP64, GPU_FP32/FP64 y WMMA FP16/BF16 (Kahan local o compensación espacial)
│       ├── run_stencil_tc.sbatch         # barrido de métricas dentro del horizonte finito (checkpoints, energía NVML)
│       ├── run_stencil_horizon.sbatch    # medición del horizonte de overflow real por formato
│       ├── stencil_jobs.sh               # orquestador de la campaña de cierre (sub-campañas, ver --help)
│       └── tools/
│           ├── extract_csv.py            # separa el log de cada job en CSV de drift/horizonte/energía/resumen
│           ├── power_sampling.h          # energía GPU vía contador NVML (2 lecturas, sin hilo de muestreo)
│           └── README.md                 # semántica detallada de columnas CSV y de las rutas de referencia
├── tools/
│   └── common_ncu.sh        # Definiciones compartidas de perfilado con Nsight Compute
├── README.md
└── .gitignore
```

> Fase 4 se desarrolla en la rama `fase4-estadistica-variabilidad` y aún no se integra a `main`.

## Ambiente Requerido

- GPU NVIDIA con soporte para Tensor Cores (Volta, Turing, Ampere o superior)
- CUDA Toolkit 11.0 o superior
- cuBLAS y cuDNN compatible con CUDA
- Herramientas de profiling de NVIDIA

## Compilación y Ejecución

La compilación y ejecución de los kernels CUDA se realiza en el clúster PACCA vía SLURM, no en local:

```bash
# Desde el directorio de la fase correspondiente
sbatch run_stencil_tc.sbatch
```

Cada `.sbatch` invoca `nvcc` con los flags de arquitectura y enlazado (cuBLAS, cuDNN, NVML) que correspondan a esa fase.

## Métricas Principales

- **Throughput (TFLOPS/GFLOPS)**: Operaciones en punto flotante por segundo
- **Latencia**: Tiempo de ejecución por iteración y total
- **Horizonte de overflow (n\*)**: Iteración en la que una ruta deja de ser finita
- **EDP (Energy-Delay Product)** y **energía por iteración**: Producto energía × tiempo y consumo GPU normalizado
- **Error Numérico**: Desviación (L₂, L∞) respecto al patrón FP64 (referencia)

## Equipo

### Director
- **Gilberto Javier Díaz Toro**, Ph.D.
  - Escuela de Ingeniería de Sistemas e Informática - UIS

### Autores / Investigadores
- **Karen Dayana Mateus Gomez** (Código: 2212765)
  - Escuela de Ingeniería de Sistemas e Informática - UIS
  
- **William Andrés Urrutia Torres** (Código: 2220058)
  - Escuela de Ingeniería de Sistemas e Informática - UIS

### Institución
**Universidad Industrial de Santander (UIS)**
- Facultad de Ingenierías Físicomecánicas
- Escuela de Ingeniería de Sistemas e Informática

---

**Fecha de Presentación**: Bucaramanga, 09 de Abril de 2026  
**Modalidad**: Trabajo de Investigación
