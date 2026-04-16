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
2. **Fase 2**: Integración de precisión mixta y activación de Tensor Cores
3. **Fase 3**: Cuantificación del drift numérico y suma compensada (Kahan)
4. **Fase 4**: Telemetría energética y análisis del Frente de Pareto

## Herramientas Utilizadas

- **Compilador**: NVIDIA nvcc (CUDA)
- **Bibliotecas**: cuBLAS, cuDNN, CUTLASS
- **Profiling**: NVIDIA Nsight Compute
- **Telemetría**: NVML (GPU), RAPL (CPU)
- **Métricas**: Normas L₂ y L∞, Energy-Delay Product (EDP)

## Estructura del Repositorio

```
mixed-precision-hpc-tensor-cores-project/
├── Fase_1/          # Línea base analítica
│   ├── GEMM/
│   ├── Convolution/
│   └── Stencil/
├── Fase_2/          # Precisión mixta y Tensor Cores
│   └── GEMM/
├── README.md
└── .gitignore
```

## Ambiente Requerido

- GPU NVIDIA con soporte para Tensor Cores (Volta, Turing, Ampere o superior)
- CUDA Toolkit 11.0 o superior
- cuBLAS y cuDNN compatible con CUDA
- Herramientas de profiling de NVIDIA

## Compilación

```bash
# Con nvcc en el directorio respectivo
nvcc -O3 kernel.cu -o kernel_executable -lcublas
```

## Métricas Principales

- **Throughput (TFLOPS)**: Operaciones en punto flotante por segundo
- **Latencia**: Tiempo de ejecución
- **EDP (Energy-Delay Product)**: Producto energía × tiempo
- **Error Numérico**: Desviación respecto a FP64 (referencia)

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
