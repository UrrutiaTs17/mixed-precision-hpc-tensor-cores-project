# Fase_1/GEMM

Línea base de Fase 1: GEMM densa `C = A*B` en FP32 o FP64, comparando **CPU (OpenBLAS)** contra **GPU (cuBLAS clásico, sin Tensor Cores)**. Es el punto de referencia contra el que se contrastan las rutas con Tensor Cores de [`Fase_2/GEMM`](../../Fase_2/GEMM).

Migrado desde `old/Fase_1/GEMM/gemm_compare_balanced.cu` (el único binario vigente de esa carpeta; `gemm_benchmark.cu` y `gemm_compare.cu` eran iteraciones tempranas descartadas, ver `old/README.md`). El comportamiento numérico es idéntico al original — este README documenta la migración, no un rediseño.

## Archivos

- `gemm_baseline.cu` — el binario.
- `run_gemm_fase1.sbatch` — script de lanzamiento en PACCA (SLURM).

## Compilación

```bash
nvcc -std=c++17 -O3 gemm_baseline.cu -o gemm_baseline \
     -I/usr/include/openblas -lcublas -lopenblas \
     -gencode arch=compute_80,code=sm_80 --allow-unsupported-compiler
```

En PACCA, usa `run_gemm_fase1.sbatch` en vez de compilar a mano — resuelve la ruta de `nvcc`/OpenBLAS igual que el resto de los `.sbatch` del proyecto (no hay `module load cuda`, ver comentarios del script).

## Flags de CLI

| Flag | Default | Descripción |
|---|---|---|
| `--m M` | 2048 | Filas de A y de C. |
| `--n N` | 2048 | Columnas de B y de C. |
| `--k K` | 2048 | Columnas de A / filas de B. |
| `--iters I` | 10 | Iteraciones cronometradas, promediadas. |
| `--seed S` | 42 | Semilla del generador de números aleatorios usado para llenar A y B. |
| `--double` | (apagado) | Usa FP64 en vez de FP32. |
| `--help`, `-h` | — | Imprime la ayuda y termina. |

Ejemplos:

```bash
./gemm_baseline --m 4096 --n 4096 --k 4096 --iters 30
./gemm_baseline --double --m 2048 --n 2048 --k 2048 --iters 10 --seed 7
```

## Qué imprime

1. **Características de la GPU activa**: nombre, compute capability, memoria global, SMs, hilos por bloque, warp size, bus de memoria, memoria compartida por bloque.
2. **Configuración del experimento**: precisión, dimensiones, iteraciones, semilla RNG, disposición de memoria (column-major, la misma para BLAS y cuBLAS).
3. **Resultados**: tiempo medio y rendimiento (TFLOP/s) de CPU BLAS y de GPU cuBLAS, el speedup GPU/CPU, y dos métricas de error entre ambos resultados (error absoluto máximo y error relativo L2). No hay una tercera ruta ni una referencia FP64 separada en esta fase — CPU y GPU se comparan directamente entre sí, ambas en la misma precisión (`--double` corre ambas rutas en FP64, sin `--double` ambas corren en FP32).

El binario no escribe ningún archivo — todo el resultado es la salida estándar; el `.sbatch` la redirige a `logs/`.

## Tamaños de campaña

`run_gemm_fase1.sbatch` barre por defecto `GEMM_SIZES="1024 2048 4096 8192"`, el mismo conjunto usado en Fase 3 y Fase 4 y dimensionado para la A100 de PACCA. `SMOKE_TEST=1` conserva una única corrida reducida de validación; esa salida no pertenece al dataset experimental.
