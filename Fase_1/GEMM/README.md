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

## Nota importante: no ejecutar solo con FP64 puro en tamaños grandes sin barrer el rango del plan de tesis

El plan de trabajo de grado promete un rango de tamaños **512–4096** para GEMM (ver `README.md` de la raíz del repo, sección "Kernels evaluados"). El `.sbatch` histórico (`old/Fase_1/GEMM/run_gemm_fase1.sbatch`) solo corría dos tamaños, **12288³ y 32768³**, muy por encima de ese rango — un hallazgo real de la auditoría previa del proyecto. A esos tamaños, la referencia de CPU (OpenBLAS `dgemm`) por sí sola toma minutos por iteración (ver el comentario dentro del `.sbatch` histórico: ~1 minuto/iteración en 16 núcleos para M=N=K=32768 en FP64, ~35 minutos las 30 iteraciones), lo que hace esas corridas costosas y no representativas del rango que la tesis realmente necesita caracterizar.

`run_gemm_fase1.sbatch` (la versión nueva, en esta misma carpeta) **por defecto sí barre el rango 512–4096** (`GEMM_SIZES` por defecto: `512 1024 2048 4096`). Los tamaños históricos grandes siguen disponibles, pero como **opt-in explícito** (`INCLUDE_LEGACY_SIZES=1`), no como comportamiento por defecto. Si vas a lanzar `gemm_baseline --double` a mano (sin pasar por el `.sbatch`), evita repetir el error histórico: no lo lances directamente a 12288/32768 sin antes haber barrido tamaños dentro de 512–4096 — a esos tamaños grandes, cada corrida FP64 es cara y una sola corrida no dice nada sobre cómo escala el comportamiento dentro del rango prometido.
