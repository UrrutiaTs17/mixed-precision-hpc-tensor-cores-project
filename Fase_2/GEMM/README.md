# Fase_2/GEMM

Activación de Tensor Cores para GEMM densa (`C = A*B`), sin encadenamiento de iteraciones (eso empieza en `Fase_3/GEMM`). Migrado desde `old/Fase_2/GEMM/gemm_tensor_activation.cu` (código ya auditado) — el único cambio de fondo es que las macros de validación CUDA/cuBLAS y las utilidades de cronometraje/error ya no están duplicadas dentro de este archivo: vienen de [`common/cuda_checks.cuh`](../../common/cuda_checks.cuh) y [`common/metrics.cuh`](../../common/metrics.cuh). La lógica numérica de cada kernel es idéntica a la versión anterior.

## Archivos

- `gemm_tensor_activation.cu` — el binario (se compila como `gemm_tc`, ver más abajo).
- `run_gemm_tc.sbatch` — script de lanzamiento en PACCA (SLURM), incluye perfilado opcional con Nsight Compute.

## Las cinco rutas

El binario compara, en la misma corrida, hasta cinco formas de calcular la misma GEMM:

1. **CPU OpenBLAS** (`cblas_sgemm`/`cblas_dgemm`) — referencia de CPU, sin GPU.
2. **GPU cuBLAS clásico** (`cublasSgemm`/`cublasDgemm`, con `CUBLAS_PEDANTIC_MATH` para desactivar TF32) — GPU sin Tensor Cores, FP32 o FP64 puro.
3. **GPU cuBLAS con Tensor Cores** (`cublasGemmEx`, operandos FP16 o BF16, acumulación FP32) — misma librería que la ruta 2, pero enrutada explícitamente por Tensor Cores.
4. **GPU WMMA propio** (kernel `wmma_gemm_kernel`, API `nvcuda::wmma` + pipeline `cp.async` de triple buffer, FP16 o BF16) — implementación propia sobre Tensor Cores, no una librería.
5. **GPU CUTLASS** (`cutlass::gemm::device::Gemm`, API 2.x de CUTLASS, operandos FP16 o BF16, acumulación FP32; activada con `--cutlass`) — implementación de referencia OFICIAL de NVIDIA sobre Tensor Cores, pero expuesta como plantilla de C++ en vez de biblioteca binaria cerrada. Es opcional porque CUTLASS es una dependencia externa header-only que hay que clonar aparte — ver "CUTLASS: dependencia opcional" más abajo y `REQUIREMENTS.md`.

Con `--double` solo corren las rutas 1 y 2 (las rutas 3, 4 y 5 son inherentemente FP16/BF16, no existe una versión FP64 de Tensor Cores en este proyecto). Todas las rutas FP32 se comparan contra una referencia FP64 calculada con `cblas_dgemm` sobre las mismas entradas casteadas a `double` — ver `ErrorMetrics` en `common/metrics.cuh`.

La ruta WMMA (4) requiere `m` y `n` múltiplos de 64 y `k` múltiplo de 32 (tamaño de tile del kernel); si no se cumple, el binario termina con un mensaje de error antes de lanzar el kernel, no falla en silencio.

La ruta CUTLASS (5) requiere compute capability ≥ 8.0: su template fija `ArchTag = cutlass::arch::Sm80` (ver "CUTLASS: dependencia opcional"), así que además de necesitar el binario compilado con soporte CUTLASS, la GPU debe ser Ampere o superior — igual que BF16, pero por una razón distinta (el `ArchTag` del template, no el tipo de dato).

## Compilación

```bash
nvcc -std=c++17 gemm_tensor_activation.cu -o gemm_tc \
     -I/usr/include/openblas -lcublas -lopenblas \
     -gencode arch=compute_80,code=sm_80 --allow-unsupported-compiler
```

Para habilitar además la ruta 5 (CUTLASS), agrega `-I$CUTLASS_DIR/include` (y probablemente `--expt-relaxed-constexpr`, ver más abajo):

```bash
nvcc -std=c++17 gemm_tensor_activation.cu -o gemm_tc \
     -I/usr/include/openblas -I$CUTLASS_DIR/include -lcublas -lopenblas \
     -gencode arch=compute_80,code=sm_80 --expt-relaxed-constexpr --allow-unsupported-compiler
```

En PACCA, usa `run_gemm_tc.sbatch` — además de compilar, valida con `cuobjdump --dump-sass` que el binario realmente contiene instrucciones `HMMA` antes de perfilar con Nsight Compute (si no las contiene, típicamente `m`/`n`/`k` no eran divisibles y el kernel WMMA no se activó).

## CUTLASS: dependencia opcional

`gemm_tensor_activation.cu` incluye los headers de CUTLASS con `#if __has_include(<cutlass/gemm/device/gemm.h>)`: si `-I$CUTLASS_DIR/include` no está en el path de compilación, el binario compila igual con las cuatro rutas 1-4, y pedir `--cutlass` en tiempo de ejecución termina el proceso con un mensaje explicando cómo habilitarlo, en vez de fallar la compilación para todos. Ver `REQUIREMENTS.md` para la URL del repositorio y la versión recomendada.

Compilado y verificado en GPU Ampere+ real (CUDA 13.3, CUTLASS v2.11.0, `sm_89`), `--cutlass` incluido. Los parámetros de forma del template (`ThreadblockShape`, `WarpShape`, `InstructionShape`, etapas del pipeline, `EpilogueOp`) siguen el patrón de `examples/08_turing_tensorop_gemm.cu` del repositorio `NVIDIA/cutlass`, cambiando `ArchTag` de Turing a `Sm80`. Requiere `--expt-relaxed-constexpr` (`run_gemm_tc.sbatch` ya lo agrega por defecto). CUTLASS v2.11.0 necesita además un parche de una línea para compilar con GCC moderno (`matrix.h`, ver `REQUIREMENTS.md`, sección CUTLASS).

## Flags de CLI

| Flag | Default | Descripción |
|---|---|---|
| `--m M` | 4096 | Filas de A y de C. |
| `--n N` | 4096 | Columnas de B y de C. |
| `--k K` | 4096 | Columnas de A / filas de B. |
| `--iters I` | 20 | Iteraciones cronometradas, promediadas. |
| `--double` | (apagado) | Usa FP64 (solo rutas 1 y 2). |
| `--tc-format fp16\|bf16\|both` | `fp16` | Formato de las rutas 3, 4 y 5. `both` corre FP16 y BF16 en la misma invocación. BF16 requiere compute capability ≥ 8.0 (Ampere o superior); si la GPU no lo soporta y se pide BF16, el binario termina con un mensaje de error. |
| `--cutlass` | (apagado) | Activa la ruta 5 (CUTLASS) para los formatos seleccionados por `--tc-format`. Requiere compute capability ≥ 8.0 y que el binario se haya compilado con `-I$CUTLASS_DIR/include`; si no, termina con un mensaje de error explicando cómo habilitarlo. |
| `--help`, `-h` | — | Imprime la ayuda y termina. |

Ejemplos:

```bash
./gemm_tc
./gemm_tc --m 4096 --n 4096 --k 4096 --iters 10
./gemm_tc --double --m 2048 --n 2048 --k 2048 --iters 5
./gemm_tc --m 1024 --n 1024 --k 1024 --iters 5 --tc-format bf16
./gemm_tc --m 2048 --n 2048 --k 2048 --iters 10 --tc-format both
./gemm_tc --m 2048 --n 2048 --k 2048 --iters 10 --tc-format both --cutlass
```

No hay flags para tolerancias ni rutas de salida porque el binario no aplica ningún criterio de aceptación ni escribe archivos: solo imprime métricas a stdout, y es el `.sbatch` quien decide dónde queda ese log y (opcionalmente) el reporte de `ncu`.

## Qué imprime

1. **Características de la GPU activa**: nombre, compute capability, memoria, SMs, relojes de GPU/memoria, bus de memoria, memoria compartida por bloque, y si la GPU soporta Tensor Cores FP16 (`major >= 7`) y TF32 (`major >= 8`).
2. **Configuración**: precisión, dimensiones (M, N, K), iteraciones.
3. **Resultados FP32** (o **FP64** con `--double`): tiempo medio, rendimiento (GFLOP/s y TFLOP/s), speedup contra CPU y contra cuBLAS clásico según corresponda, y error (máximo absoluto y relativo L2) contra la referencia FP64 — de cada ruta que corrió (1 y 2 siempre; 3 y 4 en FP16 y/o BF16 según `--tc-format`; 5 igual que 3 y 4 pero solo si se pasó `--cutlass`).

## Cómo interpretar los resultados — comparaciones justas

Reportar tres comparaciones separadas, nunca combinadas en un solo número de "speedup de Tensor Cores":

- **(a) cuBLAS FP32 vs. cuBLAS con Tensor Cores** (ruta 2 vs. ruta 3) = efecto puro de precisión, misma librería.
- **(b) cuBLAS FP32 vs. WMMA propio** (ruta 2 vs. ruta 4) = brecha de calidad de implementación, no efecto de precisión; un kernel de estudiante nunca va a igualar a cuBLAS en throughput puro, así que esta comparación **no** debe usarse como "el" número de aceleración por Tensor Cores.
- **(c) cuBLAS FP32 vs. CUTLASS** (ruta 2 vs. ruta 5) = brecha de implementación de una plantilla oficial afinada de NVIDIA — un punto intermedio entre (a) [librería FP32 vs. librería TC] y (b) [librería FP32 vs. WMMA propio]: CUTLASS no tiene el tuning específico de cuBLAS, pero sí mucho más que un kernel artesanal.

La salida del binario ya separa estos números (`Speedup TC vs GPU clasico`, `Speedup WMMA vs GPU clasico` y `Speedup CUTLASS vs GPU clasico` son líneas distintas, más `Speedup WMMA vs cuBLAS TC` y `Speedup CUTLASS vs cuBLAS TC`/`Speedup CUTLASS vs WMMA custom` para ver las brechas de implementación directamente) — el error está en agregarlos o citar solo uno de ellos como si respondiera la misma pregunta que los otros. Al escribir resultados en el documento de tesis, mantén (a), (b) y (c) como cifras separadas, con su propia interpretación cada una.

## BF16: disponible en el binario, ausente por defecto en las corridas históricas

`--tc-format bf16` existe en el binario desde el inicio de esta fase, pero el `.sbatch` histórico (`old/Fase_2/GEMM/run_gemm_tc.sbatch`) solo invocaba `./gemm_tc` sin `--tc-format`, es decir **siempre corría con el default `fp16`** — BF16 nunca se ejecutó en esas campañas pese a estar soportado. El `run_gemm_tc.sbatch` nuevo (esta carpeta) corrige eso: por defecto pasa `--tc-format both` (variable de entorno `TC_FORMAT`, ver ese script), así que cada corrida ahora cubre FP16 y BF16 en la misma invocación, salvo que se sobrescriba explícitamente.
