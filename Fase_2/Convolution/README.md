# Fase_2/Convolution

Activación de Tensor Cores para el kernel de Convolución 2D, sin encadenar iteraciones (eso empieza en Fase 3). Compara hasta cinco rutas de convolución 2D hacia adelante sobre la misma entrada y el mismo filtro: cuatro corren siempre, la quinta (CUTLASS) es opt-in vía `--cutlass`.

## Las rutas

1. **CPU — im2col + OpenBLAS.** Referencia de correctitud, no de rendimiento. La convolución se reescribe como `Y[K, outH·outW] = W[K, C·R·S] · col[C·R·S, outH·outW]` vía `im2col`, resuelta con `cblas_sgemm`/`cblas_dgemm`. FP32 o FP64 según `--double`.

2. **GPU — cuDNN clásico, sin Tensor Cores.** `cudnnConvolutionForward` con `cudnnSetConvolutionMathType(convDesc, CUDNN_FMA_MATH)`. Esta llamada es la que garantiza que esta ruta es un baseline FP32 escalar de verdad: sin ella, cuDNN activa TF32 por su cuenta en cualquier GPU Ampere o superior (`CUDNN_DEFAULT_MATH`), y la "línea base sin Tensor Cores" terminaría corriendo parcialmente en Tensor Cores. Es la misma llamada, con el mismo propósito, que usa `Fase_1/Convolution/conv_baseline.cu` — ver ese README para la medición que motivó fijarla así (79.1 TFLOP/s medidos con TF32 activo por accidente, 4x el pico real de FP32 escalar en A100).

3. **GPU — cuDNN con Tensor Cores, FP16 y/o BF16.** Tres pasos obligatorios para que cuDNN use Tensor Cores (documentados en el código junto a `benchmark_gpu_tensor_cores_conv`/`benchmark_gpu_tensor_cores_conv_bf16`):
   1. Descriptores de entrada y filtro en `CUDNN_DATA_HALF` (FP16) o `CUDNN_DATA_BFLOAT16` (BF16) — los operandos de la multiplicación pasan a 16 bits.
   2. `computeType = CUDNN_DATA_FLOAT` en `cudnnSetConvolution2dDescriptor` — la acumulación de productos parciales se hace en 32 bits, evitando desbordamiento.
   3. **`cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH)`** — sin esta llamada exacta, cuDNN puede seguir eligiendo un algoritmo escalar aunque los tensores ya estén en 16 bits; es la activación explícita, no una consecuencia automática del tipo de dato. Aparece dos veces en el archivo, una por formato (`benchmark_gpu_tensor_cores_conv` para FP16, `benchmark_gpu_tensor_cores_conv_bf16` para BF16) porque cada una construye su propio `cudnnConvolutionDescriptor_t` — no es código duplicado por descuido, cuDNN no ofrece una forma de compartir el descriptor entre dos tipos de dato distintos.

   Selección de formato con `--tc-format fp16|bf16|both` (default `fp16` en el binario; el `.sbatch` usa `both` — ver "Gap histórico" más abajo). BF16 requiere compute capability ≥ 8.0 (Ampere o superior); el binario verifica esto en tiempo de ejecución y aborta con un mensaje claro si no se cumple.

4. **GPU — im2col en GPU + kernel WMMA propio.** No pasa por cuDNN: un kernel propio (`im2col_fp16_kernel`) construye la matriz `col` en FP16 directamente en GPU, y otro (`wmma_gemm_kernel`) calcula `Y = W · col` usando la API WMMA (`nvcuda::wmma`) con un pipeline `cp.async` de 3 etapas (Ampere sm_80+). Requiere `K` múltiplo de 64, `outH·outW` múltiplo de 64 y `C·R·S` múltiplo de 32 (tile 64×64, `kKStep=32`); si la forma no cumple estas condiciones, la ruta se omite con un aviso y el resto del reporte sigue sin ella.

5. **GPU — CUTLASS `ImplicitGemmConvolution` (FP16/BF16), opt-in vía `--cutlass`.** Usa la plantilla de convolución implícita de CUTLASS (`cutlass::conv::kernel::DefaultConv2dFprop` + `cutlass::conv::device::ImplicitGemmConvolution`, API 2.x clásica, no CuTe/3.x), instanciada con Tensor Op para Ampere (`Sm80`, tile de bloque 128×128×32, tile de warp 64×64×32, instrucción `mma.sync` 16×8×16). El código (`run_cutlass_conv`/`run_cutlass_conv_bf16`, implementados sobre un `run_cutlass_conv_impl<ElementIO>` templado compartido) sigue la estructura de `examples/16_ampere_tensorop_conv2dfprop` del repositorio oficial `github.com/NVIDIA/cutlass` casi literalmente, siguiendo la recomendación de no improvisar parámetros de template de convolución de CUTLASS — es la API con más superficie de error de las cinco rutas.

   **Layout NHWC, no NCHW — y su conversión.** El resto de este archivo (CPU, cuDNN, WMMA) trabaja en NCHW (activación/salida `[N,C,H,W]`) y KCRS (filtro `[K,C,R,S]`). CUTLASS conv exige `cutlass::layout::TensorNHWC` para los tres tensores: activación `[N,H,W,C]`, filtro `[K,R,S,C]` ("KRSC", la misma clase de layout reinterpretada) y salida `[N,outH,outW,K]`. La ruta 5 por lo tanto:
   1. Sube `x` (NCHW) y `w` (KCRS) a GPU en FP32.
   2. Los transpone a NHWC/KRSC con cast a FP16 o BF16 (`convert_nchw_to_nhwc_kernel<ElementDst>`, un kernel genérico parametrizado por las 4 dimensiones — funciona igual para activación y filtro porque ambos son arreglos 4D con el eje de canales en la posición 1).
   3. Corre CUTLASS sobre esos buffers NHWC/KRSC; la salida queda en NHWC FP32.
   4. Convierte la salida NHWC → NCHW (`convert_nhwc_to_nchw_float_kernel`) antes de copiarla a host, para que sea comparable elemento a elemento contra `y_ref`/`y_cpu`/`y_gpu`/`y_tc`/`y_wmma` (todos NCHW).

   Este paso de conversión es el punto real de mayor riesgo de la ruta 5: un error de índices en cualquiera de los dos kernels de transposición produciría una salida con error numérico alto sin fallar la compilación ni abortar la ejecución — por eso la ruta compara explícitamente contra la referencia FP64 y contra la CPU FP32, igual que las otras cuatro, en vez de asumir que "corrió sin abortar" es suficiente evidencia de corrección.

   Compilado y verificado en GPU Ampere+ real (CUDA 13.3, CUTLASS v2.11.0, `sm_89`): las cinco rutas corren limpio con `--cutlass`, FP16 y BF16. `problem_size.output_size()` (de `Conv2dProblemSize`) devuelve `int64_t` — el conteo total de elementos `N·P·Q·K` — no un `cutlass::Tensor4DCoord`; la forma 4D de salida se construye a partir de los campos `N`/`P`/`Q`/`K` de `problem_size` directamente. El resto del template (parámetros de `DefaultConv2dFprop`, orden de argumentos de `Conv2dProblemSize`, campos de `ImplicitGemm::Arguments`) sigue el patrón de `examples/16_ampere_tensorop_conv2dfprop` del repositorio `NVIDIA/cutlass`. CUTLASS v2.11.0 requiere además un parche de una línea para compilar con GCC moderno (`matrix.h`, ver `REQUIREMENTS.md`, sección CUTLASS). La red de seguridad en tiempo de ejecución (comparar `N/P/Q/K` de CUTLASS contra `compute_output_dims()` propio, y `can_implement()` antes de correr) confirma que ambas formas coinciden.

   Requiere `--cutlass` explícito (desactivado por defecto) y reutiliza `--tc-format` (no agrega un flag de formato separado) para elegir FP16/BF16/ambos, igual que la ruta 3.

Con `--double` solo corren las rutas 1 y 2: las rutas 3, 4 y 5 son intrínsecamente de 16 bits de entrada.

## Cómo interpretar los resultados — comparaciones justas

Reportar comparaciones separadas, nunca combinadas en un solo número de "speedup de Tensor Cores": (a) cuDNN FP32 vs. cuDNN con Tensor Cores (ruta 2 vs. ruta 3) = efecto puro de precisión, misma librería; (b) cuDNN FP32 vs. WMMA propio (ruta 2 vs. ruta 4) = brecha de calidad de implementación, no efecto de precisión; un kernel de estudiante nunca va a igualar a cuDNN en throughput puro, así que esta comparación NO debe usarse como "el" número de aceleración por Tensor Cores.

Una tercera comparación, mencionada aquí por completitud aunque el objetivo de esta fase es más limitado, es (c') WMMA propio vs. cuDNN Tensor Core (ruta 4 vs. ruta 3, impresa como "Speedup WMMA vs cuDNN TC" cuando corrió la ruta FP16): mide qué tan cerca está la implementación WMMA propia de una librería de producción que también usa Tensor Cores — es la comparación más informativa sobre la calidad del kernel propio, distinta de (a) y (b).

Una cuarta comparación, disponible cuando se corre con `--cutlass`, es **(c) librería FP32 vs. CUTLASS (ruta 2 vs. ruta 5)** = brecha de implementación de una plantilla oficial afinada de NVIDIA — un punto intermedio entre (a) [librería FP32 vs. librería TC] y (b) [librería FP32 vs. WMMA propio]: CUTLASS es código de producción como cuDNN (comparación (a)), pero expuesto como plantilla que el usuario instancia y no como un algoritmo que la librería elige automáticamente, más parecido en ese sentido al esfuerzo de instanciación manual de la ruta WMMA (comparación (b)) aunque con un kernel mucho más afinado. Se imprime como "Speedup CUTLASS FP16/BF16 vs FP32 escalar" en el reporte.

El binario ya imprime las métricas separadas por ruta (tiempo, GFLOP/s, error vs. referencia FP64, error vs. CPU FP32) precisamente para permitir armar estas comparaciones por separado; el error está en el análisis posterior que las combine en un solo número sin decir contra cuál de ellas se está comparando.

## Gap histórico: BF16 nunca se corría por defecto

El `.sbatch` anterior (`old/Fase_2/Convolution/run_conv_tc.sbatch`) invocaba el binario sin `--tc-format`, es decir, siempre con el default del binario (`fp16`) — pese a que `benchmark_gpu_tensor_cores_conv_bf16` siempre existió y el binario ya soportaba `--tc-format bf16`/`--tc-format both`. En la práctica, ninguna corrida automatizada de Fase 2 generó nunca datos de la ruta Tensor Core BF16.

El nuevo `run_conv_tc.sbatch` corrige esto: la variable de entorno `TC_FORMAT` (`fp16` | `bf16` | `both`) controla el flag `--tc-format`, y su default es **`both`** — cada envío del job corre FP16 y BF16 salvo que se pida explícitamente lo contrario con `--export=ALL,TC_FORMAT=fp16`.

## Flags de CLI del binario (`conv_tc`)

```
conv_tc [--N N] [--C C] [--H H] [--W W] [--K K] [--R R] [--S S]
        [--pad_h P] [--pad_w P] [--stride_h S] [--stride_w S]
        [--dilation_h D] [--dilation_w D] [--iters I] [--double]
        [--tc-format fp16|bf16|both] [--cutlass]
```

Defaults del binario (cuando se invoca sin argumentos): `N=1, C=32, H=128, W=128, K=64, R=3, S=3, pad=1, stride=1, dilation=1, iters=20, tc-format=fp16, cutlass=desactivado`. Los defaults del `.sbatch` son otros (ver tabla abajo), pensados para comparabilidad con Fase 1.

`--cutlass` activa la ruta 5 (desactivada por defecto — no se agrega al `.sbatch` como comportamiento por defecto todavía, ver tabla de variables de `run_conv_tc.sbatch` más abajo); reutiliza `--tc-format` para elegir el/los formato(s) de la ruta 5.

### Ejemplos

```bash
./conv_tc
./conv_tc --N 1 --C 64 --H 224 --W 224 --K 64 --R 3 --S 3 --iters 10
./conv_tc --double --N 1 --C 16 --H 64 --W 64 --K 32 --R 3 --S 3
./conv_tc --N 1 --C 64 --H 64 --W 64 --K 64 --R 3 --S 3 --iters 2 --tc-format bf16
./conv_tc --N 1 --C 1024 --H 256 --W 256 --K 1024 --R 3 --S 3 --iters 10 --tc-format both
./conv_tc --N 1 --C 64 --H 64 --W 64 --K 64 --R 3 --S 3 --iters 2 --tc-format both --cutlass
```

## Qué produce

Salida por `stdout`: características de la GPU, configuración del experimento, y un reporte por ruta con tiempo medio, GFLOP/s/TFLOP/s, speedup vs. CPU, error máximo absoluto y error relativo L2 — contra la referencia FP64 (ground truth, calculada aparte con `im2col` + `cblas_dgemm` sobre las mismas entradas casteadas a `double`) y contra la CPU FP32 (trazabilidad con corridas previas). Si `RUN_NCU=1` (default fuera de `SMOKE_TEST`), además perfila el kernel `wmma_gemm_kernel` con Nsight Compute y deja un reporte `.ncu-rep` en `OUTPUT_DIR`.

## `run_conv_tc.sbatch`

Compila y corre `conv_tensor_activation.cu` vía SLURM, valida que el binario compilado contenga instrucciones HMMA (Tensor Core) antes de perfilar, y perfila el kernel WMMA con Nsight Compute. Nada está hardcodeado en el cuerpo del script: formato Tensor Core, forma del problema, iteraciones y directorio de salida son variables de entorno con default documentado (patrón `VAR="${VAR:-default}"`), igual que en `Fase_1/Convolution/run_conv_fase1.sbatch`.

| Variable | Default | Qué controla |
|---|---|---|
| `TC_FORMAT` | `both` | `fp16`, `bf16` o `both` — pasado directo como `--tc-format` (rige también la ruta 5 si `RUN_CUTLASS=1`). |
| `RUN_CUTLASS` | `1` | Si es `1`, agrega `--cutlass` a la invocación del binario (ruta 5, CUTLASS `ImplicitGemmConvolution`). Si es `0`, el binario corre solo las rutas 1-4, igual que antes de agregar esta ruta. |
| `CUTLASS_DIR` | `$HOME/cutlass/include` | Directorio `include/` de una copia clonada de `github.com/NVIDIA/cutlass` (header-only, serie 2.x) — se pasa como `-I${CUTLASS_DIR}` a `nvcc`. Ver `REQUIREMENTS.md`. |
| `N`, `C`, `H`, `W` | `1, 1024, 256, 256` | Forma del tensor de entrada. |
| `K`, `R`, `S` | `1024, 3, 3` | Forma del filtro. |
| `PAD_H`, `PAD_W` | `1, 1` | Padding. |
| `STRIDE_H`, `STRIDE_W` | `1, 1` | Stride. |
| `DILATION_H`, `DILATION_W` | `1, 1` | Dilatación. |
| `ITERS` | `10` | Iteraciones medidas. |
| `SMOKE_TEST` | `0` | Si es `1`, sustituye los defaults de forma por `C=H=W=K=64` (cumple las condiciones de divisibilidad de la ruta WMMA), `ITERS=2`, y desactiva NCU por defecto (`RUN_NCU=0`). |
| `RUN_NCU` | `1` (`0` si `SMOKE_TEST=1`) | Si perfila con Nsight Compute tras la corrida normal. |
| `NCU_MODE` | `quick` | `quick` (`--metrics`, rápido) o `full` (`--set`, caracterización completa). |
| `NCU_SET`, `NCU_KERNEL_REGEX`, `NCU_LAUNCH_SKIP`, `NCU_LAUNCH_COUNT` | ver script | Parámetros de la invocación de `ncu`. |
| `WARMUP_ITERS` | `3` | Debe coincidir con `kWarmupIters` en `conv_tensor_activation.cu` — determina cuántos lanzamientos saltar antes de perfilar. No es un parámetro libre: cambiarlo sin cambiar también la constante en el `.cu` desalinea el `--launch-skip` de NCU con los warmups reales. |
| `OUTPUT_DIR` | `logs` | Directorio para el binario compilado y los reportes `.ncu-rep` (no los logs de SLURM en sí — ver nota abajo). |
| `CUDA_ARCH` | `80` | Arquitectura objetivo (`80`=A100, `86`=RTX 3050, `70`=V100). |
| `NVCC`, `NCU`, `CUOBJDUMP`, `CUDNN_ROOT`, `CUDNN_INC`, `CUDNN_LIBS`, `OPENBLAS_DIR`, `OPENBLAS_INC`, `OPENBLAS_LIBS` | rutas de PACCA | Toolchain; sobrescribibles para correr en otro clúster. |

Los defaults de forma son los mismos que usa por defecto `Fase_1/Convolution/run_conv_fase1.sbatch` (primera de las dos formas históricas: `N=1, C=K=1024, H=W=256`), para que el baseline FP32 de Fase 1 sea directamente comparable con la ruta Tensor Core de aquí. La segunda forma histórica (`C=K=2048`) se reproduce con `--export=ALL,C=2048,K=2048`.

Nota sobre `#SBATCH --output`/`--error`: esas dos líneas son literales porque SLURM las procesa al momento del `sbatch`, antes de que el script (y por tanto `OUTPUT_DIR`) exista. Para cambiarlas: `sbatch --output=otra/ruta_%j.out --error=otra/ruta_%j.err run_conv_tc.sbatch`.

### Ejemplos

```bash
sbatch run_conv_tc.sbatch                                     # FP16 + BF16, forma por defecto, con NCU
sbatch --export=ALL,SMOKE_TEST=1 run_conv_tc.sbatch           # validacion rapida, sin NCU
sbatch --export=ALL,TC_FORMAT=fp16 run_conv_tc.sbatch         # solo FP16
sbatch --export=ALL,C=2048,K=2048 run_conv_tc.sbatch          # 2a forma historica
sbatch --export=ALL,RUN_NCU=0 run_conv_tc.sbatch              # sin perfilado NCU
sbatch --export=ALL,NCU_MODE=full run_conv_tc.sbatch          # caracterizacion NCU completa
sbatch --export=ALL,RUN_CUTLASS=0 run_conv_tc.sbatch           # sin la ruta 5 (CUTLASS)
sbatch --export=ALL,CUTLASS_DIR=/otra/ruta/cutlass/include run_conv_tc.sbatch
```

## Migración desde `old/`

`conv_tensor_activation.cu` es una migración de `old/Fase_2/Convolution/conv_tensor_activation.cu` **sin ningún cambio de lógica numérica ni de estructura** — la migración es, literalmente, un cambio de dónde vienen las macros y clases de infraestructura:

- `CHECK_CUDA`/`CHECK_CUDNN` (antes en `old/Fase_2/common.cuh`) ahora vienen de `common/cuda_checks.cuh`.
- `CudaEventTimer`/`Metrics`/`ErrorMetrics`/`compare_fp64_ref_vs_fp32`/`compare_float_vectors`/`compare_double_vectors` (antes también en `old/Fase_2/common.cuh`) ahora vienen de `common/metrics.cuh`.

`old/Fase_2/common.cuh` era un header compartido por los tres kernels de la Fase 2 anterior (GEMM, Convolución, Stencil), cada uno con su propia copia física del archivo. `common/cuda_checks.cuh` y `common/metrics.cuh` son la extracción de ese mismo contenido a un solo lugar para las cuatro fases — mismos structs, mismos campos, misma fórmula de error en `compare_*` (verificado campo por campo contra `old/Fase_2/common.cuh` antes de migrar); ningún llamador de este archivo necesitó cambiar. El resto del archivo — los cuatro `benchmark_*`, los kernels `im2col_fp16_kernel`/`wmma_gemm_kernel`, la orquestación en `run_experiment_float`/`run_experiment_double` — es idéntico byte a byte al original.

`run_conv_tc.sbatch` reemplaza el barrido fijo de dos formas (`RUNS=(...)`) por una sola forma parametrizada por variables de entorno (ver tabla arriba); la segunda forma histórica sigue siendo alcanzable pasando `C=2048,K=2048` explícitamente. El resto de la lógica del script (resolución de cuDNN/OpenBLAS, validación HMMA, invocación de NCU) no cambió.
