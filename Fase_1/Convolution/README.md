# Fase_1/Convolution

Línea base de Fase 1 para el kernel de Convolución 2D: CPU (im2col + OpenBLAS) contra GPU (cuDNN), **sin Tensor Cores**. Es la referencia contra la que Fase 2 (`Fase_2/Convolution/`) mide el efecto de activarlos — ver la sección "Cómo se relaciona con Fase 2" más abajo.

## Qué hace `conv_baseline.cu`

Compara dos rutas de convolución 2D hacia adelante, en la misma precisión (FP32 o FP64, según `--double`):

1. **CPU**: la convolución se reescribe como GEMM vía `im2col` (`Y[K, outH·outW] = W[K, C·R·S] · col[C·R·S, outH·outW]`) y se resuelve con `cblas_sgemm`/`cblas_dgemm` de OpenBLAS.
2. **GPU**: `cudnnConvolutionForward`, con Tensor Cores **desactivados explícitamente**.

### Por qué TF32 está apagado a propósito

En una GPU Ampere o superior, si no se toca `cudnnSetConvolutionMathType`, cuDNN usa `CUDNN_DEFAULT_MATH` y activa TF32 por su cuenta en cualquier convolución FP32 — es decir, la "línea base FP32" terminaría corriendo parcialmente en Tensor Cores sin que el código lo pida. Esto invalidaría el propósito de Fase 1: medir el punto de partida *sin* precisión mixta.

`conv_baseline.cu` fuerza `cudnnSetConvolutionMathType(convDesc, CUDNN_FMA_MATH)` en la ruta GPU FP32 (ver el comentario en `run_gpu_cudnn_float`) — esto obliga a cuDNN a la unidad escalar FP32 (techo ~19.5 TFLOP/s en A100), no a TF32 (~156 TFLOP/s). Es la única línea del archivo que hace que este binario sea realmente un baseline "sin Tensor Cores"; si se mueve o se quita, cualquier comparación de velocidad hecha con este binario deja de significar lo que dice significar.

FP64 no tiene este problema: no existe una ruta TF32 para doble precisión, así que `run_gpu_cudnn_double` no necesita (ni tiene) el equivalente.

## Flags de CLI

```
conv_baseline [opciones]
  --double            usar FP64 (por defecto FP32)
  --n <int>           batch size                  (default 1)
  --c <int>           canales de entrada           (default 64)
  --h <int>           alto de entrada               (default 64)
  --w <int>           ancho de entrada               (default 64)
  --k <int>           canales de salida / filtros   (default 64)
  --r <int>           alto del filtro                (default 3)
  --s <int>           ancho del filtro                (default 3)
  --pad_h <int>       padding vertical                 (default 1)
  --pad_w <int>       padding horizontal                (default 1)
  --stride_h <int>    stride vertical                    (default 1)
  --stride_w <int>    stride horizontal                   (default 1)
  --dilation_h <int>  dilatacion vertical                  (default 1)
  --dilation_w <int>  dilatacion horizontal                 (default 1)
  --iters <int>       iteraciones para promediar el tiempo  (default 10)
```

El binario sin argumentos ejecuta el primer tamaño del conjunto oficial; el `.sbatch` recorre los cuatro tamaños espaciales.

### Ejemplos

```bash
# Configuracion por defecto del binario, FP32.
./conv_baseline

# Una de las formas de campaña, comparable con las demás fases.
./conv_baseline --n 1 --c 64 --h 256 --w 256 --k 64 --r 3 --s 3 \
    --pad_h 1 --pad_w 1 --stride_h 1 --stride_w 1 \
    --dilation_h 1 --dilation_w 1 --iters 10

# La misma corrida, en FP64.
./conv_baseline --double --n 1 --c 64 --h 512 --w 512 --k 64 --r 3 --s 3 --iters 10

```

## Qué produce

Salida por `stdout`, en texto plano (no CSV — Fase 1 es exploratoria; el post-procesamiento estructurado en CSV empieza en Fase 3):

1. Características de la GPU activa (nombre, compute capability, relojes, memoria).
2. Configuración del experimento (forma de entrada/filtro/salida, precisión).
3. El algoritmo de convolución elegido por cuDNN y el tamaño de su workspace.
4. Resultados: tiempo medio y rendimiento (GFLOP/s y TFLOP/s) de CPU y GPU, speedup GPU/CPU, y el error de la GPU respecto a la CPU (error absoluto máximo y error relativo L2) **en la misma precisión** — esta comparación es una prueba de correctitud (¿coincide GPU con CPU dentro de lo esperable por redondeo?), no una medición de pérdida de precisión por formato (eso es tema de Fase 2, donde sí hay una referencia FP64 aparte contra la que se compara FP16/BF16).

## Gap histórico: FP64 nunca se corría

El `.sbatch` anterior (`old/Fase_1/Convolution/run_conv_fase1.sbatch`) invocaba el binario **únicamente sin `--double`**, pese a que la ruta FP64 (`run_gpu_cudnn_double`/`run_cpu_conv_openblas_double`) siempre existió y siempre funcionó — es un binario, no dos. En la práctica esto significaba que la línea base FP64 de Convolución nunca se generó como parte de una corrida automatizada de Fase 1, solo se podía obtener invocando el binario a mano.

El nuevo `run_conv_fase1.sbatch` corrige esto: la variable de entorno `PRECISION` (`fp32` | `fp64` | `both`) controla qué se corre, y su default es **`both`** — cada envío del job corre FP32 y FP64 salvo que se pida explícitamente lo contrario con `--export=ALL,PRECISION=fp32`.

## `run_conv_fase1.sbatch`

Compila y corre `conv_baseline.cu` vía SLURM. Nada está hardcodeado en el cuerpo del script: la forma del problema, la precisión y el directorio de salida son variables de entorno con default documentado (patrón `VAR="${VAR:-default}"`), igual que en `Fase_2/Convolution/run_conv_tc.sbatch`.

| Variable | Default | Qué controla |
|---|---|---|
| `PRECISION` | `both` | `fp32`, `fp64` o `both` — qué rutas de precisión correr. |
| `N`, `C`, `K` | `1, 64, 64` | Lote y canales, iguales a Fase 3/4. |
| `HW_LIST` | `"64 128 256 512"` | Dominios cuadrados `H=W` a barrer. |
| `H`, `W` | — | Compatibilidad para una corrida manual; al exportarlos reemplazan `HW_LIST` y deben ser iguales. |
| `R`, `S` | `3, 3` | Forma del filtro. |
| `PAD_H`, `PAD_W` | `1, 1` | Padding. |
| `STRIDE_H`, `STRIDE_W` | `1, 1` | Stride. |
| `DILATION_H`, `DILATION_W` | `1, 1` | Dilatación. |
| `ITERS` | `10` | Iteraciones medidas por ruta. |
| `SMOKE_TEST` | `0` | Si es `1`, sustituye los defaults de forma por una convolución pequeña (`C=K=H=W=64`, `ITERS=2`) para validar que el binario compila y corre, no para medir rendimiento. |
| `OUTPUT_DIR` | `logs` | Directorio para artefactos auxiliares de la corrida (no los logs de SLURM en sí — ver nota abajo). |
| `CUDA_ARCH` | `80` | Arquitectura objetivo (`80`=A100, `86`=RTX 3050, `70`=V100). |
| `NVCC`, `CUDNN_ROOT`, `CUDNN_INC`, `CUDNN_LIBS`, `OPENBLAS_DIR`, `OPENBLAS_INC`, `OPENBLAS_LIBS` | rutas de PACCA | Toolchain; sobrescribibles para correr en otro clúster. |

Los defaults (`N=1`, `C=K=64`, `R=S=3`, `HW_LIST="64 128 256 512"`) son idénticos a los de las campañas encadenadas de Fase 3/4. `SMOKE_TEST=1` conserva únicamente `H=W=64` con dos iteraciones y su salida no se reporta como dato experimental.

Nota sobre `#SBATCH --output`/`--error`: esas dos líneas son literales (`logs/fase1_conv_baseline_%j.out`/`.err`) porque SLURM las procesa al momento del `sbatch`, antes de que el script (y por tanto `OUTPUT_DIR`) exista — no pueden depender de una variable definida dentro del script. Para cambiarlas, pásalas en la propia línea de `sbatch`: `sbatch --output=otra/ruta_%j.out --error=otra/ruta_%j.err run_conv_fase1.sbatch`.

### Ejemplos

```bash
sbatch run_conv_fase1.sbatch                                    # FP32 + FP64, forma por defecto
sbatch --export=ALL,SMOKE_TEST=1 run_conv_fase1.sbatch          # validacion rapida
sbatch --export=ALL,PRECISION=fp32 run_conv_fase1.sbatch        # solo FP32
sbatch --export=ALL,HW_LIST="64 128 256 512" run_conv_fase1.sbatch
sbatch --export=ALL,CUDNN_ROOT=/otra/ruta run_conv_fase1.sbatch # otra instalacion de cuDNN
```

## Cómo se relaciona con Fase 2

`Fase_2/Convolution/conv_tensor_activation.cu` reutiliza la misma ruta cuDNN FP32 sin Tensor Cores (misma llamada a `CUDNN_FMA_MATH`, mismo criterio de selección de algoritmo) como su propia línea base interna — de modo que ambos binarios, corridos con la misma forma de problema, deberían dar tiempos de GPU FP32 comparables. Esa es la comparación de referencia para medir el efecto de Tensor Cores en Fase 2 (ver `Fase_2/Convolution/README.md`, sección "Cómo interpretar los resultados").

## Migración desde `old/`

Este archivo es una migración de `old/Fase_1/Convolution/cudnn_conv_balanced.cu`, sin cambios de lógica numérica. Lo que cambió:

- `CHECK_CUDA`/`CHECK_CUDNN` ahora vienen de `common/cuda_checks.cuh`, y el temporizador de eventos CUDA usa `CudaEventTimer` de `common/metrics.cuh` en vez de llamadas manuales a `cudaEventCreate`/`Record`/`Synchronize`/`ElapsedTime`/`Destroy` — mismo comportamiento, sin la duplicación que tenía cada `.cu` de este proyecto antes de que existiera `common/`.
- El resultado del experimento se reorganizó de un único `struct Metrics` con siete campos planos (`cpu_ms`, `gpu_ms`, `cpu_gflops`, ...) a `ExperimentResult { Metrics cpu; Metrics gpu; ErrorMetrics err; }`, usando los tipos `Metrics`/`ErrorMetrics` de `common/metrics.cuh` y su función `compare_float_vectors`/`compare_double_vectors` en vez de las funciones locales `compute_error_float`/`compute_error_double`. El cálculo de error es el mismo (mismo `max_abs`, mismo `rel_l2`); la única diferencia de implementación es que `common/metrics.cuh` evita la división por cero con un chequeo condicional (`sq_ref > 0.0 ? ... : 0.0`) en vez de sumar un épsilon (`1e-30`/`1e-300`) al denominador — para cualquier entrada real de este binario (salida de convolución sobre ruido uniforme, nunca todo-ceros) ambos enfoques dan el mismo resultado numérico; el épsilon original solo evitaba una división por cero literal que en la práctica nunca ocurre aquí.
- Se eliminaron `x_index`/`y_index`: dos funciones `inline` definidas en el archivo original que ninguna otra función llamaba (código muerto — los índices se calculaban inline en `im2col_*_single_image` de todas formas).
- Nombre del archivo y del binario: `cudnn_conv_balanced.cu`/`cudnn_conv_balanced` → `conv_baseline.cu`/`conv_baseline`, para que el nombre describa el rol del binario (línea base) en vez de una decisión de implementación (que use cuDNN).

Todo lo demás — inicialización de datos, `im2col`, selección de algoritmo cuDNN, la llamada `CUDNN_FMA_MATH`, el esquema de medición (1 corrida de calentamiento + `iters` corridas medidas) — es idéntico al original.
