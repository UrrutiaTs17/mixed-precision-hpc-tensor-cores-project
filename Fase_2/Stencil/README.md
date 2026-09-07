# Fase_2/Stencil

Stencil 2D de 5 puntos con activacion de Tensor Cores via WMMA (`mma.h`), comparado contra la referencia CPU FP32, la referencia FP64 (ground truth) y la ruta GPU CUDA clasica (sin Tensor Cores) de Fase 1.

Migrado desde `old/Fase_2/Stencil/stencil_tensor_activation.cu` — mismo esquema numerico, sin cambios de logica. Ver las notas de migracion al inicio del `.cu` para el detalle de que cambio (de donde vienen `CHECK_CUDA`/`CudaEventTimer`/`Metrics`/`ErrorMetrics`/`compare_*`, ahora en `common/`) y que no (aritmetica).

## Por que Stencil no tiene "ruta de biblioteca"

GEMM y Convolucion, en este proyecto, comparan tres rutas: CPU, una ruta de **biblioteca** (cuBLAS/cuDNN) y una ruta **Tensor Core propia** (WMMA/cuBLASLt) — y para esas dos hay que aclarar aparte que la ruta de biblioteca no es directamente comparable contra el WMMA propio, porque cuBLAS/cuDNN llevan años de tuning que el kernel WMMA propio no tiene (brecha de esfuerzo de implementacion, no solo de formato numerico).

**Stencil no usa cuBLAS ni cuDNN.** Aqui solo hay dos rutas de GPU:

1. **GPU CUDA FP32/FP64 clasico** (sin Tensor Cores) — codigo propio, el mismo kernel de Fase 1.
2. **GPU WMMA FP16/BF16** (Tensor Core) — codigo propio, ver la seccion siguiente.

Como ambas rutas son codigo propio con el mismo nivel de esfuerzo de optimizacion, Stencil es el **unico de los tres kernels** donde comparar "con Tensor Cores" vs. "sin Tensor Cores" es una comparacion limpia de entrada: no hay que aclarar una brecha tipo libreria-vs-WMMA como en GEMM/Convolucion, porque esa brecha simplemente no existe aqui.

## La reformulacion WMMA

`wmma::mma_sync` solo sabe multiplicar dos matrices 16x16 y acumular (`C += A*B`); no existe una operacion de Tensor Core que sume cinco vecinos y reste el centro directamente. `stencil2d_wmma_kernel` reescribe el Laplaciano de 5 puntos

```
salida = 0.25*arriba + 0.25*abajo + 0.25*izquierda + 0.25*derecha - centro
```

como una suma de cinco productos de matrices contra identidades escaladas (`I` = identidad 16x16):

```
salida = izquierda*(0.25I) + derecha*(0.25I) + (0.25I)*arriba + (0.25I)*abajo + centro*(-I)
```

porque multiplicar una matriz `X` por `c*I` (por cualquiera de los dos lados) equivale a `c*X`. Encadenando cinco `mma_sync` sobre el mismo acumulador FP32 (inicializado en cero) se obtiene exactamente esa suma, cada termino calculado por el pipeline de Tensor Cores en vez de por ALUs escalares.

Es una adaptacion **didactica** para activar y validar Tensor Cores en un kernel que no es una multiplicacion de matrices por naturaleza — no es el stencil mas eficiente posible en trafico de memoria (mover un tile 5 veces por 5 `mma_sync` cuesta mas que las 4 sumas + 1 resta del kernel FP32 clasico). Lo que se mide con esta ruta es el efecto de *usar* Tensor Cores, no el de un stencil optimizado en ancho de banda. El comentario dentro de `stencil2d_wmma_kernel` en el `.cu` tiene el detalle completo (por que `full_tile` vs. tile parcial, por que shared memory por warp, por que el orden de los 5 `mma_sync`).

## Formatos Tensor Core: `both` por defecto

`Options::tc_mode` es `Both` por defecto — el binario corre **FP16 y BF16** salvo que se pida `--tc fp16`/`--tc bf16` explicitamente. A diferencia de GEMM y Convolucion (donde el codigo original solo corria un formato por defecto y eso se corrigio al migrar), Stencil **ya** ejecutaba ambos formatos por defecto en el codigo original: no hay nada que corregir aqui, se conserva tal cual.

## Flags de CLI

```
./stencil_tc [--nx NX] [--ny NY] [--iters I] [--tc fp16|bf16|both]
```

| Flag | Default | Significado |
|---|---|---|
| `--nx NX` | 4096 | Ancho de la grilla |
| `--ny NY` | 4096 | Alto de la grilla |
| `--iters I` | 20 | Iteraciones promediadas (tras 3 de calentamiento) |
| `--tc fp16\|bf16\|both` | `both` | Formato(s) Tensor Core a ejecutar |
| `--help`, `-h` | — | Imprime uso y termina |

Ejemplos:

```
./stencil_tc --nx 4096 --ny 4096 --iters 20 --tc both
./stencil_tc --nx 8192 --ny 8192 --iters 20 --tc fp16
```

## Que produce

Salida por stdout: caracteristicas de la GPU activa, configuracion del experimento (incluye geometria del tile WMMA), y para cada ruta ejecutada (GPU FP32 clasico, WMMA FP16, WMMA BF16) tiempo, GFLOP/TFLOP/s, speedups, error maximo absoluto y error relativo L2 contra la referencia FP64 y contra la CPU FP32, y el error de solo-almacenamiento en 16 bits (cuanto se pierde por guardar un resultado que internamente ya vive en FP32 dentro de `__half`/`__nv_bfloat16`). Termina con el comando `ncu` sugerido para validar Tensor Cores. No escribe archivos por si mismo; el `.sbatch` redirige stdout/stderr a `logs/` y ahi tambien caen los reportes `.ncu-rep` de Nsight Compute cuando `RUN_NCU=1`.

## Compilar

```
nvcc -std=c++17 stencil_tensor_activation.cu -o stencil_tc \
     -gencode arch=compute_80,code=sm_80
```

## El `.sbatch`: de un tamano fijo a un barrido parametrizado

`run_stencil_tc.sbatch` ejecuta un barrido parametrizado con el mismo conjunto de tamaños de Fase 3/4:

| Variable | Default | Significado |
|---|---|---|
| `STENCIL_SIZES` | `"4096 8192 16384"` | Tamanos NX=NY a barrer, separados por espacio |
| `STENCIL_ITERS` | `20` | Iteraciones promediadas por corrida |
| `STENCIL_TC_MODES` | `"both"` | Modos Tensor Core a barrer (`fp16`, `bf16`, `both`, o combinaciones como `"fp16 bf16"`) |
| `OUT_DIR` | `logs` | Directorio para logs/reportes `.ncu-rep` generados por el script |
| `CUDA_ARCH` | `80` | Arquitectura CUDA objetivo |
| `NVCC` | ruta del HPC SDK del cluster | Compilador a usar |
| `NCU` | ruta local a `ncu` | Binario de Nsight Compute |
| `RUN_NCU` | `1` (`0` si `SMOKE_TEST=1`) | Si perfila con `ncu` ademas de correr el binario |
| `NCU_MODE` | `quick` | `quick` = metricas rapidas de validacion; `full` = `--set "${NCU_SET}"` completo |
| `NCU_SET` | `full` | Set de Nsight Compute usado cuando `NCU_MODE=full` |
| `SMOKE_TEST` | `0` | Si es `1`, corre un unico tamano pequeno (512, `both`, 3 iters) sin perfilar |

Ejemplos:

```
sbatch run_stencil_tc.sbatch
sbatch --export=ALL,SMOKE_TEST=1 run_stencil_tc.sbatch
sbatch --export=ALL,STENCIL_SIZES="4096 8192 16384",STENCIL_ITERS=30 run_stencil_tc.sbatch
sbatch --export=ALL,STENCIL_TC_MODES=fp16 run_stencil_tc.sbatch
sbatch --export=ALL,RUN_NCU=0 run_stencil_tc.sbatch
```

El tamaño reducido de `512×512` queda reservado a `SMOKE_TEST=1` y no se usa como dato experimental.

**NOTA MIGRACION (corregido):** este `.sbatch` compilaba `stencil_tc` sin `-O3` — asi estaba en el original (`old/Fase_2/Stencil/run_stencil_tc.sbatch`), a diferencia del de Fase 1, que si usa `-O3`. Eso hacia que la referencia "CPU FP32 serial" (host, escalar) corriera mucho mas lenta — medido en este entorno, ~25x mas lenta que con `-O3` a igualdad de tamano/maquina — lo que inflaba artificialmente todos los "Speedup vs CPU" que imprime el binario (no afectaba la correctitud numerica: los campos de error son identicos con y sin `-O3`, solo el rendimiento reportado de la referencia CPU). **Se agrego `-O3` a la compilacion para que Fase 2 sea consistente con Fase 1.** Cualquier "Speedup vs CPU"/"Speedup TC * vs CPU" generado con campanas anteriores a este cambio queda obsoleto y debe volver a correrse.

`NCU_QUICK_METRICS` (usado cuando `NCU_MODE=quick`) viene de `tools/common_ncu.sh`, compartido entre los `.sbatch` de todas las fases que perfilan con `ncu` — este script lo busca en `../../tools/common_ncu.sh` y, si no existe, sigue corriendo sin perfilado quick (advertencia, no error): la migracion de `tools/` al layout nuevo esta fuera del alcance de este directorio.
