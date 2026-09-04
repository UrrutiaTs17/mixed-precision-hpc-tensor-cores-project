# Fase 3 — Convolución: encadenamiento genuino

`conv_chained.cu` encadena Convolución 2D de verdad: `X(n+1) = conv(X(n), W)`, con `W` un filtro **fijo** aplicado repetidamente — un proceso de suavizado/difusión iterativo. Corrige la corrección #13 del documento *Plan de Precisión Mixta*, aplicada aquí a Convolución (ver `Fase_3/GEMM/README.md` para la misma corrección en GEMM — es el mismo razonamiento).

Sin precedente en el proyecto anterior. Compilado y verificado en GPU Ampere+ real (`sm_89`).

## El filtro W: el mismo Laplaciano de Stencil, no uno genérico

3×3, coeficientes exactos (0.25 vecino, −1.0 centro, 0 en las esquinas) — el mismo operador que usa Stencil, expresado como convolución en vez de diferencias finitas. Misma razón que la matriz de Hadamard de GEMM (ver `Fase_3/GEMM/README.md`): coeficientes exactos en cualquier formato evitan el error de representación del operador que ninguna compensación del estado puede corregir.

## Por qué 64 canales, no 1

La convolución se expresa como GEMM vía im2col (`Y[K,outH·outW] = W[K,C·R·S] · col[C·R·S,outH·outW]`, la misma técnica que ya usa la ruta 4 de `Fase_2/Convolution`). Con un solo canal (`C=K=1`, análogo al campo escalar de Stencil), la dimensión `M` del GEMM subyacente sería 1 — el kernel WMMA de `common/wmma_gemm.cuh` tiene tiles de salida de 64×64, así que `M=1` desperdiciaría 63 de cada 64 filas del tile y daría cifras de rendimiento engañosas (miden ocupación pésima del kernel, no el efecto de precisión que el proyecto quiere caracterizar).

En su lugar, `W` se construye **bloque-diagonal**: 64 canales, cada uno con el *mismo* filtro de 5 puntos aplicado de forma independiente (`W[k,c,·,·] = 0` salvo `c==k`) — matemáticamente idéntico a 64 simulaciones de Stencil corriendo en paralelo, con aprovechamiento completo del tile de 64. No cambia la ciencia (cada canal evoluciona de forma independiente e idéntica); solo evita medir un artefacto de mal dimensionamiento del kernel en vez del fenómeno real.

## Compensación por linealidad y referencia FP64

Mismo patrón que GEMM: `conv(T+comp, W) = conv(T,W) + conv(comp,W)` (la convolución es lineal), `comp` en `float` (sin ancla — eso es Fase 4). La referencia FP64 usa `cublasDgemm` sobre el `im2col` en `double`, con la misma inversión de operandos por la convención column-major de cuBLAS que ya documenta `Fase_3/GEMM/README.md` — la derivación completa, con las dimensiones específicas de este archivo (`Ncol`, `kChannels`, `kCRS`), está en el comentario junto a `gpu_fp64_conv_step()` en el `.cu`.

**Misma corrección que GEMM** (ver `Fase_3/GEMM/README.md`): `comp` se siembra desde el redondeo real de `x0→T` (`seed_comp_from_double_kernel`), no desde cero — de lo contrario la ruta `_comp` arrastra un piso de error evitable desde la primera iteración.

**Mismo punto de mayor riesgo que GEMM**: si el orden de operandos de `cublasDgemm` estuviera al revés, el binario compilaría y correría igual, comparando peras con manzanas sin ningún error visible. Verificar con `--hw` chico (p. ej. 64, el mínimo) contra una referencia independiente antes de confiar en cualquier resultado.

## `im2col`: tres variantes, no una

A diferencia de GEMM (donde el estado y la corrección pasan por el mismo `wmma_gemm_kernel` sin transformación previa), Convolución necesita reconstruir la matriz `col` en cada paso, para tres fuentes de datos distintas: el estado `T` (sin conversión, solo reordenamiento — `im2col_tc_kernel`), el residuo de compensación en `float` (con conversión a `T` — `im2col_float_to_tc_kernel`), y la referencia en `double` (`im2col_double_kernel`). Las tres implementan la misma indexación (`pad=1, stride=1, dilation=1`, padding "SAME" — imprescindible para que la salida tenga la misma forma que la entrada y pueda alimentar la siguiente iteración), solo cambia el tipo.

## Uso

```bash
./conv_chained --hw 64 --iters 20 --tc fp16 --comp on
./conv_chained --hw 128 --iters 10 --tc both --checkpoint-every 5
```

| Flag | Default | Qué hace |
|---|---|---|
| `--hw` | 64 | Alto y ancho del campo espacial (`H=W=hw`, debe hacer que `hw²` sea múltiplo de 64). 64 canales fijos — ver "Por qué 64 canales" arriba. |
| `--iters` | 20 | Iteraciones encadenadas. |
| `--tc` | both | `fp16`, `bf16` o `both`. |
| `--comp` | off | Compensación por linealidad. |
| `--checkpoint-every` | 0 | 0 = solo compara contra la referencia FP64 en la última iteración. |

## Salida

Mismo esquema `CSV_DRIFT`/`CSV_SUMMARY` que `Fase_3/GEMM/gemm_chained.cu` (ver ese README) — incluye `window_reliable`/`gpu_segments` desde el diseño, con la misma exclusión de las pausas de checkpoint de la ventana de energía/tiempo.

## Qué falta

- **`Fase_4/Convolution/`**: ✅ hecho — extensión con el ancla FP64 (`Fase_4/Convolution/conv_chained.cu`).
- **`run_conv_chained.sbatch`**: ✅ hecho — lanzador parametrizado, ver el propio `.sbatch` de esta carpeta.
- **Post-proceso de CSV**: ✅ hecho — `../tools/extract_csv_chained.py` (`Fase_3/tools/README.md`), ya integrado al final del `.sbatch`.
- **Scripts de gate** (comparar K=0/K=1 contra la referencia FP64 antes de confiar en una campaña con ancla): todavía no migrados/escritos — ver `Fase_3/tools/README.md`, sección "Qué falta".
- **Campaña real en PACCA**: compilado y verificado con `--hw` chico en GPU Ampere+; falta correr el barrido de tamaños que promete el plan.
