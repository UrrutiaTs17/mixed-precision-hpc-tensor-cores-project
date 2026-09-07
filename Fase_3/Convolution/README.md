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

**Mismo punto de mayor riesgo que GEMM** — ✅ verificado. Si el orden de operandos de `cublasDgemm` estuviera al revés, el binario compilaría y correría igual, comparando peras con manzanas sin ningún error visible. Esa verificación ya existe y ya pasó:

```bash
python3 ../tools/verificar_orden_operandos_conv.py --hw 64
```

`Fase_3/tools/verificar_orden_operandos_conv.py` extrae de este archivo — sin reimplementarlos — las constantes (`kChannels`, `kFilterR`, `kFilterS`, `kCRS`), `grid1d()`, `build_block_diagonal_filter()`, `im2col_double_kernel()` y `gpu_fp64_conv_step()`, los compila en un binario mínimo y compara contra dos referencias NumPy independientes entre sí. Verifica **cuatro** cosas, no una:

1. **Orden de operandos** de `cublasDgemm` (el análogo exacto de GEMM).
2. **Indexación del `im2col`**: una transposición `r↔s` o leer el filtro volteado (convolución en vez de correlación) darían un resultado finito y plausible. Con el Laplaciano de 5 puntos, que es *simétrico*, ninguno de los dos es visible — por eso el script corre un segundo caso con un filtro deliberadamente **asimétrico**, que es la única forma de distinguirlos.
3. **Padding "SAME"** (`pad=1, stride=1, dilation=1`): las celdas del borde deben leer ceros. Un `wrap`/`replicate` dejaría el interior exacto y solo alteraría el anillo exterior, diluido en una norma global — el script reporta el error del **borde por separado** para que un fallo de padding se nombre como tal.
4. **Estructura bloque-diagonal** de `W`: un filtro que mezclara canales seguiría dando números finitos, y la afirmación "64 simulaciones de Stencil independientes" dejaría de ser cierta sin que nada lo delatara.

Resultado en GPU Ampere real (`sm_86`, 2026-09-06): pasa las cuatro. `rel_linf = 2.2e-16` contra la correlación con padding de ceros, frente a `1.04` si el filtro estuviera volteado y `0.48` si estuviera transpuesto; bloques fuera de la diagonal exactamente `0`; error del borde (`2.0e-16`) indistinguible del del interior (`2.2e-16`).

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

Mismo esquema `CSV_DRIFT`/`CSV_SUMMARY` que `Fase_3/GEMM/gemm_chained.cu` (ver ese README), con `hw` en la columna de tamaño y `anchor_every` como última columna — que en este binario vale siempre `0`, porque Fase 3 no tiene ancla. La ruta `GPU_FP64` se publica también como fila propia.

## ⚠️ Los números de tiempo/energía anteriores a 2026-09-06 no son utilizables

Exactamente el mismo problema, y la misma corrección, que documenta `Fase_3/GEMM/README.md` en su sección homónima — **léela ahí**, con la tabla de antes/después medida en GPU real. En resumen: `t_iter_ms`, `t_total_ms`, `gflops` y `energy_gpu_j` (a) no distinguían `_none` de `_comp` ni excluían el costo de la referencia FP64, y (b) descontaban del tiempo medido el cómputo que el `cudaMemcpy` del checkpoint absorbía al esperar la cola asíncrona. Los números de **error** (`rel_l2`, `rel_linf`) nunca estuvieron afectados.

Ahora se mide en tres fases separadas (referencia FP64 → ruta `_none` → ruta `_comp`), cada una con su cronómetro y su ventana de `PowerBuffer`. Verificado en GPU real a `--hw 128 --iters 20`: `FP16_comp` = 2.07× `FP16_none`, y `GPU_FP64` aparte a 17.09 ms/iter. Lo vigila `Fase_4/tools/gate4_medicion.py`.

**Costo de memoria del host**: los snapshots FP64 de referencia, `num_checkpoints × kChannels·hw² × 8 B` (el binario lo imprime al arrancar). A `hw=512` con `CHECKPOINT_EVERY=5` e `ITERS=80` son 16 × 128 MiB = 2.1 GiB.

## Campaña por defecto

`run_conv_chained.sbatch` corre, si no se le exporta nada, el barrido completo:

| Variable | Default | Nota |
|---|---|---|
| `HW_LIST` | `64 128 256 512` | `hw²` debe ser múltiplo de 64. Techo por memoria — ver abajo. |
| `ITERS_LIST` | `20 40 80` | **Nueva**: reemplaza al escalar `ITERS`, que sigue funcionando y gana si se exporta. |
| `COMP_LIST` | `off on` | |
| `TC_FORMAT` | `both` | |
| `SMOKE_TEST` | `0` | `1` recorta a `hw=64`, 3 iteraciones y `RUN_NCU=0`. |

### Presupuesto de memoria — el cálculo corregido

Una versión anterior de la auditoría trataba el presupuesto de este kernel como "menos predecible" que el de GEMM, asumiendo que pasaba por cuDNN/CUTLASS. **Eso es cierto para Fase 2** (`Fase_2/Convolution/conv_tensor_activation.cu` sí tiene rutas cuDNN y CUTLASS), pero **no** para este archivo: `conv_chained.cu` no incluye `cudnn.h` ni CUTLASS en absoluto — reescribe la convolución como GEMM con un `im2col` propio y la resuelve con el mismo kernel WMMA y las mismas llamadas `cublasDgemm` que GEMM. Todos los buffers son `cudaMalloc` de tamaño conocido, así que el presupuesto se calcula con la misma precisión que el de GEMM.

El factor que hay que no olvidar es el de **canales**: el estado escala como `kChannels·hw² = 64·hw²`, y el buffer `im2col` como `kCRS·hw² = 576·hw²`, que es el término dominante. Por elemento de campo, con `--tc both`, `--comp on` y ancla activa:

| Concepto | B/elemento de campo |
|---|---|
| referencia FP64 (`d_x64_in/out`) | 16 |
| `im2col` en `double` (`d_col64`, 9× el campo) | **72** |
| ruta sin comp (`2×T` + `d_col` + `t_raw`) | 26 |
| ruta con comp (`+2×comp` + `comp_col` + `comp_raw`) | 56 |
| ancla FP64 (4 buffers `double`) | 32 |
| **total** | **202** |

= **12.9 KB por celda espacial**. De ahí:

| `hw` | Memoria | Veredicto |
|---|---|---|
| 256 | 0.85 GB | |
| 512 | 3.39 GB | margen 12× — **techo de la campaña** |
| 1024 | 13.6 GB | margen 2.9×: techo real bajo el criterio de 2× |
| 2048 | 54.2 GB | imposible |

**Referencia cruzada con GEMM** (la que pedía la auditoría): a `hw=256` el campo tiene `64·256² = 4.19M` elementos, el mismo orden que GEMM a `N=2048` (`4.19M`). Pero Convolución gasta 202 B/elemento contra los 90 B/elemento de GEMM — 2.2×, por el `im2col` en `double` —, así que el equivalente de `N=8192` (67.1M elementos, 6.0 GB) es `hw=1024` (67.1M elementos, 13.6 GB).

**Por qué el límite queda en 512 y no en 1024**: 512 da cuatro puntos de tamaño (los mismos cuatro que GEMM) con 12× de margen, y el salto a 1024 no agrega un régimen nuevo — solo consume el margen. Si hiciera falta el punto grande, 1024 está calculado y cumple el criterio de 2×; 2048 no.

## Qué falta

- **`Fase_4/Convolution/`**: ✅ hecho — extensión con el ancla FP64.
- **`run_conv_chained.sbatch`**: ✅ hecho — ahora con `HW_LIST` ampliado, `ITERS_LIST` y `SMOKE_TEST`.
- **Post-proceso de CSV**: ✅ hecho — `../tools/extract_csv_chained.py`.
- **Verificación del orden de operandos, `im2col`, padding y filtro**: ✅ hecha y **pasada en GPU real** — ver arriba.
- **Scripts de gate** K=0/K=1: ✅ hechos — `Fase_4/tools/gate3_ancla.py` y `Fase_4/Convolution/gate3_ancla.sbatch`.
- **Medición por ruta**: ✅ corregida — ver la advertencia de arriba, y `Fase_4/tools/gate4_medicion.py`.
- **Campaña real en PACCA**: compilado y verificado con `--hw` chico; falta el barrido completo.
