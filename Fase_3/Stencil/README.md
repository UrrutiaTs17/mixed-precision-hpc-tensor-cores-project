# Fase_3/Stencil

Stencil 2D de 5 puntos con encadenamiento genuino: cada ruta aplica el operador
`--iters` veces sobre su **propia** salida (`salida(i) -> entrada(i+1)`), no
sobre el input original repetido, así que el drift numérico medido es el que
se acumula de verdad a través de iteraciones reales — no un artefacto de
relanzar la misma operación sobre el mismo buffer (eso es lo que hacía
`Fase_2/Stencil`, y sigue siendo válido para medir throughput puro, pero no
para medir deriva).

## Procedencia de este archivo

`stencil_tensor_activation.cu` es una **relocalización sin cambios
numéricos** de `old/Fase_4/Stencil/stencil_tensor_activation.cu` — la versión
de la rama `fase4-estadistica-variabilidad` del histórico del proyecto, que
nunca se fusionó a `main`. Se migró esa versión (6501 líneas) en vez de la que
`main` tenía en `old/Fase_3/Stencil/` (3728 líneas, formulación WMMA con 5
`mma_sync` contra matrices identidad escaladas) porque es estrictamente más
avanzada: reformulación `Y = X·H + V·X` con solo 2 `mma.sync` en vez de 5,
tiling ensanchado a 16×64 (`kTileW`), `ldmatrix`/`mma.sync.m16n8k16` explícito
con swizzle de shared memory, soporte de CUDA Graphs, y checkpointing /
archivado más completos — todo ya validado por gates de regresión en el
histórico del proyecto.

Los únicos cambios frente al original son de **includes**, para apuntar a los
headers ya migrados en `common/` en lugar de `tools/power_sampling.h` y
`../../Fase_2/common.cuh` (que no existen en esta rama):

- `#include "../../common/power_sampling.h"` en vez de `#include
  "tools/power_sampling.h"`. Son el mismo código: se comparó línea a línea
  contra la copia local del original y la única diferencia es de formato
  (indentación, ancho de línea, llaves en una sola línea) y traducción de
  comentarios al español — ninguna firma de función ni struct cambió. Ver
  "Hallazgos de la auditoría" más abajo.
- `#include "../../common/cuda_checks.cuh"` + `#include
  "../../common/metrics.cuh"` en vez de `#include "../../Fase_2/common.cuh"`.
  El header original agrupaba en un solo archivo las macros `CHECK_CUDA` /
  `CHECK_CUBLAS` / `CHECK_CUDNN`, `CudaEventTimer` y las funciones
  `compare_*`/`ErrorMetrics`; en `common/` está separado por responsabilidad
  única (validación de API vs. métricas), sin cambio de comportamiento — se
  comparó también línea a línea contra `old/Fase_2/common.cuh`.

**Lo que NO se migró aquí, a propósito:** el mecanismo de "ancla FP64" (cada
`K` iteraciones, un paso completo en FP64 en vez de en baja precisión) es una
extensión que se construye por separado en `Fase_4/Stencil/`, apoyada en este
mismo kernel — no toca este archivo. Tampoco se migraron los scripts de
campaña/validación de Fase 4 (`gate1_regresion.sbatch`, `validar_*.sbatch`,
`lanzar_campana_pareto3d.sh`, `tools/analizar_variabilidad.py`, etc.): esos
van a `Fase_4/Stencil/`.

## Rutas ejecutadas

Cada corrida compila un único binario (`stencil_tc`) y ejecuta, en este orden,
las rutas habilitadas por los flags:

| Ruta | Qué es | Se puede desactivar con |
|---|---|---|
| `CPU_FP32` | Referencia serial en CPU, `float`. Siempre corre. | — |
| `CPU_FP64` | Segunda pasada FP64 serial en CPU — referencia de **costo** del patrón de oro IEEE 754, no del ground truth (ver más abajo). | `--cpu-fp64 off` |
| `GPU_FP32` | CUDA clásico en GPU, `float`, sin Tensor Cores. | — |
| `GPU_FP64` | CUDA clásico en GPU, `double`, sin Tensor Cores — referencia de máxima precisión **en GPU** (denominador GPU-vs-GPU de los speedups). | `--fp64-gpu off` |
| `WMMA_FP16` / `WMMA_BF16` | Tensor Cores vía `mma.sync.m16n8k16`, operandos FP16 o BF16, acumulación FP32. Con `--tc both` corren ambos. | `--tc fp16\|bf16\|both` |

Además, **antes** de cualquier ruta anterior, se calcula la **referencia
FP64** (`compute_cpu_stencil_fp64`, en CPU, encadenada por el mismo número de
`--iters` que las rutas comparadas): es el ground truth contra el que se mide
`rel_l2`/`rel_linf`/`max_abs` de todas las demás rutas. No confundir con la
ruta `CPU_FP64` de la tabla — esa es una **segunda** pasada FP64, separada a
propósito para no heredar la instrumentación extra del ground truth (norma
`||u^n||_inf` por iteración, test de finitud), que si se cronometrara
reportaría un costo de FP64 en CPU varias veces más alto del real.

Las rutas WMMA aceptan además `--execution-mode normal|graph` (o
`--cuda-graph` como alias corto): en `graph`, los tramos de iteraciones que no
piden checkpoint se agrupan en bloques de `--graph-block` (por defecto 32,
debe ser par) y se reproducen desde un `cudaGraphExec_t` preinstanciado, para
sacar del camino medido el costo de CPU de lanzar un kernel por iteración. No
cambia el kernel, la fórmula, los coeficientes ni el ping-pong — solo quién
paga el lanzamiento.

## Compensación del redondeo de almacenamiento

Las rutas WMMA guardan el estado en FP16/BF16 entre iteraciones aunque el
Tensor Core acumule internamente en FP32: ese "guardar" redondea, y ese
redondeo (no el del `mma.sync` en sí) es lo que las tres políticas de
`--kahan` / `--spatial-comp` atacan.

- **`none`** (`--kahan off --spatial-comp off`, por defecto en el binario):
  sin compensación. Comportamiento histórico, byte a byte.
- **`kahan_local`** (`--kahan on`): cuantizador con retroalimentación de error
  (noise shaping de primer orden) por celda — `comp[idx]` guarda el residuo
  que dejó la escritura anterior de **esa misma** celda, y se pre-resta antes
  de volver a redondear.

  El nombre histórico (`--kahan`) es engañoso: **no** es la suma compensada
  de Kahan clásica, que exige un acumulador vivo al que sumar incrementos —
  aquí el valor se recalcula entero desde los 5 vecinos en cada iteración, no
  hay tal acumulador. Y su efecto **depende del operador**, no es un
  no-op universal:

  - Bajo `--op-mode diffusive` (campo suave, residuos correlacionados entre
    iteraciones) reduce el error medido entre 24 % y 26 % frente a no
    compensar.
  - Bajo `--op-mode stress` (el operador histórico; el modo de Nyquist se
    duplica cada iteración y decorrelaciona el residuo) lo **empeora** un
    6-9 %: sin correlación temporal que explotar, la pre-resta solo inyecta
    ruido extra.

  Esto refina — no contradice — el hallazgo original de Fase 3 (documentado
  en `old/Fase_3/Stencil/tools/README.md`): bajo el operador de estrés, único
  disponible en esa fase, Kahan local resultaba indistinguible de no
  compensar (`rel_l2` igual hasta la 4ª cifra, horizonte de overflow sin
  mover). La razón estructural es la misma en ambos operadores y es la que
  explica por qué el `--kahan on` **clásico, ingenuo** (sin retroalimentación,
  `comp = val - Q(val)` sin reincorporar el residuo en la recurrencia) sí
  resultó indistinguible cuando se probó: el residuo de una celda solo lo
  relee **ella misma** en la iteración siguiente, nunca sus 4 vecinas, que son
  las que de verdad usan ese valor para calcular su propio resultado. Sin
  retroalimentación explícita, reconstruir el valor en el momento de leerlo
  (readout) solo deshace el redondeo de la **última** escritura, y ese
  término se diluye rápido según se acumula error (medido: -47.8 % a 1
  iteración, ya -6.3 % a 5, entre 0.00004 % y 0.003 % a 20).

- **`spatial`** (`--spatial-comp on`): compensación **espacial** (error
  feedback de vecinos). Cada celda reincorpora los residuos de sus 4 vecinas
  **y** el propio antes de sumar — no solo el suyo. Es la que de verdad
  funciona: baja `rel_l2` entre 5 y 6 órdenes de magnitud frente a no
  compensar, y lleva el horizonte de overflow de BF16 al mismo que FP32
  clásico. Cuesta 5 lecturas globales FP32 y 1 escritura FP32 extra por celda
  por iteración, y duplica el buffer de residuos (ping-pong) — no es gratis,
  pero es más barato que `kahan_local` (~1.56x vs. ~1.97x en `t_iter_ms`
  medido).

  **Por qué esta sí funciona y la local no**: el operador del stencil es
  **lineal**, `L(v + c) = L(v) + L(c)`, con `v` el estado guardado en FP16/BF16
  y `c` el residuo de redondeo acumulado en FP32. Eso significa que no hace
  falta re-alimentar el valor ya corregido (`v + c`) a través del Tensor Core
  para obtener el resultado correcto — basta con calcular la corrección
  `L(c)` aparte, en FP32, con la misma fórmula de 5 puntos, y sumarla al
  resultado de `L(v)` que el Tensor Core ya calculó. Es exactamente lo que
  hace `wmma_epilogue_value`/`wide_epilogue_value` en el kernel: el término
  principal sale del `mma.sync` sobre el estado cuantizado, y la corrección
  espacial se suma después, en FP32, sin tocar la ruta de Tensor Core.
  `--kahan on` no puede aprovechar esto porque su convención de signo (`Q(y) -
  y`, referida a un `y` ya pre-corregido) no es la del residuo puro `val -
  Q(val)` que la linealidad exige — sumarlo directamente da `2Q(y) - y`, que
  se pasa de largo en vez de corregir.

  `--kahan on` y `--spatial-comp on` son **mutuamente excluyentes** (son dos
  políticas alternativas del mismo problema, no capas acumulables); el
  binario rechaza la combinación con un error, no la ignora en silencio.

Las rutas WMMA bajo compensación espacial se reportan con sufijo `_SP`
(`WMMA_FP16_SP` / `WMMA_BF16_SP`) para que sus filas nunca se confundan con
las de `--kahan off|on` al mezclar corridas en el mismo CSV — la columna
`kahan` de esas filas sigue siendo `off`.

## Horizonte de overflow: medido vs. predicho

Bajo `--op-mode stress` (el operador histórico), el modo de Nyquist
`g(π,π) = -2` se **duplica** en cada iteración: toda ruta de precisión
reducida termina desbordando el rango de su formato tarde o temprano. El
programa reporta ese fenómeno de dos formas distintas y las mantiene
separadas a propósito:

- **Horizonte medido** (`h_medido`, columna `n_star`/`measured_n`): la
  primera iteración exacta en la que algún punto interior de la ruta deja de
  ser finito — un hecho observado, vía `atomicMin` sobre un contador
  compartido por todos los bloques del kernel (una sola atómica por bloque,
  no por hilo, para no serializar el kernel al divergir). `-1` (o "ninguna")
  significa que la ruta se mantuvo finita hasta `--iters`.
- **Horizonte predicho** (`h_predicho`): una **proyección**, calculada de
  antemano y sin correr ninguna ruta de baja precisión, a partir de un ajuste
  log-lineal `log2(||u^n||_inf) = log2(A) + n·log2(λ)` sobre la referencia
  FP64 en su régimen asintótico (60%-90% de las iteraciones finitas). Con la
  semilla efectiva `A` del ajuste y el máximo representable de cada formato,
  `n*_T = log2(FMT_MAX_T) - log2(semilla_T)`.

  El ajuste **se niega a imprimir una predicción** si la ventana asintótica
  tiene menos de 30 puntos finitos (`kMinOverflowFitPoints`): con menos
  puntos el ajuste degenera sobre el transitorio inicial y puede variar 5
  órdenes de magnitud según `--iters` — un número bien formado mostrando algo
  sin significado en vez de nada. Ver `fit_overflow_model` y
  `print_overflow_horizon` en el `.cu`. Tampoco hay predicción si el operador
  activo **no amplifica** el modo de Nyquist (`--op-mode diffusive` con su
  `alpha` por defecto, donde `|g(π,π)| <= 1`): sin crecimiento no hay formato
  que desborde, y la fila CSV_HORIZON sale con estado
  `contractive_operator`, no con un número.

Las dos cifras conviven en el mismo CSV_HORIZON para poder auditar la
calidad del modelo (columnas `lambda`, `r_squared`, `n_points`) contra lo que
de verdad ocurrió, en vez de confiar ciegamente en la proyección.

## Flags de CLI

```
./stencil_tc [--nx NX] [--ny NY] [--iters I] [--tc fp16|bf16|both]
             [--checkpoint-every K] [--csv RUTA] [--profile-only]
             [--kahan off|on] [--spatial-comp off|on]
             [--fp64-gpu off|on] [--cpu-fp64 off|on]
             [--op-mode stress|diffusive] [--alpha A]
             [--ci-mode legacy|monomode] [--ci-p P] [--ci-amplitude A]
             [--checkpoint-iters "1,2,5,..."] [--archive-iters "..."]
             [--archive-dir RUTA]
             [--execution-mode normal|graph] [--cuda-graph] [--graph-block B]
```

| Flag | Default | Qué hace |
|---|---|---|
| `--nx`, `--ny` | 2048, 2048 | Dimensiones de la malla (>= 3). |
| `--iters` | 20 | Iteraciones encadenadas. |
| `--tc` | `both` | Formato Tensor Core: `fp16`, `bf16` o `both`. |
| `--checkpoint-every K` | 0 (off) | Cada K iteraciones, compara cada ruta contra un snapshot FP64 de esa iteración (`CSV_DRIFT`/`CSV_ONSET`). Es también la única forma de que `storage_rel` sea evaluable si una ruta diverge antes de `--iters`. Mutuamente excluyente con `--checkpoint-iters`. |
| `--checkpoint-iters "1,2,5,..."` | vacío | Igual que arriba pero en iteraciones exactas, no múltiplos; la referencia FP64 se vuelca a disco (`ReferenceSpill`) en vez de acumularse en RAM — necesario para mallas grandes con muchos checkpoints. Emite `CSV_CKPT`/`CSV_NORM`. |
| `--archive-iters "320,640"` | vacío | Subconjunto de `--checkpoint-iters`: vuelca a disco el campo completo de cada ruta en esas iteraciones (binario crudo + `manifest.json` con sha256). |
| `--archive-dir RUTA` | `archive` | Dónde van esos ficheros. Debe existir (no se crea). |
| `--csv RUTA` | vacío (sin CSV) | Agrega una fila por ruta/configuración a un CSV persistente (esquema `kCsvHeader`, distinto del stdout parseable `CSV_SUMMARY`). |
| `--profile-only` | off | Corre solo `GPU_FP32` + la ruta TC de `--tc` (no admite `both`), sin referencias CPU/FP64 ni métricas de error — para perfilar con `ncu` sin pagar el costo de calcular las referencias. |
| `--kahan off\|on` | off | Ver "Compensación del redondeo de almacenamiento". |
| `--spatial-comp off\|on` | off | Ídem. Mutuamente excluyente con `--kahan on`. |
| `--fp64-gpu off\|on` | on | Corre `GPU_FP64`. |
| `--cpu-fp64 off\|on` | on | Corre `CPU_FP64`. |
| `--op-mode stress\|diffusive` | `stress` | Operador de 5 puntos activo. Ver más abajo. |
| `--alpha A` | 0.1875 (3/16) | Solo con `--op-mode diffusive`; error si se pasa bajo `stress`. |
| `--ci-mode legacy\|monomode` | `legacy` | Condición inicial. `legacy`: la histórica de fases 1-3. `monomode`: un único modo propio del Laplaciano discreto — necesaria bajo el operador difusivo (la CI legacy decae solo ~1.2% en 640 iteraciones bajo ese operador). |
| `--ci-p P`, `--ci-amplitude A` | 168, 1.0 | Solo con `--ci-mode monomode`. |
| `--execution-mode normal\|graph`, `--cuda-graph` | `normal` | Ver "Rutas ejecutadas". |
| `--graph-block B` | 32 | Solo con `--execution-mode graph`; debe ser par y > 0. |

`--op-mode`: `stress` (por defecto) es `out = 0.25*(u+d+l+r) - center`, el
operador histórico de las fases 1-3, con `g(π,π) = -2` — divergente por
diseño, es el fenómeno que mide el horizonte de overflow. `diffusive` es
`out = alpha*(u+d+l+r) + (1-4*alpha)*center`, estable con el `alpha` por
defecto (`g(π,π) = -0.5`), y es el operador con el que `rel_l2` se mantiene
finito el tiempo suficiente para construir el eje de error de un frente de
Pareto — bajo `stress`, toda ruta de precisión reducida termina en NaN.

### Ejemplos

```bash
# Corrida por defecto: 2048^2, 20 iteraciones, ambos formatos TC, sin compensación.
./stencil_tc

# Malla grande con checkpoints cada 5 iteraciones y CSV de resumen.
./stencil_tc --nx 4096 --ny 4096 --iters 20 --tc both --checkpoint-every 5 --csv results/run.csv

# Compensación espacial en FP16.
./stencil_tc --nx 4096 --ny 4096 --iters 20 --tc fp16 --spatial-comp on

# Operador difusivo con condición inicial monomodo.
./stencil_tc --nx 16384 --ny 16384 --iters 640 --tc both --op-mode diffusive --ci-mode monomode

# Horizonte de overflow con CUDA Graphs.
./stencil_tc --nx 4096 --ny 4096 --iters 320 --tc fp16 --execution-mode graph --graph-block 64

# Perfilado con Nsight Compute (ver el hint que el propio binario imprime al terminar).
./stencil_tc --nx 4096 --ny 4096 --iters 20 --tc fp16 --profile-only
```

## Compilación

```bash
nvcc -std=c++17 stencil_tensor_activation.cu -o stencil_tc \
     -gencode arch=compute_80,code=sm_80 \
     -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
     -DUSE_NVML_TELEMETRY -lnvidia-ml -lpthread   # opcional, telemetría GPU
```

`arch=compute_80,code=sm_80` es para A100 (Ampere). El kernel WMMA usa
`mma.sync.aligned.m16n8k16` explícito en `sm_80+`; en `sm_70`/`sm_75` cae a un
respaldo con `wmma::` clásico (mismo resultado, sin el camino ancho
optimizado). `-ffp-contract=off` en las rutas de CPU es necesario para que su
aritmética siga siendo bit a bit comparable con el histórico — ver el
comentario junto a `stencil2d_fp32_kernel` en el `.cu` sobre por qué el
device sí usa `fmaf`/`fma` explícito.

## Esquema de columnas CSV

El binario emite, por stdout, líneas `CSV_*` parseables (una familia de
tokens por tipo de dato) más, opcionalmente, un CSV persistente vía `--csv`.
La semántica de columnas descrita en `old/Fase_3/Stencil/tools/README.md`
sigue siendo válida — reproducida y ampliada aquí:

- **`CSV_SUMMARY`**: una fila por ruta/configuración con tiempos, error,
  energía y las columnas de contexto del operador (`op_mode`, `alpha`,
  `ci_mode`, `ci_p`, `cell_updates_per_s`, `energy_per_cell_update_j`,
  `reference_role`, `execution_mode`). `speedup_cpu`/`speedup_fp32` **no**
  son especulación: son `t_total_ms` crudo de cada fila, y las razones se
  calculan aguas abajo, en el análisis.
- **`CSV_ENERGY`**: energía GPU/CPU/total, EDP, J/GFLOP, `energy_gpu_j_per_iter`
  (comparable entre corridas con distinto `--iters`) y `window_reliable`
  (`1` solo si la ventana de energía es lo bastante larga, ver más abajo).
- **`CSV_DRIFT`** (cadencia `--checkpoint-every`) / **`CSV_CKPT`** (cadencia
  `--checkpoint-iters`, con `rel_linf` que `CSV_DRIFT` no lleva): error de
  cada ruta contra el snapshot FP64 de esa iteración exacta. `NONFINITE` en
  vez de un número cuando la referencia o la ruta ya divergieron — nunca un
  cero o un NaN silencioso.
  **Qué compara exactamente, que no es obvio**: `d_out_fp32`, el acumulador
  FP32 *antes* del redondeo de almacenamiento (es el "ancla de no-regresión"
  del kernel). Lo que realmente se propaga entre iteraciones es el buffer `T`
  de 16 bits, y ese error vive en las columnas `rel_l2_prop`/`rel_linf_prop`
  de `CSV_SUMMARY`. La distinción importa al comparar contra GEMM/Convolución:
  el `rel_l2` de `CSV_DRIFT` de aquellos kernels mide el buffer cuantizado,
  o sea el análogo de `rel_l2_prop` de aquí, **no** de este `rel_l2`. Es
  también la razón por la que el gate K=1 del ancla usa criterios distintos
  por kernel — ver `Fase_4/tools/README.md`.

**`anchor_every`** es la última columna de `CSV_DRIFT`, `CSV_SUMMARY` y
`CSV_ENERGY`. En este binario vale siempre `0` (Fase 3 no tiene ancla); existe
para que el esquema sea idéntico al de `Fase_4/Stencil`, que sí la usa, porque
`run_full_pipeline.sh` concatena los `results/` de las dos fases en el mismo
análisis.
- **`CSV_NORM`**: `||u^n||_2` y `||u^n||_inf` de la referencia FP64 en cada
  checkpoint — dan escala a `CSV_DRIFT`/`CSV_CKPT` (un `rel_l2` que crece no
  distingue "la ruta se degrada" de "la referencia se encoge", y bajo el
  operador difusivo la referencia sí se encoge de forma sistemática).
- **`CSV_HORIZON`**: `predicho`/`medido` por formato, más `lambda`,
  `r_squared`, `n_points`, `A` (semilla efectiva) y `estado`
  (`ok`/`insufficient_points`/`contractive_operator`) — ver "Horizonte de
  overflow" arriba.
- **`CSV_STORE`**: error relativo de **guardar** el estado en 16 bits
  (`store_rel_norm`, `store_rel_max_guarded`), independiente del error contra
  FP64 acumulado por la recurrencia completa.
- **`CSV_ONSET`**: primera iteración de checkpoint en la que cada ruta deja
  de ser finita — granularidad de checkpoint, no la iteración exacta (esa es
  `CSV_DRIFT`/`CSV_HORIZON`); es una cota superior del horizonte real.
- **`CSV_REGION`**: marca `begin`/`end` (timestamp de pared en ns) de la
  ventana cronometrada de cada ruta, para alinear un muestreador de potencia
  externo si hiciera falta.
- **`CSV_ARCHIVE`** / **`CSV_CI_SHA256`**: metadatos del archivado de campos
  completos y huella de la condición inicial (determinismo verificable entre
  corridas).

Los nombres de ruta (`CPU_FP32`, `CPU_FP64`, `GPU_FP32`, `GPU_FP64`,
`WMMA_FP16`/`WMMA_BF16`, o con sufijo `_SP` bajo compensación espacial) son
consistentes en todas las familias de token.

## Qué significa cada métrica

- **`rel_l2`**: `||ref - test||_2 / ||ref||_2`, con `ref` la referencia FP64
  (ground truth). Es la métrica primaria de exactitud.
- **`rel_linf`**: `max|ref - test| / ||ref||_inf` — el análogo en norma
  infinito; más sensible a un único punto que se descarrila que `rel_l2`.
- **`max_abs`**: `max|ref - test|`, sin normalizar. Sin la norma de la
  referencia al lado (`CSV_NORM`), un `max_abs` no dice si es mucho o poco.
- **`storage_rel`** (`store_rel_norm`/`store_rel_max_guarded`): cuánto se
  pierde **solo** por el redondeo de guardar el estado en FP16/BF16
  (`Q(u) - u`), aislado del error acumulado por toda la recurrencia. Sirve
  para separar "el formato de 16 bits no puede representar esto" de "el
  error se acumuló iteración tras iteración".
- **Horizonte de overflow** (`n*`, medido/predicho): ver la sección dedicada
  arriba.
- **`drift`**: cómo crecen `rel_l2`/`rel_linf`/`max_abs` de una ruta a
  través de checkpoints sucesivos — la cantidad que este archivo existe para
  medir (de ahí "Fase 3: encadenamiento genuino y drift").
- **Ventana de energía confiable** (`window_reliable`): el contador de
  energía de NVML avanza en saltos discretos (~20-25 ms de refresco del
  sensor onboard), así que una ventana corta tiene un error de cuantización
  del mismo orden que el valor medido. `window_reliable=1` exige
  `time_total_s >= 0.5 s × <número de segmentos de energía>` (el
  checkpointing parte la ventana en varios segmentos, y cada uno paga su
  propio salto de cuantización). Filas con `window_reliable=0` deben
  excluirse de cualquier promedio de energía aguas abajo.

## Hallazgos de la auditoría (para tener en cuenta al usar este kernel)

0. **`NX_LIST`/`NY_LIST`** (nuevo). `run_stencil_tc.sbatch` ya no corre un solo
   tamaño: barre `4096 8192 16384` por defecto, los mismos tres de la campaña
   de drift ya publicada. Las dos listas se recorren **emparejadas**, no como
   producto cartesiano — todo el análisis aguas abajo asume mallas cuadradas
   (`Fase_4/tools/common_analysis.py` colapsa `(nx, ny)` a una sola columna
   `size` usando solo `nx`), así que un producto 3×3 generaría 6 corridas
   rectangulares mal etiquetadas sin avisar. Listas de distinta longitud dan
   error explícito. `NX`/`NY` siguen funcionando y **ganan** si se exportan:
   `NX=8192 NY=8192 bash run_stencil_tc.sbatch` corre un solo tamaño, igual
   que antes de este cambio.

1. **Rango de `NX`/`NY` nunca ejercitado por debajo de 4096²**. Los flags
   `--nx`/`--ny` no tienen ningún límite superior a lo que el hardware
   soporte, y los dos `.sbatch` de esta carpeta ya los exponen como variables
   de entorno sin valor hardcodeado — técnicamente **sí** se puede correr a
   512²/1024²/2048² hoy. Lo que no existe es evidencia de que el histórico
   del proyecto haya corrido alguna campaña real en ese rango: todas las
   corridas documentadas en los comentarios del `.cu` y de los `.sbatch`
   (presupuestos de memoria, calibración de `--checkpoint-every`, tiempos de
   `#SBATCH --time`) asumen 4096²-16384². Antes de usar este binario para
   generar datos a 512²-2048² conviene revalidar al menos el horizonte de
   overflow medido en ese rango (la condición inicial legacy tiene modos
   dominantes de frecuencia fija en radianes/celda, así que su comportamiento
   relativo al tamaño de malla no está garantizado que escale igual que a los
   tamaños grandes).
2. **`common/power_sampling.h` vs. la copia local del original**: se
   comparó línea a línea (`diff -bB`) contra `old/Fase_3/Stencil/tools/power_sampling.h`
   (que es exactamente lo que `old/Fase_4/Stencil/tools/power_sampling.h`
   incluía, vía symlink). La única diferencia es de formato — indentación de
   2 vs. 4 espacios, ancho de línea, llaves de una función en la misma línea
   o no — y traducción de comentarios de inglés a español. **Ninguna firma
   de función, struct, macro o constante cambió.** Es seguro usar la copia de
   `common/` como reemplazo directo.
3. **Kahan local no es universalmente "indistinguible de no compensar"**:
   el hallazgo de Fase 3 (`old/Fase_3/Stencil/tools/README.md`) se estableció
   únicamente bajo `--op-mode stress`, el único operador disponible en esa
   fase. Con el operador difusivo que Fase 4 agregó, `--kahan on` sí mueve la
   aguja — para mejor (-24% a -26% de error) porque el operador difusivo
   produce residuos correlacionados que el noise shaping puede explotar. El
   hallazgo cualitativo no cambia (`spatial` sigue siendo la única política
   que reduce el error en **ambos** operadores, y por menos costo), pero
   "indistinguible" deja de ser una descripción correcta en general — ver
   "Compensación del redondeo de almacenamiento" arriba.

No se encontró ningún `// NOTA MIGRACIÓN:` que documentar en el `.cu`: no se
identificó ningún bug real durante la migración (la lógica numérica se
copió sin tocar; los únicos cambios son los dos `#include` descritos arriba).
