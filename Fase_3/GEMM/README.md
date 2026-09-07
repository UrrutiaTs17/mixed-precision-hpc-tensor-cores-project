# Fase 3 — GEMM: encadenamiento genuino

`gemm_chained.cu` encadena GEMM de verdad: `X(n+1) = X(n) · A`, con `A` un operador **fijo** aplicado repetidamente — la misma estructura que la iteración de potencias para autovalores. Corrige la corrección #13 del documento *Plan de Precisión Mixta*: GEMM se medía hasta Fase 2 como una llamada suelta, sin representar cómo se usa realmente en HPC clásico (solucionadores iterativos, capas apiladas), donde el error de una operación se propaga a la siguiente.

Sin precedente en el proyecto anterior — a diferencia de Stencil (que ya llegaba encadenado desde `old/Fase_4`), esto se escribió desde cero siguiendo la sección 02 del plan. Compilado y verificado en GPU Ampere+ real (`sm_89`).

## El operador A: por qué una matriz de Hadamard, no una ortogonal genérica

`A = c · H`, con `H` la matriz de Hadamard de Sylvester (entradas exactamente `±1`, construcción recursiva, requiere `N` potencia de 2 — los tamaños 512/1024/2048/4096 del rango del plan lo son todos) y `c` una potencia de 2 exacta.

Esto no es una elección arbitraria — corrige un hallazgo real de la segunda ronda de revisión del plan: una matriz ortogonal genérica (de una descomposición QR aleatoria) tiene entradas que **no son exactamente representables en FP16/BF16**. Al castear esa matriz a Tensor Cores, cada entrada se redondea, introduciendo un error de *operador* sistemático — el mismo en cada iteración — que ninguna compensación del *estado* puede corregir, porque `comp` solo rastrea el redondeo de `X`, nunca el de `A`. Con `A = c·H` (entradas `±c`, `c` potencia de 2), `A_T = A` bit a bit al castear a cualquier formato — el único error que queda es exactamente el que `comp` está diseñado para compensar.

`c` se elige para que el factor de amplificación por iteración (`λ = c·√N`, exacto en aritmética ideal — `H` es exactamente ortogonal salvo por el factor `√N`) quede cerca de `--target-lambda` (default 1.1), análogo al `λ≈2` de Stencil, pero sin necesitar que `√N` en sí sea una potencia de 2.

## Compensación por linealidad — sin ancla todavía

`X·A` es lineal, igual que el Laplaciano de Stencil: `(T+comp)·A = T·A + comp·A`. La compensación (`--comp on`) explota esto exactamente como la compensación espacial de Stencil: el término principal se calcula con Tensor Cores sobre `T`, la corrección se calcula **también** con Tensor Cores pero sobre `comp` (truncado a `T` — es pequeño, así que ese redondeo es de segundo orden), y se suman.

`comp` se guarda en `float` — este archivo es Fase 3, no incluye el ancla FP64. Eso es `Fase_4/GEMM/gemm_chained.cu` (ya existe), que ensancha `comp` a `double` siguiendo el mismo patrón que `Fase_4/Stencil/stencil_tensor_activation.cu` — con kernels locales propios, no vía `common/chained_precision.cuh` (ver `common/README.md`, sección de ese header, sobre por qué terminó sin usarse).

`comp` se siembra desde el redondeo *real* de `x0→T` (`seed_comp_from_double_kernel`), no desde cero — mismo patrón que `Fase_4/Stencil`. Sembrar `comp` en cero descartaría el error de la primerísima conversión `x0→T`, que quedaría sin corregir para siempre (se propaga amplificado por `A` en cada iteración) en vez de quedar capturado desde el principio, como exige la propiedad de reconstrucción `Q(v)+comp=v` que el resto del mecanismo asume ya válida desde `t=0`. El gate K=1 de `Fase_4/GEMM/README.md` depende de esta siembra para converger a la referencia FP64.

## El punto de mayor riesgo: orden de operandos en la referencia FP64 — ✅ verificado

cuBLAS es *column-major*; el kernel WMMA de `common/wmma_gemm.cuh` es *row-major*. Para que ambas rutas calculen exactamente `X·A` sobre el mismo buffer sin transponer nada explícitamente, la llamada a `cublasDgemm` invierte el orden de los operandos (`A` primero, `X` segundo) — ver la derivación completa en el comentario de `gpu_fp64_step()` en el `.cu`. Si esto estuviera al revés, el binario compilaría y correría igual, pero compararía peras con manzanas sin ningún error visible.

Esa verificación **ya existe y ya pasó**:

```bash
python3 ../tools/verificar_orden_operandos_gemm.py --n 32
```

`Fase_3/tools/verificar_orden_operandos_gemm.py` extrae el texto de `gpu_fp64_step()` de este archivo — sin reimplementarlo, para que verifique el código que corre en la campaña y no una copia —, lo compila en un binario mínimo que expone el resultado crudo, y lo compara contra `X @ A` calculado explícitamente en NumPy. Exige además que la función sea idéntica en Fase 3 y Fase 4.

Resultado en GPU Ampere real (`sm_86`, 2026-09-06): `rel_linf = 0.0` contra `X·A`, con las hipótesis alternativas a distancia `1.28` (`A·X`), `1.12` (`Xᵀ·A`) y `1.23` (`(X·A)ᵀ`). El orden de operandos es correcto.

Ese script es también el paso 1 de `tools/validacion_preliminar.sbatch`, que hay que correr antes de cualquier campaña.

## Kernel WMMA: compartido con Fase 2, no reimplementado

`common/wmma_gemm.cuh` (extraído de `Fase_2/GEMM/gemm_tensor_activation.cu`, ya migrado y verificado) — ver `common/README.md`. Este archivo reutiliza `wmma_gemm_kernel<T>` tal cual, tratando el estado encadenado `X` como el primer operando y `A` como el segundo.

## Energía: ventana correcta desde el diseño, no como parche posterior

El trabajo de checkpoint (copia D2H + comparación en host) queda **excluido** de la ventana de tiempo/energía medida — se pausa `PowerBuffer` y se resta `checkpoint_pause_s` del tiempo total, exactamente el mismo problema y el mismo arreglo que ya documenta `Fase_3/Stencil/stencil_tensor_activation.cu`. También reporta `window_reliable` (mismo criterio de 500 ms/tramo que Stencil, vía `kEnergyWindowReliableSeconds` de `common/power_sampling.h`) y el número de tramos, para que un `--checkpoint-every` chico no produzca una cifra de energía silenciosamente poco confiable.

## Uso

```bash
./gemm_chained --n 1024 --iters 20 --tc fp16 --comp on
./gemm_chained --n 4096 --iters 10 --tc both --checkpoint-every 5
```

| Flag | Default | Qué hace |
|---|---|---|
| `--n` | 1024 | Tamaño de la matriz cuadrada (debe ser potencia de 2). |
| `--iters` | 20 | Iteraciones encadenadas. |
| `--tc` | both | `fp16`, `bf16` o `both`. |
| `--comp` | off | Compensación por linealidad. |
| `--checkpoint-every` | 0 | 0 = solo compara contra la referencia FP64 en la última iteración. |
| `--target-lambda` | 1.1 | Factor de amplificación objetivo por iteración (ver elección de `A`). |
| `--seed` | 42 | Semilla del estado inicial `X⁰` (uniforme en [-1,1]). |

## Salida (CSV_DRIFT / CSV_SUMMARY)

```
CSV_DRIFT,<formato>_<none|comp>,n,iter,rel_l2,rel_linf,solution_finite,anchor_every
CSV_SUMMARY,<ruta>,n,iters,t_iter_ms,t_total_ms,gflops,energy_gpu_j,window_reliable,gpu_segments,anchor_every
```

`<ruta>` es `<formato>_none`, `<formato>_comp` o **`GPU_FP64`** — la trayectoria de referencia, que ahora se publica como una fila propia (ver abajo). Es el punto de Pareto "todo en FP64" que a este kernel le faltaba; Stencil ya lo tenía.

## ⚠️ Los números de tiempo/energía anteriores a 2026-09-06 no son utilizables

Hasta esa fecha, `t_iter_ms`, `t_total_ms`, `gflops` y `energy_gpu_j` de este binario estaban mal **de dos formas independientes**. Cualquier campaña corrida antes hay que volver a correrla; los números de **error** (`rel_l2`, `rel_linf`) nunca estuvieron afectados y siguen siendo válidos.

**1. No distinguían una ruta de otra.** Un solo cronómetro envolvía las tres trayectorias de cada iteración (referencia FP64 + WMMA sin compensación + WMMA con compensación) y el mismo número se imprimía en las dos filas. Los dos `PowerBuffer` se abrían y cerraban en los mismos instantes sobre `nvmlDeviceGetTotalEnergyConsumption`, que es un contador **de todo el dispositivo**, así que las dos columnas de energía eran literalmente el mismo valor. Además, ambos incluían el costo de la referencia FP64.

**2. Descontaban el cómputo del tiempo medido.** La pausa de checkpoint se abría *antes* de sincronizar. Como los lanzamientos de kernel son asíncronos, el `cudaMemcpy` D2H del checkpoint bloqueaba esperando toda la cola pendiente, esa espera caía dentro de la pausa, y se restaba del total. El resultado eran tiempos absurdamente bajos.

Medido en la misma GPU (RTX 3050, `sm_86`), `--n 2048 --iters 20 --tc both --comp on`:

| | antes | después |
|---|---|---|
| `FP16_none` | 2.925 ms/iter → 5 873 GFLOPS | 16.91 ms/iter → 1 016 GFLOPS |
| `FP16_comp` | 2.925 ms/iter (idéntico) | 35.68 ms/iter → 482 GFLOPS (2.11× de `_none`) |
| `BF16_none` | 1.045 ms/iter → **16 440 GFLOPS** | 16.93 ms/iter → 1 015 GFLOPS |
| `GPU_FP64` | no se reportaba | 167.6 ms/iter → 102.5 GFLOPS |

16 440 GFLOPS en una RTX 3050 era la señal más visible de que algo estaba mal: es un orden de magnitud por encima de lo que ese kernel puede alcanzar en esa tarjeta.

Lo verifica `Fase_4/tools/gate4_medicion.py`, que corre en `tools/validacion_preliminar.sbatch`.

## Cómo se mide ahora

Tres **fases separadas**, cada una con su propio cronómetro y su propia ventana de `PowerBuffer`:

1. **Referencia FP64** — una sola vez por invocación (no una por formato: la trayectoria no depende de `T`). Publica la fila `GPU_FP64` y guarda en RAM del host un snapshot del estado por cada checkpoint.
2. **Ruta `_none`** — su propia ventana; compara contra los snapshots guardados, sin recalcular nada dentro de la ventana medida.
3. **Ruta `_comp`** — ídem.

**Por qué fases y no tres cronómetros en un bucle único**: el *tiempo* sí se podía separar con eventos CUDA dentro del bucle. La *energía* no — el contador de NVML es de todo el dispositivo y su cuantización (~20-25 ms de GPU cargada) es mayor que el tramo de una trayectoria en una iteración, así que sumar cientos de tramos cuantizados no da nada utilizable. Atribuir energía a una ruta exige darle una ventana **contigua** propia.

**Costo**: los snapshots FP64 en RAM del host (`num_checkpoints × N² × 8 B`; el binario lo imprime al arrancar). A `N=8192` con `CHECKPOINT_EVERY=5` e `ITERS=80` son 16 × 537 MiB = 8.6 GiB, dentro de `--mem=128G`. Es el mismo patrón que Stencil ya usaba (`ckpt.fp64_checkpoints`).

**Lo que no arregla**: las fases corren una detrás de otra, así que la última ve una GPU más caliente que la primera — la misma limitación de aislamiento térmico que el plan ya documenta (Etapa 5).

**`gflops` usa siempre los FLOPs útiles** (un `X·A` por iteración), también en `_comp`, que hace un segundo producto para la corrección: esa ruta entrega el mismo resultado útil a mayor costo, así que su `gflops` más bajo es lo que hay que reportar, no un artefacto.

`anchor_every` es la última columna y en **este** binario vale siempre `0`: Fase 3 no tiene ancla. La columna existe igual para que el esquema sea idéntico al de `Fase_4/GEMM` — `run_full_pipeline.sh` concatena los `results/` de las dos fases en el mismo análisis, y dos esquemas distintos obligarían a `Fase_4/tools/common_analysis.py` a ramificar por fase.

**No coincide con el esquema de `Fase_3/Stencil`** (columnas `nx`/`ny` en vez de `n`, distinta semántica de rutas). El post-proceso es `../tools/extract_csv_chained.py`, ya integrado al final del `.sbatch`.

## Campaña por defecto

`run_gemm_chained.sbatch` corre, si no se le exporta nada, el barrido completo:

| Variable | Default | Nota |
|---|---|---|
| `N_LIST` | `1024 2048 4096 8192` | Potencias de 2. Techo por presupuesto de memoria — ver abajo. |
| `ITERS_LIST` | `20 40 80` | **Nueva**: reemplaza al escalar `ITERS`, que sigue funcionando y gana si se exporta (`ITERS=20 bash …` = una sola pasada, igual que antes). |
| `COMP_LIST` | `off on` | |
| `TC_FORMAT` | `both` | |
| `SMOKE_TEST` | `0` | `1` recorta a `N=1024`, 3 iteraciones y `RUN_NCU=0`. |

**Presupuesto de memoria (A100-PCIE-40GB, 39.49 GiB)** — no subir `N_LIST` sin rehacer esta cuenta, que está detallada en el propio `.sbatch`. Con `--tc both`, `--comp on` y ancla activa son **90 B por elemento** de la matriz `N×N`:

| `N` | Memoria | Veredicto |
|---|---|---|
| 4096 | 1.51 GB | |
| 8192 | 6.04 GB | 15 % de la tarjeta, margen 6.5× — **techo de la campaña** |
| 16384 | 24.2 GB | 61 %, sin margen de seguridad — **descartado** |
| 32768 | 96.6 GB | imposible |

El perfilado NCU se restringe a la primera pasada de `ITERS_LIST`: `wmma_gemm_kernel` es el mismo para cualquier `--iters`, así que repetirlo no produce un dato nuevo.

## Qué falta

- **`Fase_4/GEMM/`**: ✅ hecho — extensión con el ancla FP64 (`Fase_4/GEMM/gemm_chained.cu`).
- **`Fase_3/Convolution/`**: ✅ hecho.
- **`run_gemm_chained.sbatch`**: ✅ hecho — lanzador parametrizado, ahora con `ITERS_LIST` y `SMOKE_TEST`.
- **Post-proceso de CSV**: ✅ hecho — `../tools/extract_csv_chained.py`.
- **Verificación del orden de operandos**: ✅ hecho y **pasado en GPU real** — ver arriba.
- **Scripts de gate** K=0/K=1: ✅ hechos — `Fase_4/tools/gate3_ancla.py` y `Fase_4/GEMM/gate3_ancla.sbatch`.
- **Medición por ruta**: ✅ corregida — ver la advertencia de arriba. El gate que la vigila es `Fase_4/tools/gate4_medicion.py`.
- **Campaña real en PACCA**: compilado y verificado con `N` chico en GPU Ampere+; falta el barrido completo.
