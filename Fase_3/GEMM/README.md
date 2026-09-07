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
CSV_DRIFT,<formato>_<none|comp>,n,iter,rel_l2,rel_linf,solution_finite
CSV_SUMMARY,<formato>_<none|comp>,n,iters,t_iter_ms,t_total_ms,gflops,energy_gpu_j,window_reliable,gpu_segments
```

**No coincide con el esquema de `Fase_3/Stencil/tools/extract_csv.py`** (esa herramienta es específica de Stencil — columnas `nx`/`ny` en vez de `n`, distinta semántica de rutas). Post-procesar estos CSV es trabajo pendiente — ver "Qué falta" abajo.

## Qué falta

- **`Fase_4/GEMM/`**: ✅ hecho — extensión con el ancla FP64 (`Fase_4/GEMM/gemm_chained.cu`).
- **`Fase_3/Convolution/`**: ✅ hecho.
- **`run_gemm_chained.sbatch`**: ✅ hecho — lanzador parametrizado (`N_LIST`, `COMP_LIST`, `TC_FORMAT`, etc. vía `--export`), ver el propio `.sbatch` de esta carpeta.
- **Post-proceso de CSV**: ✅ hecho — `../tools/extract_csv_chained.py` (`Fase_3/tools/README.md`), ya integrado al final del `.sbatch`.
- **Scripts de gate** (comparar K=0/K=1 contra la referencia FP64 antes de confiar en una campaña con ancla): todavía no migrados/escritos — ver `Fase_3/tools/README.md`, sección "Qué falta".
- **Campaña real en PACCA**: compilado y verificado con `N` chico en GPU Ampere+; falta correr el barrido de tamaños que promete el plan (512–4096).
