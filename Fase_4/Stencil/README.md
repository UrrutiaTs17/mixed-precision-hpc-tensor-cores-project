# Fase 4 — Stencil: ancla FP64

Extiende `Fase_3/Stencil/stencil_tensor_activation.cu` (copiado como base — ver la nota de cabecera del archivo) con el mecanismo de **ancla FP64**: cada `K` iteraciones, el paso que normalmente calcularía el kernel WMMA se recalcula completo en FP64 y el resultado reemplaza al de la ruta rápida. Especificado y corregido (dos rondas de revisión crítica) en el documento **Plan de Precisión Mixta**, secciones 01 y 02 — esa es la referencia normativa; si algo aquí no coincide con lo que el plan describe, el plan tiene razón.

## Por qué existe

La compensación (Kahan local, compensación espacial — ver `Fase_3/Stencil/README.md`) tiene un techo: como máximo recupera precisión cercana a FP32, porque no puede compensar más información de la que el Tensor Core realmente calculó. El ancla es el mecanismo para acercarse más a FP64 sin pagar su costo completo — recalculando exactamente en FP64 solo *una de cada K* iteraciones, en vez de todas.

## Qué corrige y qué no

- **Sí corrige**: en el paso de ancla no se introduce ningún error de redondeo por almacenamiento nuevo.
- **No corrige**: el drift acumulado en las iteraciones anteriores al ancla sigue presente en el campo de entrada — el ancla no reconstruye la trayectoria FP64 verdadera desde el inicio (eso costaría lo mismo que correr todo en FP64). Si frenar la introducción de error nuevo en cada ancla basta para contener el crecimiento total, o si el error heredado domina igual, es una pregunta empírica — para eso está el barrido de `K`, no para asumir la respuesta.

## Uso

```
--anchor-every K
```

Entero ≥ 0, por defecto `0` (deshabilitado — comportamiento idéntico a Fase 3). `K > 0` activa el ancla.

**Requisitos** (validados en `parse_args`, el binario aborta con un mensaje explicativo si no se cumplen):
- `--spatial-comp on` — el residuo de compensación se re-siembra en `double`, y esa reconstrucción solo está implementada sobre la convención de `CompMode::Spatial` (`Q(v) + comp = v`).
- `--execution-mode normal` (el default) — incompatible con `--execution-mode graph`: los grafos capturan de antemano una secuencia fija de lanzamientos WMMA, y el ancla necesita decidir en cada iteración si toca FP64 o WMMA. Soportarlo exigiría un tercer grafo dedicado, fuera de alcance de esta primera implementación.

Ejemplo:

```bash
./stencil_tc --nx 4096 --ny 4096 --iters 40 --tc fp16 --spatial-comp on --anchor-every 8
```

## Validación — correr esto ANTES de confiar en cualquier resultado

Estos tres gates están descritos en detalle en la sección 01/02 del documento de plan. Los dos primeros **ya están automatizados**:

```bash
sbatch gate3_ancla.sbatch          # o: bash gate3_ancla.sbatch, sin SLURM
```

`gate3_ancla.sbatch` compila los **dos** binarios (Fase 3 y este), corre las tres pasadas y le pasa los logs a `../tools/gate3_ancla.py --kernel stencil`.

1. **`--anchor-every 1`** (ancla en cada iteración). Aquí el gate tiene **dos criterios**, cada uno sobre la columna que le corresponde, y la distinción no es un tecnicismo:
   - **(a)** `rel_l2`/`rel_linf` deben alcanzar el nivel de la ruta `GPU_FP64` de la *misma* corrida. Con K=1 el paso se sustituye entero por `stencil2d_fp64_kernel`, así que la trayectoria anclada **es** la de `GPU_FP64`; el gate admite un factor de holgura (`--factor-gpu-fp64`, default 10) en vez de exigir identidad bit a bit, para no fallar por el *narrowing* a `float` del readout.
   - **(b)** `rel_l2_prop`/`rel_linf_prop` deben quedar bajo la cota de cuantización del formato (`2^-11` en FP16, `2^-8` en BF16).

   **Por qué dos y no uno**: en Stencil, `CSV_DRIFT.rel_l2` compara contra `d_out_fp32` — el acumulador FP32 *sin* el redondeo de almacenamiento, "ancla de no-regresión". **No es el mismo objeto** que mide `CSV_DRIFT` en GEMM/Convolución (allí es el buffer `T` cuantizado). El análogo real de aquella columna es `rel_l2_prop`, que sí mide el estado propagado en 16 bits. Aplicar el criterio de un kernel al otro es el error fácil de cometer, y es la razón por la que `gate3_ancla.py` no tiene un solo umbral global.
2. **`--anchor-every 0`**: la ruta debe ser bit-idéntica a `Fase_3/Stencil/stencil_tc` con los mismos parámetros. El gate compara `CSV_DRIFT`, `CSV_SUMMARY`, `CSV_STORE`, `CSV_HORIZON` y `CSV_ONSET` campo a campo, separando las columnas de tiempo/energía (que se reportan como desviación relativa pero **no deciden**: dos corridas del mismo binario ya difieren ahí por ruido).
3. **Barrido de `K` intermedios**: una vez 1 y 2 pasan, recién ahí tienen sentido los resultados de `K` intermedios — son los que responden la pregunta real del objetivo 4. `ANCHOR_LIST` por defecto ya los incluye (`0 1 8 32`).

**Estado de la verificación**: `gate3_ancla.py` se ejercitó de punta a punta en GPU Ampere real (`sm_86`) para GEMM y Convolución, positiva y negativamente. Stencil **no** se pudo ejercitar localmente: su `.cu` tiene un `static_assert(sizeof(long) >= 8)` que rechaza Windows a propósito (`ReferenceSpill` necesita `fseek`/`ftell` de 64 bits). La primera corrida real de este gate será en PACCA.

## Costo de memoria — léelo antes de dimensionar una campaña

Habilitar el ancla ensancha el residuo de compensación de `float` (4 bytes/celda) a `double` (8 bytes/celda). El par `(T, comp)` con ancla activo pasa a ocupar **10 bytes/celda en FP16** — más que los 8 bytes de FP64 puro. No es un descuido: en cuanto se usa el ancla, la ventaja deja de ser de ancho de banda de memoria y pasa a ser puramente de throughput de cómputo (el término principal se sigue calculando con Tensor Cores). Vale la pena medir si esa ventaja de cómputo sigue compensando el costo de memoria adicional — es exactamente el tipo de resultado que debe salir del barrido de `K`, no asumirse de antemano.

## Diseño experimental (barrido de K)

Ver sección 01/02 del plan para el detalle completo. En resumen: escala logarítmica relativa al horizonte de overflow sin ancla de cada formato, nunca por encima de él (un `K` mayor que el horizonte nunca llega a dispararse antes del overflow). Ejemplo orientativo: `K ∈ {2, 4, 8, 16, ∞}` para FP16 (horizonte ≈29); `K ∈ {2, 4, 8, 16, 32, 64, ∞}` para BF16 (horizonte ≈142 con compensación espacial). **Calibra `K` con el costo real de un paso FP64 vs. un paso Tensor Core medido en este kernel** — no asumas que la escala de otro kernel (GEMM, Convolución) transfiere directamente; la brecha FP64/TC es muy distinta según si el kernel es memory-bound (Stencil) o compute-bound.

## Cómo lanzar la campaña

`run_stencil_tc.sbatch` en esta carpeta ya expone `ANCHOR_LIST` (valores de `K` separados por espacio, default `"0"` = deshabilitado) además de todos los parámetros heredados de Fase 3 (`NX`/`NY`/`ITERS_LIST`/`TC_FORMAT`/`SPATIAL_COMP`/etc.). Valida `ANCHOR_LIST` contra `SPATIAL_COMP` al arrancar (falla rápido con un mensaje claro si se pide `K>0` sin `--spatial-comp on`, en vez de dejar que el binario aborte a mitad de una campaña larga). El perfilado NCU no se repite por cada valor de `K` — el ancla no modifica el kernel WMMA que NCU perfila (ver "Sobre qué base de código se construyó" abajo), así que perfilarlo más de una vez no agrega información.

```bash
sbatch run_stencil_tc.sbatch                                      # ANCHOR_LIST=0, igual que Fase 3
sbatch --export=ALL,ANCHOR_LIST="0 1" run_stencil_tc.sbatch       # gates K=0 / K=1 (correr primero)
sbatch --export=ALL,ANCHOR_LIST="2 4 8 16" run_stencil_tc.sbatch  # barrido real
```

## Qué falta (no construido todavía en esta sesión)

- **`gate3_ancla.sbatch`**: ✅ hecho — está en esta carpeta, y la comparación la hace `../tools/gate3_ancla.py --kernel stencil`. Un solo script cubre los tres kernels: la lógica de las dos puertas es la misma y solo cambia el esquema de columnas, aislado en una tabla de datos (ver `Fase_4/tools/README.md` para el razonamiento de esa decisión).
- **Columna `anchor_every` en el CSV**: ✅ hecha — `CSV_DRIFT`, `CSV_SUMMARY` y `CSV_ENERGY` la traen como **última** columna (al final, para no correr ningún índice posicional que las herramientas ya usaban). En Stencil es contexto de **invocación completa**: sale de `g_anchor_every_csv`, que `main()` fija una sola vez, así que todas las filas de una corrida comparten el valor — incluidas las de `GPU_FP64`/`CPU_FP64`, que nunca ejecutan el ancla. Es correcto: aquí el ancla es un parámetro de la corrida. **No** es la semántica de GEMM/Convolución, donde varía por ruta dentro de la misma corrida; esa diferencia está codificada en la estructura del código a propósito, ver `Fase_4/tools/README.md`.
- **Campaña por defecto**: `run_stencil_tc.sbatch` ahora barre `NX_LIST`/`NY_LIST` (`4096 8192 16384`, los mismos tres tamaños de la campaña de drift ya publicada — sección 3.3.2 del documento de resultados) además de `ITERS_LIST` (`10 50 100 120`) y `ANCHOR_LIST` (`0 1 8 32`). `NX`/`NY` siguen funcionando y ganan si se exportan. Las dos listas de tamaño se recorren **emparejadas**, no como producto cartesiano: todo el análisis aguas abajo asume mallas cuadradas (`common_analysis.py` colapsa `(nx, ny)` a una sola columna `size` usando solo `nx`), y un producto 3×3 generaría 6 corridas rectangulares mal etiquetadas. Lista de distinta longitud = error explícito, no silencio.
- **Presupuesto de memoria**: 52 B/celda en la ruta WMMA con compensación espacial y ancla → 14.0 GB a `16384²` (35 % de la A100-40GB). `20480²` (21.8 GB) queda justo en el límite del criterio de 2× y se **descarta**; `32768²` (55.8 GB) excede la memoria física. Ojo con la memoria de **host**, que a `16384` es el recurso más ajustado: con `CHECKPOINT_EVERY=5` e `ITERS=120` son 24 checkpoints FP64 en RAM = 51.5 GB.
- **Perfilado NCU**: ahora también restringido a un solo tamaño de malla (`NCU_NX`/`NCU_NY`, por defecto el primero de las listas), además de a un solo `ITERS`, una política de `KAHAN_LIST` y un `K` de `ANCHOR_LIST`. Perfilar `stencil2d_wmma_kernel` a cada tamaño no dice nada nuevo sobre Tensor Cores — solo sobre escala de memoria, que ya se mide con `t_iter_ms` y las columnas de energía — y triplicaría la parte más cara del job.
- **Frente de Pareto 3D y estadística** (etapas 7-9 del plan): dependen de que la campaña de datos con ancla ya exista.

## Sobre qué base de código se construyó

Este archivo parte de `Fase_3/Stencil/stencil_tensor_activation.cu`, que a su vez migra la versión más avanzada del kernel (rama histórica `fase4-estadistica-variabilidad`, con la reformulación `Y = X·H + V·X`, tiling ensanchado, `ldmatrix`/`mma.sync` explícito y CUDA Graphs). Busca el marcador `ANCLA FP64:` en el `.cu` para ubicar cada pieza del mecanismo — son las únicas secciones que difieren de `Fase_3/Stencil/stencil_tensor_activation.cu`.
