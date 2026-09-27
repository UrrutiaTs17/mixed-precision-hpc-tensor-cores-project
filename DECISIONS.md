# DECISIONS.md — pre-registro de la campaña corregida (post-auditoría de `campana_holder_20260917`, job 7145)

Este documento se commitea **antes** de cualquier `sbatch` de la campaña
nueva (ver `MANIFIESTO.md` para el hash del commit que lo introduce). Fija
las reglas de emisión/medición para que no se decidan post-hoc mirando los
resultados. La campaña nueva escribe en un directorio propio; no sobrescribe
`results/` de la campaña 7145.

## 1. Tolerancias y parámetros físicos

- **Tolerancia GPU_FP64 vs. CPU_FP64**: `1e-12` (el límite teórico de
  redondeo en doble precisión es `2.8e-15`; se deja margen de ~3 órdenes de
  magnitud para acumulación de operaciones, no porque se espere alcanzarlo).
- **Operador difusivo estable**: `alpha = 3/16`. Cualquier corrida de esta
  campaña que se declare bajo este operador debe verificarlo contra su
  propio log de arranque (ver §5, assert A8 de `audit_coverage.py`) — abort
  si no coincide. No confundir con el operador `stress` de la campaña 7145
  (`OP_MODE=stress`, usado para horizonte de desbordamiento, no para el
  trilema tiempo-energía-error).

## 2. Rejilla de iteraciones por kernel y tamaño

Reutiliza la calibración ya hecha y validada en la campaña 7145 (documentada
en `Fase_4/tools/README.md` y verificada empíricamente: MIN de `t_iter_ms`
observado, con margen ~1.3–1.4×, nunca la mediana):

| Kernel | Pase numérico (iters) | Pase energético dedicado (iters, por tamaño) |
|---|---|---|
| GEMM | 20, 40, 80 (todos los N) | N=1024,2048 → 24000; N=4096,8192 → 500 |
| Convolución 2D | 20, 40, 80 (todos los HW) | HW=64,128 → 37000; HW=256,512 → 2500 |
| Stencil 2D (`stress`) | 10, 50, 100, 120 (todas las mallas) | nx=4096 → 4000; nx=8192,16384 → 1500 |
| Stencil 2D (`alpha=3/16`, matriz A8) | por definir en el job exploratorio del Paso 5.1 | 4096²→4000; 8192²→1500; 16384²→1500 (mismas ventanas que `stress`, K∈{0,1,8,32}) |
| Variabilidad (8 réplicas, punto fijo) | GEMM/Conv/Stencil: 20, 40 | *(sin pase energético dedicado — r1..r8 no tienen ventanas fiables, ver README de la campaña 7145)* |

Prohibido fijar `ITERS_LIST` a ciegas: todo tamaño nuevo que no esté en esta
tabla debe calibrarse primero con un job exploratorio (mismo método: MIN
`t_iter_ms` real + margen) antes de entrar a la campaña de producción.

## 3. Criterio `window_reliable` (ya implementado, se documenta, no se cambia)

```
window_reliable = gpu_valid
                   && gpu_segment_count > 0
                   && time_total_s >= kEnergyWindowReliableSeconds * gpu_segment_count
```
con `kEnergyWindowReliableSeconds = 0.500` (`common/power_sampling.h`).
Verificado contra el código antes de este documento — no requiere cambios,
solo queda fijado aquí como criterio pre-registrado.

## 4. Regla de error no evaluable (pendiente de implementar, Paso 2 del plan)

En `common/metrics.cuh::compare_sequences`, cuando la referencia FP64 no es
finita (`reference_finite == false`):
- `rel_l2`, `rel_linf`, `l2_abs` deben quedar en **`NaN`**, nunca `0.0`.
- Se emite `error_evaluable = 0` (nueva columna) con `motivo_exclusion` de
  texto corto: `"reference_non_finite"` si la referencia no es finita,
  `"solution_non_finite"` si la solución no lo es, `""` en caso contrario.
- Esto reemplaza el parche equivalente que hoy vive solo en la capa de
  análisis (`Fase_4/analysis/build_canonical.py`, `Fase_4/tools/common_analysis.py`):
  se corrige en la fuente para que ningún consumidor futuro tenga que
  reinventarlo.

## 5. Columnas nuevas del esquema (diff contra el esquema de 7145)

| Columna | Kernel(s) | Reemplaza / complementa |
|---|---|---|
| `error_evaluable` | los 3 | nueva (ver §4) |
| `motivo_exclusion` | los 3 | nueva (ver §4) |
| `device` (`cpu`\|`gpu`) | los 3 | hoy se infiere del prefijo de `route` en el extractor; pasa a ser explícita desde el `.cu` |
| `gpu_valid` | los 3 | idem — hoy implícito, pasa a columna real |
| `comp_scheme` | los 3 | Stencil: `none`\|`kahan_local`\|`spatial` (hoy `KAHAN_LIST` queda pisado en silencio por `SPATIAL_COMP=on`, ver §6). GEMM/Conv: `none`\|`local` (mapeado desde `--comp off/on`) |
| `n_cpu_fp64_invocaciones` | Stencil | nueva (ver §7) |
| ~~`speedup_cpu`~~ | los 3 | **eliminada** del extractor y de todo consumidor (ya no se usa en ningún análisis, ver memoria del proyecto) |

Cualquier `.cu` cuya cabecera de `CSV_SUMMARY`/`CSV_DRIFT`/`CSV_ENERGY` no
coincida con este esquema hace abortar el extractor (Paso 2.4 del plan).

## 6. `comp_scheme` en Stencil: dos invocaciones, no una

El binario de Stencil **rechaza** `--kahan on --spatial-comp on` (son
tratamientos alternativos, no combinables — confirmado en el código, no es
un bug del binario). La campaña nueva cubre las tres celdas de `comp_scheme`
lanzando **dos invocaciones separadas** de `run_stencil_tc.sbatch`:

1. `SPATIAL_COMP=off KAHAN_LIST="off on"` → cubre `comp_scheme=none` y
   `comp_scheme=kahan_local`.
2. `SPATIAL_COMP=on` (fuerza `KAHAN_LIST=off` internamente, como ya hace hoy)
   → cubre `comp_scheme=spatial`.

El script deja de forzar `KAHAN_LIST=off` en silencio cuando se le pide
`kahan_local` explícitamente por `.env`; en su lugar, aborta si el valor
efectivo con el que corrió el binario difiere del solicitado (fila
`comp_scheme` real vs. variable de entorno pedida).

## 7. `CPU_FP64`: una invocación por celda, no por ruta GPU

Regla revisada (el número original de "30" no tiene fuente verificable en
este repositorio ni en los CSV de 7145 — ver nota al final): **CPU_FP64 debe
ejecutarse exactamente una vez por celda única `(size, iters)` del bloque
energético**, no una vez por cada ruta GPU evaluada en esa celda. Se emite
`n_cpu_fp64_invocaciones` (contador real, no calculado post-hoc) y
`audit_coverage.py` aborta si:

```
n_cpu_fp64_invocaciones / n_celdas_unicas(size, iters) > 1
```

**Evidencia medida contra 7145** (Paso 0 del plan, `energy_stencil_7145.csv`):
en esa campaña `CPU_FP64` aparece exactamente **1 vez por cada combinación
`(nx, iters, anchor_every)`** — 52 filas sobre 52 combinaciones — es decir,
la razón ahí es **1.0, no >1**. La campaña 7145 **no reproduce** un
sobre-relanzamiento de `CPU_FP64`; si acaso, en el pase estrictamente
dedicado de energía (`iters` grande) falta en 2 de 3 mallas (`nx=8192` y
`nx=16384` a `iters=1500`), un problema de cobertura, no de duplicación. No
encontré en el repositorio ni en la memoria del proyecto una campaña donde
`CPU_FP64` corra "90 veces en vez de 30" — si ese hallazgo viene de otra
corrida o de una conversación con el director que no está en este repo,
hace falta identificar esa fuente para no perder la evidencia original del
defecto. El assert relativo (§7, fórmula de arriba) se implementa igual;
esta nota solo documenta que la evidencia numérica específica de "90 vs 30"
no se pudo verificar contra 7145.

## 8. Otras reglas de emisión (Paso 2 del plan, sin cambios de diseño)

- `CHECKPOINT_EVERY` desactivado (`0`) o `>= iters` en toda pasada
  energética — **ya implementado** en `run_gemm_chained.sbatch`; verificar
  Convolución y Stencil antes de dar el punto por cerrado.
- `ARCHIVE_DIR` apunta al scratch local del nodo (`$SLURM_TMPDIR` o
  equivalente); copia única al directorio final de la campaña al terminar.
  Nunca escritura directa a NFS del repo durante la ejecución.
- Rejilla de iteraciones numérica y energética alineada por configuración,
  de modo que exista intersección no vacía — ya es así de facto (unión por
  `size, route, K`, no por `iters`; ver `merge_energy_into_stencil` en
  `Fase_4/tools/common_analysis.py`), se deja documentado explícitamente en
  el script en vez de quedar implícito en el pipeline de análisis.

## 9. Orden de ejecución (Paso 5 del plan, prioridad Fase 4)

1. Job exploratorio de horizonte para la matriz Stencil `alpha=3/16`
   (`first_nonfinite` por ruta/tamaño bajo el operador difusivo).
2. Campaña energética + multiobjetivo de Fase 4: Stencil `alpha=3/16`
   completo, luego GEMM/Conv si el tiempo de GPU lo permite en la misma
   tanda.
3. Campaña de variabilidad (8 réplicas), con las dos invocaciones
   Kahan/spatial separadas (§6) para que `kahan_local` tenga datos reales
   por primera vez.
4. `audit_coverage.py` al final de cada job, exit code ≠0 si falla.
5. (Menor prioridad, si el tiempo lo permite) relanzamiento de exactitud
   Fase 1–3 — la auditoría del plan del proyecto ya encontró que Fase 1–3
   cumple en general.
