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

---

## Errata y precisiones de implementación (2026-09-28, posteriores al pre-registro `1ae99cf`)

Se añaden como sección aparte, sin reescribir lo pre-registrado. Ninguna cambia un
umbral ni un parámetro físico; corrigen una medición mal planteada y fijan cómo se
implementó lo decidido.

1. **§7 (CPU_FP64) — corrección de la evidencia.** La medición de §7 contó `CPU_FP64` por
   `(nx, iters, anchor_every)` y dio razón 1.0: ese conteo ya *escondía* el defecto, porque
   `CPU_FP64` no depende de K. Contado por celda `(nx, ny, iters)`, que es la regla de §7,
   `energy_stencil_7145.csv` tiene **52 invocaciones en 13 celdas, máximo 4 por celda**: la
   referencia se relanza una vez por cada K de `ANCHOR_LIST` (0, 1, 8, 32). Es el mismo
   defecto que el hallazgo previo ("90× en vez de 30×", razón 3 con los K de la variabilidad:
   0, 1, 5). Reproducido con `extract_csv.py --max-cpu-fp64-per-cell 1` y con
   `tools/audit_coverage.py` (sección "CPU_FP64"). Corrección en `run_stencil_tc.sbatch`:
   con `CAMPANA_STRICT=1` (o `CPU_FP64_ONCE=1`) `--cpu-fp64` se pasa solo en la primera pasada
   de `KAHAN_LIST` y de `ANCHOR_LIST`. Es un requisito *por invocación del script*: la
   segunda invocación (`SPATIAL_COMP=on`, §6) debe lanzarse con `CPU_FP64=off`, porque la
   referencia ya salió en la primera.
2. **§5 (columnas) — dónde se emite cada una.** `error_evaluable` y `motivo_exclusion` los
   emiten los binarios GEMM/Conv de Fase 4 (tras `anchor_every`, vía `common/metrics.cuh`).
   En Stencil el binario ya imprime `NaN` para todo error no evaluable
   (`fmt_csv_error_num`), no emite `0.0`: sus columnas se derivan de los tokens crudos
   (`NONFINITE` en los 4 campos = referencia no finita; en los 3 de error = solución no
   finita). `device`, `gpu_valid`, `comp_scheme` y `n_cpu_fp64_invocaciones` se derivan en los
   extractores a partir de lo que el binario ya emite (sufijo `_SP` de la ruta, columna
   `kahan`, `energy_window_reliable`), sin tocar `stencil_tensor_activation.cu`. Un log
   anterior al fix con `solution_finite=0` y `rel_l2` numérico parcial se anula a `NaN` y se
   cuenta (7145: 270 filas en GEMM, 684 en Conv).
3. **§5 (`comp_scheme`) — GEMM y Convolución.** Solo tienen dos esquemas (`none`, `local`);
   `kahan_local` y `spatial` son de Stencil. El assert A1 exige `{none, kahan_local, spatial}`
   en Stencil y `{none, local}` en GEMM/Conv.
4. **A2 y el desbordamiento de la referencia FP64.** Con operadores amplificantes (Convolución:
   ×2 por iteración) la referencia FP64 desborda tras ~1000 iteraciones, muy antes de las
   ventanas energéticas de §2 (2500–37000); ahí no existe error evaluable a los iters
   energéticos *por la física del operador*, que esta campaña no cambia. `audit_coverage.py`
   trae `--a2-policy explained` (default: esas configuraciones se listan como *exentas* con su
   causa y no hacen fallar A2) y `--a2-policy strict` (fallan igual). Decisión de política
   pendiente del responsable; el reporte imprime ambas cuentas. Para el operador difusivo
   `alpha=3/16` (contractivo) el problema no existe.
5. **A2/B para Stencil `alpha=3/16` — alineación de rejillas.** El pase energético
   (`RUN_KIND=energy`, `CHECKPOINT_EVERY=0`) nunca emite filas de error en Stencil (estructural).
   El error a los iters energéticos (4000/1500/1500) sale de una pasada numérica *a la misma
   ventana* (`RUN_KIND=numeric`, `ITERS_LIST=<ventana>`, `CHECKPOINT_EVERY` grande), además de
   la numérica corta (10 50 100 120). Sin esa pasada A2 y A8 no pueden pasar.
6. **Gate por job vs. por campaña.** `audit_coverage.py --mode job` (al final de cada job) da
   `N/A` a A2 en jobs solo-energía y a A8 (necesitan la unión de jobs); `--mode campaign`
   (sobre la unión de todos los directorios) no admite ningún `N/A`. En job, A1 se evalúa contra
   `--expect-schemes` (una invocación no puede traer los tres esquemas de Stencil).
7. **Estado al escribir esto:** implementados y probados contra los logs de 7145
   (re-extracción idéntica en las columnas comunes; el gate falla en A1, A2, A4, A8 como
   se esperaba): `common/metrics.cuh`, `Fase_4/{GEMM,Convolution}/*_chained.cu` (sin compilar
   aún: se valida en PACCA con `SMOKE_TEST=1`), extractores, `tools/audit_coverage.py`,
   `Fase_4/Stencil/run_stencil_tc.sbatch`. **Pendiente:** `run_gemm_chained.sbatch` y
   `run_conv_chained.sbatch` (modo estricto/ARCHIVE_DIR/auditoría), lanzador de la campaña.

8. **Ampliación de K del ancla (2026-09-28, pre-registrada ANTES de lanzar estos jobs).** La
   campaña original barre solo tres K no nulos por kernel (Stencil 1, 8, 32; GEMM/Conv 1, 5, 20),
   pocos niveles para la regresión de K como variable ordinal (`log(K+1)`) y para la guía por
   tolerancia. Ningún binario limita K más allá de `K >= 0` (el ancla exige `--comp on` en
   GEMM/Conv y `--spatial-comp on` en Stencil). Costo medido en 7145 (t_iter relativo a K=0, BF16,
   pase dedicado): GEMM 1.03–1.33×, Conv 0.86–1.03×, Stencil 3.0–3.3× (K=1), 1.7–1.8× (K=8), 1.6×
   (K=32). Se añaden, sin retirar ni cambiar nada de lo anterior:
   - **Stencil `alpha=3/16`** (grupo `kext`, `SPATIAL_COMP=on`, `CPU_FP64=off` porque la referencia
     ya sale en el grupo `sp`): `K ∈ {2, 4, 16, 64, 128}`, mismas cinco pasadas (numérica corta,
     numérica a la ventana S y L, energía S y L). Con `sp`, la escala completa queda
     `{0, 1, 2, 4, 8, 16, 32, 64, 128}`. Un K mayor que los `iters` de la pasada no ancla nunca
     (p. ej. 64 y 128 solo actúan en las ventanas de 1500/4000 y en `iters` ≥ 100): es esperado,
     no un defecto.
   - **GEMM y Convolución** (campaña completa de Fase 4 con el binario corregido, en un solo
     barrido, sin duplicar K): `ANCHOR_LIST = {0, 1, 2, 5, 10, 20, 40}` (nuevos: 2, 10, 40).
     Numérica `20 40 80` + energía dedicada por cola (GEMM: N=1024,2048 → 24000; N=4096,8192 →
     500; Conv: HW=64,128 → 37000; HW=256,512 → 2500). El pase de energía de GEMM/Conv ya emite
     el error final (checkpoint incondicional de la última iteración), así que no hacen falta
     pasadas numéricas a la ventana como en Stencil.
   - **Sin cambios** en A1–A8: A8 sigue exigiendo la matriz `{0, 1, 8, 32}` (las K nuevas son
     aditivas). Los K nuevos de `kext` van en un checkout aparte (`~/campana_v2b`) para no alterar
     el código que ejecutan los jobs 7716–7725 ya encolados.
   - **Variabilidad (8 réplicas):** se hará con la escala ampliada (GEMM/Conv `{0,1,2,5,10,20}`,
     Stencil `{0,1,2,4,8}` en su invocación spatial); se pre-registra aquí y se lanza después.
   - **Gate:** `audit_coverage.py` amplía la exención física de A2 a GEMM/Conv y a Stencil
     `stress` cuando lo que desborda a los iters energéticos es la referencia FP64 **o** la
     solución de 16 bits (operadores amplificantes); en `alpha=3/16` (contractivo) no se exime.
     A3 se evalúa solo en el horizonte numérico (`iter ≤ 80`).
