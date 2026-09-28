# MANIFIESTO.md — campaña corregida (post-auditoría de `campana_holder_20260917`, job 7145)

Registro de procedencia de la campaña que corrige los 15 defectos de datos
detectados. No sobrescribe nada de la campaña 7145 (`results/` de esa
campaña queda intacto); esta campaña escribe en `RESULTS_DIR_V2` (ver
`campana.env`).

## Pre-registro

| Campo | Valor |
|---|---|
| Commit de pre-registro | `1ae99cf19559e9c97a9bb97a6d0274f6c732d7c0` |
| Timestamp del commit | `2026-09-27T14:00:21-05:00` |
| Rama | `main` |
| Mensaje | "Pre-registro de la campana corregida: tolerancias, rejilla de iters, esquema nuevo" |
| Archivo pre-registrado | `DECISIONS.md` (mismo commit) |

Cualquier `sbatch` de producción de esta campaña debe tener un timestamp de
lanzamiento **posterior** al de arriba. Se verifica comparando la fecha del
job en `sacct` contra el timestamp del commit.

## Commits relacionados (contexto, no forman parte del pre-registro en sí)

| Commit | Contenido |
|---|---|
| `801312a` | Fixes de merge (drift GEMM/Conv, energía Stencil) en `Fase_4/tools/common_analysis.py` — corrige en la capa de análisis lo que el Paso 2 de `DECISIONS.md` corrige en la fuente (`metrics.cuh`) |
| `8d1c0ce` | Pipeline de figuras F3-F6 sobre la campaña 7145 (`Fase_4/analysis/`) |
| `cf34888` | Scripts de lanzamiento/reetiquetado dentro de un holder compartido (`tools/*_dentro_holder.sh`) |

## Diff de esquema de columnas: 7145 vs. esta campaña

Se completa en el Paso 2 del plan (cuando los `.cu`/extractores estén
modificados) y se referencia aquí con el commit que lo introduce. Hasta
entonces, el esquema objetivo está descrito en `DECISIONS.md` §5.

| | 7145 | Campaña corregida |
|---|---|---|
| Commit de los `.cu`/extractores | *(el de la campaña 7145, sin fecha registrada aquí)* | *(pendiente — se llena en el Paso 2)* |
| `error_evaluable`, `motivo_exclusion` | no | sí |
| `device`, `gpu_valid` explícitos | no (inferidos en el extractor) | sí (columna real) |
| `comp_scheme` | no (solo `kahan`/`SPATIAL_COMP` crudos) | sí (`none`\|`kahan_local`\|`spatial` en Stencil; `none`\|`local` en GEMM/Conv) |
| `n_cpu_fp64_invocaciones` | no | sí (Stencil) |
| `speedup_cpu` | sí | eliminada |

## Bitácora de jobs (se llena conforme se lanza cada paso)

| Fecha | Paso del plan | Job(s) PACCA | Estado | Nota |
|---|---|---|---|---|
| 2026-09-28 | Paso 4 — smoke (rama `campana-v2`, commit `931507a`, checkout limpio `~/campana_v2`) | 7710 (Stencil `sp`), 7711 (Stencil `off`, FALLO por diseño: ancla K>0 exige spatial), 7714 (Stencil `off`, rehecho con `ANCHOR_LIST=0`), 7712 (GEMM), 7713 (Conv), 7715 (GEMM N=2048 × 2200 it) | COMPLETED (7711 FAILED esperado) | Compila con `metrics.cuh` nuevo; esquema estricto 38/16/8 campos OK; `CPU_FP64` 1 vez por celda; `comp_scheme` efectivo = pedido; 7715: la referencia FP64 desborda a iter 2200 → `rel_l2=NaN`, `error_evaluable=0`, `motivo=reference_non_finite` (el binario viejo daba 0.0) |
| 2026-09-28 | Paso 5.1/5.2 — Stencil α=3/16 (`CI_MODE=monomode`, p=168), commit `3bb0c0d`, `tools/lanzar_campana_fase4.sh`, salida `~/campana_v2_out/fase4_a316/` | grupo `sp`: 7716 num_corta, 7717 num_S, 7718 num_L, 7719 en_S, 7720 en_L; grupo `off`: 7721 num_corta, 7722 num_S, 7723 num_L, 7724 en_S, 7725 en_L (cadena `afterany`) | en cola | `CI_MODE=monomode` elegido por coherencia con la tesis ("condición inicial controlada", Fase 4) y con la memoria del proyecto (monomodo agregado para que el eje de error sea construible); ver `JOBS.tsv` |
| 2026-09-28 | Paso 5.2 (ext.) — ampliación de K (DECISIONS.md, errata 8), commit `c9d111f`, checkout `~/campana_v2b`, salida `~/campana_v2_out/fase4_kext/` | Stencil `kext` (K=2,4,16,64,128): 7726 num_corta, 7727 num_S, 7728 num_L, 7729 en_S, 7730 en_L; GEMM (K=0,1,2,5,10,20,40): 7731 num, 7732 en_A, 7733 en_B; Conv: 7734 num, 7735 en_A, 7736 en_B (cadena `afterany` tras 7725) | en cola | Ampliación pre-registrada antes de lanzar; ver `JOBS.tsv` |
| 2026-09-28 | CANCELACIÓN de 7716–7736 (DECISIONS.md, errata 9) | 7716–7736 | CANCELLED (7716 tras 26 min; el resto sin empezar) | Salidas parciales movidas a `~/campana_v2_out/cancelado_20260928/` |
| 2026-09-28 | Relanzamiento ancla primero, commit `e8b2b66`, checkout `~/campana_v2b`, salida `~/campana_v2_out/fase4_v2/` | Stencil `spk` (K=0,1,2,4,8,16,32,64,128; `CPU_FP64=on`): 7737 num_corta, 7738 num_S, 7739 num_L, 7740 en_S, 7741 en_L; GEMM (K=0,1,2,5,10,20,40): 7742 num, 7743 en_A, 7744 en_B; Conv: 7745 num, 7746 en_A, 7747 en_B; Stencil `off` (none, kahan_local; K=0): 7748 num_corta, 7749 num_S, 7750 num_L, 7751 en_S, 7752 en_L | 7737 corriendo, resto en cola (cadena `afterany`) | `CI_MODE=monomode`; ver `JOBS.tsv` |
| 2026-09-28 | PAUSA por instrucción del responsable | 7737–7752 | CANCELLED (7737 tras 38 min; el resto sin empezar) | Estructura intacta: código en `campana-v2` (`e8b2b66`), checkout `~/campana_v2b`, salidas parciales en `~/campana_v2_out/cancelado_20260928/fase4_v2_pausa/` (con su `JOBS.tsv`). **Para relanzar exactamente lo mismo**, desde `~/campana_v2b` (`git pull --ff-only origin campana-v2`): `O=$HOME/campana_v2_out/fase4_v2; KERNELS_RUN="stencil gemm conv" GROUPS_RUN=spk CI_MODE=monomode OUT_BASE=$O bash tools/lanzar_campana_fase4.sh; LAST=$(tail -1 $O/JOBS.tsv \| cut -f4); KERNELS_RUN=stencil GROUPS_RUN=off CI_MODE=monomode OUT_BASE=$O AFTER_JOB=$LAST bash tools/lanzar_campana_fase4.sh` |
| *(pendiente)* | Paso 5.3 — variabilidad (8 réplicas) con la escala de K ampliada | — | — | — |

## Criterio de aceptación (recordatorio, ver también `DECISIONS.md`)

- `tools/audit_coverage.py` sale con exit 0 y los 8 asserts (A1-A8) en verde,
  con `audit_report.md` adjunto.
- Este archivo (`MANIFIESTO.md`) y `DECISIONS.md` commiteados con timestamp
  anterior al primer `sbatch` de producción — cumplido arriba.
- Diff de esquema de columnas documentado — parcial, se completa en el
  Paso 2.
- Lista explícita de celdas que sigan vacías, si las hay — se llena junto
  con `audit_report.md` al final de la campaña.
