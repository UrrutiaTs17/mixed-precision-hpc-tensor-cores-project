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
| 2026-09-28 | Hallazgo: Stencil `stress` no tiene ninguna celda con T,E fiables Y error finito simultáneos (ver abajo) | — | — | `horizon_stencil_7145.csv` (job 7145, fuente NO consumida por `build_canonical.py` per decisión previa, usada aquí solo como diagnóstico): ajuste exponencial r²=0.999999, λ≈1.96/iter, tamaño-independiente. Horizonte de divergencia predicho: FP16≈29 (medido), BF16≈136, FP32≈139, **FP64≈1035** — todos por debajo de las ventanas de energía (1500–4000 it). Con el operador `stress` NINGUNA ruta (ni la referencia FP64 exacta) tiene una ventana de energía numéricamente válida; F5/F6/F8/F9 (Stencil) quedan pendientes de rehacerse con datos `diffusive` α=3/16 en vez de `stress`. |
| 2026-09-28 | Relanzamiento vía holder ajeno (`hyp_holder_cfg`, job 7757, proyecto Hyperion del responsable, inactivo) en vez de `sbatch` — el nodo GPU único estaba ocupado por 7757/7758 | GRUPO `spk` (stencil+gemm+conv) y luego Stencil `off`, como pasos `srun --jobid=7757 --overlap` (nuevo `tools/lanzar_campana_fase4_dentro_holder.sh`, commit `1dfd107`) | REEMPLAZADO (ver fila siguiente) | Encolados primero como sbatch (7759–7774), cancelados al detectar el holder inactivo; smoke de `spk/num_corta` confirmó operador `diffusive` α=3/16 **contractivo** (`g(π,π)=-0.5`, `CSV_HORIZON` con estado `contractive_operator`, `CSV_ONSET` todos en -1) — sin horizonte de desbordamiento, resuelve el hallazgo de arriba |
| 2026-09-28 | Corrección de alcance por instrucción del responsable: GEMM/Conv NO se relanzan (7145 ya sustenta un Pareto valido para ellos: N=1024/4096 con error acotado hasta el final del pase de energia, N=8192 correctamente excluido por divergencia real, N=2048 con el defecto de cero espurio ya diagnosticado pero no usado porque el error se evalua en h=40, no en la ventana de energia). Se mata el paso `spk/num_corta` en curso (7:48, solo Stencil perdido) y se relanza **solo Stencil**, `spk` seguido de `off` en una sola invocacion | Stencil `spk`+`off` (10 pasos totales), `srun --jobid=7757 --overlap` | REEMPLAZADO (ver fila siguiente) | Bitácora de pasos: `~/campana_v2_out/fase4_v2/JOBS.tsv`; logs por paso en `logs_holder_run/v2_*.log`; log del driver `logs_holder_run/driver_v2_stencil_only_*.log` |
| 2026-09-28 | Recorte de K por instrucción del responsable (urgencia de tiempo): K completo (9 valores, 0,1,2,4,8,16,32,64,128) sin terminar `num_corta` tras 38 min en el intento anterior con esta config. Se mata el paso en curso (33s, nada perdido) y se relanza con `K_STENCIL_FULL="0 1 2 32 64 128"` (6 valores: extremos bajos+altos, sin 4/8/16) | Stencil `spk`+`off`, K∈{0,1,2,32,64,128} | en curso | Los K omitidos (4,8,16) se agregarían despues sin repetir el resto, si hace falta. Log: `logs_holder_run/driver_v2_stencil_k6_*.log` |
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
