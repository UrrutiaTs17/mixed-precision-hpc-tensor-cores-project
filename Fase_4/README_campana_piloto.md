# Campana piloto de variabilidad — Fase 4 — Stencil 2D (16384^2)

Piloto de **5 replicas** para estimar la dispersion (std, CV%, IC) de tiempo,
GFLOP/s, energia GPU (NVML) y energia CPU (RAPL). No es una campana numerica.
**No es la campana definitiva**: el numero final de replicas se decide despues
de revisar el CV de tiempo, energia GPU y energia CPU.

Calibrada sobre el commit `8d4e0e8`, los mismos jobs **5200** (`SPATIAL_COMP=on`)
y **5201** (`SPATIAL_COMP=off`, `KAHAN_LIST="off on"`) del 2026-08-16, que ya
corrieron esta configuracion exacta en `paccaA100`.

## Diseno

| | |
| --- | --- |
| Malla | NX=NY=16384 |
| Horizontes | ITERS = 320, 640 |
| Checkpoints | `CHECKPOINT_EVERY=0` |
| Perfilado | `RUN_NCU=0` |
| Precisiones | FP16 y BF16 (acum. FP32, WMMA) — **una sola invocacion** via `--tc both` |
| Tratamientos | `none`, `kahan_local`, `spatial` (mutuamente excluyentes) |
| Replicas | 5 |
| Exclusividad | `--exclusive` (ya en `run_stencil_tc.sbatch`) |

**10 jobs SLURM · 30 invocaciones del binario · 60 observaciones WMMA.**

| Bloque | Env | Tratamientos | Invocaciones | Obs WMMA | Walltime |
| --- | --- | --- | --- | --- | --- |
| A | `SPATIAL_COMP=off KAHAN_LIST="off on"` | `none` + `kahan_local` | 4 | 8 | `02:30:00` |
| B | `SPATIAL_COMP=on` (fuerza `KAHAN_LIST=off`) | `spatial` | 2 | 4 | `01:30:00` |

`kahan=on + spatial=on` es imposible por tres barreras independientes: el
lanzador valida cada combinacion, el `.sbatch` fuerza `KAHAN_LIST=off` cuando
`SPATIAL_COMP=on`, y el binario aborta esa pareja en `parse_args`.

### Walltime

Derivado del costo observado en 5200/5201 a esta misma escala:
invocacion de 320 iters ≈ 12 min, de 640 iters ≈ 23 min (la parte cara son las
referencias CPU FP32/FP64, no la GPU). Bloque A ≈ 73 min → `02:30:00` (+105 %);
bloque B ≈ 36 min → `01:30:00` (+150 %). Ambos por encima del 30 % exigido.
Un timeout deja el job **sin CSV** (la extraccion corre al final del script),
por eso el margen es amplio y la validacion comprueba `State=COMPLETED`.

## 1. Copiar a PACCA y lanzar

```bash
# Desde la maquina local. Confirme antes la ruta real del repo en PACCA.
scp -r Fase_4 latorresn@hpc.unicartagena.edu.co:~/mixed-precision-hpc-tensor-cores-project/

ssh latorresn@hpc.unicartagena.edu.co
cd ~/mixed-precision-hpc-tensor-cores-project
git rev-parse HEAD          # debe ser 8d4e0e82efaf264ec2ad74b3f8bb172d614f4847
git status --porcelain      # debe estar limpio en archivos versionados

./Fase_4/lanzar_piloto_variabilidad.sh
```

El lanzador aborta si el arbol esta sucio o el commit no coincide, crea
`Fase_4/f4_piloto_variabilidad_<FECHA>/` (directorio nuevo, nunca sobrescribe),
escribe el manifiesto **antes** del primer `sbatch`, envia los 10 jobs e
imprime el mapa JobID → replica/tratamiento y el `squeue` inicial.

## 2. Seguimiento

Sustituya `<CAMPANA>` por el ID que imprime el lanzador.

```bash
# Cola, filtrada por usuario
squeue -u "$(whoami)" -o "%.10i %.22j %.9P %.8T %.10M %.10l %.6D %R"

# Solo esta campana
squeue -u "$(whoami)" -n "$(cut -d, -f2 Fase_4/<CAMPANA>/manifiesto_jobs.csv | tail -n +2 | paste -sd,)"

# Estado final: JobID, JobName, State, ExitCode, Elapsed
sacct -j "$(cut -d, -f1 Fase_4/<CAMPANA>/manifiesto_jobs.csv | tail -n +2 | paste -sd,)" \
      -X --format=JobID,JobName%22,State,ExitCode,Elapsed,Start,End

# Log en vivo de un job concreto
tail -f Fase_4/<CAMPANA>/logs/mixed_precision_stencil_tc_f3_<JOBID>.out
tail -f Fase_4/<CAMPANA>/logs/mixed_precision_stencil_tc_f3_<JOBID>.err

# Log en vivo del ultimo job que haya arrancado
tail -f "$(ls -t Fase_4/<CAMPANA>/logs/*.out | head -1)"

# Progreso: corridas completadas de las 30 esperadas
grep -h '^Corrida:' Fase_4/<CAMPANA>/results/run_*.log | wc -l

# CSV ya generados (uno por job terminado)
ls -l Fase_4/<CAMPANA>/results/energy_stencil_*.csv
```

## 3. Analisis (al terminar)

```bash
python3 Fase_4/tools/analizar_variabilidad.py \
    --campana Fase_4/<CAMPANA> \
    --replicas-esperadas 5 --anidado "" \
    --csv-salida Fase_4/<CAMPANA>/results/estadisticos_variabilidad.csv
```

Agrupa por **(precision, compensacion, iteraciones)** y da, por metrica,
`n`, media, `std` muestral (ddof=1), mediana, min, max y `CV% = 100*std/media`,
listando siempre las observaciones individuales con su replica y JobID.
Metricas: tiempo total, t/iter, GFLOP/s, energia GPU, energia CPU, energia
total, energia GPU por iteracion, EDP, J/GFLOP y potencia media GPU y total.
Cierra con un panorama de CV% — el criterio para dimensionar la campana
definitiva.

## 4. Validacion (al terminar)

```bash
python3 Fase_4/tools/validar_campana.py --campana Fase_4/<CAMPANA> --replicas 5
```

Comprueba: 5 replicas completas en las 12 celdas · FP16 y BF16 presentes · los
tres tratamientos presentes · ambos horizontes presentes · ningun CSV vacio o
ausente · sin duplicados · tiempos > 0 · energia GPU y CPU no negativas ·
`energy_window_reliable=1` en toda observacion · `State=COMPLETED` y
`ExitCode=0` en los 10 jobs (avisa aparte de los `TIMEOUT`) · **ausencia total
de corridas `kahan=on + spatial=on`**. Sale con codigo != 0 si algo falla.

## Interpretacion

**`rel_l2`, `rel_linf` y `max_abs` saldran NaN en las 60 observaciones.** A 320
y 640 iteraciones la solucion supera `first_nonfinite` (la referencia FP64
diverge hacia ~1045). Es el comportamiento esperado, **no** es un fallo del job
ni motivo para descartar la energia medida: la campana energetica y la numerica
son independientes. Por eso la validacion no comprueba metricas de error.

Cada invocacion emite ademas filas `CPU_FP32`, `CPU_FP64`, `GPU_FP32` y
`GPU_FP64`. Se conservan en los CSV pero no cuentan dentro de las 60
observaciones WMMA; el analisis y la validacion las ignoran por construccion.

La frecuencia de GPU no se fija (`nvidia-smi -lgc` esta denegado y el modo
persistente no persiste en este nodo): el ruido termico se trata por repeticion
estadistica, que es justamente el proposito de esta campana.
