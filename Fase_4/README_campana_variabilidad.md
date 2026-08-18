# Campana de variabilidad experimental — Fase 4 — Stencil 2D (16384²), 15 réplicas

Campaña **energética y de reproducibilidad**, no numérica. La Fase 3 cerró la
parte numérica (drift, error, compensación); lo que falta es caracterizar la
**variabilidad experimental**: las corridas previas son esencialmente una
observación por combinación, así que no permiten estimar desviación estándar,
CV, intervalos de confianza ni error experimental.

Se mide la dispersión de: tiempo de ejecución, rendimiento (GFLOP/s), energía
GPU (NVML), energía CPU (RAPL), potencia media y derivadas (EDP, J/GFLOP)
sobre **15 réplicas independientes lanzadas en una sola operación**.

## Diseño

| | |
| --- | --- |
| Malla | NX = NY = 16384 |
| Horizontes | ITERS = 320 y 640 |
| Checkpoints | `CHECKPOINT_EVERY=0` (desactivados) |
| Perfilado | `RUN_NCU=0` (NCU altera tiempos y energía: fuera) |
| Precisiones | FP16 y BF16 (acum. FP32, WMMA) — **una sola invocación** vía `--tc both` |
| Tratamientos | `none`, `kahan_local`, `spatial` (mutuamente excluyentes) |
| Réplicas | 15 |
| Exclusividad | `--exclusive` en ambos arrays |
| Frecuencia GPU | **sin fijar** (`nvidia-smi -lgc` denegado, el modo persistente no persiste); el ruido térmico se trata por repetición estadística |

**2 precisiones × 3 tratamientos × 2 horizontes × 15 réplicas = 180
observaciones WMMA.** Cada invocación emite además filas `CPU_FP32`,
`CPU_FP64`, `GPU_FP32` y `GPU_FP64`: se conservan pero no cuentan dentro de
las 180 (el análisis y la validación las ignoran por construcción).

## Dos job arrays de 15 tareas

`SLURM_ARRAY_TASK_ID` **es** el número de réplica (1..15) en ambos arrays.

| Array | Bloque | Entorno | Tratamientos | Invocaciones/tarea | Obs WMMA/tarea | Walltime |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | A | `SPATIAL_COMP=off KAHAN_LIST="off on"` | `none` + `kahan_local` | 4 | 8 | `02:30:00` |
| 2 | B | `SPATIAL_COMP=on` (fuerza `KAHAN_LIST=off`) | `spatial` | 2 | 4 | `01:30:00` |

**30 tareas SLURM · 90 invocaciones del binario · 180 observaciones WMMA.**

`kahan=on + spatial=on` es imposible por **cuatro barreras independientes**:
(1) el diseño en dos arrays disjuntos — el array que pone `spatial=on` nunca
pide `kahan=on`; (2) `cuerpo_tarea.sh` aborta esa combinación antes de
ejecutar; (3) `run_stencil_tc.sbatch` fuerza `KAHAN_LIST=off` cuando
`SPATIAL_COMP=on`; (4) el binario la rechaza en `parse_args`.

### Walltime

Derivado del costo observado en los jobs **5200/5201** a esta misma escala
(16384², `ITERS_LIST="320 640"`, `CHECKPOINT_EVERY=0`, `RUN_NCU=0`, mismo
commit, `paccaA100`), donde el costo dominante son las referencias CPU
FP32/FP64, no la GPU: invocación de 320 iters ≈ 12 min, de 640 iters ≈ 23 min.
Bloque A ≈ 73 min → `02:30:00` (**+105 %**); bloque B ≈ 38 min → `01:30:00`
(**+137 %**). Ambos por encima del 30 % exigido. Un timeout deja la tarea
**sin CSV** (la extracción corre al final), por eso el margen es amplio y la
validación comprueba `State=COMPLETED`.

### Aislamiento entre tareas

Cada tarea trabaja en `replicas/r<NN>/<bloque>/`: su propia copia del fuente,
su propio binario compilado, su propio `logs/` y `results/`. Dentro de la
tarea, `SLURM_JOB_ID` es el id único de esa tarea del array, así que
`results/run_<JOB>.log` y los CSV `*_stencil_<JOB>.csv` tampoco colisionan, y
la columna `job_id` —ya presente en el esquema, que **no se modifica**— basta
para rastrear cada observación. Ninguna observación se promedia al guardar.

## 1. Lanzar (en PACCA)

```bash
cd ~/mixed-precision-hpc-tensor-cores-project    # ruta del repo en PACCA
git rev-parse HEAD          # debe ser 8d4e0e82efaf264ec2ad74b3f8bb172d614f4847
git status --porcelain      # limpio en archivos versionados

./Fase_4/lanzar_campana_variabilidad.sh
```

El lanzador, en este orden: verifica rama, commit y que los fuentes que congela
estén limpios —tolera los artefactos que dejan las corridas previas
(`stencil_tc`, `logs/`, `results/`, reportes de `ncu`) y acepta un HEAD
posterior al commit calibrado siempre que los fuentes de la campaña sean
idénticos a los que midieron los jobs 5200/5201—; consulta los límites **reales** (`scontrol show partition`,
`scontrol show config` para `MaxArraySize`, `sacctmgr` para `MaxSubmitJobs`) y
aborta si el walltime no cabe en el `MaxTime` de la partición o si el array
excede `MaxArraySize`; crea `Fase_4/f4_variabilidad_15r_<FECHA>/` (directorio
**nuevo**, nunca sobrescribe) con los 30 directorios de réplica y los fuentes
congelados; escribe el **manifiesto antes del primer `sbatch`**; envía los dos
arrays; e imprime ID de campaña, los dos JobID, número de tareas, invocaciones,
observaciones esperadas, el mapeo *array task ID → réplica → tratamiento* y el
`squeue` inicial. No bloquea la terminal.

Parámetros sobrescribibles por entorno: `REPLICAS`, `THROTTLE` (por defecto 2,
se baja solo al número de nodos de la partición), `WALL_A`, `WALL_B`,
`PARTICION`, `MUESTREO_GPU_S`, `CAMPANA`.

> **Throttle**: `--array=1-15%<THROTTLE>` limita cuántas tareas corren a la vez.
> Con `--exclusive` cada tarea toma un nodo entero, así que además impide que
> dos réplicas compitan por el mismo hardware.

## 2. Seguimiento

```bash
# Fija la campaña mas reciente en una variable y usa rutas reales
CAMP=$(ls -td Fase_4/f4_variabilidad_15r_* | head -1); echo "$CAMP"
JIDS=$(tail -n +2 "$CAMP/manifiesto_jobs.csv" | cut -d, -f1 | sort -u | paste -sd,)

# Cola, por usuario, con el estado de cada tarea del array
squeue -u "$(whoami)" -o "%.18i %.20j %.9P %.8T %.10M %.10l %.6D %R"

# Solo esta campaña, tarea por tarea (-r expande los arrays)
squeue -u "$(whoami)" -j "$JIDS" -r -o "%.18i %.20j %.8T %.10M %.10l %.6D %R"

# Estado final, incluyendo subtareas del array
sacct -j "$JIDS" -X --format=JobID%18,JobName%20,State,ExitCode,Elapsed,Start,End,NodeList

# Solo lo que no terminó bien
sacct -j "$JIDS" -X -n -P --format=JobID,State,ExitCode | grep -v 'COMPLETED|0:0'

# Log en vivo de una tarea concreta (bloque A, réplica 7)
tail -f "$CAMP"/logs/f4var_A_*_7.out
tail -f "$CAMP"/logs/f4var_A_*_7.err

# Log en vivo de la última tarea que haya arrancado
tail -f "$(ls -t "$CAMP"/logs/*.out | head -1)"

# Log crudo del binario de esa réplica (el que alimenta a extract_csv)
tail -f "$CAMP"/replicas/r07/A/results/run_*.log

# Progreso: corridas completadas de las 90 esperadas
grep -h '^Corrida:' "$CAMP"/replicas/*/*/results/run_*.log | wc -l

# Tareas ya registradas por sí mismas (réplica, nodo, inicio, fin, rc)
column -s, -t "$CAMP/ejecuciones.csv"

# CSV de energía ya generados (uno por tarea terminada, de 30)
ls -1 "$CAMP"/replicas/*/*/results/energy_stencil_*.csv | wc -l
```

## 3. Re-lanzamiento selectivo

Si falla la réplica 7 del bloque A, se relanza **sola**, sin tocar el resto:

```bash
./Fase_4/relanzar_replica.sh "$CAMP" 7 A
```

Archiva las salidas de la corrida fallida en
`replicas/r07/A/fallidos/<sello>/` —nunca las borra, y así no se cuentan como
duplicado— y reenvía con `--array=7` y el walltime del manifiesto. En crudo
sería:

```bash
cd "$CAMP" && sbatch --export=ALL --array=7 --time=02:30:00 array_bloqueA.sbatch
```

## 4. Análisis (al terminar)

```bash
python3 Fase_4/tools/analizar_variabilidad.py \
    --campana "$CAMP" \
    --anidado 5,10,15 \
    --csv-salida "$CAMP/estadisticos_variabilidad.csv" \
    --csv-deriva "$CAMP/deriva_temporal.csv" | tee "$CAMP/analisis_variabilidad.txt"
```

Da tres cosas:

1. **Estadísticos por celda** — agrupa por (precisión, compensación,
   iteraciones) y calcula `n`, media, `std` muestral (**ddof=1**), mediana,
   min, max y `CV% = 100·std/media` sobre tiempo, t/iter, GFLOP/s, energía GPU,
   energía CPU, energía total, energía GPU por iteración, EDP, J/GFLOP y
   potencia media GPU y total. **Las observaciones individuales siempre quedan
   listadas** con su réplica, `job_id` y nodo.
2. **Análisis anidado** — los mismos estadísticos con las primeras 5 réplicas,
   luego las primeras 10, luego las 15, más una tabla de cuánto se mueve el CV
   al pasar de un nivel al siguiente. Es el argumento para justificar el tamaño
   muestral en el reporte final: si el CV deja de moverse, `n` ya basta.
3. **Deriva temporal** — cada observación normalizada por la media de su celda,
   promediada por réplica; compara la mitad temprana contra la tardía
   (`delta %` frente al CV de la celda) e incorpora temperatura y clock medios
   por réplica desde la telemetría de contexto. Avisa si las réplicas no
   corrieron todas en el mismo nodo, porque entonces el CV mezcla variabilidad
   temporal con variabilidad de hardware.

## 5. Validación (al terminar)

```bash
python3 Fase_4/tools/validar_campana.py --campana "$CAMP"   # --replicas 15 por defecto
```

Comprueba: 15 réplicas completas en las 12 celdas · FP16 y BF16 presentes · los
tres tratamientos presentes · ambos horizontes presentes · ningún CSV ausente o
vacío · sin duplicados · tiempos > 0 · energía GPU y CPU no negativas ·
`energy_window_reliable=1` en toda observación · `rc=0` en todas las tareas
registradas · `State=COMPLETED` y `ExitCode=0` en las 30 tareas de los arrays
(avisa aparte de los `TIMEOUT`) · **ausencia total de corridas `kahan=on +
spatial=on`**. Sale con código ≠ 0 si algo falla.

## Telemetría de contexto (clock y temperatura)

El binario **no** registra clock ni temperatura por corrida, y su
implementación de NVML/RAPL no se toca. Para poder detectar deriva térmica a
posteriori, cada tarea deja aparte:

- `replicas/r<NN>/<bloque>/telemetria_gpu_<JOB>.csv` — un único proceso
  `nvidia-smi` que duerme entre muestras (`MUESTREO_GPU_S`, 60 s por defecto;
  `MUESTREO_GPU_S=0` lo desactiva).
- `replicas/r<NN>/<bloque>/telemetria_hitos_<JOB>.csv` — instantáneas
  `pre_run` / `post_run`.

No entra en ninguna ventana de tiempo ni de energía: son ficheros paralelos,
fuera del binario, y no alteran el esquema CSV.

## Interpretación

**`rel_l2`, `rel_linf` y `max_abs` saldrán NaN en las 180 observaciones.** A
320 y 640 iteraciones la solución supera `first_nonfinite` (la referencia FP64
diverge hacia ~1045). Es el comportamiento esperado, **no** es un fallo del job
ni motivo para descartar la energía medida: la campaña energética y la numérica
son independientes. Por eso la validación no comprueba métricas de error.

## Qué NO toca esta campaña

Ni la matemática del stencil, ni los kernels CUDA, ni FP16/BF16, ni la
acumulación FP32, ni Kahan, ni la compensación espacial, ni FP64, ni la
definición de error, ni `first_nonfinite`, ni los checkpoints, ni el esquema
CSV, ni la implementación de NVML o de RAPL. El warm-up y las transferencias
D2H siguen fuera de las ventanas de tiempo y energía tal como ya estaban: los
`.sbatch` de la campaña sólo fijan variables de entorno que
`run_stencil_tc.sbatch` ya aceptaba.

## Ficheros

| Fichero | Qué es |
| --- | --- |
| `lanzar_campana_variabilidad.sh` | Lanzador: verifica, congela, escribe el manifiesto y envía los dos arrays |
| `plantillas/array_bloqueA.sbatch` | Array 1 — `none` + `kahan_local` |
| `plantillas/array_bloqueB.sbatch` | Array 2 — `spatial` |
| `plantillas/cuerpo_tarea.sh` | Cuerpo común de una tarea: aísla la réplica y delega en el `.sbatch` de Fase 3 |
| `relanzar_replica.sh` | Re-lanzamiento selectivo de una réplica |
| `tools/analizar_variabilidad.py` | Estadísticos, anidado y deriva temporal |
| `tools/validar_campana.py` | Validación de integridad |
