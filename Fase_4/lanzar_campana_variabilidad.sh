#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Fase 4 - Campana de VARIABILIDAD experimental (Stencil 2D, 16384^2)
#
# 15 replicas independientes en UN SOLO envio, mediante DOS job arrays de 15
# tareas (SLURM_ARRAY_TASK_ID == numero de replica).
#
# Que mide: la DISPERSION (std muestral, CV%, IC) de tiempo, GFLOP/s, energia
# GPU (NVML), energia CPU (RAPL), potencia media y derivadas (EDP, J/GFLOP).
# Las corridas previas son una sola observacion por combinacion, asi que no
# permiten estimar error experimental. Campana ENERGETICA y de
# REPRODUCIBILIDAD, no numerica: a ITERS=320/640 el error sale NaN por diseno
# (la solucion supera first_nonfinite) y eso NO invalida la energia medida.
#
# Diseno: 2 precisiones (FP16/BF16, una sola invocacion via --tc both)
#       x 3 tratamientos (none / kahan_local / spatial)
#       x 2 horizontes (ITERS 320 / 640)
#       x 15 replicas = 180 observaciones WMMA.
#
# ARRAY 1 (bloque A): SPATIAL_COMP=off, KAHAN_LIST="off on" -> none + kahan_local
# ARRAY 2 (bloque B): SPATIAL_COMP=on  (fuerza KAHAN_LIST=off) -> spatial
# Los dos arrays son disjuntos: kahan=on + spatial=on no puede ocurrir.
#
# Se ejecuta EN PACCA, desde cualquier punto del repositorio.
# ---------------------------------------------------------------------------
set -euo pipefail

# --- Parametros de la campana (todos sobrescribibles por entorno) -----------
REPLICAS="${REPLICAS:-15}"
# Throttle del array: cuantas tareas pueden correr a la vez. Con --exclusive
# cada tarea toma un nodo entero, asi que esto limita ademas cuantas replicas
# compiten por hardware. Si la particion tiene menos nodos, se baja solo.
THROTTLE="${THROTTLE:-2}"
NX_C="${NX:-16384}"
NY_C="${NY:-16384}"
ITERS_C="${ITERS_LIST:-320 640}"
CKPT_C="${CHECKPOINT_EVERY:-0}"      # checkpoints desactivados
NCU_C="${RUN_NCU:-0}"                # NCU altera tiempos y energia: fuera
FP64_GPU_C="${FP64_GPU:-on}"
CPU_FP64_C="${CPU_FP64:-on}"
PARTICION="${PARTICION:-GPU}"
MUESTREO_GPU_S="${MUESTREO_GPU_S:-60}"

# Walltime por tarea, derivado del costo OBSERVADO en los jobs 5200/5201
# (16384^2, ITERS_LIST="320 640", CHECKPOINT_EVERY=0, RUN_NCU=0, mismo commit,
# nodo paccaA100), donde el costo dominante son las referencias CPU FP32/FP64,
# no la GPU:
#   invocacion ITERS=320 -> ~12 min ;  invocacion ITERS=640 -> ~23 min
#   bloque A = 4 invocaciones ~= 70 min + compilacion ~3 min ~= 73 min
#       -> 02:30:00  (+105% de margen)
#   bloque B = 2 invocaciones ~= 35 min + compilacion ~3 min ~= 38 min
#       -> 01:30:00  (+137% de margen)
# Ambos superan con holgura el 30% exigido. Un timeout deja la tarea SIN CSV
# (la extraccion corre al final), por eso el margen es amplio y la validacion
# comprueba State=COMPLETED.
WALL_A="${WALL_A:-02:30:00}"
WALL_B="${WALL_B:-01:30:00}"

COMMIT_CALIBRADO="8d4e0e82efaf264ec2ad74b3f8bb172d614f4847"

msg() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# --- 0. Utilidades ----------------------------------------------------------
# Convierte un walltime de SLURM a segundos. Formatos: UNLIMITED/INFINITE (->
# vacio), D-HH:MM:SS, HH:MM:SS, MM:SS, MM.
slurm_a_segundos() {
    local t="${1:-}" dias=0
    case "${t}" in
        ""|UNLIMITED|INFINITE|unlimited|infinite|n/a|N/A) return 1 ;;
    esac
    if [[ "${t}" == *-* ]]; then
        dias="${t%%-*}"
        t="${t#*-}"
    fi
    local IFS=:
    # shellcheck disable=SC2206
    local partes=(${t})
    local h=0 m=0 s=0
    case "${#partes[@]}" in
        3) h="${partes[0]}"; m="${partes[1]}"; s="${partes[2]}" ;;
        2) m="${partes[0]}"; s="${partes[1]}" ;;
        1) m="${partes[0]}" ;;
        *) return 1 ;;
    esac
    [[ "${dias}${h}${m}${s}" =~ ^[0-9]+$ ]] || return 1
    printf '%d\n' "$((10#${dias} * 86400 + 10#${h} * 3600 + 10#${m} * 60 + 10#${s}))"
}

# --- 1. Procedencia: rama, commit, arbol -----------------------------------
ROOT="$(git rev-parse --show-toplevel)" || die "no estas dentro del repositorio git"
cd "${ROOT}"
RAMA="$(git rev-parse --abbrev-ref HEAD)"
COMMIT="$(git rev-parse HEAD)"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    msg "Cambios locales sobre archivos VERSIONADOS (pueden alterar la campana):"
    git status --porcelain --untracked-files=no
    die "arbol sucio; no se lanza. Revise o descarte los cambios usted mismo (este script no toca su arbol)."
fi

# Los fuentes que la campana congela deben estar limpios ademas de versionados:
# un fichero sin seguimiento ahi dentro podria cambiar lo que se compila.
FUENTES=(Fase_3/Stencil Fase_2/common.cuh tools/common_ncu.sh)

# ...salvo los ARTEFACTOS que toda corrida previa deja en Fase_3/Stencil (el
# binario compilado, logs/, results/, reportes de ncu). En PACCA estan siempre
# presentes y no alteran lo que se compila: se listan, pero no abortan.
es_artefacto() {
    case "$1" in
        Fase_3/Stencil/stencil_tc|\
        Fase_3/Stencil/logs|Fase_3/Stencil/logs/|Fase_3/Stencil/logs/*|\
        Fase_3/Stencil/results|Fase_3/Stencil/results/|Fase_3/Stencil/results/*|\
        *.out|*.err|*.log|*.ncu-rep|*.nsys-rep|*.qdrep|*__pycache__*) return 0 ;;
    esac
    return 1
}

SUCIO_FUENTES=""
ARTEFACTOS=""
while IFS= read -r _linea; do
    [[ -z "${_linea}" ]] && continue
    _estado="${_linea:0:2}"
    _ruta="${_linea:3}"
    _ruta="${_ruta#\"}"; _ruta="${_ruta%\"}"
    if [[ "${_estado}" == "??" ]] && es_artefacto "${_ruta}"; then
        ARTEFACTOS+="${_linea}"$'\n'
    else
        SUCIO_FUENTES+="${_linea}"$'\n'
    fi
done < <(git status --porcelain -- "${FUENTES[@]}")

if [[ -n "${SUCIO_FUENTES}" ]]; then
    msg "Cambios (o ficheros sin seguimiento) en los fuentes de la campana:"
    printf '%s' "${SUCIO_FUENTES}"
    die "los fuentes que se congelan no estan limpios; no se lanza."
fi
if [[ -n "${ARTEFACTOS}" ]]; then
    msg "[i] Artefactos de corridas anteriores en los fuentes (no se congelan, no bloquean):"
    printf '%s' "${ARTEFACTOS}" | sed 's/^/      /'
fi

# Calibracion del walltime: lo que importa no es que HEAD sea exactamente el
# commit calibrado, sino que los FUENTES de la campana sean los mismos que
# midieron los jobs 5200/5201. Un commit posterior que solo toque Fase_4 (o el
# README) no invalida nada.
if [[ "${COMMIT}" != "${COMMIT_CALIBRADO}" ]]; then
    if git rev-parse --verify --quiet "${COMMIT_CALIBRADO}^{commit}" >/dev/null &&
       git diff --quiet "${COMMIT_CALIBRADO}" HEAD -- "${FUENTES[@]}"; then
        msg "[i] HEAD=${COMMIT}"
        msg "    != commit calibrado ${COMMIT_CALIBRADO}, pero los fuentes de la"
        msg "    campana son IDENTICOS a los de ese commit: el walltime calibrado"
        msg "    sobre los jobs 5200/5201 sigue siendo valido."
    else
        msg "AVISO: HEAD=${COMMIT}"
        msg "       Los fuentes difieren de ${COMMIT_CALIBRADO}, sobre el que se"
        msg "       calibro el walltime (jobs 5200/5201)."
        msg "       Exporte ACEPTAR_COMMIT_DISTINTO=1 si aun asi desea lanzar."
        [[ "${ACEPTAR_COMMIT_DISTINTO:-0}" == "1" ]] || die "fuentes distintos a los calibrados"
    fi
fi

command -v sbatch >/dev/null || die "sbatch no disponible: ejecute este script en PACCA"

# --- 2. Limites REALES de la particion y de la cuenta ----------------------
msg "=============================================================="
msg " Limites de la particion ${PARTICION} y de la cuenta"
msg "=============================================================="
PART_INFO="$(scontrol show partition "${PARTICION}" 2>/dev/null || true)"
[[ -n "${PART_INFO}" ]] || die "la particion '${PARTICION}' no existe (scontrol show partition)"
printf '%s\n' "${PART_INFO}" | sed 's/^/  /'

MAXTIME="$(printf '%s\n' "${PART_INFO}" | grep -oE 'MaxTime=[^ ]+' | head -1 | cut -d= -f2 || true)"
NODOS_PART="$(printf '%s\n' "${PART_INFO}" | grep -oE 'TotalNodes=[0-9]+' | head -1 | cut -d= -f2 || true)"
msg ""
msg "  MaxTime de la particion : ${MAXTIME:-desconocido}"
msg "  Nodos de la particion   : ${NODOS_PART:-desconocido}"

MAXARRAY="$(scontrol show config 2>/dev/null | grep -iE '^MaxArraySize' | awk -F= '{gsub(/ /,"",$2); print $2}' || true)"
msg "  MaxArraySize            : ${MAXARRAY:-desconocido}"

msg ""
msg "  Asociaciones de $(whoami) (sacctmgr):"
sacctmgr -n -P show assoc where user="$(whoami)" \
    format=Cluster,Account,Partition,MaxJobs,MaxSubmitJobs,GrpJobs,GrpSubmitJobs,MaxWall 2>/dev/null \
    | sed 's/^/    /' || msg "    (sacctmgr no disponible o sin datos)"
msg ""
msg "  Jobs propios ahora en cola: $(squeue -h -u "$(whoami)" 2>/dev/null | wc -l)"
msg ""

# Comprobaciones duras sobre esos limites
if [[ -n "${MAXARRAY}" && "${MAXARRAY}" =~ ^[0-9]+$ ]] && (( REPLICAS > MAXARRAY )); then
    die "REPLICAS=${REPLICAS} supera MaxArraySize=${MAXARRAY}"
fi
if MAX_S="$(slurm_a_segundos "${MAXTIME}")"; then
    for par in "A:${WALL_A}" "B:${WALL_B}"; do
        blq="${par%%:*}"; wall="${par#*:}"
        W_S="$(slurm_a_segundos "${wall}")" || die "walltime ilegible: ${wall}"
        (( W_S <= MAX_S )) || die "el walltime del bloque ${blq} (${wall}) supera MaxTime=${MAXTIME} de la particion"
    done
    msg "  [OK] ${WALL_A} y ${WALL_B} caben en MaxTime=${MAXTIME}"
else
    msg "  [i] MaxTime sin limite util (${MAXTIME:-desconocido}); no hay nada que verificar contra el."
fi

# Con --exclusive no tiene sentido permitir mas tareas simultaneas que nodos.
if [[ -n "${NODOS_PART}" && "${NODOS_PART}" =~ ^[0-9]+$ ]] && (( THROTTLE > NODOS_PART )); then
    msg "  [i] THROTTLE=${THROTTLE} > nodos de la particion (${NODOS_PART}); se baja a ${NODOS_PART}."
    THROTTLE="${NODOS_PART}"
fi
(( THROTTLE >= 1 )) || die "THROTTLE debe ser >= 1"

TAREAS_TOTALES=$(( REPLICAS * 2 ))
LIMITE_ENVIO="$(sacctmgr -n -P show assoc where user="$(whoami)" format=MaxSubmitJobs 2>/dev/null \
                | grep -oE '^[0-9]+' | sort -n | head -1 || true)"
if [[ -n "${LIMITE_ENVIO}" ]] && (( TAREAS_TOTALES > LIMITE_ENVIO )); then
    msg "AVISO: la campana envia ${TAREAS_TOTALES} tareas y MaxSubmitJobs=${LIMITE_ENVIO}."
    msg "       SLURM puede rechazar el segundo array. Reduzca REPLICAS o pida ampliacion."
    [[ "${ACEPTAR_LIMITE_ENVIO:-0}" == "1" ]] || die "limite de envio insuficiente (exporte ACEPTAR_LIMITE_ENVIO=1 para forzar)"
fi

# --- 3. Directorio de campana NUEVO (jamas sobrescribe) ---------------------
CAMPANA="${CAMPANA:-f4_variabilidad_15r_$(date +%Y%m%d_%H%M%S)}"
CDIR="${ROOT}/Fase_4/${CAMPANA}"
[[ -e "${CDIR}" ]] && die "el directorio de campana ya existe: ${CDIR}"
mkdir -p "${CDIR}/logs" "${CDIR}/replicas/Fase_2" "${CDIR}/replicas/tools"

SRC="${ROOT}/Fase_3/Stencil"
PLANT="${ROOT}/Fase_4/plantillas"

# Fuentes congelados en el nivel "replicas/": el .cu incluye
# "../../Fase_2/common.cuh" (relativo al PROPIO .cu, que vive en
# replicas/rNN/<bloque>/) y run_stencil_tc.sbatch busca
# "../../tools/common_ncu.sh" (relativo a su cwd, el mismo directorio). Ambas
# rutas resuelven exactamente en replicas/.
cp "${ROOT}/Fase_2/common.cuh"        "${CDIR}/replicas/Fase_2/"
cp "${ROOT}/tools/common_ncu.sh"      "${CDIR}/replicas/tools/"

SHA_CU="$(sha256sum "${SRC}/stencil_tensor_activation.cu" | awk '{print $1}')"

# Un directorio de trabajo por (replica, bloque): cada tarea del array compila
# su propio binario, escribe su propio results/ y su propio log crudo. Ninguna
# tarea comparte fichero con otra.
for r in $(seq 1 "${REPLICAS}"); do
    RID="$(printf 'r%02d' "${r}")"
    for blq in A B; do
        TD="${CDIR}/replicas/${RID}/${blq}"
        mkdir -p "${TD}/tools" "${TD}/logs" "${TD}/results"
        cp "${SRC}/stencil_tensor_activation.cu" "${TD}/"
        cp "${SRC}/run_stencil_tc.sbatch"        "${TD}/"
        cp "${SRC}/tools/extract_csv.py"         "${TD}/tools/"
        cp "${SRC}/tools/power_sampling.h"       "${TD}/tools/"
        cp "${ROOT}/tools/common_ncu.sh"         "${TD}/"
        [[ "$(sha256sum "${TD}/stencil_tensor_activation.cu" | awk '{print $1}')" == "${SHA_CU}" ]] \
            || die "la copia del fuente en ${TD} no coincide con el original"
    done
done

cp "${PLANT}/cuerpo_tarea.sh"      "${CDIR}/"
cp "${PLANT}/array_bloqueA.sbatch" "${CDIR}/"
cp "${PLANT}/array_bloqueB.sbatch" "${CDIR}/"

cat > "${CDIR}/campana.env" <<ENV_EOF
# Parametros congelados de la campana. Lo lee cada tarea del array.
CAMPANA_ID="${CAMPANA}"
CAMPANA_DIR="${CDIR}"
REPO_ROOT="${ROOT}"
RAMA="${RAMA}"
COMMIT="${COMMIT}"
SHA_CU="${SHA_CU}"
REPLICAS="${REPLICAS}"
NX="${NX_C}"
NY="${NY_C}"
ITERS_LIST="${ITERS_C}"
CHECKPOINT_EVERY="${CKPT_C}"
RUN_NCU="${NCU_C}"
FP64_GPU="${FP64_GPU_C}"
CPU_FP64="${CPU_FP64_C}"
MUESTREO_GPU_S="${MUESTREO_GPU_S}"
ENV_EOF

# --- 4. MANIFIESTO, escrito ANTES del primer sbatch -------------------------
MAPA="${CDIR}/manifiesto_jobs.csv"
echo "array_job_id,array_nombre,bloque,array_task_id,replica,tratamientos,precisiones,iters,nx,ny,kahan_list,spatial_comp,checkpoint_every,run_ncu,walltime,throttle,invocaciones,obs_wmma,dir_replica" > "${MAPA}"

TS_LANZAMIENTO="$(date -Iseconds)"
cat > "${CDIR}/MANIFIESTO.md" <<MAN_EOF
# Campana de variabilidad experimental - Fase 4 - Stencil 2D

- **ID de campana**: \`${CAMPANA}\`
- **Timestamp de lanzamiento**: ${TS_LANZAMIENTO}
- **Rama**: \`${RAMA}\`
- **Commit**: \`${COMMIT}\`
- **Estado del arbol**: limpio en archivos versionados y en los fuentes congelados
- **sha256 del .cu congelado**: \`${SHA_CU}\`
- **Directorio**: \`Fase_4/${CAMPANA}/\`
- **Particion**: \`${PARTICION}\` (MaxTime=${MAXTIME:-desconocido}, nodos=${NODOS_PART:-desconocido}, MaxArraySize=${MAXARRAY:-desconocido})

## Proposito
Caracterizar la **variabilidad experimental** (std muestral ddof=1, CV%, IC) de
tiempo, GFLOP/s, energia GPU (NVML), energia CPU (RAPL), potencia media y
derivadas (EDP, J/GFLOP). Las corridas previas son esencialmente **una
observacion por combinacion**, de modo que no permiten estimar desviacion
estandar ni error experimental. Campana **energetica y de reproducibilidad**,
no numerica.

## Parametros fijos
| Parametro | Valor |
| --- | --- |
| NX x NY | ${NX_C} x ${NY_C} |
| ITERS | ${ITERS_C} |
| CHECKPOINT_EVERY | ${CKPT_C} (desactivados) |
| RUN_NCU | ${NCU_C} (NCU altera tiempo y energia: fuera) |
| FP64_GPU / CPU_FP64 | ${FP64_GPU_C} / ${CPU_FP64_C} |
| Replicas | ${REPLICAS} |
| Exclusividad | \`--exclusive\` en ambos arrays |
| Frecuencia GPU | **sin fijar** (nvidia-smi -lgc denegado, el modo persistente no persiste); el ruido termico se trata por repeticion estadistica |
| Telemetria de contexto | \`nvidia-smi\` cada ${MUESTREO_GPU_S}s + instantaneas pre/post, fuera de la medicion, para detectar deriva termica |

## Factores
- **Precision** (2): FP16 y BF16, acumulacion FP32, WMMA. Ambas en **una sola
  invocacion** del binario mediante \`--tc both\`.
- **Compensacion** (3, mutuamente excluyentes):
  | Tratamiento | Flags | Etiqueta en el CSV |
  | --- | --- | --- |
  | none | \`--kahan off --spatial-comp off\` | \`WMMA_FP16\`/\`WMMA_BF16\` con \`kahan=off\` |
  | kahan_local | \`--kahan on --spatial-comp off\` | \`WMMA_FP16\`/\`WMMA_BF16\` con \`kahan=on\` |
  | spatial | \`--kahan off --spatial-comp on\` | \`WMMA_FP16_SP\`/\`WMMA_BF16_SP\` |
- **Horizonte** (2): ITERS = 320 y 640.

## Estructura de los dos job arrays
\`SLURM_ARRAY_TASK_ID\` **es** el numero de replica (1..${REPLICAS}) en ambos arrays.

| Array | Bloque | Entorno | Tratamientos | Invocaciones/tarea | Obs WMMA/tarea | Walltime |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | A | \`SPATIAL_COMP=off KAHAN_LIST="off on"\` | none + kahan_local | 4 | 8 | ${WALL_A} |
| 2 | B | \`SPATIAL_COMP=on\` (fuerza \`KAHAN_LIST=off\`) | spatial | 2 | 4 | ${WALL_B} |

Throttle aplicado: \`--array=1-${REPLICAS}%${THROTTLE}\` en cada array.

**Cuatro barreras** contra \`kahan=on + spatial=on\`: (1) el diseno en dos arrays
disjuntos -- el array que pone \`spatial=on\` nunca pide \`kahan=on\`; (2)
\`cuerpo_tarea.sh\` aborta esa combinacion antes de ejecutar; (3)
\`run_stencil_tc.sbatch\` fuerza \`KAHAN_LIST=off\` cuando \`SPATIAL_COMP=on\`;
(4) el binario la rechaza en \`parse_args\`.

## Conteo esperado
- Tareas SLURM: **${TAREAS_TOTALES}** (2 arrays x ${REPLICAS} tareas)
- Invocaciones del binario: **$(( REPLICAS * 6 ))** (${REPLICAS} x (4 + 2))
- Observaciones WMMA: **$(( REPLICAS * 12 ))** = 2 precisiones x 3 tratamientos x 2 horizontes x ${REPLICAS} replicas

Cada invocacion emite ademas filas \`CPU_FP32\`, \`CPU_FP64\`, \`GPU_FP32\` y
\`GPU_FP64\`. Se conservan, pero **no** cuentan dentro de las $(( REPLICAS * 12 )).

## Independencia y trazabilidad
Cada tarea del array trabaja en su **propio directorio**
\`replicas/r<NN>/<bloque>/\` (fuente, binario, \`logs/\`, \`results/\`), asi que no
hay colision de nombres entre tareas. Dentro de cada tarea el
\`SLURM_JOB_ID\` es unico, de modo que \`results/run_<JOB>.log\` y los CSV
\`*_stencil_<JOB>.csv\` tambien lo son, y la columna \`job_id\` -- ya presente en
el esquema, que **no se modifica** -- identifica la observacion. El mapa
job -> replica se registra en \`ejecuciones.csv\` (lo escribe cada tarea al
arrancar, con nodo y timestamps) y la ruta del CSV lo dice por si sola.
Ninguna observacion se promedia ni se descarta al guardar.

## Advertencia de interpretacion
A 320 y 640 iteraciones **rel_l2 / rel_linf / max_abs saldran NaN**: la solucion
supera \`first_nonfinite\`. Es esperado, **no** es un fallo del job ni motivo
para descartar la energia medida. La campana energetica y la numerica son
independientes.

## Salidas
- Logs SLURM por tarea: \`logs/f4var_{A,B}_<ARRAYJOBID>_<TASKID>.{out,err}\`
- Log crudo por tarea: \`replicas/r<NN>/<bloque>/results/run_<JOBID>.log\`
- CSV por tarea: \`replicas/r<NN>/<bloque>/results/{summary,energy,drift,store,horizon}_stencil_<JOBID>.csv\`
- Telemetria de contexto: \`replicas/r<NN>/<bloque>/telemetria_{gpu,hitos}_<JOBID>.csv\`
- Registro de ejecucion: \`ejecuciones.csv\`
MAN_EOF

msg "=============================================================="
msg " Campana : ${CAMPANA}"
msg " Rama    : ${RAMA}"
msg " Commit  : ${COMMIT}"
msg " Destino : ${CDIR}"
msg " Manifiesto escrito ANTES del primer sbatch."
msg "=============================================================="
msg ""

# --- 5. Envio de los dos arrays --------------------------------------------
cd "${CDIR}"

enviar_array() {
    local bloque="$1" script="$2" wall="$3" tratamientos="$4" ninv="$5" nobs="$6"
    local spatial="$7" kahan_list="$8"

    # Barrera de diseno tambien en el lanzador.
    if [[ "${spatial}" == "on" && " ${kahan_list} " == *" on "* ]]; then
        die "combinacion prohibida kahan=on + spatial=on (bloque ${bloque})"
    fi

    local jid
    jid=$(sbatch --parsable --export=ALL \
            --partition="${PARTICION}" \
            --array="1-${REPLICAS}%${THROTTLE}" \
            --time="${wall}" \
            "${script}")
    local nombre
    nombre=$(grep -m1 -oE '^#SBATCH --job-name=.*' "${script}" | cut -d= -f2)

    for r in $(seq 1 "${REPLICAS}"); do
        printf '%s,%s,%s,%d,%d,%s,FP16+BF16,%s,%d,%d,%s,%s,%d,%d,%s,%d,%d,%d,%s\n' \
            "${jid}" "${nombre}" "${bloque}" "${r}" "${r}" "${tratamientos}" \
            "${ITERS_C// /+}" "${NX_C}" "${NY_C}" "${kahan_list// /+}" "${spatial}" \
            "${CKPT_C}" "${NCU_C}" "${wall}" "${THROTTLE}" "${ninv}" "${nobs}" \
            "replicas/$(printf 'r%02d' "${r}")/${bloque}" >> "${MAPA}"
    done

    printf '%s\n' "${jid}"
}

msg "Enviando ARRAY 1 (bloque A: none + kahan_local)..."
JID_A="$(enviar_array A array_bloqueA.sbatch "${WALL_A}" "none+kahan_local" 4 8 off "off on")"
msg "  JobID de array: ${JID_A}   tareas 1-${REPLICAS} (throttle %${THROTTLE})  walltime ${WALL_A}"

msg "Enviando ARRAY 2 (bloque B: spatial)..."
JID_B="$(enviar_array B array_bloqueB.sbatch "${WALL_B}" "spatial" 2 4 on "off")"
msg "  JobID de array: ${JID_B}   tareas 1-${REPLICAS} (throttle %${THROTTLE})  walltime ${WALL_B}"

# Los JobID reales se conocen despues del envio: se anaden al manifiesto.
{
    echo ""
    echo "## Job IDs (anadidos tras el envio)"
    echo ""
    echo "| Array | Bloque | Tratamientos | JobID | Tareas | Walltime |"
    echo "| --- | --- | --- | --- | --- | --- |"
    echo "| 1 | A | none + kahan_local | \`${JID_A}\` | \`${JID_A}_[1-${REPLICAS}]\` | ${WALL_A} |"
    echo "| 2 | B | spatial | \`${JID_B}\` | \`${JID_B}_[1-${REPLICAS}]\` | ${WALL_B} |"
    echo ""
    echo "Mapeo: \`array task id == numero de replica\`, identico en ambos arrays."
} >> "${CDIR}/MANIFIESTO.md"

# --- 6. Reporte -------------------------------------------------------------
msg ""
msg "=============================================================="
msg " ID de campana                 : ${CAMPANA}"
msg " Array 1 (bloque A)            : JobID ${JID_A}   tareas ${JID_A}_[1-${REPLICAS}]"
msg " Array 2 (bloque B)            : JobID ${JID_B}   tareas ${JID_B}_[1-${REPLICAS}]"
msg " Tareas SLURM totales          : ${TAREAS_TOTALES}"
msg " Invocaciones del binario      : $(( REPLICAS * 6 ))"
msg " Observaciones WMMA esperadas  : $(( REPLICAS * 12 ))  (2 prec x 3 trat x 2 iters x ${REPLICAS} rep)"
msg " Manifiesto                    : ${CDIR}/MANIFIESTO.md"
msg " Mapa de tareas                : ${MAPA}"
msg "=============================================================="
msg ""
msg "Mapeo array task ID -> replica -> tratamiento:"
msg ""
printf '  %-14s %-14s %-22s %-12s %s\n' "TAREA" "REPLICA" "TRATAMIENTOS" "ITERS" "DIRECTORIO"
for r in $(seq 1 "${REPLICAS}"); do
    RID="$(printf 'r%02d' "${r}")"
    printf '  %-14s %-14s %-22s %-12s %s\n' \
        "${JID_A}_${r}" "${r} (${RID})" "none+kahan_local" "${ITERS_C// /,}" "replicas/${RID}/A"
    printf '  %-14s %-14s %-22s %-12s %s\n' \
        "${JID_B}_${r}" "${r} (${RID})" "spatial" "${ITERS_C// /,}" "replicas/${RID}/B"
done

msg ""
msg "Estado inicial en cola:"
squeue -u "$(whoami)" -o "%.18i %.20j %.9P %.8T %.10M %.10l %.6D %R" || true
msg ""
msg "Seguimiento (no bloquea):"
msg "  squeue -u \$(whoami) -j ${JID_A},${JID_B} -o \"%.18i %.20j %.8T %.10M %.10l %R\""
msg "  sacct -j ${JID_A},${JID_B} -X --format=JobID%18,JobName%20,State,ExitCode,Elapsed,Start,End,NodeList"
msg "  tail -f ${CDIR}/logs/f4var_A_${JID_A}_1.out"
