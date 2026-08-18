# ---------------------------------------------------------------------------
# Cuerpo COMUN de una tarea de job array de la campana de variabilidad (Fase 4).
#
# Lo sourcean array_bloqueA.sbatch y array_bloqueB.sbatch DESPUES de fijar:
#   BLOQUE        A | B
#   TRATAMIENTOS  etiqueta legible del bloque
#   SPATIAL_COMP  off | on
#   KAHAN_LIST    "off on" | "off"
#   INVOCACIONES  invocaciones del binario que hace el bloque
#   OBS_WMMA      observaciones WMMA que produce el bloque
#
# No mide nada por si mismo: prepara el entorno de la replica, delega en el
# run_stencil_tc.sbatch congelado de Fase 3 (que compila y ejecuta) y registra
# la identidad de la tarea. La medicion de tiempo y energia vive dentro del
# binario y no se toca.
# ---------------------------------------------------------------------------
set -euo pipefail

CDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${CDIR}/campana.env" ]] || {
    echo "ERROR: no encuentro ${CDIR}/campana.env (envie desde el dir de campana)" >&2
    exit 90
}
# shellcheck source=/dev/null
source "${CDIR}/campana.env"

# --- Barrera de diseno ------------------------------------------------------
# kahan=on + spatial=on NO existe en este diseno. Es la tercera de cuatro
# barreras: (1) el diseno en dos arrays disjuntos, (2) esta comprobacion,
# (3) run_stencil_tc.sbatch fuerza KAHAN_LIST=off si SPATIAL_COMP=on,
# (4) el binario aborta esa pareja en parse_args.
if [[ "${SPATIAL_COMP}" == "on" && " ${KAHAN_LIST} " == *" on "* ]]; then
    echo "ERROR: combinacion prohibida kahan=on + spatial=on (bloque ${BLOQUE})" >&2
    exit 91
fi

TASK="${SLURM_ARRAY_TASK_ID:?este script solo corre como job array}"
REPLICA="$(printf 'r%02d' "${TASK}")"
TAREA_DIR="${CDIR}/replicas/${REPLICA}/${BLOQUE}"
[[ -d "${TAREA_DIR}" ]] || {
    echo "ERROR: falta el directorio de replica ${TAREA_DIR}" >&2
    exit 92
}

JOB="${SLURM_JOB_ID:-manual}"
ARRAY_JOB="${SLURM_ARRAY_JOB_ID:-${JOB}}"
NODO="$(hostname)"
INICIO="$(date -Iseconds)"

echo "=============================================================="
echo " Fase 4 | campana ${CAMPANA_ID}"
echo " bloque=${BLOQUE}  tratamientos=${TRATAMIENTOS}"
echo " replica=${TASK} (${REPLICA})   [array task id == numero de replica]"
echo " array_job_id=${ARRAY_JOB}  array_task_id=${TASK}  job_id=${JOB}"
echo " nodo=${NODO}  inicio=${INICIO}"
echo " dir de trabajo=${TAREA_DIR}"
echo " NX=${NX} NY=${NY} ITERS_LIST=${ITERS_LIST} CHECKPOINT_EVERY=${CHECKPOINT_EVERY}"
echo " RUN_NCU=${RUN_NCU} SPATIAL_COMP=${SPATIAL_COMP} KAHAN_LIST=${KAHAN_LIST}"
echo "=============================================================="

# --- Telemetria de CONTEXTO (clock / temperatura) ---------------------------
# NO es la medicion: la energia la miden NVML y RAPL dentro del binario y esa
# implementacion no se toca. Esto existe solo para poder detectar DERIVA
# TERMICA a posteriori entre replicas separadas por horas. Es un unico proceso
# nvidia-smi que duerme entre muestras (MUESTREO_GPU_S segundos, 0 = apagado),
# mas dos instantaneas en los bordes del job.
TELEM="${TAREA_DIR}/telemetria_gpu_${JOB}.csv"
HITOS="${TAREA_DIR}/telemetria_hitos_${JOB}.csv"
CAMPOS_SMI="timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,pstate"
SAMPLER_PID=""

instantanea() {
    local etiqueta="$1"
    [[ -s "${HITOS}" ]] || echo "hito,timestamp,clocks_sm_mhz,clocks_mem_mhz,temp_gpu_c,power_w,util_gpu,pstate" > "${HITOS}"
    nvidia-smi --query-gpu="${CAMPOS_SMI}" --format=csv,noheader 2>/dev/null \
        | sed "s/^/${etiqueta},/" >> "${HITOS}" || true
}

detener_sampler() {
    if [[ -n "${SAMPLER_PID}" ]]; then
        kill "${SAMPLER_PID}" 2>/dev/null || true
        wait "${SAMPLER_PID}" 2>/dev/null || true
    fi
}
trap detener_sampler EXIT

if [[ "${MUESTREO_GPU_S:-0}" -gt 0 ]]; then
    nvidia-smi --query-gpu="${CAMPOS_SMI}" --format=csv \
        -l "${MUESTREO_GPU_S}" > "${TELEM}" 2>/dev/null &
    SAMPLER_PID=$!
    echo "Telemetria de contexto: ${TELEM} (cada ${MUESTREO_GPU_S}s, pid ${SAMPLER_PID})"
else
    echo "Telemetria de contexto periodica desactivada (MUESTREO_GPU_S=0)"
fi

# --- Registro de la ejecucion (mapa job_id -> replica, antes de correr) -----
REG="${CDIR}/ejecuciones.csv"
{
    flock 9
    [[ -s "${REG}" ]] || echo "array_job_id,array_task_id,replica,bloque,job_id,nodo,inicio,fin,rc,tratamientos,spatial_comp,kahan_list,iters_list,nx,ny,invocaciones,obs_wmma" > "${REG}"
    printf '%s,%s,%s,%s,%s,%s,%s,,,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${ARRAY_JOB}" "${TASK}" "${TASK}" "${BLOQUE}" "${JOB}" "${NODO}" "${INICIO}" \
        "${TRATAMIENTOS}" "${SPATIAL_COMP}" "${KAHAN_LIST// /+}" "${ITERS_LIST// /+}" \
        "${NX}" "${NY}" "${INVOCACIONES}" "${OBS_WMMA}" >> "${REG}"
} 9>"${CDIR}/.ejecuciones.lock"

# --- Ejecucion --------------------------------------------------------------
# run_stencil_tc.sbatch se ejecuta como script bash: sus directivas #SBATCH se
# ignoran (las del array mandan) y hace cd a SLURM_SUBMIT_DIR, que aqui se
# redefine SOLO para el hijo y apunta al directorio de esta replica. Dentro,
# JOB_ID = SLURM_JOB_ID = el id UNICO de esta tarea del array, asi que
# results/run_<JOB>.log y los CSV no colisionan con ninguna otra tarea, y la
# columna job_id del CSV queda rastreable hasta la replica sin tocar el esquema.
instantanea "pre_run"

set +e
env SLURM_SUBMIT_DIR="${TAREA_DIR}" \
    NX="${NX}" NY="${NY}" ITERS_LIST="${ITERS_LIST}" \
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY}" RUN_NCU="${RUN_NCU}" \
    SPATIAL_COMP="${SPATIAL_COMP}" KAHAN_LIST="${KAHAN_LIST}" \
    FP64_GPU="${FP64_GPU}" CPU_FP64="${CPU_FP64}" \
    bash "${TAREA_DIR}/run_stencil_tc.sbatch"
RC=$?
set -e

instantanea "post_run"
detener_sampler
SAMPLER_PID=""
FIN="$(date -Iseconds)"

# --- Cierre del registro ----------------------------------------------------
cat > "${TAREA_DIR}/tarea_${JOB}.env" <<REG_EOF
CAMPANA_ID=${CAMPANA_ID}
BLOQUE=${BLOQUE}
TRATAMIENTOS=${TRATAMIENTOS}
REPLICA=${TASK}
ARRAY_JOB_ID=${ARRAY_JOB}
ARRAY_TASK_ID=${TASK}
JOB_ID=${JOB}
NODO=${NODO}
INICIO=${INICIO}
FIN=${FIN}
RC=${RC}
NX=${NX}
NY=${NY}
ITERS_LIST=${ITERS_LIST}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY}
RUN_NCU=${RUN_NCU}
SPATIAL_COMP=${SPATIAL_COMP}
KAHAN_LIST=${KAHAN_LIST}
FP64_GPU=${FP64_GPU}
CPU_FP64=${CPU_FP64}
COMMIT=${COMMIT}
SHA_CU=${SHA_CU}
REG_EOF

{
    flock 9
    python3 - "${REG}" "${JOB}" "${FIN}" "${RC}" <<'PY_EOF'
import csv, sys
ruta, job, fin, rc = sys.argv[1:5]
with open(ruta, newline="") as fh:
    filas = list(csv.reader(fh))
for f in filas[1:]:
    if len(f) > 8 and f[4] == job:
        f[7], f[8] = fin, rc
with open(ruta, "w", newline="") as fh:
    csv.writer(fh).writerows(filas)
PY_EOF
} 9>"${CDIR}/.ejecuciones.lock"

echo
echo "=============================================================="
echo " Fin replica ${TASK} bloque ${BLOQUE} | job ${JOB} | rc=${RC} | ${FIN}"
echo " CSV en ${TAREA_DIR}/results/"
echo "=============================================================="
exit "${RC}"
