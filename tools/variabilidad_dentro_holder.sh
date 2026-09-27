#!/bin/bash
# Corre la campana de variabilidad (replicas para ANOVA/Tukey) como steps de
# `srun` DENTRO de una asignacion ya existente (el holder), en vez de jobs
# `sbatch` independientes como hace tools/lanzar_campana_variabilidad.sh.
#
# PROBLEMA QUE ESTO NO RESUELVE SOLO: cada replica, al correr con
# `srun --jobid=$JOBID`, hereda el MISMO SLURM_JOB_ID -- y
# run_statistics.py cuenta replicas por job_id distinto (Dataset.cells(),
# n_replicas=nunique(job_id)). Sin arreglo aparte, run_statistics.py veria
# TODAS las replicas de esta campana como "1 sola", exactamente el problema
# que esta campana existe para resolver.
#
# SOLUCION: cada replica escribe a su propio subdirectorio
# (results/variabilidad/r<N>/), y variabilidad_relabel_job_id.py (corre
# despues, fuera del holder, sin GPU) reescribe la columna job_id de cada
# CSV a un valor sintetico unico por replica (p.ej. "7145-r3") antes de
# pasarlo a run_statistics.py. Las mediciones en si SI son replicas
# independientes de verdad (cada srun relanza el binario entero desde cero,
# misma logica que un sbatch separado) -- lo unico compartido es la
# ETIQUETA de asignacion SLURM, no la ejecucion.
#
# Uso:
#   JOBID=7145 REPLICAS=8 nohup bash tools/variabilidad_dentro_holder.sh \
#       > "logs_holder_run/variabilidad_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
#   disown

set -uo pipefail

JOBID="${JOBID:?export JOBID=<id del holder>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNDIR="${REPO_ROOT}/logs_holder_run"
mkdir -p "${RUNDIR}"

REPLICAS="${REPLICAS:-8}"
RUN_GEMM="${RUN_GEMM:-1}"
RUN_CONV="${RUN_CONV:-1}"
RUN_STENCIL="${RUN_STENCIL:-1}"

# Mismos puntos fijos, chicos y baratos, que tools/lanzar_campana_variabilidad.sh
REPL_GEMM_N_LIST="${REPL_GEMM_N_LIST:-1024}"
REPL_GEMM_ITERS_LIST="${REPL_GEMM_ITERS_LIST:-20 40}"
REPL_GEMM_ANCHOR_LIST="${REPL_GEMM_ANCHOR_LIST:-0 1 5}"
REPL_GEMM_COMP_LIST="${REPL_GEMM_COMP_LIST:-off on}"

REPL_CONV_HW_LIST="${REPL_CONV_HW_LIST:-64}"
REPL_CONV_ITERS_LIST="${REPL_CONV_ITERS_LIST:-20 40}"
REPL_CONV_ANCHOR_LIST="${REPL_CONV_ANCHOR_LIST:-0 1 5}"
REPL_CONV_COMP_LIST="${REPL_CONV_COMP_LIST:-off on}"

REPL_STENCIL_NX_LIST="${REPL_STENCIL_NX_LIST:-1024}"
REPL_STENCIL_ITERS_LIST="${REPL_STENCIL_ITERS_LIST:-20 40}"
REPL_STENCIL_ANCHOR_LIST="${REPL_STENCIL_ANCHOR_LIST:-0 1 5}"
REPL_STENCIL_KAHAN_LIST="${REPL_STENCIL_KAHAN_LIST:-off on}"
REPL_STENCIL_SPATIAL_COMP="${REPL_STENCIL_SPATIAL_COMP:-on}"
REPL_STENCIL_CPU_FP64="${REPL_STENCIL_CPU_FP64:-on}"

STATUS_LOG="${RUNDIR}/variabilidad_estado_$(date +%Y%m%d_%H%M%S).log"
estado() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${STATUS_LOG}"; }

holder_vivo() { sacct -j "${JOBID}" --format=State -n -X 2>/dev/null | grep -q RUNNING; }

declare -a WORKLIST=()
for ((r = 1; r <= REPLICAS; r++)); do
    [[ "${RUN_GEMM}" == "1" ]]    && WORKLIST+=("gemm:${r}")
    [[ "${RUN_CONV}" == "1" ]]    && WORKLIST+=("conv:${r}")
    [[ "${RUN_STENCIL}" == "1" ]] && WORKLIST+=("stencil:${r}")
done
if command -v shuf >/dev/null 2>&1; then
    mapfile -t WORKLIST < <(printf '%s\n' "${WORKLIST[@]}" | shuf)
fi

estado "=== Campana de variabilidad dentro del holder ${JOBID}: ${#WORKLIST[@]} replicas ==="

correr_replica() {
    local kernel="$1" rep="$2" dir="$3" script="$4" extra_env="$5"
    local vardir="results/variabilidad/r${rep}"
    local logsdir="results/variabilidad/r${rep}/logs"
    local log="${RUNDIR}/variab_${kernel}_r${rep}.log"

    if ! holder_vivo; then
        estado "ALERTA: holder ${JOBID} ya no esta RUNNING -- se detiene la campana."
        return 2
    fi
    estado "INICIA: ${kernel} replica ${rep} (${vardir})"
    # LOGS_DIR tambien por replica, no solo RESULTS_DIR: todas comparten
    # SLURM_JOB_ID=JOBID (steps del mismo holder), asi que sin esto el log
    # crudo (run_<job_id>.log) se ACUMULARIA entre replicas (mismo
    # JOBID+LOGS_DIR default) y cada extract_csv reprocesaria TODO lo
    # acumulado hasta ese punto, no solo su propia replica.
    if srun --jobid="${JOBID}" --job-name="variab_${kernel}_r${rep}" \
         bash -c "cd '${REPO_ROOT}/${dir}' && RESULTS_DIR='${vardir}' LOGS_DIR='${logsdir}' ${extra_env} bash '${script}'" \
         > "${log}" 2>&1; then
        estado "OK: ${kernel} replica ${rep}"
        return 0
    else
        local rc=$?
        estado "FALLO (rc=${rc}): ${kernel} replica ${rep} -> ${log} -- se continua."
        return 1
    fi
}

for item in "${WORKLIST[@]}"; do
    kernel="${item%%:*}"; rep="${item##*:}"
    case "${kernel}" in
        gemm)
            correr_replica gemm "${rep}" Fase_4/GEMM run_gemm_chained.sbatch \
                "N_LIST=\"${REPL_GEMM_N_LIST}\" ITERS_LIST=\"${REPL_GEMM_ITERS_LIST}\" ANCHOR_LIST=\"${REPL_GEMM_ANCHOR_LIST}\" COMP_LIST=\"${REPL_GEMM_COMP_LIST}\""
            ;;
        conv)
            correr_replica conv "${rep}" Fase_4/Convolution run_conv_chained.sbatch \
                "HW_LIST=\"${REPL_CONV_HW_LIST}\" ITERS_LIST=\"${REPL_CONV_ITERS_LIST}\" ANCHOR_LIST=\"${REPL_CONV_ANCHOR_LIST}\" COMP_LIST=\"${REPL_CONV_COMP_LIST}\""
            ;;
        stencil)
            correr_replica stencil "${rep}" Fase_4/Stencil run_stencil_tc.sbatch \
                "NX_LIST=\"${REPL_STENCIL_NX_LIST}\" NY_LIST=\"${REPL_STENCIL_NX_LIST}\" ITERS_LIST=\"${REPL_STENCIL_ITERS_LIST}\" ANCHOR_LIST=\"${REPL_STENCIL_ANCHOR_LIST}\" KAHAN_LIST=\"${REPL_STENCIL_KAHAN_LIST}\" SPATIAL_COMP=\"${REPL_STENCIL_SPATIAL_COMP}\" CPU_FP64=\"${REPL_STENCIL_CPU_FP64}\""
            ;;
    esac
    rc=$?
    if [[ "${rc}" -eq 2 ]]; then exit 0; fi
done

estado "=== Campana de variabilidad completa. Detalle en ${RUNDIR}/ ==="
