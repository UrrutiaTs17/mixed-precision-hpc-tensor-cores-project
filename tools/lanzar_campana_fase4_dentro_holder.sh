#!/bin/bash
# tools/lanzar_campana_fase4_dentro_holder.sh -- misma campana y mismos pasos
# que tools/lanzar_campana_fase4.sh, pero como job-steps de `srun --overlap`
# DENTRO de una asignacion SLURM ya existente (JOBID), en vez de encolar jobs
# nuevos con `sbatch` (que esperarian en cola si el nodo GPU esta ocupado por
# esa asignacion). Mismo patron que tools/run_dentro_holder.sh (holder del
# equipo, campana_holder_20260917): un step que falla NO tumba el holder, se
# registra y se sigue con el siguiente (sin `set -e`).
#
# Uso (desde la raiz del checkout, SIEMPRE detached -- esto corre horas):
#   JOBID=7757 CI_MODE=monomode OUT_BASE=$HOME/campana_v2_out/fase4_v2 \
#       KERNELS_RUN="stencil gemm conv" GROUPS_RUN=spk \
#       nohup bash tools/lanzar_campana_fase4_dentro_holder.sh \
#       > logs_holder_run/driver_v2_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#   disown
#
# Para detener la SECUENCIA (no el holder): touch logs_holder_run/HOLDER_RUN_STOP
#
# Mismas variables que lanzar_campana_fase4.sh (CI_MODE, OUT_BASE, GROUPS_RUN,
# KERNELS_RUN, AFTER_JOB se ignora aqui -- la secuencia ya es serial) mas JOBID
# (obligatoria: el holder donde correr).
set -uo pipefail

: "${JOBID:?export JOBID=<id del holder>, ej. JOBID=7757}"
: "${CI_MODE:?defina CI_MODE=monomode|legacy}"
: "${OUT_BASE:?defina OUT_BASE=directorio propio de esta campana}"
case "${OUT_BASE%/}" in results|*/results) echo "ERROR: OUT_BASE no puede terminar en results/" >&2; exit 2;; esac
GROUPS_RUN="${GROUPS_RUN:-sp off}"
KERNELS_RUN="${KERNELS_RUN:-stencil}"
K_STENCIL_EXT="${K_STENCIL_EXT:-2 4 16 64 128}"
K_STENCIL_FULL="${K_STENCIL_FULL:-0 1 2 4 8 16 32 64 128}"
K_CHAINED_ALL="${K_CHAINED_ALL:-0 1 2 5 10 20 40}"
ALPHA_CAMPANA="0.1875"

TIER_S_NX="4096";        TIER_S_ITERS="4000"
TIER_L_NX="8192 16384";  TIER_L_ITERS="1500"
NUM_ITERS_CORTA="10 50 100 120"
CKPT_CORTA="5"
CKPT_VENTANA_S="500"
CKPT_VENTANA_L="250"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNDIR="${REPO_ROOT}/logs_holder_run"
STOP_FILE="${RUNDIR}/HOLDER_RUN_STOP"
mkdir -p "${RUNDIR}"
rm -f "${STOP_FILE}"
mkdir -p "${OUT_BASE}"
JOBS_TSV="${OUT_BASE}/JOBS.tsv"
[[ -s "${JOBS_TSV}" ]] || printf 'fecha\tgrupo\tnombre\tholder_step\testado\texport\n' > "${JOBS_TSV}"

estado() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
holder_vivo() { sacct -j "${JOBID}" --format=State -n -X 2>/dev/null | grep -q RUNNING; }

# Traduce el mismo `encolar()` de lanzar_campana_fase4.sh a un step srun.
correr() {   # correr <dir> <script> <grupo> <nombre> <VAR=val ...>
    local kdir_run="$1" kscript="$2" grupo="$3" nombre="$4"; shift 4
    local log="${RUNDIR}/v2_${grupo}_${nombre}.log"
    if [[ -f "${STOP_FILE}" ]]; then
        estado "HOLDER_RUN_STOP detectado -- se detiene la SECUENCIA (el holder ${JOBID} sigue vivo)."
        exit 0
    fi
    if ! holder_vivo; then
        estado "ALERTA: el holder ${JOBID} ya no esta RUNNING segun sacct -- se detiene la secuencia, sin tocar nada."
        exit 0
    fi
    local kv=(CAMPANA_STRICT=1 TC_FORMAT=both "ARCHIVE_DIR=/tmp/campana_v2_${grupo}_${nombre}" \
        "FINAL_RESULTS_DIR=${OUT_BASE%/}/${grupo}_${nombre}")
    kv+=("$@")
    estado "INICIA: v2_${grupo}_${nombre} (dir=${kdir_run} script=${kscript})"
    if srun --jobid="${JOBID}" --overlap --job-name="v2_${grupo}_${nombre}" \
         bash -c "cd '${kdir_run}' && exec env $(printf '%q ' "${kv[@]}") bash '${kscript}'" \
         > "${log}" 2>&1; then
        estado "OK: v2_${grupo}_${nombre} -> ${log}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${grupo}" "${nombre}" "${JOBID}" "OK" "${kv[*]}" >> "${JOBS_TSV}"
    else
        local rc=$?
        estado "FALLO (rc=${rc}): v2_${grupo}_${nombre} -> ${log} -- el holder ${JOBID} sigue vivo, se continua con el siguiente paso."
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${grupo}" "${nombre}" "${JOBID}" "FALLO(rc=${rc})" "${kv[*]}" >> "${JOBS_TSV}"
    fi
}

estado "=== Inicio de secuencia v2 dentro del holder ${JOBID} (PID driver=$$) ==="

for kern in ${KERNELS_RUN}; do
  case "${kern}" in
  stencil)
    KDIR_RUN="${REPO_ROOT}/Fase_4/Stencil"; KSCRIPT="run_stencil_tc.sbatch"
    OPFLAGS=(OP_MODE=diffusive "ALPHA=${ALPHA_CAMPANA}" "CI_MODE=${CI_MODE}")
    for g in ${GROUPS_RUN}; do
        case "${g}" in
            sp)   COMUN=(SPATIAL_COMP=on "ANCHOR_LIST=0 1 8 32" CPU_FP64=on) ;;
            off)  COMUN=(SPATIAL_COMP=off "KAHAN_LIST=off on" ANCHOR_LIST=0 CPU_FP64=off) ;;
            spk)  COMUN=(SPATIAL_COMP=on "ANCHOR_LIST=${K_STENCIL_FULL}" CPU_FP64=on) ;;
            kext) COMUN=(SPATIAL_COMP=on "ANCHOR_LIST=${K_STENCIL_EXT}" CPU_FP64=off) ;;
            *)    estado "grupo desconocido: ${g}"; exit 2 ;;
        esac
        correr "${KDIR_RUN}" "${KSCRIPT}" "${g}" num_corta "${OPFLAGS[@]}" "${COMUN[@]}" \
            RUN_KIND=numeric "NX_LIST=4096 8192 16384" "ITERS_LIST=${NUM_ITERS_CORTA}" "CHECKPOINT_EVERY=${CKPT_CORTA}"
        correr "${KDIR_RUN}" "${KSCRIPT}" "${g}" num_S "${OPFLAGS[@]}" "${COMUN[@]}" \
            RUN_KIND=numeric "NX_LIST=${TIER_S_NX}" "ITERS_LIST=${TIER_S_ITERS}" "CHECKPOINT_EVERY=${CKPT_VENTANA_S}"
        correr "${KDIR_RUN}" "${KSCRIPT}" "${g}" num_L "${OPFLAGS[@]}" "${COMUN[@]}" \
            RUN_KIND=numeric "NX_LIST=${TIER_L_NX}" "ITERS_LIST=${TIER_L_ITERS}" "CHECKPOINT_EVERY=${CKPT_VENTANA_L}"
        correr "${KDIR_RUN}" "${KSCRIPT}" "${g}" en_S "${OPFLAGS[@]}" "${COMUN[@]}" \
            RUN_KIND=energy "NX_LIST=${TIER_S_NX}" "ITERS_LIST=${TIER_S_ITERS}"
        correr "${KDIR_RUN}" "${KSCRIPT}" "${g}" en_L "${OPFLAGS[@]}" "${COMUN[@]}" \
            RUN_KIND=energy "NX_LIST=${TIER_L_NX}" "ITERS_LIST=${TIER_L_ITERS}"
    done ;;
  gemm)
    KDIR_RUN="${REPO_ROOT}/Fase_4/GEMM"; KSCRIPT="run_gemm_chained.sbatch"
    COMUN=("ANCHOR_LIST=${K_CHAINED_ALL}" "COMP_LIST=off on" RUN_NCU=0)
    correr "${KDIR_RUN}" "${KSCRIPT}" gemm num "${COMUN[@]}" RUN_KIND=numeric "N_LIST=1024 2048 4096 8192" "ITERS_LIST=20 40 80" CHECKPOINT_EVERY=5
    correr "${KDIR_RUN}" "${KSCRIPT}" gemm en_A "${COMUN[@]}" RUN_KIND=energy "N_LIST=1024 2048" "ITERS_LIST=24000"
    correr "${KDIR_RUN}" "${KSCRIPT}" gemm en_B "${COMUN[@]}" RUN_KIND=energy "N_LIST=4096 8192" "ITERS_LIST=500" ;;
  conv)
    KDIR_RUN="${REPO_ROOT}/Fase_4/Convolution"; KSCRIPT="run_conv_chained.sbatch"
    COMUN=("ANCHOR_LIST=${K_CHAINED_ALL}" "COMP_LIST=off on" RUN_NCU=0)
    correr "${KDIR_RUN}" "${KSCRIPT}" conv num "${COMUN[@]}" RUN_KIND=numeric "HW_LIST=64 128 256 512" "ITERS_LIST=20 40 80" CHECKPOINT_EVERY=5
    correr "${KDIR_RUN}" "${KSCRIPT}" conv en_A "${COMUN[@]}" RUN_KIND=energy "HW_LIST=64 128" "ITERS_LIST=37000"
    correr "${KDIR_RUN}" "${KSCRIPT}" conv en_B "${COMUN[@]}" RUN_KIND=energy "HW_LIST=256 512" "ITERS_LIST=2500" ;;
  *) estado "kernel desconocido: ${kern}"; exit 2 ;;
  esac
done

estado "=== Secuencia v2 completa dentro del holder ${JOBID}. Bitacora: ${JOBS_TSV} ==="
estado "Gate de campana: python3 tools/audit_coverage.py --results-dir ${OUT_BASE} --recursive --mode campaign --out ${OUT_BASE}/audit_report.md"
