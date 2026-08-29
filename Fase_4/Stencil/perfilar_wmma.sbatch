#!/bin/bash
#SBATCH --job-name=f4_perfil_wmma
#SBATCH --output=logs/f4_perfil_wmma_%j.out
#SBATCH --error=logs/f4_perfil_wmma_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --exclusive
#SBATCH --time=01:30:00

# ===========================================================================
# PERFILADO: por que el kernel WMMA se queda al ~41 % del ancho de banda
# mientras GPU_FP32 llega al ~75 %.
#
# La campana de alineacion + vectorizacion (jobs 6719-6729) cerro solo entre
# 2.7 % y 7.5 % de ese hueco. La hipotesis microarquitectonica de la alineacion
# esta MEDIDA y es insuficiente: la causa esta en otro sitio y este job existe
# para encontrarla con contadores, no con razonamiento.
#
# Se perfila el kernel WMMA y, en la MISMA corrida, stencil2d_fp32_kernel como
# referencia: la pregunta no es "cuanto rinde WMMA" sino "en que se diferencia
# de la ruta que si satura".
#
# Metricas: se reutilizan los tres grupos ya definidos y probados en
# tools/common_ncu.sh (quick, coalescing, warp stalls). El cuarto grupo
# -- conflictos de banco y limitantes de ocupancia -- es NUEVO y por eso va en
# una invocacion aparte y tolerante a fallo: si algun nombre de metrica no
# existe en esta version de ncu, se pierde ese grupo y no el job entero.
#
# Uso:
#   sbatch perfilar_wmma.sbatch
#   sbatch --export=ALL,NX=16384 perfilar_wmma.sbatch
# ===========================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

REPO_ROOT="$(git rev-parse --show-toplevel)"
CUDA_ARCH="${CUDA_ARCH:-80}"
NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"
NX="${NX:-4096}"
ITERS="${ITERS:-10}"

JOB_ID="${SLURM_JOB_ID:-manual}"
WORK="perfil_wmma_${JOB_ID}"
mkdir -p "${WORK}/csv" "${WORK}/salidas"

echo "Nodo        : $(hostname)"
echo "Fecha       : $(date -Iseconds)"
echo "Commit      : $(git -C "${REPO_ROOT}" rev-parse HEAD)"
echo "Malla       : ${NX}^2 · ${ITERS} iteraciones"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

# --- ncu: misma convencion de ruta que run_stencil_tc.sbatch de Fase_3 -------
NCU="${NCU:-${HOME}/fcrojasv/nsight_compute/ncu}"
if [[ ! -x "${NCU}" ]]; then
    _d="$(dirname "${NVCC}")"
    for cand in "$(command -v ncu || true)" "${_d}/ncu" \
                "${_d}/../../cuda/bin/ncu" "${_d}"/../../cuda/*/bin/ncu \
                /opt/nvidia/nsight-compute/*/ncu; do
        if [[ -n "${cand}" && -x "${cand}" ]]; then NCU="${cand}"; break; fi
    done
fi
if [[ ! -x "${NCU}" ]]; then
    echo "ERROR: no se encontro ncu. Exporta NCU=/ruta/a/ncu y relanza."
    exit 1
fi
echo "ncu: ${NCU}  ($("${NCU}" --version 2>/dev/null | head -1 || echo version desconocida))"

# Definiciones compartidas: NCU_QUICK_METRICS, NCU_WARP_STALL_METRICS,
# NCU_COALESCING_METRICS y ncu_run(). Se declaran una sola vez en tools/.
for _c in "${REPO_ROOT}/tools/common_ncu.sh" "../../tools/common_ncu.sh"; do
    if [[ -f "${_c}" ]]; then . "${_c}"; COMMON_NCU="${_c}"; break; fi
done
if [[ -z "${COMMON_NCU:-}" ]]; then echo "ERROR: no se encontro tools/common_ncu.sh"; exit 1; fi
echo "metricas: ${COMMON_NCU}"
echo

# Grupo NUEVO. Responde dos preguntas que los otros tres no cubren:
#   - conflictos de banco en shared: la vectorizacion introdujo accesos de
#     128 bits a shared (LDS.128 del acumulador). Si generan conflictos, parte
#     del beneficio se esta pagando ahi.
#   - que limita la ocupancia: shared, registros, warps o bloques. El kernel
#     WMMA usa 32-40 registros y 6656/11776 B de shared por bloque; saber cual
#     es el techo decide si vale la pena tocar el reparto de shared.
NCU_EXTRA_METRICS="l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps,launch__occupancy_limit_blocks,launch__waves_per_multiprocessor,gpu__time_duration.sum"

NVML_LINK_FLAG="${NVML_LINK_FLAG:--lnvidia-ml}"
if "${NVCC}" -E -x cu - 2>/dev/null <<< '#include <nvml.h>' >/dev/null &&
   (ldconfig -p 2>/dev/null | grep -Eq 'lib(nvidia-ml|nvml)\.so' ||
    [[ -f /usr/lib64/libnvidia-ml.so || -f /usr/lib64/libnvml.so ]]); then
    NVML_FLAGS=(-DUSE_NVML_TELEMETRY "${NVML_LINK_FLAG}" -lpthread)
else
    NVML_FLAGS=(-lpthread)
fi

# ===========================================================================
# ETAPA 1 — Binario
# ===========================================================================
echo "############################################################"
echo "# ETAPA 1: compilacion"
echo "############################################################"
BIN="${PWD}/${WORK}/stencil_tc"
( cd "${REPO_ROOT}/Fase_3/Stencil" && "${NVCC}" -std=c++17 stencil_tensor_activation.cu -o "${BIN}" \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
    --allow-unsupported-compiler "${NVML_FLAGS[@]}" )
echo "  sha256: $(sha256sum "${BIN}" | awk '{print $1}')"
echo

# ===========================================================================
# ETAPA 2 — Perfilado
# ===========================================================================
echo "############################################################"
echo "# ETAPA 2: perfilado"
echo "############################################################"
# Se saltan las kWarmupIters=3 primeras invocaciones y se perfila UNA sola en
# regimen estacionario: perfilar un warm-up mediria el arranque, no el kernel.
SKIP="${NCU_LAUNCH_SKIP:-3}"
BASE_ARGS=(--nx "${NX}" --ny "${NX}" --iters "${ITERS}" --profile-only
           --op-mode diffusive --alpha 0.1875 --ci-mode monomode --ci-p 5)

perfilar() {   # $1 = etiqueta, $2 = regex de kernel, $3.. = flags del binario
    local etq="$1" regex="$2"; shift 2
    local pasos=0 ok=0
    for grupo in QUICK STALL EXTRA; do
        case "${grupo}" in
            QUICK) MET="${NCU_QUICK_METRICS},${NCU_COALESCING_METRICS}" ;;
            STALL) MET="${NCU_WARP_STALL_METRICS}" ;;
            EXTRA) MET="${NCU_EXTRA_METRICS}" ;;
        esac
        pasos=$((pasos + 1))
        local csv="${PWD}/${WORK}/csv/${etq}_${grupo}.csv"
        if ncu_run "${NCU}" --csv --page raw \
                --target-processes all \
                --kernel-name "regex:${regex}" \
                --launch-skip "${SKIP}" --launch-count 1 \
                --metrics "${MET}" \
                "${BIN}" "$@" > "${csv}" 2> "${PWD}/${WORK}/salidas/${etq}_${grupo}.err"; then
            ok=$((ok + 1))
        else
            echo "     AVISO: grupo ${grupo} fallo en ${etq} (ver salidas/${etq}_${grupo}.err)"
            rm -f "${csv}"
        fi
    done
    echo "  ${etq}: ${ok}/${pasos} grupos de metricas"
}

RX_WMMA=".*stencil2d_wmma_kernel.*"
RX_FP32=".*stencil2d_fp32_kernel.*"

# GPU_FP32 es la REFERENCIA: la ruta que si llega al 75 %. Se perfila desde la
# misma invocacion del binario, con la misma malla y el mismo operador.
perfilar "FP32_ref"     "${RX_FP32}" "${BASE_ARGS[@]}" --tc fp16
perfilar "WMMA_FP16_off"     "${RX_WMMA}" "${BASE_ARGS[@]}" --tc fp16
perfilar "WMMA_BF16_off"     "${RX_WMMA}" "${BASE_ARGS[@]}" --tc bf16
perfilar "WMMA_FP16_local"   "${RX_WMMA}" "${BASE_ARGS[@]}" --tc fp16 --kahan on
perfilar "WMMA_FP16_spatial" "${RX_WMMA}" "${BASE_ARGS[@]}" --tc fp16 --spatial-comp on
echo

# ===========================================================================
# ETAPA 3 — Tabla comparativa
# ===========================================================================
echo "############################################################"
echo "# ETAPA 3: comparativa (FP32 es la referencia que SI satura)"
echo "############################################################"
# El resumen vive en tools/resumir_ncu.py y NO empotrado aqui, para que un
# fallo de parseo se arregle y se re-ejecute sobre los CSV ya generados sin
# repetir el perfilado. Paso en el job 6730: ncu escribio los cinco casos
# correctamente y la tabla salio vacia porque el parser esperaba el formato
# largo (--page details) y --page raw emite el ancho.
RESUMIDOR="${REPO_ROOT}/Fase_4/Stencil/tools/resumir_ncu.py"
if [[ -f "${RESUMIDOR}" ]]; then
    python3 "${RESUMIDOR}" "${PWD}/${WORK}/csv" || {
        echo "  El resumidor fallo. Los CSV siguen intactos; reintenta con:"
        echo "    python3 ${RESUMIDOR} ${PWD}/${WORK}/csv --debug"
    }
else
    echo "  AVISO: no se encontro ${RESUMIDOR}; los CSV crudos quedan sin resumir."
fi

echo
echo "Como leerla:"
echo "  La columna FP32_ref es la ruta que llega al ~75 %. Cada fila donde WMMA"
echo "  se separa mucho de ella es una hipotesis sobre el 41 %."
echo "  - DRAM % bajo + long_scoreboard alto -> latency-bound, no"
echo "    bandwidth-bound: el kernel espera, no satura. La solucion seria mas"
echo "    paralelismo en vuelo, no menos bytes."
echo "  - sectores/request > 4 en ld/st -> accesos aun descoalescidos."
echo "  - barrier alto -> los __syncwarp() del tile dominan; el reparto"
echo "    1 warp = 1 tile seria el sospechoso."
echo "  - conflictos de banco altos -> el precio de los accesos de 128 bits a"
echo "    shared que introdujo la vectorizacion."
echo "  - 'limite ocup.' mas bajo = el recurso que la esta capando."

echo
echo "CSV crudos: ${PWD}/${WORK}/csv/"
