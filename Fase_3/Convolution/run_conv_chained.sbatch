#!/bin/bash
#SBATCH --job-name=mixed_precision_conv_chained_f3
#SBATCH --output=logs/mixed_precision_conv_chained_f3_%j.out
#SBATCH --error=logs/mixed_precision_conv_chained_f3_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=24:00:00

# Lanzador parametrizado de conv_chained.cu (Fase 3: encadenamiento genuino
# X(n+1)=conv(X(n),W), 64 canales bloque-diagonal, ver README.md de esta
# carpeta). Mismo patron que Fase_3/GEMM/run_gemm_chained.sbatch -- nada
# hardcodeado, todo via --export.
#
#   sbatch run_conv_chained.sbatch
#   sbatch --export=ALL,HW_LIST="64 128 256",COMP_LIST="off on" run_conv_chained.sbatch
#   sbatch --export=ALL,TC_FORMAT=fp16,CHECKPOINT_EVERY=1 run_conv_chained.sbatch

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f conv_chained.cu && -f Convolution/conv_chained.cu ]]; then
    cd Convolution
fi

LOGS_DIR="${LOGS_DIR:-logs}"
RESULTS_DIR="${RESULTS_DIR:-results}"
mkdir -p "${LOGS_DIR}" "${RESULTS_DIR}"
JOB_ID="${SLURM_JOB_ID:-manual}"
RESULTS_LOG="${RESULTS_DIR}/run_${JOB_ID}.log"

# --- Deteccion portable del toolchain (tools/detect_toolchain.sh) --------
# Corre igual bajo SLURM en PACCA o con `bash` directo en cualquier otra
# maquina con el entorno conda de environment.yml activo -- ver el
# comentario de cabecera de ese script.
for _toolchain in "../../tools/detect_toolchain.sh" "./detect_toolchain.sh"; do
    if [[ -f "${_toolchain}" ]]; then
        # shellcheck source=/dev/null
        source "${_toolchain}"
        break
    fi
done
CUDA_ARCH="$(detect_cuda_arch)"  # A100=80, RTX 3050=86, V100=70, Ada=89, Hopper=90
NVCC="$(detect_nvcc)"
NCU="$(detect_ncu || true)"
if [[ -z "${NCU}" ]]; then
    echo "AVISO: Nsight Compute (ncu) no encontrado -- se deshabilita el perfilado (RUN_NCU forzado a 0)." >&2
    RUN_NCU=0
fi

# NVML opcional -- mismo bloque de deteccion que el resto del proyecto.
NVML_FLAGS=()
NVML_LINK_FLAG="${NVML_LINK_FLAG:--lnvidia-ml}"
if [[ "${DISABLE_NVML_TELEMETRY:-0}" != "1" ]] &&
   "${NVCC}" -E -x cu - 2>/dev/null <<< '#include <nvml.h>' >/dev/null &&
   (ldconfig -p 2>/dev/null | grep -Eq 'lib(nvidia-ml|nvml)\.so' ||
    [[ -f /usr/lib64/libnvidia-ml.so || -f /usr/lib64/libnvml.so ||
       -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so ||
       -f /usr/lib/x86_64-linux-gnu/libnvml.so ]]); then
    NVML_FLAGS=(-DUSE_NVML_TELEMETRY "${NVML_LINK_FLAG}" -lpthread)
    echo "NVML habilitado para telemetria GPU."
else
    NVML_FLAGS=(-lpthread)
    echo "NVML no disponible para el enlace; se usara el stub de telemetria GPU."
fi

# --- Barrido: tamano espacial, formato, compensacion --------------------------
# HW debe hacer que hw*hw sea multiplo de 64 (dimension N del GEMM
# subyacente via im2col) -- 64, 128, 192, 256... funcionan (ver validacion en
# parse_args de conv_chained.cu).
HW_LIST="${HW_LIST:-64}"
ITERS="${ITERS:-20}"
TC_FORMAT="${TC_FORMAT:-both}"          # fp16 | bf16 | both
COMP_LIST="${COMP_LIST:-off on}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
SEED="${SEED:-42}"

# --- Perfilado NCU opcional (un solo kernel de interes: wmma_gemm_kernel) ---
RUN_NCU="${RUN_NCU:-0}"
NCU_MODE="${NCU_MODE:-quick}"
NCU_SET="${NCU_SET:-full}"
NCU_KERNEL_REGEX="${NCU_KERNEL_REGEX:-.*wmma_gemm_kernel.*}"

echo "Nodo: $(hostname)"
echo "Fecha: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-no definido}"
echo "Arquitectura CUDA: sm_${CUDA_ARCH}"
echo "NVCC=${NVCC}"
echo "RUN_NCU=${RUN_NCU}"

nvidia-smi || true

echo "Compilando conv_chained (Fase_3/Convolution, incluye common/ por ruta relativa)..."
"${NVCC}" -std=c++17 conv_chained.cu -o conv_chained \
    -lcublas -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
    --allow-unsupported-compiler "${NVML_FLAGS[@]}"

GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo desconocido)"
if git diff --quiet 2>/dev/null; then GIT_STATUS="limpio"; else GIT_STATUS="sucio"; fi
BUILD_DATE="$(date -Iseconds)"
BIN_SHA256="$(sha256sum conv_chained | awk '{print $1}')"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "GIT_STATUS=${GIT_STATUS}"
echo "BUILD_DATE=${BUILD_DATE}"
echo "BIN_SHA256=${BIN_SHA256}"
echo "RUN_PARAMS=HW_LIST=${HW_LIST} ITERS=${ITERS} TC_FORMAT=${TC_FORMAT} COMP_LIST=${COMP_LIST} CHECKPOINT_EVERY=${CHECKPOINT_EVERY} SEED=${SEED}"

_nvcc_dir="$(dirname "${NVCC}")"
if [[ -z "${CUOBJDUMP:-}" ]]; then
    if command -v cuobjdump >/dev/null 2>&1; then
        CUOBJDUMP="$(command -v cuobjdump)"
    else
        for _cand in "${_nvcc_dir}/cuobjdump" "${_nvcc_dir}/../../cuda/bin/cuobjdump" \
                     "${_nvcc_dir}"/../../cuda/*/bin/cuobjdump; do
            [[ -x "${_cand}" ]] && { CUOBJDUMP="${_cand}"; break; }
        done
    fi
fi
if [[ -x "${CUOBJDUMP:-}" ]] || command -v "${CUOBJDUMP:-}" >/dev/null 2>&1; then
    HMMA_COUNT=$("${CUOBJDUMP}" --dump-sass conv_chained | grep -c HMMA || true)
    echo "Instrucciones HMMA detectadas en binario: ${HMMA_COUNT} (cuobjdump: ${CUOBJDUMP})"
    if [[ "${RUN_NCU}" == "1" && "${HMMA_COUNT}" -eq 0 ]]; then
        echo "ERROR: binario sin instrucciones HMMA; revisar wmma_gemm_kernel antes de perfilar." >&2
        exit 1
    fi
else
    echo "AVISO: no se encontro cuobjdump; se omite la validacion HMMA." >&2
fi

for _common_ncu in "../../tools/common_ncu.sh" "./common_ncu.sh"; do
    if [[ -f "${_common_ncu}" ]]; then
        # shellcheck source=/dev/null
        source "${_common_ncu}"
        break
    fi
done

echo "Ejecutando conv_chained (Fase 3: encadenamiento genuino, sin ancla)..."

{
for HW in ${HW_LIST}; do
    for COMP in ${COMP_LIST}; do
        echo
        echo "============================================================"
        echo "Corrida: HW=${HW}, ITERS=${ITERS}, TC=${TC_FORMAT}, COMP=${COMP}"
        echo "============================================================"
        ./conv_chained --hw "${HW}" --iters "${ITERS}" --tc "${TC_FORMAT}" \
            --comp "${COMP}" --checkpoint-every "${CHECKPOINT_EVERY}" --seed "${SEED}"

        if [[ "${RUN_NCU}" == "1" ]]; then
            if [[ "${NCU_MODE}" == "full" ]]; then
                NCU_CAPTURE_ARGS=(--set "${NCU_SET}")
            else
                NCU_CAPTURE_ARGS=(--metrics "${NCU_QUICK_METRICS:-}")
            fi
            REPORT="${LOGS_DIR}/ncu_conv_hw${HW}_comp${COMP}_job${SLURM_JOB_ID:-manual}"
            echo "Perfilando wmma_gemm_kernel con Nsight Compute (HW=${HW}, COMP=${COMP})..."
            ncu_run "${NCU}" "${NCU_CAPTURE_ARGS[@]}" --target-processes all \
                --kernel-name "regex:${NCU_KERNEL_REGEX}" \
                --launch-count 1 --export "${REPORT}" --force-overwrite \
                ./conv_chained --hw "${HW}" --iters 1 --tc "${TC_FORMAT}" --comp "${COMP}"
        fi
    done
done
} 2>&1 | tee -a "${RESULTS_LOG}"

EXTRACT_CSV_SCRIPT="${EXTRACT_CSV_SCRIPT:-}"
if [[ -z "${EXTRACT_CSV_SCRIPT}" ]]; then
    for _cand in "../tools/extract_csv_chained.py" "tools/extract_csv_chained.py" \
                 "../../tools/extract_csv_chained.py"; do
        if [[ -f "${_cand}" ]]; then
            EXTRACT_CSV_SCRIPT="${_cand}"
            break
        fi
    done
fi

if [[ -n "${EXTRACT_CSV_SCRIPT}" ]]; then
    echo
    echo "Extrayendo CSVs desde ${RESULTS_LOG} (via ${EXTRACT_CSV_SCRIPT})..."
    python3 "${EXTRACT_CSV_SCRIPT}" \
        --input "${RESULTS_LOG}" --outdir "${RESULTS_DIR}" --job-id "${JOB_ID}" --kernel conv
else
    echo
    echo "AVISO: extract_csv_chained.py no encontrado (se probo ../tools/, tools/ y" \
         "../../tools/ relativos a esta carpeta). Se omite la extraccion; el log crudo" \
         "con los tokens CSV_* queda en ${RESULTS_LOG}." >&2
fi
