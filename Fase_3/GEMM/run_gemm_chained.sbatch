#!/bin/bash
#SBATCH --job-name=mixed_precision_gemm_chained_f3
#SBATCH --output=logs/mixed_precision_gemm_chained_f3_%j.out
#SBATCH --error=logs/mixed_precision_gemm_chained_f3_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=24:00:00

# Lanzador parametrizado de gemm_chained.cu (Fase 3: encadenamiento genuino
# X(n+1)=X(n)*A, ver README.md de esta carpeta). Ningun tamano, formato o
# politica de compensacion queda hardcodeado -- todo se controla con
# variables de entorno via --export, con los mismos defaults documentados
# abajo si no se pasa nada.
#
#   sbatch run_gemm_chained.sbatch
#   sbatch --export=ALL,N_LIST="512 1024 2048 4096",COMP_LIST="off on" run_gemm_chained.sbatch
#   sbatch --export=ALL,TC_FORMAT=fp16,CHECKPOINT_EVERY=1 run_gemm_chained.sbatch
#
# Mismo patron de deteccion de NVML/toolchain que Fase_4/Stencil/run_stencil_tc.sbatch
# -- ver ese archivo si algo de esta seccion inicial no es obvio.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f gemm_chained.cu && -f GEMM/gemm_chained.cu ]]; then
    cd GEMM
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

# NVML es opcional -- ver common/power_sampling.h. Mismo bloque de deteccion
# que Fase_4/Stencil/run_stencil_tc.sbatch, copiado tal cual (no depende de
# que kernel se compile).
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

# --- Barrido: tamanos, formato, compensacion ---------------------------------
N_LIST="${N_LIST:-1024}"                 # cada valor debe ser potencia de 2
ITERS="${ITERS:-20}"
TC_FORMAT="${TC_FORMAT:-both}"           # fp16 | bf16 | both
COMP_LIST="${COMP_LIST:-off on}"         # compensacion por linealidad
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
SEED="${SEED:-42}"
TARGET_LAMBDA="${TARGET_LAMBDA:-1.1}"

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

echo "Compilando gemm_chained (Fase_3/GEMM, incluye common/ por ruta relativa)..."
"${NVCC}" -std=c++17 gemm_chained.cu -o gemm_chained \
    -lcublas -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
    --allow-unsupported-compiler "${NVML_FLAGS[@]}"

# Procedencia del binario/corrida -- mismo patron que Fase_4/Stencil.
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo desconocido)"
if git diff --quiet 2>/dev/null; then GIT_STATUS="limpio"; else GIT_STATUS="sucio"; fi
BUILD_DATE="$(date -Iseconds)"
BIN_SHA256="$(sha256sum gemm_chained | awk '{print $1}')"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "GIT_STATUS=${GIT_STATUS}"
echo "BUILD_DATE=${BUILD_DATE}"
echo "BIN_SHA256=${BIN_SHA256}"
echo "RUN_PARAMS=N_LIST=${N_LIST} ITERS=${ITERS} TC_FORMAT=${TC_FORMAT} COMP_LIST=${COMP_LIST} CHECKPOINT_EVERY=${CHECKPOINT_EVERY} SEED=${SEED} TARGET_LAMBDA=${TARGET_LAMBDA}"

# Validacion SASS pre-perfilado -- mismo patron que Fase_4/Stencil.
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
    HMMA_COUNT=$("${CUOBJDUMP}" --dump-sass gemm_chained | grep -c HMMA || true)
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

echo "Ejecutando gemm_chained (Fase 3: encadenamiento genuino, sin ancla)..."

{
for N in ${N_LIST}; do
    for COMP in ${COMP_LIST}; do
        echo
        echo "============================================================"
        echo "Corrida: N=${N}, ITERS=${ITERS}, TC=${TC_FORMAT}, COMP=${COMP}"
        echo "============================================================"
        ./gemm_chained --n "${N}" --iters "${ITERS}" --tc "${TC_FORMAT}" \
            --comp "${COMP}" --checkpoint-every "${CHECKPOINT_EVERY}" \
            --seed "${SEED}" --target-lambda "${TARGET_LAMBDA}"

        if [[ "${RUN_NCU}" == "1" ]]; then
            if [[ "${NCU_MODE}" == "full" ]]; then
                NCU_CAPTURE_ARGS=(--set "${NCU_SET}")
            else
                NCU_CAPTURE_ARGS=(--metrics "${NCU_QUICK_METRICS:-}")
            fi
            REPORT="${LOGS_DIR}/ncu_gemm_n${N}_comp${COMP}_job${SLURM_JOB_ID:-manual}"
            echo "Perfilando wmma_gemm_kernel con Nsight Compute (N=${N}, COMP=${COMP})..."
            ncu_run "${NCU}" "${NCU_CAPTURE_ARGS[@]}" --target-processes all \
                --kernel-name "regex:${NCU_KERNEL_REGEX}" \
                --launch-count 1 --export "${REPORT}" --force-overwrite \
                ./gemm_chained --n "${N}" --iters 1 --tc "${TC_FORMAT}" --comp "${COMP}"
        fi
    done
done
} 2>&1 | tee -a "${RESULTS_LOG}"

# Post-proceso de CSV: extract_csv_chained.py (esquema propio de gemm_chained/
# conv_chained, no el de Stencil -- ver Fase_3/tools/README.md). Se busca en
# varias rutas candidatas y, si no se encuentra, se avisa y se omite en vez de
# fallar el job -- el log crudo queda disponible para post-procesar despues.
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
        --input "${RESULTS_LOG}" --outdir "${RESULTS_DIR}" --job-id "${JOB_ID}" --kernel gemm
else
    echo
    echo "AVISO: extract_csv_chained.py no encontrado (se probo ../tools/, tools/ y" \
         "../../tools/ relativos a esta carpeta). Se omite la extraccion; el log crudo" \
         "con los tokens CSV_* queda en ${RESULTS_LOG}." >&2
fi
