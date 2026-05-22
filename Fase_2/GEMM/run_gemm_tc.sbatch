#!/bin/bash
#SBATCH --job-name=mixed_precision_gemm_tc
#SBATCH --output=logs/mixed_precision_gemm_tc_%j.out
#SBATCH --error=logs/mixed_precision_gemm_tc_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=08:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f gemm_tensor_activation.cu && -f GEMM/gemm_tensor_activation.cu ]]; then
    cd GEMM
fi
mkdir -p logs

# Carga de modulos: en PACCA no hay modulo "cuda"; nvcc/ncu se toman por ruta.
if command -v module >/dev/null 2>&1; then
    module load openblas || true
fi

# Arquitectura por defecto para A100.
# Puedes sobrescribirla con:
#   sbatch --export=ALL,CUDA_ARCH=86 run_gemm_tc.sbatch
CUDA_ARCH="${CUDA_ARCH:-80}"  # A100=80, RTX 3050=86, V100=70

NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"
NCU="${NCU:-${HOME}/fcrojasv/nsight_compute/ncu}"
normalize_include_flags() {
    local raw="${1:-}"
    local item
    local normalized=()

    for item in ${raw}; do
        if [[ "${item}" == -I* || "${item}" == -* ]]; then
            normalized+=("${item}")
        else
            normalized+=("-I${item}")
        fi
    done

    if [[ "${#normalized[@]}" -gt 0 ]]; then
        printf '%s ' "${normalized[@]}"
    fi
}

# On OpenHPC, OPENBLAS_DIR is reliably set by the module; derive flags from it
# to avoid getting the raw path passed to nvcc without -I.
if [[ -n "${OPENBLAS_DIR:-}" ]]; then
    OPENBLAS_INC="-I${OPENBLAS_DIR}/include"
    OPENBLAS_LIBS="-L${OPENBLAS_DIR}/lib -lopenblas"
else
    OPENBLAS_INC="$(normalize_include_flags "${OPENBLAS_INC:--I/usr/include/openblas}")"
    OPENBLAS_LIBS="${OPENBLAS_LIBS:--lopenblas}"
fi
SMOKE_TEST="${SMOKE_TEST:-0}"
if [[ "${SMOKE_TEST}" == "1" ]]; then
    RUN_NCU="${RUN_NCU:-0}"
else
    RUN_NCU="${RUN_NCU:-1}"
fi
NCU_SET="${NCU_SET:-full}"
NCU_KERNEL_REGEX="${NCU_KERNEL_REGEX:-.*wmma_gemm_kernel.*}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-3}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-1}"

echo "Nodo: $(hostname)"
echo "Fecha: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-no definido}"
echo "Arquitectura CUDA: sm_${CUDA_ARCH}"
echo "NVCC=${NVCC}"
echo "NCU=${NCU}"
echo "SMOKE_TEST=${SMOKE_TEST}"
echo "RUN_NCU=${RUN_NCU}"

nvidia-smi || true

echo "Compilando gemm_tc..."
"${NVCC}" -std=c++17 gemm_tensor_activation.cu -o gemm_tc \
    ${OPENBLAS_INC} -lcublas ${OPENBLAS_LIBS} \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    --allow-unsupported-compiler

echo "Ejecutando benchmarks registrados en Primeras_mediciones_rendimiento_PACCA.txt..."

if [[ "${SMOKE_TEST}" == "1" ]]; then
    declare -a RUNS=(
        "1024 1024 1024 2"
    )
else
    declare -a RUNS=(
        "12288 12288 12288 30"
        "32768 32768 32768 30"
    )
fi

for run in "${RUNS[@]}"; do
    read -r M N K ITERS <<< "${run}"
    echo
    echo "============================================================"
    echo "Corrida: M=${M}, N=${N}, K=${K}, ITERS=${ITERS}"
    echo "============================================================"
    ./gemm_tc --m "${M}" --n "${N}" --k "${K}" --iters "${ITERS}"

    if [[ "${RUN_NCU}" == "1" ]]; then
        REPORT="logs/ncu_wmma_m${M}_n${N}_k${K}_job${SLURM_JOB_ID:-manual}"
        echo
        echo "Perfilando WMMA con Nsight Compute..."
        echo "Reporte: ${REPORT}.ncu-rep"
        "${NCU}" --set "${NCU_SET}" \
            --target-processes all \
            --kernel-name "regex:${NCU_KERNEL_REGEX}" \
            --launch-skip "${NCU_LAUNCH_SKIP}" \
            --launch-count "${NCU_LAUNCH_COUNT}" \
            --export "${REPORT}" \
            --force-overwrite \
            ./gemm_tc --m "${M}" --n "${N}" --k "${K}" --iters "${ITERS}"
    fi
done
