#!/bin/bash
#SBATCH --job-name=fase1_gemm_baseline
#SBATCH --output=logs/fase1_gemm_baseline_%j.out
#SBATCH --error=logs/fase1_gemm_baseline_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=24:00:00

# Baseline de Fase 1 (CPU BLAS vs cuBLAS clasico, sin Tensor Cores) en PACCA.
# Mismo patron de resolucion de toolchain que los .sbatch de Fase 2: nvcc y las
# librerias se toman por ruta, no por "module load cuda" (no existe en PACCA).
#
#   sbatch run_gemm_fase1.sbatch
#   sbatch --export=ALL,SMOKE_TEST=1 run_gemm_fase1.sbatch
#   sbatch --export=ALL,CUDA_ARCH=86 run_gemm_fase1.sbatch

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f gemm_compare_balanced.cu && -f GEMM/gemm_compare_balanced.cu ]]; then
    cd GEMM
fi
mkdir -p logs

if command -v module >/dev/null 2>&1; then
    module load openblas || true
fi

CUDA_ARCH="${CUDA_ARCH:-80}"  # A100=80, RTX 3050=86, V100=70
NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"

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

# En OpenHPC el modulo define OPENBLAS_DIR; derivar los flags de ahi evita que
# la ruta cruda llegue a nvcc sin -I.
if [[ -n "${OPENBLAS_DIR:-}" ]]; then
    OPENBLAS_INC="-I${OPENBLAS_DIR}/include"
    OPENBLAS_LIBS="-L${OPENBLAS_DIR}/lib -lopenblas"
else
    OPENBLAS_INC="$(normalize_include_flags "${OPENBLAS_INC:--I/usr/include/openblas}")"
    OPENBLAS_LIBS="${OPENBLAS_LIBS:--lopenblas}"
fi

SMOKE_TEST="${SMOKE_TEST:-0}"

echo "Nodo: $(hostname)"
echo "Fecha: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-no definido}"
echo "Arquitectura CUDA: sm_${CUDA_ARCH}"
echo "NVCC=${NVCC}"
echo "SMOKE_TEST=${SMOKE_TEST}"

nvidia-smi || true

echo "Compilando gemm_compare_balanced..."
"${NVCC}" -std=c++17 -O3 gemm_compare_balanced.cu -o gemm_compare_balanced \
    ${OPENBLAS_INC} -lcublas ${OPENBLAS_LIBS} \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    --allow-unsupported-compiler

# Dimensiones alineadas con las corridas de Fase 2 para que los baselines de
# ambas fases sean directamente comparables.
# FP64 a 32768 si cabe: las tres matrices ocupan 3*32768^2*8 B = 25.8 GiB de los
# 39.5 GiB de la A100 (y 34.4 GiB en host contando A, B, C_cpu y C_gpu, dentro
# de los 128 G pedidos). El costo real es la referencia OpenBLAS: 7.04e13 FLOP
# por iteracion, ~1 min por iteracion en 16 nucleos, ~35 min las 30 iteraciones.
if [[ "${SMOKE_TEST}" == "1" ]]; then
    declare -a RUNS=(
        "1024 1024 1024 2 fp32"
    )
else
    declare -a RUNS=(
        "12288 12288 12288 30 fp32"
        "12288 12288 12288 30 fp64"
        "32768 32768 32768 30 fp32"
        "32768 32768 32768 30 fp64"
    )
fi

for run in "${RUNS[@]}"; do
    read -r M N K ITERS PRECISION <<< "${run}"
    echo
    echo "============================================================"
    echo "Corrida: M=${M}, N=${N}, K=${K}, ITERS=${ITERS}, ${PRECISION}"
    echo "============================================================"
    if [[ "${PRECISION}" == "fp64" ]]; then
        ./gemm_compare_balanced --double \
            --m "${M}" --n "${N}" --k "${K}" --iters "${ITERS}"
    else
        ./gemm_compare_balanced \
            --m "${M}" --n "${N}" --k "${K}" --iters "${ITERS}"
    fi
done
