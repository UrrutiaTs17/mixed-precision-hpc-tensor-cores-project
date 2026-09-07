#!/bin/bash
#SBATCH --job-name=fase1_stencil_baseline
#SBATCH --output=logs/fase1_stencil_baseline_%j.out
#SBATCH --error=logs/fase1_stencil_baseline_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=12:00:00

# Baseline de Fase 1 (CPU FP32/FP64 vs GPU CUDA clasico, sin Tensor Cores).
# Sin dependencias externas: stencil2d_baseline solo usa cuda_runtime.
#
#   sbatch run_stencil_fase1.sbatch
#   sbatch --export=ALL,SMOKE_TEST=1 run_stencil_fase1.sbatch
#   sbatch --export=ALL,CUDA_ARCH=86 run_stencil_fase1.sbatch

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f stencil2d_baseline.cu && -f Stencil2D/stencil2d_baseline.cu ]]; then
    cd Stencil2D
fi
mkdir -p logs

CUDA_ARCH="${CUDA_ARCH:-80}"  # A100=80, RTX 3050=86, V100=70
NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"

SMOKE_TEST="${SMOKE_TEST:-0}"

echo "Nodo: $(hostname)"
echo "Fecha: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-no definido}"
echo "Arquitectura CUDA: sm_${CUDA_ARCH}"
echo "NVCC=${NVCC}"
echo "SMOKE_TEST=${SMOKE_TEST}"

nvidia-smi || true

echo "Compilando stencil2d_baseline..."
"${NVCC}" -std=c++17 -O3 stencil2d_baseline.cu -o stencil2d_baseline \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    --allow-unsupported-compiler

# 4096x4096 es la malla de referencia del proyecto: la misma que usan los
# .sbatch de Fase 2 y Fase 3, para que los tiempos sean comparables entre fases.
if [[ "${SMOKE_TEST}" == "1" ]]; then
    declare -a RUNS=(
        "512 512 3 fp32"
    )
else
    declare -a RUNS=(
        "4096 4096 20 fp32"
        "4096 4096 20 fp64"
    )
fi

for run in "${RUNS[@]}"; do
    read -r NX NY ITERS PRECISION <<< "${run}"
    echo
    echo "============================================================"
    echo "Corrida: NX=${NX}, NY=${NY}, ITERS=${ITERS}, ${PRECISION}"
    echo "============================================================"
    if [[ "${PRECISION}" == "fp64" ]]; then
        ./stencil2d_baseline --double --nx "${NX}" --ny "${NY}" --iters "${ITERS}"
    else
        ./stencil2d_baseline --nx "${NX}" --ny "${NY}" --iters "${ITERS}"
    fi
done
