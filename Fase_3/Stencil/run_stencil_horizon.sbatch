#!/bin/bash
#SBATCH --job-name=mixed_precision_stencil_horizon_f3
#SBATCH --output=logs/mixed_precision_stencil_horizon_f3_%j.out
#SBATCH --error=logs/mixed_precision_stencil_horizon_f3_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --exclusive
#SBATCH --time=01:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f stencil_tensor_activation.cu && -f Stencil/stencil_tensor_activation.cu ]]; then
    cd Stencil
fi
mkdir -p logs

# Job separado del barrido de metricas (run_stencil_tc.sbatch): ese barrido
# corre a 16384^2 dentro de la ventana finita (ITERS_LIST <= 120, todas las
# metricas L2/Linf/etc. comparables). Este job mide el horizonte de overflow
# REAL por formato (n*, impreso via "Horizonte de overflow (predicho /
# medido)" en stencil_tensor_activation.cu) a 4096^2 con iters=1200,
# suficiente para capturar incluso FP64 (prediccion n* ~1038 dada la
# componente Nyquist de la condicion inicial). A 16384^2 el mismo barrido
# costaria ~1038 x 7.4s de CPU FP64 encadenada (inviable); a 4096^2 (1/16 de
# los puntos) ese costo cabe holgado en la ventana de este job.
# No corre ncu: el objetivo es el texto de n* por stdout (primera iteracion
# no finita de cada ruta), no perfilar Tensor Cores -- para eso esta
# run_stencil_tc.sbatch.

# Arquitectura por defecto para A100.
# Puedes sobrescribirla con:
#   sbatch --export=ALL,CUDA_ARCH=86 run_stencil_horizon.sbatch
CUDA_ARCH="${CUDA_ARCH:-80}"  # A100=80, RTX 3050=86, V100=70

NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"

# Dimensiones y horizonte del barrido: NX/NY 16x mas chicos que el barrido de
# metricas para que la referencia CPU FP64 encadenada sea viable a
# ITERS=1200; TC=both corre FP16 y BF16 en la misma invocacion.
NX="${NX:-4096}"
NY="${NY:-4096}"
ITERS="${ITERS:-1200}"
TC="${TC:-both}"

echo "Nodo: $(hostname)"
echo "Fecha: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-no definido}"
echo "Arquitectura CUDA: sm_${CUDA_ARCH}"
echo "NVCC=${NVCC}"

nvidia-smi || true

echo "Compilando stencil_tc (Fase_3/Stencil, incluye Fase_2/common.cuh por ruta relativa)..."
"${NVCC}" -std=c++17 stencil_tensor_activation.cu -o stencil_tc \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
    --allow-unsupported-compiler

echo
echo "============================================================"
echo "Barrido de horizonte: NX=${NX}, NY=${NY}, ITERS=${ITERS}, TC=${TC}"
echo "============================================================"
TIMEFORMAT=">>> Tiempo real de la corrida: %R s (user %U, sys %S)"
{ time ./stencil_tc --nx "${NX}" --ny "${NY}" --iters "${ITERS}" --tc "${TC}"; } 2>&1
