#!/bin/bash
#SBATCH --job-name=f4_anclaje_reflinf
#SBATCH --output=logs/f4_anclaje_%j.out
#SBATCH --error=logs/f4_anclaje_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=01:00:00
# ===========================================================================
# ANCLAJE de la extrapolacion de ref_linf al tamano REAL de la campana.
# ===========================================================================
# Punto unico: 16384^2, diffusive, monomode, ci_p=168, alpha=3/16, 640 iters.
#
# Por que existe: la escalera 1024/2048/4096/8192 a p/nx constante reprodujo
# g^640 con desviacion <=0.01% en los cuatro puntos, pero 16384^2 nunca corrio.
# Esto cierra la extrapolacion con un punto medido antes de comprometer las
# replicas de la campana.
#
# ADVERTENCIA: el binario NO tiene modo "solo referencia CPU". No existe
# bandera para saltarse las rutas GPU. Lo mas cerca que se puede dejar es:
#   --fp64-gpu off   quita la ruta GPU_FP64 (la mas cara en VRAM: 8 B/celda)
#   --cpu-fp64 off   quita la SEGUNDA pasada FP64 de CPU (la de cronometraje);
#                    el ground truth FP64 se calcula igual, es lo que se mide
#   --tc fp16        una sola ruta WMMA en vez de las dos
# Con eso siguen corriendo CPU_FP32, GPU_FP32 y WMMA_FP16. Es el minimo
# alcanzable sin tocar el .cu, y el .cu esta congelado para este anclaje.
#
# Costo esperado, extrapolado del punto de 8192^2 medido en local
# (174 s de pared, 3.4 GB de RSS): ~4x en celdas -> ~12 min y ~14 GB de RSS.
# El --time=01:00:00 y los 128G son holgura, no estimacion.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
JOB_ID="${SLURM_JOB_ID:-manual}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
SRC_DIR="${REPO_ROOT}/Fase_4/Stencil"
NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"
CUDA_ARCH="${CUDA_ARCH:-80}"

BIN="${PWD}/results/stencil_anclaje_${JOB_ID}"
echo "Compilando desde ${SRC_DIR} ..."
( cd "${SRC_DIR}" && "${NVCC}" -std=c++17 stencil_tensor_activation.cu -o "${BIN}" \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
    --allow-unsupported-compiler -lpthread )

echo "=============== PROCEDENCIA ==============="
echo "git_commit : $(git -C "${REPO_ROOT}" rev-parse HEAD)"
for f in "${SRC_DIR}/stencil_tensor_activation.cu" "${REPO_ROOT}/Fase_2/common.cuh" \
         "${SRC_DIR}/tools/power_sampling.h"; do
    echo "sha256     : $(sha256sum "${f}" | awk '{print $1}')  ${f#${REPO_ROOT}/}"
done
echo "nvcc       : ${NVCC}   arch: sm_${CUDA_ARCH}"
echo "==========================================="

CSV="${PWD}/results/anclaje_reflinf_${JOB_ID}.csv"
/usr/bin/time -v "${BIN}" \
    --nx 16384 --ny 16384 --iters 640 --tc fp16 \
    --op-mode diffusive --alpha 0.1875 \
    --ci-mode monomode --ci-p 168 --ci-amplitude 1.0 \
    --fp64-gpu off --cpu-fp64 off \
    --csv "${CSV}"

echo
echo "############ CRITERIO DE ACEPTACION ############"
echo "g^640 predicho para 16384^2 / ci_p=168 : 3.691e-01"
echo "ref_linf medido:"
awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="ref_linf") c=i; next} {print "   "$c}' "${CSV}"
echo "Se acepta si cae en [3.6e-01, 3.8e-01]. Un colapso de ordenes de"
echo "magnitud (p.ej. 1e-08) refutaria la extrapolacion y hay que parar."
