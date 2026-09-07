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
# Sin dependencias externas: stencil_baseline solo usa cuda_runtime.
#
# Nada de lo que se barre esta fijo en el script: tamanos, iteraciones y
# precisiones se controlan por variable de entorno (patron VAR="${VAR:-default}").
# La campana reportable usa el mismo conjunto dimensionado para PACCA que
# Fase 3/4. El tamano reducido queda exclusivamente en SMOKE_TEST.
#
#   sbatch run_stencil_fase1.sbatch
#   sbatch --export=ALL,SMOKE_TEST=1 run_stencil_fase1.sbatch
#   sbatch --export=ALL,CUDA_ARCH=86 run_stencil_fase1.sbatch
#   sbatch --export=ALL,STENCIL_SIZES="4096 8192 16384",STENCIL_ITERS=30 run_stencil_fase1.sbatch
#   sbatch --export=ALL,STENCIL_PRECISIONS=fp32 run_stencil_fase1.sbatch

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f stencil_baseline.cu && -f Stencil/stencil_baseline.cu ]]; then
    cd Stencil
fi

# Directorio de salida para logs/artefactos generados por este script (no
# confundir con #SBATCH --output/--error de arriba: SLURM resuelve esas dos
# rutas al momento de encolar el job, antes de que el script corra, asi que
# no pueden depender de una variable definida aqui dentro).
OUT_DIR="${OUT_DIR:-logs}"
mkdir -p "${OUT_DIR}"

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

SMOKE_TEST="${SMOKE_TEST:-0}"

# Tamanos NX=NY (grilla cuadrada) a barrer, separados por espacio. Es el mismo
# conjunto dimensionado para la A100 de PACCA que usan Fase 3 y Fase 4.
STENCIL_SIZES="${STENCIL_SIZES:-4096 8192 16384}"

# Iteraciones promediadas por corrida (--iters de stencil_baseline).
STENCIL_ITERS="${STENCIL_ITERS:-20}"

# Precisiones a correr por cada tamano: "fp32", "fp64", o ambas separadas
# por espacio (default). fp32 = sin --double; fp64 = con --double.
STENCIL_PRECISIONS="${STENCIL_PRECISIONS:-fp32 fp64}"

echo "Nodo: $(hostname)"
echo "Fecha: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-no definido}"
echo "Arquitectura CUDA: sm_${CUDA_ARCH}"
echo "NVCC=${NVCC}"
echo "SMOKE_TEST=${SMOKE_TEST}"
echo "OUT_DIR=${OUT_DIR}"
echo "STENCIL_SIZES=${STENCIL_SIZES}"
echo "STENCIL_ITERS=${STENCIL_ITERS}"
echo "STENCIL_PRECISIONS=${STENCIL_PRECISIONS}"

nvidia-smi || true

echo "Compilando stencil_baseline..."
"${NVCC}" -std=c++17 -O3 stencil_baseline.cu -o stencil_baseline \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    --allow-unsupported-compiler

# SMOKE_TEST=1 corre un unico tamano pequeno para validar rapido que el
# binario compila y ejecuta (por ejemplo tras cambiar de nodo/arquitectura),
# sin esperar el barrido completo.
if [[ "${SMOKE_TEST}" == "1" ]]; then
    RUN_SIZES="512"
    RUN_ITERS=3
    RUN_PRECISIONS="fp32"
else
    RUN_SIZES="${STENCIL_SIZES}"
    RUN_ITERS="${STENCIL_ITERS}"
    RUN_PRECISIONS="${STENCIL_PRECISIONS}"
fi

for SIZE in ${RUN_SIZES}; do
    for PRECISION in ${RUN_PRECISIONS}; do
        echo
        echo "============================================================"
        echo "Corrida: NX=${SIZE}, NY=${SIZE}, ITERS=${RUN_ITERS}, ${PRECISION}"
        echo "============================================================"
        if [[ "${PRECISION}" == "fp64" ]]; then
            ./stencil_baseline --double --nx "${SIZE}" --ny "${SIZE}" --iters "${RUN_ITERS}"
        else
            ./stencil_baseline --nx "${SIZE}" --ny "${SIZE}" --iters "${RUN_ITERS}"
        fi
    done
done
