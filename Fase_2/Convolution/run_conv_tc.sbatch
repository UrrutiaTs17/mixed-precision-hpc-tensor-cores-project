#!/bin/bash
#SBATCH --job-name=mixed_precision_conv_tc
#SBATCH --output=logs/mixed_precision_conv_tc_%j.out
#SBATCH --error=logs/mixed_precision_conv_tc_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=08:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f conv_tensor_activation.cu && -f Convolution/conv_tensor_activation.cu ]]; then
    cd Convolution
fi
mkdir -p logs

# Carga de modulos: en PACCA no hay modulo "cuda"; nvcc/ncu se toman por ruta.
if command -v module >/dev/null 2>&1; then
    module load openblas || true
fi

# Arquitectura por defecto para A100.
# Puedes sobrescribirla con:
#   sbatch --export=ALL,CUDA_ARCH=86 run_conv_tc.sbatch
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

OPENBLAS_INC="$(normalize_include_flags "${OPENBLAS_INC:--I/usr/include/openblas}")"
OPENBLAS_LIBS="${OPENBLAS_LIBS:--lopenblas}"
CUDNN_INC="$(normalize_include_flags "${CUDNN_INC:-}")"
CUDNN_LIBS="${CUDNN_LIBS:--lcudnn}"
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

echo "Compilando conv_tc..."
"${NVCC}" -std=c++17 conv_tensor_activation.cu -o conv_tc \
    ${OPENBLAS_INC} ${CUDNN_INC} ${CUDNN_LIBS} ${OPENBLAS_LIBS} \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    --allow-unsupported-compiler

echo "Ejecutando benchmarks de convolucion 2D con Tensor Cores..."

if [[ "${SMOKE_TEST}" == "1" ]]; then
    declare -a RUNS=(
        "1 32 32 32 32 3 3 1 1 1 1 1 1 2"
    )
else
    declare -a RUNS=(
        "1 32 128 128 64 3 3 1 1 1 1 1 1 20"
        "1 64 224 224 64 3 3 1 1 1 1 1 1 10"
    )
fi

for run in "${RUNS[@]}"; do
    read -r N C H W K R S PAD_H PAD_W STRIDE_H STRIDE_W DILATION_H DILATION_W ITERS <<< "${run}"
    echo
    echo "============================================================"
    echo "Corrida: N=${N}, C=${C}, H=${H}, W=${W}, K=${K}, R=${R}, S=${S}, ITERS=${ITERS}"
    echo "Padding: (${PAD_H}, ${PAD_W}), stride: (${STRIDE_H}, ${STRIDE_W}), dilation: (${DILATION_H}, ${DILATION_W})"
    echo "============================================================"
    ./conv_tc \
        --N "${N}" --C "${C}" --H "${H}" --W "${W}" \
        --K "${K}" --R "${R}" --S "${S}" \
        --pad_h "${PAD_H}" --pad_w "${PAD_W}" \
        --stride_h "${STRIDE_H}" --stride_w "${STRIDE_W}" \
        --dilation_h "${DILATION_H}" --dilation_w "${DILATION_W}" \
        --iters "${ITERS}"

    if [[ "${RUN_NCU}" == "1" ]]; then
        REPORT="logs/ncu_wmma_conv_N${N}_C${C}_H${H}_W${W}_K${K}_R${R}_S${S}_job${SLURM_JOB_ID:-manual}"
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
            ./conv_tc \
                --N "${N}" --C "${C}" --H "${H}" --W "${W}" \
                --K "${K}" --R "${R}" --S "${S}" \
                --pad_h "${PAD_H}" --pad_w "${PAD_W}" \
                --stride_h "${STRIDE_H}" --stride_w "${STRIDE_W}" \
                --dilation_h "${DILATION_H}" --dilation_w "${DILATION_W}" \
                --iters "${ITERS}"
    fi
done
