#!/bin/bash
#SBATCH --job-name=fase1_conv_baseline
#SBATCH --output=logs/fase1_conv_baseline_%j.out
#SBATCH --error=logs/fase1_conv_baseline_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=24:00:00

# Baseline de Fase 1 (CPU vs cuDNN sin Tensor Cores) en PACCA.
# Reutiliza la misma resolucion de cuDNN que Fase_2/Convolution: en PACCA no hay
# cuDNN del sistema ni modulo, la unica instalacion compatible con CUDA 12 es la
# cuDNN 9.x (wheel cu12) del entorno Analytics.
#
#   sbatch run_conv_fase1.sbatch
#   sbatch --export=ALL,SMOKE_TEST=1 run_conv_fase1.sbatch
#   sbatch --export=ALL,CUDNN_ROOT=/otra/ruta run_conv_fase1.sbatch

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f cudnn_conv_balanced.cu && -f Convolution/cudnn_conv_balanced.cu ]]; then
    cd Convolution
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

if [[ -n "${OPENBLAS_DIR:-}" ]]; then
    OPENBLAS_INC="-I${OPENBLAS_DIR}/include"
    OPENBLAS_LIBS="-L${OPENBLAS_DIR}/lib -lopenblas"
else
    OPENBLAS_INC="$(normalize_include_flags "${OPENBLAS_INC:--I/usr/include/openblas}")"
    OPENBLAS_LIBS="${OPENBLAS_LIBS:--lopenblas}"
fi

CUDNN_ROOT="${CUDNN_ROOT:-/opt/ohpc/pub/Analytics/anaconda3/envs/hybrid-profiler/lib/python3.10/site-packages/nvidia/cudnn}"

if [[ -z "${CUDNN_INC:-}" && -f "${CUDNN_ROOT}/include/cudnn.h" ]]; then
    CUDNN_INC="-I${CUDNN_ROOT}/include"
else
    CUDNN_INC="$(normalize_include_flags "${CUDNN_INC:-}")"
fi

if [[ -z "${CUDNN_LIBS:-}" ]]; then
    if [[ -e "${CUDNN_ROOT}/lib/libcudnn.so" ]]; then
        CUDNN_LIBS="-L${CUDNN_ROOT}/lib -lcudnn"
    elif compgen -G "${CUDNN_ROOT}/lib/libcudnn.so.*" >/dev/null; then
        # El wheel de cuDNN 9 solo trae libcudnn.so.9 (sin el symlink .so), por
        # lo que -lcudnn fallaria: se enlaza el soname exacto con -l:.
        _cudnn_soname="$(basename "$(ls -1 "${CUDNN_ROOT}"/lib/libcudnn.so.* | sort -V | tail -1)")"
        CUDNN_LIBS="-L${CUDNN_ROOT}/lib -l:${_cudnn_soname}"
    else
        CUDNN_LIBS="-lcudnn"
    fi
fi

# cuDNN 9 despacha a sub-librerias (cnn/graph/ops...) del mismo stack cu12.
if [[ -d "${CUDNN_ROOT}/lib" ]]; then
    LD_LIBRARY_PATH="${CUDNN_ROOT}/lib:${LD_LIBRARY_PATH:-}"
    for _libdir in "${CUDNN_ROOT}"/../*/lib; do
        [[ -d "${_libdir}" ]] && LD_LIBRARY_PATH="${_libdir}:${LD_LIBRARY_PATH}"
    done
    export LD_LIBRARY_PATH
fi

SMOKE_TEST="${SMOKE_TEST:-0}"

echo "Nodo: $(hostname)"
echo "Fecha: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-no definido}"
echo "Arquitectura CUDA: sm_${CUDA_ARCH}"
echo "NVCC=${NVCC}"
echo "SMOKE_TEST=${SMOKE_TEST}"

nvidia-smi || true

echo "Compilando cudnn_conv_balanced..."
"${NVCC}" -std=c++17 -O3 cudnn_conv_balanced.cu -o cudnn_conv_balanced \
    ${OPENBLAS_INC} ${CUDNN_INC} ${CUDNN_LIBS} ${OPENBLAS_LIBS} \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    --allow-unsupported-compiler

# Mismas formas que las corridas de Fase 2, para comparar el baseline cuDNN FP32
# contra la ruta Tensor Core de Fase 2 sobre la misma convolucion.
# El baseline FP32 desactiva TF32 desde el codigo (cudnnSetConvolutionMathType
# con CUDNN_FMA_MATH en cudnn_conv_balanced.cu), no por NVIDIA_TF32_OVERRIDE:
# asi la linea base sigue siendo FP32 escalar aunque el binario se corra a mano
# fuera de este sbatch.
if [[ "${SMOKE_TEST}" == "1" ]]; then
    declare -a RUNS=(
        "1 64 64 64 64 3 3 1 1 1 1 1 1 2 fp32"
    )
else
    declare -a RUNS=(
        "1 1024 256 256 1024 3 3 1 1 1 1 1 1 10 fp32"
        "1 2048 256 256 2048 3 3 1 1 1 1 1 1 5 fp32"
    )
fi

for run in "${RUNS[@]}"; do
    read -r N C H W K R S PAD_H PAD_W STRIDE_H STRIDE_W DILATION_H DILATION_W ITERS PRECISION <<< "${run}"
    echo
    echo "============================================================"
    echo "Corrida: N=${N}, C=${C}, H=${H}, W=${W}, K=${K}, R=${R}, S=${S}, ITERS=${ITERS}, ${PRECISION}"
    echo "Padding: (${PAD_H}, ${PAD_W}), stride: (${STRIDE_H}, ${STRIDE_W}), dilation: (${DILATION_H}, ${DILATION_W})"
    echo "============================================================"
    # Se evita expandir un array vacio ("${DOUBLE_FLAG[@]}"): bajo `set -u` eso
    # aborta el job en bash < 4.4.
    if [[ "${PRECISION}" == "fp64" ]]; then
        ./cudnn_conv_balanced --double \
            --n "${N}" --c "${C}" --h "${H}" --w "${W}" \
            --k "${K}" --r "${R}" --s "${S}" \
            --pad_h "${PAD_H}" --pad_w "${PAD_W}" \
            --stride_h "${STRIDE_H}" --stride_w "${STRIDE_W}" \
            --dilation_h "${DILATION_H}" --dilation_w "${DILATION_W}" \
            --iters "${ITERS}"
    else
        ./cudnn_conv_balanced \
            --n "${N}" --c "${C}" --h "${H}" --w "${W}" \
            --k "${K}" --r "${R}" --s "${S}" \
            --pad_h "${PAD_H}" --pad_w "${PAD_W}" \
            --stride_h "${STRIDE_H}" --stride_w "${STRIDE_W}" \
            --dilation_h "${DILATION_H}" --dilation_w "${DILATION_W}" \
            --iters "${ITERS}"
    fi
done
