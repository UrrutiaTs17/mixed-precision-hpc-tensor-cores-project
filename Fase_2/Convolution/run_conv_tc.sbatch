#!/bin/bash
#SBATCH --job-name=mixed_precision_conv_tc
#SBATCH --output=logs/mixed_precision_conv_tc_%j.out
#SBATCH --error=logs/mixed_precision_conv_tc_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=24:00:00

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

# On OpenHPC, OPENBLAS_DIR is reliably set by the module; derive flags from it
# to avoid getting the raw path passed to nvcc without -I.
if [[ -n "${OPENBLAS_DIR:-}" ]]; then
    OPENBLAS_INC="-I${OPENBLAS_DIR}/include"
    OPENBLAS_LIBS="-L${OPENBLAS_DIR}/lib -lopenblas"
else
    OPENBLAS_INC="$(normalize_include_flags "${OPENBLAS_INC:--I/usr/include/openblas}")"
    OPENBLAS_LIBS="${OPENBLAS_LIBS:--lopenblas}"
fi
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

# Definiciones compartidas de perfilado (NCU_QUICK_METRICS). Se declara una
# sola vez en Fase_2/common_ncu.sh y se reutiliza entre los .sbatch.
for _common_ncu in "../common_ncu.sh" "./common_ncu.sh"; do
    if [[ -f "${_common_ncu}" ]]; then
        # shellcheck source=/dev/null
        source "${_common_ncu}"
        break
    fi
done

# Separacion validacion/caracterizacion en NCU:
#   quick -> --metrics "${NCU_QUICK_METRICS}" (rapido, valida Tensor Cores)
#   full  -> --set "${NCU_SET}"               (caracterizacion completa)
NCU_MODE="${NCU_MODE:-quick}"
TIMEFORMAT=">>> Tiempo real de la llamada ncu (NCU_MODE=${NCU_MODE}): %R s (user %U, sys %S)"

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

# Validacion SASS pre-perfilado: si el binario no contiene instrucciones HMMA,
# el kernel WMMA no se activo (tipicamente por dimensiones no divisibles) y
# perfilar seria inutil. CUOBJDUMP se resuelve primero por PATH y, si no esta,
# se asume junto a NVCC (tipico en instalaciones HPC SDK).
CUOBJDUMP="${CUOBJDUMP:-$(command -v cuobjdump || echo "$(dirname "${NVCC}")/cuobjdump")}"
HMMA_COUNT=$("${CUOBJDUMP}" --dump-sass conv_tc | grep -c HMMA || true)
echo "Instrucciones HMMA detectadas en binario: ${HMMA_COUNT}"
if [[ "${RUN_NCU}" == "1" && "${HMMA_COUNT}" -eq 0 ]]; then
    echo "ERROR: binario sin instrucciones HMMA; revisar condiciones" \
         "de divisibilidad M/N/K antes de perfilar." >&2
    exit 1
fi

echo "Ejecutando benchmarks de convolucion 2D con Tensor Cores..."

if [[ "${SMOKE_TEST}" == "1" ]]; then
    # N=1, C=64, H=W=64, K=64: outH*outW=4096 (mult 64), K=64 (mult 64),
    # C*R*S=576 (mult 32) -> WMMA path activa.
    declare -a RUNS=(
        "1 64 64 64 64 3 3 1 1 1 1 1 1 2"
    )
else
    # N=1 en ambas corridas: con launch-skip=3, NCU salta exactamente los 3
    # warmups del benchmark (1 llamada wmma_gemm_kernel por iteracion con N=1).
    # Run 1: M=1024, N=65536, K=9216  -> ~1236 GFLOPs -> ~6.6 ms por kernel en A100
    # Run 2: M=2048, N=65536, K=18432 -> ~4948 GFLOPs -> ~26.5 ms por kernel en A100
    declare -a RUNS=(
        "1 1024 256 256 1024 3 3 1 1 1 1 1 1 10"
        "1 2048 256 256 2048 3 3 1 1 1 1 1 1 5"
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
        # Con batch N>1 el benchmark lanza N invocaciones del kernel WMMA por
        # iteracion, por lo que hay que saltar WARMUP_ITERS*N lanzamientos para
        # no perfilar los warmups. WARMUP_ITERS debe coincidir con kWarmupIters
        # en conv_tensor_activation.cu.
        WARMUP_ITERS=3
        NCU_LAUNCH_SKIP_RUN=$((WARMUP_ITERS * N))
        if [[ "${NCU_MODE}" == "full" ]]; then
            NCU_CAPTURE_ARGS=(--set "${NCU_SET}")
        else
            NCU_CAPTURE_ARGS=(--metrics "${NCU_QUICK_METRICS}")
        fi
        echo
        echo "Perfilando WMMA con Nsight Compute (NCU_MODE=${NCU_MODE}, launch-skip=${NCU_LAUNCH_SKIP_RUN})..."
        echo "Reporte: ${REPORT}.ncu-rep"
        # `time` (y stderr de ncu) se redirigen al .out para correlacionar
        # duracion vs #SBATCH --time en corridas con --set full.
        { time "${NCU}" "${NCU_CAPTURE_ARGS[@]}" \
            --target-processes all \
            --kernel-name "regex:${NCU_KERNEL_REGEX}" \
            --launch-skip "${NCU_LAUNCH_SKIP_RUN}" \
            --launch-count "${NCU_LAUNCH_COUNT}" \
            --export "${REPORT}" \
            --force-overwrite \
            ./conv_tc \
                --N "${N}" --C "${C}" --H "${H}" --W "${W}" \
                --K "${K}" --R "${R}" --S "${S}" \
                --pad_h "${PAD_H}" --pad_w "${PAD_W}" \
                --stride_h "${STRIDE_H}" --stride_w "${STRIDE_W}" \
                --dilation_h "${DILATION_H}" --dilation_w "${DILATION_W}" \
                --iters "${ITERS}"; } 2>&1
    fi
done
