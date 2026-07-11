#!/bin/bash
#SBATCH --job-name=mixed_precision_stencil_tc
#SBATCH --output=logs/mixed_precision_stencil_tc_%j.out
#SBATCH --error=logs/mixed_precision_stencil_tc_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=48:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f stencil_tensor_activation.cu && -f Stencil/stencil_tensor_activation.cu ]]; then
    cd Stencil
fi
mkdir -p logs

# Sin dependencias externas: stencil_tc solo usa mma.h / cuda_fp16.h / cuda_bf16.h,
# por lo que no se cargan modulos (OpenBLAS/cuDNN no aplican aqui).

# Arquitectura por defecto para A100.
# Puedes sobrescribirla con:
#   sbatch --export=ALL,CUDA_ARCH=86 run_stencil_tc.sbatch
CUDA_ARCH="${CUDA_ARCH:-80}"  # A100=80, RTX 3050=86, V100=70

NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"
NCU="${NCU:-${HOME}/fcrojasv/nsight_compute/ncu}"

SMOKE_TEST="${SMOKE_TEST:-0}"
if [[ "${SMOKE_TEST}" == "1" ]]; then
    RUN_NCU="${RUN_NCU:-0}"
else
    RUN_NCU="${RUN_NCU:-1}"
fi
NCU_SET="${NCU_SET:-full}"
# El kernel WMMA es un template sobre T (__half / __nv_bfloat16); ncu filtra por
# nombre demangled, asi que este regex cubre ambas instanciaciones:
#   (anonymous namespace)::stencil2d_wmma_kernel<__half>(...)
#   (anonymous namespace)::stencil2d_wmma_kernel<__nv_bfloat16>(...)
NCU_KERNEL_REGEX="${NCU_KERNEL_REGEX:-.*stencil2d_wmma_kernel.*}"
# Debe coincidir con kWarmupIters en stencil_tensor_activation.cu.
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

echo "Compilando stencil_tc..."
"${NVCC}" -std=c++17 stencil_tensor_activation.cu -o stencil_tc \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    --allow-unsupported-compiler

# Validacion SASS pre-perfilado: si el binario no contiene instrucciones HMMA,
# las instanciaciones WMMA (FP16 y BF16 emiten ambas HMMA en Ampere) no
# generaron codigo Tensor Core y perfilar seria inutil. CUOBJDUMP se busca por
# PATH, junto a NVCC, o en el arbol cuda/ del HPC SDK (layout tipico
# .../<version>/cuda/bin/cuobjdump).
_nvcc_dir="$(dirname "${NVCC}")"
if [[ -z "${CUOBJDUMP:-}" ]]; then
    if command -v cuobjdump >/dev/null 2>&1; then
        CUOBJDUMP="$(command -v cuobjdump)"
    else
        for _cand in \
            "${_nvcc_dir}/cuobjdump" \
            "${_nvcc_dir}/../../cuda/bin/cuobjdump" \
            "${_nvcc_dir}"/../../cuda/*/bin/cuobjdump; do
            if [[ -x "${_cand}" ]]; then
                CUOBJDUMP="${_cand}"
                break
            fi
        done
    fi
fi

if [[ -x "${CUOBJDUMP:-}" ]] || command -v "${CUOBJDUMP:-}" >/dev/null 2>&1; then
    HMMA_COUNT=$("${CUOBJDUMP}" --dump-sass stencil_tc | grep -c HMMA || true)
    echo "Instrucciones HMMA detectadas en binario: ${HMMA_COUNT} (cuobjdump: ${CUOBJDUMP})"
    if [[ "${RUN_NCU}" == "1" && "${HMMA_COUNT}" -eq 0 ]]; then
        echo "ERROR: binario sin instrucciones HMMA; revisar la instanciacion" \
             "del kernel stencil2d_wmma_kernel antes de perfilar." >&2
        exit 1
    fi
else
    echo "AVISO: no se encontro cuobjdump; se omite la validacion HMMA." \
         "Exporta CUOBJDUMP=/ruta/a/cuobjdump para habilitarla." >&2
fi

echo "Ejecutando benchmarks de stencil 2D con Tensor Cores..."

if [[ "${SMOKE_TEST}" == "1" ]]; then
    declare -a RUNS=(
        "512 512 3 both"
    )
else
    # Dimensiones de referencia del proyecto (mismas del hint Nsight en el .cu).
    declare -a RUNS=(
        "4096 4096 20 both"
    )
fi

for run in "${RUNS[@]}"; do
    read -r NX NY ITERS TC <<< "${run}"
    echo
    echo "============================================================"
    echo "Corrida: NX=${NX}, NY=${NY}, ITERS=${ITERS}, TC=${TC}"
    echo "============================================================"
    ./stencil_tc --nx "${NX}" --ny "${NY}" --iters "${ITERS}" --tc "${TC}"

    if [[ "${RUN_NCU}" == "1" ]]; then
        # Con --tc both el binario lanza primero todos los kernels FP16 y luego
        # los BF16; como --launch-skip cuenta lanzamientos que matchean el regex
        # de forma global, un solo pase de ncu con skip=3 solo capturaria la
        # instanciacion FP16. Se perfila cada tipo por separado (--tc fp16 /
        # --tc bf16): un reporte por precision con skip/count deterministas.
        if [[ "${TC}" == "both" ]]; then
            NCU_TC_TYPES=(fp16 bf16)
        else
            NCU_TC_TYPES=("${TC}")
        fi
        if [[ "${NCU_MODE}" == "full" ]]; then
            NCU_CAPTURE_ARGS=(--set "${NCU_SET}")
        else
            NCU_CAPTURE_ARGS=(--metrics "${NCU_QUICK_METRICS}")
        fi
        for TC_TYPE in "${NCU_TC_TYPES[@]}"; do
            REPORT="logs/ncu_wmma_stencil_nx${NX}_ny${NY}_${TC_TYPE}_job${SLURM_JOB_ID:-manual}"
            echo
            echo "Perfilando WMMA ${TC_TYPE} con Nsight Compute (NCU_MODE=${NCU_MODE})..."
            echo "Reporte: ${REPORT}.ncu-rep"
            # `time` (y stderr de ncu) se redirigen al .out para correlacionar
            # duracion vs #SBATCH --time en corridas con --set full.
            { time "${NCU}" "${NCU_CAPTURE_ARGS[@]}" \
                --target-processes all \
                --kernel-name "regex:${NCU_KERNEL_REGEX}" \
                --launch-skip "${NCU_LAUNCH_SKIP}" \
                --launch-count "${NCU_LAUNCH_COUNT}" \
                --export "${REPORT}" \
                --force-overwrite \
                ./stencil_tc --nx "${NX}" --ny "${NY}" --iters "${ITERS}" --tc "${TC_TYPE}"; } 2>&1
        done
    fi
done
