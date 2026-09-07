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

# Fase 2 Tensor Core (WMMA) para Stencil 2D. Nada de lo que se barre esta
# fijo en el script: tamanos, formato Tensor Core e iteraciones se controlan
# por variable de entorno (patron VAR="${VAR:-default}"). La campana reportable
# usa el mismo conjunto dimensionado para PACCA que Fase 3/4; el tamano
# reducido queda exclusivamente en SMOKE_TEST.
#
#   sbatch run_stencil_tc.sbatch
#   sbatch --export=ALL,SMOKE_TEST=1 run_stencil_tc.sbatch
#   sbatch --export=ALL,CUDA_ARCH=86 run_stencil_tc.sbatch
#   sbatch --export=ALL,STENCIL_SIZES="4096 8192 16384",STENCIL_ITERS=30 run_stencil_tc.sbatch
#   sbatch --export=ALL,STENCIL_TC_MODES=fp16 run_stencil_tc.sbatch
#   sbatch --export=ALL,RUN_NCU=0 run_stencil_tc.sbatch

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f stencil_tensor_activation.cu && -f Stencil/stencil_tensor_activation.cu ]]; then
    cd Stencil
fi

# Directorio de salida para logs/reportes generados por este script (no
# confundir con #SBATCH --output/--error de arriba: SLURM resuelve esas dos
# rutas al momento de encolar el job, antes de que el script corra, asi que
# no pueden depender de una variable definida aqui dentro).
OUT_DIR="${OUT_DIR:-logs}"
mkdir -p "${OUT_DIR}"

# Sin dependencias externas: stencil_tc solo usa mma.h / cuda_fp16.h / cuda_bf16.h,
# por lo que no se cargan modulos (OpenBLAS/cuDNN no aplican aqui).

# Arquitectura por defecto para A100.
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
NCU="$(detect_ncu || true)"
if [[ -z "${NCU}" ]]; then
    echo "AVISO: Nsight Compute (ncu) no encontrado -- se deshabilita el perfilado (RUN_NCU forzado a 0)." >&2
    RUN_NCU=0
fi

# Tamanos NX=NY (grilla cuadrada) a barrer, separados por espacio. Es el mismo
# conjunto dimensionado para la A100 de PACCA que usan Fase 3 y Fase 4.
STENCIL_SIZES="${STENCIL_SIZES:-4096 8192 16384}"

# Iteraciones promediadas por corrida (--iters de stencil_tc).
STENCIL_ITERS="${STENCIL_ITERS:-20}"

# Modos Tensor Core a barrer, separados por espacio: "fp16", "bf16", "both"
# (fp16+bf16 en una sola invocacion) o una combinacion, p. ej. "fp16 bf16"
# para forzar dos invocaciones separadas en vez de una con --tc both.
STENCIL_TC_MODES="${STENCIL_TC_MODES:-both}"

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
# sola vez en tools/common_ncu.sh (fuera de cualquier Fase_N) y se reutiliza
# entre los .sbatch de todas las fases; este script corre desde
# Fase_2/Stencil, de ahi "../../tools/common_ncu.sh" como ruta principal.
for _common_ncu in "../../tools/common_ncu.sh" "./common_ncu.sh"; do
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
echo "OUT_DIR=${OUT_DIR}"
echo "STENCIL_SIZES=${STENCIL_SIZES}"
echo "STENCIL_ITERS=${STENCIL_ITERS}"
echo "STENCIL_TC_MODES=${STENCIL_TC_MODES}"

nvidia-smi || true

echo "Compilando stencil_tc..."
# NOTA MIGRACION (actualizada): el .sbatch original de Fase 2
# (old/Fase_2/Stencil/run_stencil_tc.sbatch) compilaba sin -O3, a diferencia
# de Fase 1 -- eso hacia correr el bucle "CPU FP32 serial" (host, escalar)
# ~25x mas lento de lo real en este entorno, inflando artificialmente todos
# los "Speedup vs CPU"/"Speedup TC * vs CPU" que imprime el binario (no
# afectaba la correctitud numerica, solo el rendimiento reportado de la
# referencia CPU). Se agrega -O3 aqui para volver a Fase 2 consistente con
# Fase 1: cualquier "Speedup vs CPU" generado ANTES de este cambio queda
# obsoleto y debe volver a correrse.
"${NVCC}" -std=c++17 -O3 stencil_tensor_activation.cu -o stencil_tc \
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

# SMOKE_TEST=1 corre un unico tamano pequeno con --tc both para validar
# rapido que el binario compila y ejecuta, sin esperar el barrido completo
# ni el perfilado NCU (RUN_NCU se fuerza a 0 arriba salvo que se sobrescriba).
if [[ "${SMOKE_TEST}" == "1" ]]; then
    RUN_SIZES="512"
    RUN_ITERS=3
    RUN_TC_MODES="both"
else
    RUN_SIZES="${STENCIL_SIZES}"
    RUN_ITERS="${STENCIL_ITERS}"
    RUN_TC_MODES="${STENCIL_TC_MODES}"
fi

for SIZE in ${RUN_SIZES}; do
    for TC in ${RUN_TC_MODES}; do
        echo
        echo "============================================================"
        echo "Corrida: NX=${SIZE}, NY=${SIZE}, ITERS=${RUN_ITERS}, TC=${TC}"
        echo "============================================================"
        ./stencil_tc --nx "${SIZE}" --ny "${SIZE}" --iters "${RUN_ITERS}" --tc "${TC}"

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
                REPORT="${OUT_DIR}/ncu_wmma_stencil_nx${SIZE}_ny${SIZE}_${TC_TYPE}_job${SLURM_JOB_ID:-manual}"
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
                    ./stencil_tc --nx "${SIZE}" --ny "${SIZE}" --iters "${RUN_ITERS}" --tc "${TC_TYPE}"; } 2>&1
            done
        fi
    done
done
