#!/bin/bash
#SBATCH --job-name=mixed_precision_stencil_tc_f3
#SBATCH --output=logs/mixed_precision_stencil_tc_f3_%j.out
#SBATCH --error=logs/mixed_precision_stencil_tc_f3_%j.err
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
# Se perfilan tres kernels en invocaciones ncu SEPARADAS (un regex cada una):
# el WMMA (template sobre T = __half / __nv_bfloat16; ncu filtra por nombre
# demangled, asi que el regex cubre ambas instanciaciones), el GPU FP32
# clasico y las conversiones float<->16 bits que corren dentro de la ruta WMMA.
NCU_KERNEL_REGEX_WMMA="${NCU_KERNEL_REGEX_WMMA:-.*stencil2d_wmma_kernel.*}"
NCU_KERNEL_REGEX_FP32="${NCU_KERNEL_REGEX_FP32:-.*stencil2d_fp32_kernel.*}"
NCU_KERNEL_REGEX_CONVERT="${NCU_KERNEL_REGEX_CONVERT:-.*convert_float_to_(half|bfloat16)_kernel.*}"
# Debe coincidir con kWarmupIters en stencil_tensor_activation.cu (Fase_3/Stencil);
# se deriva del propio fuente en vez de repetir el literal para que ambos no
# puedan desincronizarse si kWarmupIters cambia.
KWARMUP_ITERS_CU="$(grep -m1 'kWarmupIters *=' stencil_tensor_activation.cu | grep -oE '[0-9]+' | head -n1)"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-${KWARMUP_ITERS_CU}}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-1}"
# Barrido de METRICAS (16384^2): 10..120 caen dentro del horizonte de
# FP32/BF16/FP16 (produce L2/Linf finitas). Se elimino 200: a este tamano
# divergen las tres rutas GPU (n* FP32~142, BF16~138, FP16~28) y toda la
# corrida sale "REFERENCIA NO FINITA" sin ningun dato comparable -- era la
# corrida mas cara del job sin aportar nada. El horizonte de divergencia en
# si (donde SI importa iters grande) se mide aparte, a 4096^2, en
# run_stencil_horizon.sbatch.
ITERS_LIST="${ITERS_LIST:-10 50 100 120}"
# Necesario para que storage_rel (error relativo de guardar en FP16/BF16) sea
# evaluable en las corridas donde la ruta diverge antes de ITERS (p.ej. FP16
# con n*~28 en corridas con ITERS>28): sin --checkpoint-every, stencil_tc solo
# puede capturar un estado FP32 finito en la ULTIMA iteracion medida, y si esa
# iteracion ya diverguio no hay forma de recuperar un estado finito posterior
# al hecho (ver comentario storage_rel_eval_iter en
# benchmark_gpu_tensor_core_stencil, stencil_tensor_activation.cu). 5 deja
# checkpoints en 5,10,15,20,25,... -- el ultimo antes de n*~28 (FP16, el
# formato que diverge mas temprano) cae en 25, bien dentro del rango finito.
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
# Restriccion de tiempo: NCU solo se corre para un valor de ITERS de la lista,
# no para todos -- mantiene el numero de reportes NCU en 5 por corrida (WMMA
# fp16/bf16, conversion fp16/bf16, FP32 clasico una sola vez).
ITERS_NCU="${ITERS_NCU:-100}"

# Definiciones compartidas de perfilado (NCU_QUICK_METRICS). Se declara una
# sola vez en Fase_2/common_ncu.sh y se reutiliza entre los .sbatch de todas
# las fases; este script corre desde Fase_3/Stencil, asi que "../common_ncu.sh"
# (Fase_3/common_ncu.sh) y "./common_ncu.sh" no existen aqui -- se agrega la
# ruta real hacia Fase_2/common_ncu.sh como ultima alternativa.
for _common_ncu in "../common_ncu.sh" "./common_ncu.sh" "../../Fase_2/common_ncu.sh"; do
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

echo "Compilando stencil_tc (Fase_3/Stencil, incluye Fase_2/common.cuh por ruta relativa)..."
"${NVCC}" -std=c++17 stencil_tensor_activation.cu -o stencil_tc \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
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

echo "Ejecutando benchmarks de stencil 2D con Tensor Cores (Fase 3: encadenamiento genuino)..."

if [[ "${SMOKE_TEST}" == "1" ]]; then
    declare -a RUNS=(
        "512 512 3 both"
    )
else
    # Dimensiones de referencia del proyecto (mismas del hint Nsight en el .cu),
    # barridas sobre ITERS_LIST dentro de la ventana finita (10..120): todas
    # las metricas (L2/Linf/gflops/etc.) quedan comparables entre corridas.
    declare -a RUNS=()
    for ITERS_VAL in ${ITERS_LIST}; do
        RUNS+=("16384 16384 ${ITERS_VAL} both")
    done
fi

# Perfila UN solo kernel (regex + tc_type que lo activa) con --profile-only en
# el binario: evita recalcular la referencia CPU FP32 (~742s) y la FP64
# encadenada (~900s) antes de llegar al kernel bajo perfil (~1723s de pared
# por llamada a ncu antes de este cambio). Usa NX/NY/ITERS/NCU_CAPTURE_ARGS
# del ambito que la invoca (bucle principal, mas abajo).
ncu_profile_kernel() {
    local label="$1" regex="$2" tc_type="$3" report_tag="$4"
    local report="logs/ncu_${report_tag}_nx${NX}_ny${NY}_job${SLURM_JOB_ID:-manual}"
    echo
    echo "Perfilando ${label} con Nsight Compute (NCU_MODE=${NCU_MODE})..."
    echo "Reporte: ${report}.ncu-rep"
    # `time` (y stderr de ncu) se redirigen al .out para correlacionar
    # duracion vs #SBATCH --time en corridas con --set full.
    { time ncu_run "${NCU}" "${NCU_CAPTURE_ARGS[@]}" \
        --target-processes all \
        --kernel-name "regex:${regex}" \
        --launch-skip "${NCU_LAUNCH_SKIP}" \
        --launch-count "${NCU_LAUNCH_COUNT}" \
        --export "${report}" \
        --force-overwrite \
        ./stencil_tc --nx "${NX}" --ny "${NY}" --iters "${ITERS}" --tc "${tc_type}" \
            --profile-only; } 2>&1
}

for run in "${RUNS[@]}"; do
    read -r NX NY ITERS TC <<< "${run}"
    echo
    echo "============================================================"
    echo "Corrida: NX=${NX}, NY=${NY}, ITERS=${ITERS}, TC=${TC}"
    echo "============================================================"
    ./stencil_tc --nx "${NX}" --ny "${NY}" --iters "${ITERS}" --tc "${TC}" \
        --checkpoint-every "${CHECKPOINT_EVERY}"

    if [[ "${RUN_NCU}" == "1" && "${ITERS}" == "${ITERS_NCU}" ]]; then
        # Con --tc both el binario lanza primero todos los kernels FP16 y luego
        # los BF16; como --launch-skip cuenta lanzamientos que matchean el regex
        # de forma global, un solo pase de ncu con skip=3 solo capturaria la
        # instanciacion FP16. Se perfila cada tipo por separado (--tc fp16 /
        # --tc bf16): un reporte por precision con skip/count deterministas.
        # --profile-only exige --tc fp16|bf16 (nunca both), asi que este split
        # ya es obligatorio, no solo una optimizacion de --launch-skip.
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

        # GPU FP32 clasico no depende de la precision TC: perfilarlo una vez
        # por corrida (no por TC_TYPE) evita duplicar tiempo de pared sin
        # informacion nueva. El valor de --tc que se le pasa es irrelevante
        # para ese kernel; se usa el primero de NCU_TC_TYPES.
        ncu_profile_kernel "GPU FP32 clasico" "${NCU_KERNEL_REGEX_FP32}" \
            "${NCU_TC_TYPES[0]}" "fp32_stencil"

        for TC_TYPE in "${NCU_TC_TYPES[@]}"; do
            ncu_profile_kernel "WMMA ${TC_TYPE}" "${NCU_KERNEL_REGEX_WMMA}" \
                "${TC_TYPE}" "wmma_stencil_${TC_TYPE}"
            ncu_profile_kernel "conversion ${TC_TYPE}" "${NCU_KERNEL_REGEX_CONVERT}" \
                "${TC_TYPE}" "convert_${TC_TYPE}"
        done
    fi
done
