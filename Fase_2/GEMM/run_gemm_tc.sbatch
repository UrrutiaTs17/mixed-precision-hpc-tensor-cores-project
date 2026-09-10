#!/bin/bash
#SBATCH --job-name=mixed_precision_gemm_tc
#SBATCH --output=logs/mixed_precision_gemm_tc_%j.out
#SBATCH --error=logs/mixed_precision_gemm_tc_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=24:00:00

# Activacion de Tensor Cores para GEMM (Fase 2). Compara cuBLAS clasico,
# cuBLAS con Tensor Cores y WMMA propio -- ver Fase_2/GEMM/README.md para la
# lista completa de rutas y como interpretar los resultados sin combinar
# comparaciones que no son equivalentes.
#
# Todo parametro que un usuario razonablemente querria cambiar sin editar
# este script se pasa por variable de entorno via --export=ALL,VAR=valor:
#
#   sbatch run_gemm_tc.sbatch
#   sbatch --export=ALL,SMOKE_TEST=1 run_gemm_tc.sbatch
#   sbatch --export=ALL,CUDA_ARCH=86 run_gemm_tc.sbatch
#   sbatch --export=ALL,GEMM_SIZES="1024 2048 4096 8192",ITERS=10 run_gemm_tc.sbatch
#   sbatch --export=ALL,TC_FORMAT=fp16 run_gemm_tc.sbatch
#   sbatch --export=ALL,RUN_NCU=0 run_gemm_tc.sbatch
#   sbatch --export=ALL,INCLUDE_LEGACY_SIZES=1 run_gemm_tc.sbatch
#   sbatch --export=ALL,RUN_CUTLASS=0 run_gemm_tc.sbatch
#   sbatch --export=ALL,CUTLASS_DIR=/ruta/a/otro/cutlass/include run_gemm_tc.sbatch
#
# NOTA sobre --output/--error: al igual que en Fase_1/GEMM/run_gemm_fase1.sbatch,
# las lineas #SBATCH las procesa `sbatch` antes de que el shell del job
# arranque y NO admiten expansion de variables de entorno. Para cambiar donde
# SLURM escribe el log del job, pasa --output/--error en la linea de `sbatch`.

set -euo pipefail

# No forzar cd a SLURM_SUBMIT_DIR: cuando este script se invoca ANIDADO
# (`bash` dentro de un job ya en marcha -- ver tools/validacion_preliminar.sbatch,
# que hace `cd` al directorio correcto ANTES de llamar `bash <este script>`),
# SLURM_SUBMIT_DIR es el del job PADRE (la raiz del repo), no el de este
# kernel, y pisaria un pwd que el llamador ya dejo correcto. Se prueba pwd
# primero; solo se cae a SLURM_SUBMIT_DIR cuando pwd no trae el fuente
# (invocacion de primer nivel via `sbatch` desde la raiz del repo).
if [[ ! -f gemm_tensor_activation.cu ]]; then
    if [[ -f GEMM/gemm_tensor_activation.cu ]]; then
        cd GEMM
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/gemm_tensor_activation.cu" ]]; then
        cd "${SLURM_SUBMIT_DIR}"
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/GEMM/gemm_tensor_activation.cu" ]]; then
        cd "${SLURM_SUBMIT_DIR}/GEMM"
    fi
fi

# Directorio donde este script deja lo que si gestiona directamente: el
# binario compilado y los reportes .ncu-rep de Nsight Compute.
RESULTS_DIR="${RESULTS_DIR:-logs}"
mkdir -p "${RESULTS_DIR}"

# Carga de modulos: en PACCA no hay modulo "cuda"; nvcc/ncu se toman por ruta.
if command -v module >/dev/null 2>&1; then
    # En PACCA (OpenHPC/Lmod) "openblas" vive bajo la familia de
    # compilador "gnu12" -- cargarlo sin eso primero falla en silencio
    # (el "|| true" se traga el error de Lmod) y deja OPENBLAS_DIR sin
    # definir, cayendo al fallback generico de abajo que no encuentra
    # cblas.h en este sistema. module load es un no-op inofensivo si
    # gnu12 ya esta activo o si el cluster no usa esta jerarquia.
    module load gnu12 || true
    module load openblas || true
fi

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
elif [[ -z "${OPENBLAS_INC:-}" && -f "/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/include/cblas.h" ]]; then
    # Ruta historica de PACCA (mismo patron que detect_nvcc en
    # tools/detect_toolchain.sh): cubre el caso en que el modulo no
    # cargo (Lmod ausente o jerarquia distinta) pero el paquete OpenHPC
    # si esta instalado.
    OPENBLAS_INC="-I/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/include"
    OPENBLAS_LIBS="${OPENBLAS_LIBS:--L/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib -lopenblas}"
else
    OPENBLAS_INC="$(normalize_include_flags "${OPENBLAS_INC:--I/usr/include/openblas}")"
    OPENBLAS_LIBS="${OPENBLAS_LIBS:--lopenblas}"
fi
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

# --- Parametros del barrido -------------------------------------------------
# GEMM_SIZES: tamanos cuadrados (M=N=K) a barrer. Por defecto cubre el rango
# 1024-8192 compartido por las cuatro fases y dimensionado para la A100 de
# PACCA. Todos los valores por defecto son multiplos de 64 y de 32, requisitos de
# divisibilidad del kernel WMMA (m,n multiplo de 64; k multiplo de 32) -- si
# cambias GEMM_SIZES, mantenlos multiplos de 64 o la ruta WMMA (4) terminara
# el binario con un error de validacion en vez de correr.
GEMM_SIZES="${GEMM_SIZES:-1024 2048 4096 8192}"
ITERS="${ITERS:-30}"

# TC_FORMAT: fp16 | bf16 | both. El .sbatch historico jamas pasaba
# --tc-format (el binario caia siempre al default fp16, ver
# Fase_2/GEMM/README.md, seccion "BF16"). El default aqui es "both" para que
# cada corrida cubra FP16 y BF16 sin necesitar dos invocaciones.
TC_FORMAT="${TC_FORMAT:-both}"

# TC_PRECISIONS: subconjunto de {fp32, fp64} a correr. fp64 solo ejercita las
# rutas 1 y 2 (CPU y cuBLAS clasico) -- ver README.md, el binario ignora
# --tc-format con --double. Por defecto solo fp32, igual que las corridas
# historicas; se deja como variable para no tener que editar el script si se
# quiere agregar una comparacion FP64 sin Tensor Cores.
TC_PRECISIONS="${TC_PRECISIONS:-fp32}"

# INCLUDE_LEGACY_SIZES=1 agrega los tamanos historicos grandes (12288, 32768)
# a continuacion del barrido normal, con su propio numero de iteraciones.
INCLUDE_LEGACY_SIZES="${INCLUDE_LEGACY_SIZES:-0}"
LEGACY_SIZES="${LEGACY_SIZES:-12288 32768}"
LEGACY_ITERS="${LEGACY_ITERS:-30}"

# RUN_CUTLASS: activa la ruta 5 (CUTLASS, cutlass::gemm::device::Gemm) ademas
# de las cuatro rutas historicas (CPU, cuBLAS clasico, cuBLAS TC, WMMA
# propio). CUTLASS es header-only: no se instala como libreria, solo hace
# falta un checkout de github.com/NVIDIA/cutlass (serie 2.x -- ver
# REQUIREMENTS.md) cuyo directorio include/ se agrega al -I de nvcc via
# CUTLASS_DIR.
#
# Por defecto CUTLASS_DIR apunta a $HOME/cutlass/include: clona CUTLASS ahi
# antes de lanzar el job --
#   git clone --branch v2.11.0 https://github.com/NVIDIA/cutlass "$HOME/cutlass"
# -- o exporta CUTLASS_DIR apuntando a otro checkout ya existente en PACCA.
#
# Si RUN_CUTLASS=1 pero CUTLASS_DIR no contiene un checkout valido, la
# compilacion de gemm_tc NO falla (gemm_tensor_activation.cu detecta la
# ausencia de CUTLASS con __has_include y compila igual las 4 rutas
# restantes), pero cada corrida con --cutlass terminara en tiempo de
# ejecucion con un mensaje de error explicando como habilitarlo. Revisa el
# log tras el primer job antes de asumir que la ruta 5 corrio.
RUN_CUTLASS="${RUN_CUTLASS:-1}"
CUTLASS_DIR="${CUTLASS_DIR:-$HOME/cutlass/include}"
if [[ "${RUN_CUTLASS}" == "1" ]]; then
    CUTLASS_INC="-I${CUTLASS_DIR}"
    # VERIFICAR EN PACCA: --expt-relaxed-constexpr lo requieren varios
    # ejemplos oficiales de CUTLASS 2.x para su metaprogramacion constexpr
    # host/device; no se pudo confirmar si esta version puntual de CUTLASS
    # lo exige (no hay nvcc ni GPU en el entorno donde se escribio esto). Si
    # la compilacion falla sin este flag, este es el primer sospechoso; si
    # sobra, no deberia romper nada quitarlo.
    CUTLASS_EXTRA_NVCC_FLAGS="--expt-relaxed-constexpr"
else
    CUTLASS_INC=""
    CUTLASS_EXTRA_NVCC_FLAGS=""
fi

# Parametros de la corrida SMOKE_TEST=1 (verificacion rapida, no un punto de
# datos experimental).
SMOKE_SIZE="${SMOKE_SIZE:-1024}"
SMOKE_ITERS="${SMOKE_ITERS:-2}"
SMOKE_PRECISION="${SMOKE_PRECISION:-fp32}"

# Definiciones compartidas de perfilado (NCU_QUICK_METRICS). Se declara una
# sola vez en tools/common_ncu.sh (fuera de cualquier Fase_N) y se reutiliza
# entre los .sbatch de todas las fases; este script corre desde Fase_2/GEMM,
# de ahi "../../tools/common_ncu.sh" como ruta principal.
#
# tools/common_ncu.sh ya esta migrado en la raiz del repositorio (fuera de
# cualquier Fase_N) -- la ruta relativa de abajo ("../../tools/common_ncu.sh"
# desde Fase_2/GEMM) lo encuentra sin pasos adicionales.
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
# Las corridas grandes (M/N/K=12288, 32768) usan quick salvo que se exporte
# explicitamente NCU_MODE=full (riesgo real de timeout SLURM con --set full).
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
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "GEMM_SIZES=${GEMM_SIZES}"
echo "ITERS=${ITERS}"
echo "TC_FORMAT=${TC_FORMAT}"
echo "TC_PRECISIONS=${TC_PRECISIONS}"
echo "INCLUDE_LEGACY_SIZES=${INCLUDE_LEGACY_SIZES}"
echo "RUN_CUTLASS=${RUN_CUTLASS}"
echo "CUTLASS_DIR=${CUTLASS_DIR}"

nvidia-smi || true

echo "Compilando gemm_tc..."
"${NVCC}" -std=c++17 -O3 gemm_tensor_activation.cu -o "${RESULTS_DIR}/gemm_tc" \
    ${OPENBLAS_INC} -lcublas ${OPENBLAS_LIBS} \
    ${CUTLASS_INC} ${CUTLASS_EXTRA_NVCC_FLAGS} \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    --allow-unsupported-compiler
BIN="${RESULTS_DIR}/gemm_tc"

# Validacion SASS pre-perfilado: si el binario no contiene instrucciones HMMA,
# el kernel WMMA no se activo (tipicamente por M/N/K no divisibles) y perfilar
# seria inutil. CUOBJDUMP se busca por PATH, junto a NVCC, o en el arbol cuda/
# del HPC SDK (layout tipico .../<version>/cuda/bin/cuobjdump).
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
    HMMA_COUNT=$("${CUOBJDUMP}" --dump-sass "${BIN}" | grep -c HMMA || true)
    echo "Instrucciones HMMA detectadas en binario: ${HMMA_COUNT} (cuobjdump: ${CUOBJDUMP})"
    if [[ "${RUN_NCU}" == "1" && "${HMMA_COUNT}" -eq 0 ]]; then
        echo "ERROR: binario sin instrucciones HMMA; revisar condiciones" \
             "de divisibilidad M/N/K antes de perfilar." >&2
        exit 1
    fi
else
    echo "AVISO: no se encontro cuobjdump; se omite la validacion HMMA." \
         "Exporta CUOBJDUMP=/ruta/a/cuobjdump para habilitarla." >&2
fi

echo "Ejecutando benchmarks..."

# Construye la lista de corridas "M ITERS PRECISION" (M=N=K, ver la nota de
# divisibilidad WMMA junto a GEMM_SIZES). Nada queda hardcodeado fuera de esta
# seccion: cambia GEMM_SIZES/ITERS/TC_PRECISIONS/LEGACY_* por variable de
# entorno para barrer otra cosa sin tocar el script.
declare -a RUNS=()
if [[ "${SMOKE_TEST}" == "1" ]]; then
    RUNS+=("${SMOKE_SIZE} ${SMOKE_ITERS} ${SMOKE_PRECISION}")
else
    for size in ${GEMM_SIZES}; do
        for precision in ${TC_PRECISIONS}; do
            RUNS+=("${size} ${ITERS} ${precision}")
        done
    done
    if [[ "${INCLUDE_LEGACY_SIZES}" == "1" ]]; then
        for size in ${LEGACY_SIZES}; do
            for precision in ${TC_PRECISIONS}; do
                RUNS+=("${size} ${LEGACY_ITERS} ${precision}")
            done
        done
    fi
fi

for run in "${RUNS[@]}"; do
    read -r SIZE RUN_ITERS PRECISION <<< "${run}"
    echo
    echo "============================================================"
    echo "Corrida: M=N=K=${SIZE}, ITERS=${RUN_ITERS}, ${PRECISION}, TC_FORMAT=${TC_FORMAT}"
    echo "============================================================"

    declare -a EXTRA_FLAGS=()
    if [[ "${PRECISION}" == "fp64" ]]; then
        EXTRA_FLAGS+=("--double")
    else
        EXTRA_FLAGS+=("--tc-format" "${TC_FORMAT}")
        # Ruta 5 (CUTLASS): solo tiene sentido junto a un formato de Tensor
        # Cores, nunca con --double (que ni siquiera pasa por esta rama).
        if [[ "${RUN_CUTLASS}" == "1" ]]; then
            EXTRA_FLAGS+=("--cutlass")
        fi
    fi

    "${BIN}" --m "${SIZE}" --n "${SIZE}" --k "${SIZE}" --iters "${RUN_ITERS}" \
        "${EXTRA_FLAGS[@]}"

    if [[ "${RUN_NCU}" == "1" && "${PRECISION}" != "fp64" ]]; then
        REPORT="${RESULTS_DIR}/ncu_wmma_m${SIZE}_n${SIZE}_k${SIZE}_job${SLURM_JOB_ID:-manual}"
        if [[ "${NCU_MODE}" == "full" ]]; then
            NCU_CAPTURE_ARGS=(--set "${NCU_SET}")
        else
            NCU_CAPTURE_ARGS=(--metrics "${NCU_QUICK_METRICS}")
        fi
        echo
        echo "Perfilando WMMA con Nsight Compute (NCU_MODE=${NCU_MODE})..."
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
            "${BIN}" --m "${SIZE}" --n "${SIZE}" --k "${SIZE}" --iters "${RUN_ITERS}" \
            "${EXTRA_FLAGS[@]}"; } 2>&1
    fi
done
