#!/bin/bash
#SBATCH --job-name=fase1_gemm_baseline
#SBATCH --output=logs/fase1_gemm_baseline_%j.out
#SBATCH --error=logs/fase1_gemm_baseline_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=24:00:00

# Baseline de Fase 1 (CPU BLAS vs cuBLAS clasico, sin Tensor Cores) en PACCA.
# Mismo patron de resolucion de toolchain que los .sbatch de Fase 2: nvcc y las
# librerias se toman por ruta, no por "module load cuda" (no existe en PACCA).
#
# Todo parametro que un usuario razonablemente querria cambiar sin editar este
# script se pasa por variable de entorno via --export=ALL,VAR=valor. Ejemplos:
#
#   sbatch run_gemm_fase1.sbatch
#   sbatch --export=ALL,SMOKE_TEST=1 run_gemm_fase1.sbatch
#   sbatch --export=ALL,CUDA_ARCH=86 run_gemm_fase1.sbatch
#   sbatch --export=ALL,GEMM_SIZES="512 1024" run_gemm_fase1.sbatch
#   sbatch --export=ALL,PRECISIONS=fp64,ITERS=10 run_gemm_fase1.sbatch
#   sbatch --export=ALL,INCLUDE_LEGACY_SIZES=1 run_gemm_fase1.sbatch
#
# NOTA sobre --output/--error: las lineas #SBATCH las procesa el propio
# comando `sbatch` ANTES de que el shell del job arranque, asi que NO admiten
# expansion de variables de entorno (--export=ALL,X=Y no las alcanza). Para
# cambiar donde SLURM escribe el log del job (no los resultados que el script
# gestiona con RESULTS_DIR), pasa --output/--error directamente en la linea de
# `sbatch`, p. ej.: sbatch --output=/otra/ruta/%j.out run_gemm_fase1.sbatch

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f gemm_baseline.cu && -f GEMM/gemm_baseline.cu ]]; then
    cd GEMM
fi

# Directorio donde este script deja lo que SI gestiona directamente (el
# binario compilado; los .out/.err de SLURM en si los define #SBATCH arriba,
# ver la nota de cabecera sobre por que esos dos no son overrideables por
# variable de entorno).
RESULTS_DIR="${RESULTS_DIR:-logs}"
mkdir -p "${RESULTS_DIR}"

if command -v module >/dev/null 2>&1; then
    module load openblas || true
fi

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

# En OpenHPC el modulo define OPENBLAS_DIR; derivar los flags de ahi evita que
# la ruta cruda llegue a nvcc sin -I.
if [[ -n "${OPENBLAS_DIR:-}" ]]; then
    OPENBLAS_INC="-I${OPENBLAS_DIR}/include"
    OPENBLAS_LIBS="-L${OPENBLAS_DIR}/lib -lopenblas"
else
    OPENBLAS_INC="$(normalize_include_flags "${OPENBLAS_INC:--I/usr/include/openblas}")"
    OPENBLAS_LIBS="${OPENBLAS_LIBS:--lopenblas}"
fi

SMOKE_TEST="${SMOKE_TEST:-0}"

# --- Parametros del barrido -------------------------------------------------
# GEMM_SIZES: tamanos cuadrados (M=N=K) a barrer. Por defecto cubre el rango
# 512-4096 que promete el plan de tesis (ver Fase_1/GEMM/README.md, seccion
# "Nota importante"). El .sbatch historico SOLO corria 12288 y 32768 -- muy
# por encima de ese rango y con un costo de referencia OpenBLAS de minutos
# por iteracion en FP64 -- por eso ya no son el default, solo un opt-in.
GEMM_SIZES="${GEMM_SIZES:-512 1024 2048 4096}"
# PRECISIONS: subconjunto de {fp32, fp64} a correr, separado por espacios.
PRECISIONS="${PRECISIONS:-fp32 fp64}"
# ITERS: iteraciones cronometradas por corrida (promediadas por el binario).
ITERS="${ITERS:-30}"

# INCLUDE_LEGACY_SIZES=1 agrega los tamanos historicos grandes (LEGACY_SIZES)
# a continuacion del barrido normal, con su propio numero de iteraciones
# (LEGACY_ITERS) porque a esos tamanos cada iteracion FP64 cuesta minutos --
# ver el comentario de costo en el .sbatch historico (old/Fase_1/GEMM/).
INCLUDE_LEGACY_SIZES="${INCLUDE_LEGACY_SIZES:-0}"
LEGACY_SIZES="${LEGACY_SIZES:-12288 32768}"
LEGACY_ITERS="${LEGACY_ITERS:-30}"

# Semilla RNG pasada a --seed en cada corrida (ver Options::seed en
# gemm_baseline.cu). Fija por defecto para reproducibilidad entre lanzamientos.
SEED="${SEED:-42}"

# Parametros de la corrida SMOKE_TEST=1 (verificacion rapida de que el
# binario compila y corre, no un punto de datos experimental).
SMOKE_SIZE="${SMOKE_SIZE:-1024}"
SMOKE_ITERS="${SMOKE_ITERS:-2}"
SMOKE_PRECISION="${SMOKE_PRECISION:-fp32}"

echo "Nodo: $(hostname)"
echo "Fecha: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-no definido}"
echo "Arquitectura CUDA: sm_${CUDA_ARCH}"
echo "NVCC=${NVCC}"
echo "SMOKE_TEST=${SMOKE_TEST}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "GEMM_SIZES=${GEMM_SIZES}"
echo "PRECISIONS=${PRECISIONS}"
echo "ITERS=${ITERS}"
echo "SEED=${SEED}"
echo "INCLUDE_LEGACY_SIZES=${INCLUDE_LEGACY_SIZES}"

nvidia-smi || true

echo "Compilando gemm_baseline..."
"${NVCC}" -std=c++17 -O3 gemm_baseline.cu -o "${RESULTS_DIR}/gemm_baseline" \
    ${OPENBLAS_INC} -lcublas ${OPENBLAS_LIBS} \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    --allow-unsupported-compiler
BIN="${RESULTS_DIR}/gemm_baseline"

# Construye la lista de corridas "M ITERS PRECISION" (M=N=K, matrices
# cuadradas en toda esta fase -- ver gemm_baseline.cu). Nada queda hardcodeado
# fuera de esta seccion: cambia GEMM_SIZES/PRECISIONS/ITERS/LEGACY_* por
# variable de entorno para barrer otra cosa sin tocar el script.
declare -a RUNS=()
if [[ "${SMOKE_TEST}" == "1" ]]; then
    RUNS+=("${SMOKE_SIZE} ${SMOKE_ITERS} ${SMOKE_PRECISION}")
else
    for size in ${GEMM_SIZES}; do
        for precision in ${PRECISIONS}; do
            RUNS+=("${size} ${ITERS} ${precision}")
        done
    done
    if [[ "${INCLUDE_LEGACY_SIZES}" == "1" ]]; then
        for size in ${LEGACY_SIZES}; do
            for precision in ${PRECISIONS}; do
                RUNS+=("${size} ${LEGACY_ITERS} ${precision}")
            done
        done
    fi
fi

for run in "${RUNS[@]}"; do
    read -r SIZE RUN_ITERS PRECISION <<< "${run}"
    echo
    echo "============================================================"
    echo "Corrida: M=N=K=${SIZE}, ITERS=${RUN_ITERS}, ${PRECISION}"
    echo "============================================================"
    if [[ "${PRECISION}" == "fp64" ]]; then
        "${BIN}" --double \
            --m "${SIZE}" --n "${SIZE}" --k "${SIZE}" --iters "${RUN_ITERS}" --seed "${SEED}"
    else
        "${BIN}" \
            --m "${SIZE}" --n "${SIZE}" --k "${SIZE}" --iters "${RUN_ITERS}" --seed "${SEED}"
    fi
done
