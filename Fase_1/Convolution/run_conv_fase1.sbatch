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
# Nada esta hardcodeado en el cuerpo del script: la forma del problema
# (entrada, filtro, batch, padding/stride/dilation), la precision a correr y
# el directorio de salida son variables de entorno con un default documentado
# mas abajo (patron VAR="${VAR:-default}"). El unico hardcodeo real que queda
# son los paths de #SBATCH --output/--error de arriba: SLURM los procesa antes
# de que este script exista, asi que no pueden depender de una variable
# definida aqui adentro; para cambiarlos, pasa --output=/--error= en la propia
# linea de sbatch.
#
# PRECISION por defecto es "both" (corre FP32 y FP64): el .sbatch historico
# (old/Fase_1/Convolution/run_conv_fase1.sbatch) solo invocaba la ruta FP32
# pese a que el binario siempre soporto --double -- ver
# Fase_1/Convolution/README.md, seccion "Gap historico: FP64 nunca se corria".
#
#   sbatch run_conv_fase1.sbatch
#   sbatch --export=ALL,SMOKE_TEST=1 run_conv_fase1.sbatch
#   sbatch --export=ALL,PRECISION=fp32 run_conv_fase1.sbatch
#   sbatch --export=ALL,HW_LIST="64 128 256 512" run_conv_fase1.sbatch
#   sbatch --export=ALL,CUDNN_ROOT=/otra/ruta run_conv_fase1.sbatch

set -euo pipefail

# No forzar cd a SLURM_SUBMIT_DIR: cuando este script se invoca ANIDADO
# (`bash` dentro de un job ya en marcha -- ver tools/validacion_preliminar.sbatch,
# que hace `cd` al directorio correcto ANTES de llamar `bash <este script>`),
# SLURM_SUBMIT_DIR es el del job PADRE (la raiz del repo), no el de este
# kernel, y pisaria un pwd que el llamador ya dejo correcto. Se prueba pwd
# primero; solo se cae a SLURM_SUBMIT_DIR cuando pwd no trae el fuente
# (invocacion de primer nivel via `sbatch` desde la raiz del repo).
if [[ ! -f conv_baseline.cu ]]; then
    if [[ -f Convolution/conv_baseline.cu ]]; then
        cd Convolution
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/conv_baseline.cu" ]]; then
        cd "${SLURM_SUBMIT_DIR}"
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/Convolution/conv_baseline.cu" ]]; then
        cd "${SLURM_SUBMIT_DIR}/Convolution"
    fi
fi

# Directorio para binarios/artefactos auxiliares de esta corrida (no para los
# logs de SLURM en si, ver la nota sobre #SBATCH --output/--error arriba).
OUTPUT_DIR="${OUTPUT_DIR:-logs}"
mkdir -p "${OUTPUT_DIR}"

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
PRECISION="${PRECISION:-both}"  # fp32 | fp64 | both

# Forma del problema: misma campana que Fase 3/4, con 64 canales y
# HW_LIST=64,128,256,512. H/W escalares siguen disponibles
# para una corrida manual y, si se exportan, reemplazan el barrido por una sola
# malla cuadrada. SMOKE_TEST conserva solo el menor tamano para validacion.
if [[ "${SMOKE_TEST}" == "1" ]]; then
    _default_hw_list="64"
    _default_iters=2
else
    _default_hw_list="64 128 256 512"
    _default_iters=10
fi
N="${N:-1}"
C="${C:-64}"
K="${K:-64}"
R="${R:-3}"
S="${S:-3}"
ITERS="${ITERS:-${_default_iters}}"
if [[ -n "${H:-}" || -n "${W:-}" ]]; then
    H="${H:-${W}}"
    W="${W:-${H}}"
    if [[ "${H}" != "${W}" ]]; then
        echo "ERROR: las campanas comparables exigen H=W (recibido H=${H}, W=${W})." >&2
        exit 2
    fi
    HW_LIST="${H}"
else
    HW_LIST="${HW_LIST:-${_default_hw_list}}"
fi
PAD_H="${PAD_H:-1}"
PAD_W="${PAD_W:-1}"
STRIDE_H="${STRIDE_H:-1}"
STRIDE_W="${STRIDE_W:-1}"
DILATION_H="${DILATION_H:-1}"
DILATION_W="${DILATION_W:-1}"

echo "Nodo: $(hostname)"
echo "Fecha: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-no definido}"
echo "Arquitectura CUDA: sm_${CUDA_ARCH}"
echo "NVCC=${NVCC}"
echo "SMOKE_TEST=${SMOKE_TEST}"
echo "PRECISION=${PRECISION}"
echo "HW_LIST=${HW_LIST}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

nvidia-smi || true

echo "Compilando conv_baseline..."
"${NVCC}" -std=c++17 -O3 conv_baseline.cu -o conv_baseline \
    ${OPENBLAS_INC} ${CUDNN_INC} ${CUDNN_LIBS} ${OPENBLAS_LIBS} \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    --allow-unsupported-compiler

# El baseline FP32 desactiva TF32 desde el codigo (cudnnSetConvolutionMathType
# con CUDNN_FMA_MATH en conv_baseline.cu), no por NVIDIA_TF32_OVERRIDE: asi la
# linea base sigue siendo FP32 escalar aunque el binario se corra a mano fuera
# de este sbatch.
#
# fp32/fp64 se lanzan con dos invocaciones completas (no con un array de
# flags condicional): bajo `set -u`, expandir un array vacio como
# "${flag[@]}" aborta el job en bash < 4.4, asi que se evita el patron.
run_fp32() {
    local hw="$1"
    ./conv_baseline \
        --n "${N}" --c "${C}" --h "${hw}" --w "${hw}" \
        --k "${K}" --r "${R}" --s "${S}" \
        --pad_h "${PAD_H}" --pad_w "${PAD_W}" \
        --stride_h "${STRIDE_H}" --stride_w "${STRIDE_W}" \
        --dilation_h "${DILATION_H}" --dilation_w "${DILATION_W}" \
        --iters "${ITERS}"
}

run_fp64() {
    local hw="$1"
    ./conv_baseline --double \
        --n "${N}" --c "${C}" --h "${hw}" --w "${hw}" \
        --k "${K}" --r "${R}" --s "${S}" \
        --pad_h "${PAD_H}" --pad_w "${PAD_W}" \
        --stride_h "${STRIDE_H}" --stride_w "${STRIDE_W}" \
        --dilation_h "${DILATION_H}" --dilation_w "${DILATION_W}" \
        --iters "${ITERS}"
}

for HW in ${HW_LIST}; do
    echo
    echo "============================================================"
    echo "Corrida: N=${N}, C=${C}, H=W=${HW}, K=${K}, R=${R}, S=${S}, ITERS=${ITERS}"
    echo "Padding: (${PAD_H}, ${PAD_W}), stride: (${STRIDE_H}, ${STRIDE_W}), dilation: (${DILATION_H}, ${DILATION_W})"
    echo "============================================================"
    case "${PRECISION}" in
        fp32)
            run_fp32 "${HW}"
            ;;
        fp64)
            run_fp64 "${HW}"
            ;;
        both)
            echo "--- Precision: FP32 ---"
            run_fp32 "${HW}"
            echo
            echo "--- Precision: FP64 ---"
            run_fp64 "${HW}"
            ;;
        *)
            echo "PRECISION invalida: '${PRECISION}' (use fp32, fp64 o both)" >&2
            exit 1
            ;;
    esac
done
