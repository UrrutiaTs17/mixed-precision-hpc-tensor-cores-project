#!/bin/bash
#SBATCH --job-name=gate3_ancla_conv
#SBATCH --output=logs/gate3_ancla_conv_%j.out
#SBATCH --error=logs/gate3_ancla_conv_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# GATE 3 -- ancla FP64 de Convolucion, automatizado (K=0 y K=1).
#
# Es lo que "Que falta" de Fase_4/Convolution/README.md pedia. Mismo diseno que
# Fase_4/GEMM/gate3_ancla.sbatch (ver su cabecera para el razonamiento
# completo): compila los DOS binarios, corre las tres pasadas y le pasa los
# logs a ../tools/gate3_ancla.py --kernel conv.
#
#   K=0 : Fase_4 --anchor-every 0 == Fase_3 en las columnas deterministas.
#   K=1 : rel_l2/rel_linf bajo la cota de cuantizacion del formato.
#
# UN DETALLE PROPIO DE CONVOLUCION que este gate cubre: el paso de ancla y la
# trayectoria de referencia comparten el mismo buffer scratch de im2col en
# double (d_col64). Es seguro -- las dos llamadas a gpu_fp64_conv_step()
# ocurren secuenciales en el stream por defecto, sin carrera de datos (ver
# Fase_4/Convolution/README.md) -- pero es la UNICA diferencia estructural del
# mecanismo de ancla frente a GEMM. Si este gate falla y el de GEMM pasa con
# los mismos parametros, ese buffer compartido es el primer sospechoso, antes
# que cualquier error de logica del ancla.
#
#   sbatch gate3_ancla.sbatch
#   sbatch --export=ALL,GATE_HW=128,GATE_ITERS=20 gate3_ancla.sbatch
#
# Codigo de salida: el de gate3_ancla.py (0 pasa / 1 falla / 2 no evaluable).

set -euo pipefail

# No forzar cd a SLURM_SUBMIT_DIR: cuando este script se invoca ANIDADO
# (`bash` dentro de un job ya en marcha -- ver tools/validacion_preliminar.sbatch,
# que hace `cd` al directorio correcto ANTES de llamar `bash <este script>`),
# SLURM_SUBMIT_DIR es el del job PADRE (la raiz del repo), no el de este
# kernel, y pisaria un pwd que el llamador ya dejo correcto. Se prueba pwd
# primero; solo se cae a SLURM_SUBMIT_DIR cuando pwd no trae el fuente
# (invocacion de primer nivel via `sbatch` desde la raiz del repo).
if [[ ! -f conv_chained.cu ]]; then
    if [[ -f Convolution/conv_chained.cu ]]; then
        cd Convolution
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/conv_chained.cu" ]]; then
        cd "${SLURM_SUBMIT_DIR}"
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/Convolution/conv_chained.cu" ]]; then
        cd "${SLURM_SUBMIT_DIR}/Convolution"
    fi
fi

LOGS_DIR="${LOGS_DIR:-logs}"
RESULTS_DIR="${RESULTS_DIR:-results}"
mkdir -p "${LOGS_DIR}" "${RESULTS_DIR}"
JOB_ID="${SLURM_JOB_ID:-manual}"
GATE_DIR="${RESULTS_DIR}/gate3_${JOB_ID}"
mkdir -p "${GATE_DIR}"
GATE_DIR_ABS="$(cd "${GATE_DIR}" && pwd)"

for _toolchain in "../../tools/detect_toolchain.sh" "./detect_toolchain.sh"; do
    if [[ -f "${_toolchain}" ]]; then
        # shellcheck source=/dev/null
        source "${_toolchain}"
        break
    fi
done
CUDA_ARCH="$(detect_cuda_arch)"
NVCC="$(detect_nvcc)"

# NVML apagado a proposito: los gates solo miran columnas deterministas.
NVCC_FLAGS=(-std=c++17 -lcublas
            -gencode "arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}"
            -Xcompiler "-O3 -funroll-loops -ffp-contract=off"
            --allow-unsupported-compiler -lpthread)

GATE_HW="${GATE_HW:-64}"         # el minimo que acepta el binario
GATE_ITERS="${GATE_ITERS:-12}"
GATE_CKPT="${GATE_CKPT:-4}"
GATE_TC="${GATE_TC:-both}"
GATE_SEED="${GATE_SEED:-42}"

echo "Nodo: $(hostname)"
echo "Arquitectura CUDA: sm_${CUDA_ARCH}"
echo "Parametros del gate: HW=${GATE_HW} ITERS=${GATE_ITERS} TC=${GATE_TC}"

echo "Compilando Fase 4 (este directorio)..."
"${NVCC}" conv_chained.cu -o "${GATE_DIR_ABS}/conv_f4" "${NVCC_FLAGS[@]}"

echo "Compilando Fase 3 (../../Fase_3/Convolution)..."
( cd ../../Fase_3/Convolution && "${NVCC}" conv_chained.cu -o "${GATE_DIR_ABS}/conv_f3" "${NVCC_FLAGS[@]}" )

COMUN=(--hw "${GATE_HW}" --iters "${GATE_ITERS}" --tc "${GATE_TC}"
       --comp on --checkpoint-every "${GATE_CKPT}" --seed "${GATE_SEED}")

echo "Corriendo Fase 3 (sin --anchor-every)..."
"${GATE_DIR_ABS}/conv_f3" "${COMUN[@]}"                  > "${GATE_DIR}/f3_k0.log" 2>&1
echo "Corriendo Fase 4 con --anchor-every 0..."
"${GATE_DIR_ABS}/conv_f4" "${COMUN[@]}" --anchor-every 0 > "${GATE_DIR}/f4_k0.log" 2>&1
echo "Corriendo Fase 4 con --anchor-every 1..."
"${GATE_DIR_ABS}/conv_f4" "${COMUN[@]}" --anchor-every 1 > "${GATE_DIR}/f4_k1.log" 2>&1

GATE_SCRIPT="${GATE_SCRIPT:-}"
if [[ -z "${GATE_SCRIPT}" ]]; then
    for _cand in "../tools/gate3_ancla.py" "tools/gate3_ancla.py" "../../Fase_4/tools/gate3_ancla.py"; do
        [[ -f "${_cand}" ]] && { GATE_SCRIPT="${_cand}"; break; }
    done
fi
if [[ -z "${GATE_SCRIPT}" ]]; then
    echo "ERROR: no se encontro gate3_ancla.py. Los logs quedan en ${GATE_DIR}" \
         "para compararlos a mano." >&2
    exit 2
fi

echo
python3 "${GATE_SCRIPT}" --kernel conv \
    --etiqueta "HW=${GATE_HW} iters=${GATE_ITERS} job=${JOB_ID}" \
    --gate0-base  "${GATE_DIR}/f3_k0.log" \
    --gate0-nuevo "${GATE_DIR}/f4_k0.log" \
    --gate1       "${GATE_DIR}/f4_k1.log"
