#!/bin/bash
#SBATCH --job-name=gate3_ancla_gemm
#SBATCH --output=logs/gate3_ancla_gemm_%j.out
#SBATCH --error=logs/gate3_ancla_gemm_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# GATE 3 -- ancla FP64 de GEMM, automatizado (K=0 y K=1).
#
# Es lo que "Que falta" de Fase_4/GEMM/README.md pedia: automatizar las dos
# puertas que hasta ahora habia que correr y comparar a mano.
#
# Compila los DOS binarios (Fase 3 y Fase 4), corre las tres pasadas que los
# gates necesitan y le pasa los logs a ../tools/gate3_ancla.py, que decide:
#
#   K=0 : Fase_4 --anchor-every 0  debe dar columnas DETERMINISTAS identicas a
#         Fase_3 con los mismos parametros.
#   K=1 : Fase_4 --anchor-every 1  debe quedar bajo la cota de cuantizacion del
#         formato (2^-11 en FP16, 2^-8 en BF16) en rel_l2 y rel_linf.
#
# Deliberadamente CHICO y CORTO (--time 00:30:00, N=256): un error de orden de
# operandos o de logica del ancla se delata igual a N=256 que a N=8192, y este
# job tiene que poder colarse en la cola antes de comprometer las horas de la
# campana. No produce datos de campana: sus resultados no se reportan.
#
#   sbatch gate3_ancla.sbatch
#   sbatch --export=ALL,GATE_N=1024,GATE_ITERS=20 gate3_ancla.sbatch
#
# Codigo de salida: el de gate3_ancla.py (0 pasa / 1 falla / 2 no evaluable).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f gemm_chained.cu && -f GEMM/gemm_chained.cu ]]; then
    cd GEMM
fi

LOGS_DIR="${LOGS_DIR:-logs}"
RESULTS_DIR="${RESULTS_DIR:-results}"
mkdir -p "${LOGS_DIR}" "${RESULTS_DIR}"
JOB_ID="${SLURM_JOB_ID:-manual}"
GATE_DIR="${RESULTS_DIR}/gate3_${JOB_ID}"
mkdir -p "${GATE_DIR}"
# Absoluto: la compilacion de Fase 3 corre desde OTRO directorio (subshell con
# cd), y un -o relativo escribiria el binario alli en vez de aqui.
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

# NVML deliberadamente APAGADO: los gates solo miran columnas deterministas
# (rel_l2/rel_linf/anchor_every), y la telemetria solo agregaria ruido a las
# columnas de medicion que de todas formas no deciden el veredicto.
NVCC_FLAGS=(-std=c++17 -lcublas
            -gencode "arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}"
            -Xcompiler "-O3 -funroll-loops -ffp-contract=off"
            --allow-unsupported-compiler -lpthread)

GATE_N="${GATE_N:-256}"          # potencia de 2 y multiplo de 64
GATE_ITERS="${GATE_ITERS:-12}"
GATE_CKPT="${GATE_CKPT:-4}"
GATE_TC="${GATE_TC:-both}"
GATE_SEED="${GATE_SEED:-42}"

echo "Nodo: $(hostname)"
echo "Arquitectura CUDA: sm_${CUDA_ARCH}"
echo "Parametros del gate: N=${GATE_N} ITERS=${GATE_ITERS} TC=${GATE_TC}"

echo "Compilando Fase 4 (este directorio)..."
"${NVCC}" gemm_chained.cu -o "${GATE_DIR}/gemm_f4" "${NVCC_FLAGS[@]}"

echo "Compilando Fase 3 (../../Fase_3/GEMM)..."
( cd ../../Fase_3/GEMM && "${NVCC}" gemm_chained.cu -o "${GATE_DIR_ABS}/gemm_f3" "${NVCC_FLAGS[@]}" )

COMUN=(--n "${GATE_N}" --iters "${GATE_ITERS}" --tc "${GATE_TC}"
       --comp on --checkpoint-every "${GATE_CKPT}" --seed "${GATE_SEED}")

echo "Corriendo Fase 3 (sin --anchor-every)..."
"${GATE_DIR}/gemm_f3" "${COMUN[@]}"                     > "${GATE_DIR}/f3_k0.log" 2>&1
echo "Corriendo Fase 4 con --anchor-every 0..."
"${GATE_DIR}/gemm_f4" "${COMUN[@]}" --anchor-every 0    > "${GATE_DIR}/f4_k0.log" 2>&1
echo "Corriendo Fase 4 con --anchor-every 1..."
"${GATE_DIR}/gemm_f4" "${COMUN[@]}" --anchor-every 1    > "${GATE_DIR}/f4_k1.log" 2>&1

GATE_SCRIPT="${GATE_SCRIPT:-}"
if [[ -z "${GATE_SCRIPT}" ]]; then
    for _cand in "../tools/gate3_ancla.py" "tools/gate3_ancla.py" "../../Fase_4/tools/gate3_ancla.py"; do
        [[ -f "${_cand}" ]] && { GATE_SCRIPT="${_cand}"; break; }
    done
fi
if [[ -z "${GATE_SCRIPT}" ]]; then
    echo "ERROR: no se encontro gate3_ancla.py (se probo ../tools/, tools/ y" \
         "../../Fase_4/tools/). Los logs quedan en ${GATE_DIR} para compararlos" \
         "a mano." >&2
    exit 2
fi

echo
python3 "${GATE_SCRIPT}" --kernel gemm \
    --etiqueta "N=${GATE_N} iters=${GATE_ITERS} job=${JOB_ID}" \
    --gate0-base  "${GATE_DIR}/f3_k0.log" \
    --gate0-nuevo "${GATE_DIR}/f4_k0.log" \
    --gate1       "${GATE_DIR}/f4_k1.log"
