#!/bin/bash
#SBATCH --job-name=gate3_ancla_stencil
#SBATCH --output=logs/gate3_ancla_stencil_%j.out
#SBATCH --error=logs/gate3_ancla_stencil_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# GATE 3 -- ancla FP64 de Stencil, automatizado (K=0 y K=1).
#
# Es el `gate3_ancla.sbatch` que Fase_4/Stencil/README.md nombra en su seccion
# "Que falta". Compila los DOS binarios (Fase 3 y Fase 4), corre las tres
# pasadas que los gates necesitan y le pasa los logs a ../tools/gate3_ancla.py.
#
#   K=0 : Fase_4 --anchor-every 0 debe dar columnas DETERMINISTAS identicas a
#         Fase_3 con los mismos parametros (CSV_DRIFT, CSV_SUMMARY, CSV_STORE,
#         CSV_HORIZON, CSV_ONSET).
#   K=1 : DOS criterios, cada uno sobre la columna que le corresponde --
#         (a) rel_l2/rel_linf (acumulador FP32) de WMMA_*_SP deben alcanzar el
#             nivel de la ruta GPU_FP64 de la MISMA corrida, y
#         (b) rel_l2_prop/rel_linf_prop (estado propagado en 16 bits) deben
#             quedar bajo la cota de cuantizacion del formato.
#         Ver el docstring de gate3_ancla.py sobre por que NO es el mismo test
#         que en GEMM/Convolucion: alli CSV_DRIFT compara contra el buffer T
#         cuantizado, aqui contra d_out_fp32, que es otro objeto.
#
# --time 01:00:00 y no 30 min como GEMM/Conv: la ruta CPU_FP64 (referencia de
# costo, --cpu-fp64 on) es serial en CPU y a esta malla ya cuesta unos minutos
# por pasada; son tres pasadas.
#
# --tc fp16 (no both) por defecto: el gate no gana nada corriendo BF16 tambien
# -- la cota de cuantizacion se evalua por formato, y con FP16 (la cota mas
# estrecha, 2^-11) el test es el mas exigente de los dos. GATE_TC=both si se
# quiere cubrir los dos.
#
#   sbatch gate3_ancla.sbatch
#   sbatch --export=ALL,GATE_NX=1024,GATE_NY=1024,GATE_TC=both gate3_ancla.sbatch
#
# Codigo de salida: el de gate3_ancla.py (0 pasa / 1 falla / 2 no evaluable).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f stencil_tensor_activation.cu && -f Stencil/stencil_tensor_activation.cu ]]; then
    cd Stencil
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
NVCC_FLAGS=(-std=c++17
            -gencode "arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}"
            -Xcompiler "-O3 -funroll-loops -ffp-contract=off"
            --allow-unsupported-compiler -lpthread)

GATE_NX="${GATE_NX:-1024}"
GATE_NY="${GATE_NY:-1024}"
GATE_ITERS="${GATE_ITERS:-20}"
GATE_CKPT="${GATE_CKPT:-5}"
GATE_TC="${GATE_TC:-fp16}"
GATE_OP_MODE="${GATE_OP_MODE:-stress}"
GATE_CI_MODE="${GATE_CI_MODE:-legacy}"

echo "Nodo: $(hostname)"
echo "Arquitectura CUDA: sm_${CUDA_ARCH}"
echo "Parametros del gate: NX=${GATE_NX} NY=${GATE_NY} ITERS=${GATE_ITERS} TC=${GATE_TC}"

echo "Compilando Fase 4 (este directorio)..."
"${NVCC}" stencil_tensor_activation.cu -o "${GATE_DIR_ABS}/stencil_f4" "${NVCC_FLAGS[@]}"

echo "Compilando Fase 3 (../../Fase_3/Stencil)..."
( cd ../../Fase_3/Stencil && "${NVCC}" stencil_tensor_activation.cu -o "${GATE_DIR_ABS}/stencil_f3" "${NVCC_FLAGS[@]}" )

# --spatial-comp on es OBLIGATORIO: el binario rechaza --anchor-every > 0 sin
# ella (el residuo del ancla se re-siembra sobre la convencion de esa politica).
# --fp64-gpu on tambien: el criterio (a) del gate K=1 compara contra esa ruta.
COMUN=(--nx "${GATE_NX}" --ny "${GATE_NY}" --iters "${GATE_ITERS}"
       --tc "${GATE_TC}" --checkpoint-every "${GATE_CKPT}"
       --kahan off --spatial-comp on --fp64-gpu on --cpu-fp64 on
       --op-mode "${GATE_OP_MODE}" --ci-mode "${GATE_CI_MODE}")

echo "Corriendo Fase 3 (sin --anchor-every)..."
"${GATE_DIR_ABS}/stencil_f3" "${COMUN[@]}"                  > "${GATE_DIR}/f3_k0.log" 2>&1
echo "Corriendo Fase 4 con --anchor-every 0..."
"${GATE_DIR_ABS}/stencil_f4" "${COMUN[@]}" --anchor-every 0 > "${GATE_DIR}/f4_k0.log" 2>&1
echo "Corriendo Fase 4 con --anchor-every 1..."
"${GATE_DIR_ABS}/stencil_f4" "${COMUN[@]}" --anchor-every 1 > "${GATE_DIR}/f4_k1.log" 2>&1

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
python3 "${GATE_SCRIPT}" --kernel stencil \
    --etiqueta "${GATE_NX}x${GATE_NY} iters=${GATE_ITERS} job=${JOB_ID}" \
    --gate0-base  "${GATE_DIR}/f3_k0.log" \
    --gate0-nuevo "${GATE_DIR}/f4_k0.log" \
    --gate1       "${GATE_DIR}/f4_k1.log"
