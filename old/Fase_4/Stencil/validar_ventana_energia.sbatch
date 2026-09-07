#!/bin/bash
#SBATCH --job-name=f4_valida_energia
#SBATCH --output=logs/f4_valener_%j.out
#SBATCH --error=logs/f4_valener_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --time=00:45:00
# ===========================================================================
# Valida el arreglo de la VENTANA DE ENERGIA en la ruta WMMA.
# ===========================================================================
# El fallo: esa ruta entra al bloque de volcado por write_fp32, cierto tambien
# en la ultima iteracion medida aunque no haya checkpointing. El cierre del
# tramo energetico estaba gateado por exclude_checkpoint_energy = false en las
# corridas RUN_KIND=energy, asi que el D2H final (~1 s a 16384^2) caia DENTRO
# de energy_j mientras t_ms_iter si lo excluia.
#
# CRITERIO, por construccion avg_power_w = energy_j / energy_wall_s, luego
# E/P ES la ventana. Se exige que coincida con t_ms_total en TODAS las rutas:
#     |E/P - t_total| / t_total <= 5 %
# Antes del arreglo la ruta WMMA daba ~3.9x a 16384^2 con iters=200.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
JOB_ID="${SLURM_JOB_ID:-manual}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
SRC_DIR="${REPO_ROOT}/Fase_4/Stencil"
NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"
CUDA_ARCH="${CUDA_ARCH:-80}"
WORK="results/valener_${JOB_ID}"; mkdir -p "${WORK}"
BIN="${PWD}/${WORK}/stencil_tc_${JOB_ID}"

echo "=============== PROCEDENCIA ==============="
echo "HEAD        : $(git -C "${REPO_ROOT}" rev-parse HEAD)"
git -C "${REPO_ROOT}" diff --quiet 2>/dev/null && echo "git_status  : limpio" || echo "git_status  : sucio"
sha256sum "${SRC_DIR}/stencil_tensor_activation.cu" | awk '{print "sha256 .cu  : "$1}'
nvidia-smi --query-gpu=name,power.limit --format=csv,noheader
echo

echo "=============== COMPILANDO ==============="
( cd "${SRC_DIR}" && "${NVCC}" -std=c++17 stencil_tensor_activation.cu -o "${BIN}" \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
    --allow-unsupported-compiler -DUSE_NVML_TELEMETRY -lnvidia-ml -lpthread ) \
  || { echo "VEREDICTO: FALLO (no compila)"; exit 1; }
echo "compilado OK"
echo

# Misma malla e iteraciones que la campana, --profile-only para que corra
# solo GPU_FP32 (control, ya estaba limpia) + la ruta WMMA (la sospechosa).
CSV="${WORK}/valener_${JOB_ID}.csv"
for TC in fp16 bf16; do
  echo "=============== CORRIDA tc=${TC} ==============="
  "${BIN}" --nx 16384 --ny 16384 --iters 200 --tc "${TC}" --kahan off \
           --profile-only --op-mode diffusive --alpha 0.1875 \
           --ci-mode monomode --ci-p 168 --csv "${CSV}"
done

echo
echo "=============== VEREDICTO ==============="
awk -F, 'NR>1 {
    t_total_s = $8/1000.0; E = $21; P = $22;
    if (E=="NA" || P=="NA" || P+0==0) { printf "  %-10s SIN TELEMETRIA (E=%s P=%s)\n", $2, E, P; next }
    win = E/P; d = (win - t_total_s)/t_total_s;
    veredicto = (d<0.05 && d>-0.05) ? "OK" : "*** DESVIADO ***";
    printf "  %-10s t_total=%7.4f s  ventana(E/P)=%7.4f s  desvio=%+7.2f %%  %s\n",
           $2, t_total_s, win, 100*d, veredicto;
    if (d>=0.05 || d<=-0.05) fallo=1;
} END { exit (fallo?1:0) }' "${CSV}"
rc=$?
echo
if [ $rc -eq 0 ]; then echo "VEREDICTO GLOBAL: LIMPIO - la ventana de energia coincide con la de tiempo";
else echo "VEREDICTO GLOBAL: FALLIDO - NO liberar 6753"; fi
exit $rc
