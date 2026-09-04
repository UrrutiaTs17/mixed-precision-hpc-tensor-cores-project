#!/bin/bash
#SBATCH --job-name=f4_anclaje_reflinf
#SBATCH --output=logs/f4_anclaje_%j.out
#SBATCH --error=logs/f4_anclaje_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=04:00:00
# ===========================================================================
# ANCLAJE de la extrapolacion de ref_linf al tamano REAL de la campana.
# ===========================================================================
# Punto unico: 16384^2, diffusive, monomode, ci_p=168, alpha=3/16, 640 iters.
#
# Por que existe: la escalera 1024/2048/4096/8192 a p/nx constante reprodujo
# g^640 con desviacion <=0.01% en los cuatro puntos, pero 16384^2 nunca corrio.
# Esto cierra la extrapolacion con un punto medido antes de comprometer las
# replicas de la campana.
#
# ADVERTENCIA: el binario NO tiene modo "solo referencia CPU". No existe
# bandera para saltarse las rutas GPU. Lo mas cerca que se puede dejar es:
#   --fp64-gpu off   quita la ruta GPU_FP64 (la mas cara en VRAM: 8 B/celda)
#   --cpu-fp64 off   quita la SEGUNDA pasada FP64 de CPU (la de cronometraje);
#                    el ground truth FP64 se calcula igual, es lo que se mide
#   --tc fp16        una sola ruta WMMA en vez de las dos
# Con eso siguen corriendo CPU_FP32, GPU_FP32 y WMMA_FP16. Es el minimo
# alcanzable sin tocar el .cu, y el .cu esta congelado para este anclaje.
#
# COSTO, recalibrado con la medida REAL del job 6746 (no con la estimacion
# local, que fallaba ~6x y mato al job 6745 por timeout a los 60 min):
#   6746 midio 75 min por invocacion de 320 iters a 16384^2 con la config
#   completa. Este job apaga FP64_GPU y la 2a pasada CPU_FP64, ~35% menos.
#     pasada 1 (640 iters, diffusive) : ~98 min
#     pasada 2 (160 iters, stress)    : ~30 min
#     compilacion                     : ~4 min      TOTAL ~2.2 h
# De ahi --time=04:00:00.
#
# SOLO_HORIZONTE=1 salta la pasada 1 y deja unicamente el horizonte (~34 min).
# Tiene sentido porque el job 6746 YA midio a 16384^2 el decaimiento del modo
# dominante: lambda = 9.984438e-01 con R^2 = 1.000000 sobre 96 puntos, que
# coincide con la formula idealizada en 2.5e-08 e implica ref_linf(640) =
# 0.3691, dentro del rango de aceptacion. Con SOLO_HORIZONTE=1 ese valor queda
# INFERIDO de una tasa medida a escala real, no medido directamente a 640
# iteraciones -- es una extrapolacion segura sobre un operador lineal de
# espectro fijo, pero es una extrapolacion y hay que declararla como tal.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
JOB_ID="${SLURM_JOB_ID:-manual}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
SRC_DIR="${REPO_ROOT}/Fase_4/Stencil"
NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"
CUDA_ARCH="${CUDA_ARCH:-80}"

BIN="${PWD}/results/stencil_anclaje_${JOB_ID}"
echo "Compilando desde ${SRC_DIR} ..."
( cd "${SRC_DIR}" && "${NVCC}" -std=c++17 stencil_tensor_activation.cu -o "${BIN}" \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
    --allow-unsupported-compiler -lpthread )

echo "=============== PROCEDENCIA ==============="
echo "git_commit : $(git -C "${REPO_ROOT}" rev-parse HEAD)"
for f in "${SRC_DIR}/stencil_tensor_activation.cu" "${REPO_ROOT}/Fase_2/common.cuh" \
         "${SRC_DIR}/tools/power_sampling.h"; do
    echo "sha256     : $(sha256sum "${f}" | awk '{print $1}')  ${f#${REPO_ROOT}/}"
done
echo "nvcc       : ${NVCC}   arch: sm_${CUDA_ARCH}"
echo "==========================================="

SOLO_HORIZONTE="${SOLO_HORIZONTE:-0}"
CSV="${PWD}/results/anclaje_reflinf_${JOB_ID}.csv"
if [[ "${SOLO_HORIZONTE}" == "1" ]]; then
    echo "### PASADA 1 OMITIDA (SOLO_HORIZONTE=1) ###"
    echo "ref_linf(640) se toma del lambda medido por el job 6746:"
    echo "   lambda = 9.984438e-01  ->  lambda^640 = 3.691e-01  (INFERIDO)"
else
/usr/bin/time -v "${BIN}" \
    --nx 16384 --ny 16384 --iters 640 --tc fp16 \
    --op-mode diffusive --alpha 0.1875 \
    --ci-mode monomode --ci-p 168 --ci-amplitude 1.0 \
    --fp64-gpu off --cpu-fp64 off \
    --csv "${CSV}"
fi


# ===========================================================================
# PASADA 2: horizonte de desbordamiento a 16384^2
# ===========================================================================
# NO se puede sacar de la pasada 1: el horizonte es un fenomeno del operador
# STRESS (g(pi,pi) = -2 amplifica el modo Nyquist hasta desbordar). Bajo
# diffusive |g| < 1 en todo el espectro y NADA diverge -- la pasada 1
# reportara "ninguna (finito hasta iters=640)" para todas las rutas.
#
# Se resuelve dentro de ESTE MISMO job (no hay un segundo sbatch): una
# invocacion extra con --op-mode stress y CI legacy. 160 iters basta para
# cruzar los tres horizontes esperados (28 / 138 / 142) sin pagar 640.
# Costo estimado ~3 min sobre el mismo nodo ya reservado.
CSV_H="${PWD}/results/anclaje_horizonte_${JOB_ID}.csv"
echo
echo "### PASADA 2: horizonte bajo stress (CI legacy, 160 iters) ###"
"${BIN}" \
    --nx 16384 --ny 16384 --iters 160 --tc both \
    --op-mode stress \
    --fp64-gpu off --cpu-fp64 off \
    --csv "${CSV_H}"

echo
echo "############ CRITERIO DE ACEPTACION ############"
echo "g^640 predicho para 16384^2 / ci_p=168 : 3.691e-01"
if [[ "${SOLO_HORIZONTE}" == "1" ]]; then
  echo "ref_linf: NO MEDIDO en este job (pasada 1 omitida); inferido = 3.691e-01"
else
echo "ref_linf medido:"
awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="ref_linf") c=i; next} {print "   "$c}' "${CSV}"
echo "Se acepta si cae en [3.6e-01, 3.8e-01]. Un colapso de ordenes de"
echo "magnitud (p.ej. 1e-08) refutaria la extrapolacion y hay que parar."
fi
echo
echo "Horizonte medido en local a 1024/2048/4096/8192 (identico en los cuatro):"
echo "   wmma_fp16 = 28   wmma_bf16 = 138   gpu_fp32 = 142"
echo "Horizonte medido AQUI a 16384^2 (columna n_star):"
awk -F, 'NR==1{for(i=1;i<=NF;i++){if($i=="formato")f=i; if($i=="n_star")n=i} next}
         $f=="wmma_fp16"||$f=="wmma_bf16"||$f=="gpu_fp32"{printf "   %-10s = %s\n",$f,$n}' "${CSV_H}"
echo "Se acepta si coinciden. Cualquier desviacion se reporta con el numero"
echo "exacto, sin redondear a \"consistente\"."
