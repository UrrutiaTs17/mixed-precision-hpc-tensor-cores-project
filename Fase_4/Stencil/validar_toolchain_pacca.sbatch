#!/bin/bash
#SBATCH --job-name=f4_validar_toolchain
#SBATCH --output=logs/f4_valtool_%j.out
#SBATCH --error=logs/f4_valtool_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
# ===========================================================================
# TAREA 1: validar Fase_4/Stencil/stencil_tensor_activation.cu contra la
# toolchain REAL del cluster (hpc_sdk 23.1), no la nvcc 13.1 local.
# ===========================================================================
# Todo lo medido en la auditoria salio de nvcc 13.1 sobre una RTX 3050. Este
# job es el unico que decide si esas cifras son citables. No lanzar el
# anclaje ni ningun piloto si este job no sale limpio.
#
# Reproduce EXACTAMENTE la configuracion en la que se tomaron los valores de
# referencia locales: 1024^2, 20 iters, diffusive, alpha=3/16, monomode,
# ci_p=168, sin GPU_FP64 ni la segunda pasada CPU_FP64.
set -uo pipefail          # sin -e: queremos recorrer TODOS los chequeos
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
JOB_ID="${SLURM_JOB_ID:-manual}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
SRC_DIR="${REPO_ROOT}/Fase_4/Stencil"
NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"
CUDA_ARCH="${CUDA_ARCH:-80}"
WORK="results/valtool_${JOB_ID}"; mkdir -p "${WORK}"
fallos=0

echo "=============== PROCEDENCIA ==============="
echo "HEAD       : $(git -C "${REPO_ROOT}" rev-parse HEAD)"
echo "nvcc       : ${NVCC}"
"${NVCC}" --version | tail -2
echo "arch       : sm_${CUDA_ARCH}"
for f in "${SRC_DIR}/stencil_tensor_activation.cu" "${REPO_ROOT}/Fase_2/common.cuh" \
         "${SRC_DIR}/tools/power_sampling.h"; do
    echo "sha256     : $(sha256sum "${f}" | awk '{print $1}')  ${f#${REPO_ROOT}/}"
done
nvidia-smi --query-gpu=name --format=csv,noheader
echo

# --------------------------------------------------------------------------
# CHEQUEO 1: la formula que se va a compilar es la esperada
# --------------------------------------------------------------------------
echo "########## CHEQUEO 1: formula de compensated_store<Local> ##########"
CU="${SRC_DIR}/stencil_tensor_activation.cu"
if grep -q 'const float y = val - comp\[idx\];' "${CU}" &&
   grep -q 'comp\[idx\] = tc_to_float(s) - y;' "${CU}"; then
    echo "  OK  formula ORIGINAL de error-feedback presente (y = val - comp[idx])"
else
    echo "  X   la formula original NO esta; el fuente no es el validado"; fallos=$((fallos+1))
fi
if grep -cq 'comp_mode == CompMode::Local' "${CU}"; then
    echo "  X   aparece CompMode::Local en la reconstruccion; deberia estar solo en Spatial"
    grep -n 'comp_mode == CompMode::Local' "${CU}"; fallos=$((fallos+1))
else
    echo "  OK  reconstruccion restringida a Spatial (sin CompMode::Local)"
fi
echo

# --------------------------------------------------------------------------
# CHEQUEO 2: compilacion, cero errores y solo los DOS warnings conocidos
# --------------------------------------------------------------------------
echo "########## CHEQUEO 2: compilacion con hpc_sdk ##########"
NVML_FLAGS=()
NVML_LINK_FLAG="${NVML_LINK_FLAG:--lnvidia-ml}"
if [[ "${DISABLE_NVML_TELEMETRY:-0}" != "1" ]] &&
   "${NVCC}" -E -x cu - 2>/dev/null <<< '#include <nvml.h>' >/dev/null &&
   (ldconfig -p 2>/dev/null | grep -Eq 'lib(nvidia-ml|nvml)\.so' ||
    [[ -f /usr/lib64/libnvidia-ml.so || -f /usr/lib64/libnvml.so ]]); then
    NVML_FLAGS=(-DUSE_NVML_TELEMETRY "${NVML_LINK_FLAG}" -lpthread)
    echo "  NVML habilitado."
else
    NVML_FLAGS=(-lpthread); echo "  NVML no disponible; stub de telemetria."
fi

BIN="${PWD}/${WORK}/stencil_tc"
CLOG="${PWD}/${WORK}/compilacion.log"
( cd "${SRC_DIR}" && "${NVCC}" -std=c++17 stencil_tensor_activation.cu -o "${BIN}" \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
    --allow-unsupported-compiler "${NVML_FLAGS[@]}" ) > "${CLOG}" 2>&1
rc=$?
if [[ ${rc} -ne 0 ]]; then
    echo "  X   COMPILACION FALLIDA (rc=${rc}). PARAR."; sed -n '1,60p' "${CLOG}"; exit 1
fi
echo "  OK  compila, rc=0"
echo "  sha256 binario: $(sha256sum "${BIN}" | awk '{print $1}')"

# OJO: se cuenta 'warning #', no 'warning' a secas. nvcc cierra el bloque de
# diagnosticos con la linea informativa
#   Remark: The warnings can be suppressed with "-diag-suppress <numero>"
# que contiene la palabra "warnings" y NO es un warning. Contarla dio un falso
# positivo en el job 6744 (3 en vez de 2) que marco FAILED una compilacion
# perfectamente limpia. La misma linea aparece en los logs locales de nvcc 13.1,
# asi que el fallo era del contador, nunca del codigo ni de la toolchain.
n_warn=$(grep -c "warning #" "${CLOG}")
echo "  warnings emitidos: ${n_warn}  (esperados 2)"
grep -q 'variable "xcol" was declared but never referenced' "${CLOG}" \
  && echo "  OK  warning conocido 1/2: xcol #177-D" \
  || { echo "  X   falta el warning conocido de xcol"; fallos=$((fallos+1)); }
grep -q 'ReferenceSpill::is_open' "${CLOG}" \
  && echo "  OK  warning conocido 2/2: ReferenceSpill::is_open #177-D" \
  || { echo "  X   falta el warning conocido de ReferenceSpill::is_open"; fallos=$((fallos+1)); }
if [[ ${n_warn} -gt 2 ]]; then
    echo "  X   HAY WARNINGS NUEVOS respecto a la linea base local:"
    grep "warning #" "${CLOG}" | grep -vE 'xcol|ReferenceSpill' | sed 's/^/      /'
    fallos=$((fallos+1))
fi
echo

# --------------------------------------------------------------------------
# CHEQUEO 3: reproduccion numerica de los valores medidos en local
# --------------------------------------------------------------------------
echo "########## CHEQUEO 3: reproduccion numerica ##########"
echo "Configuracion identica a la de las mediciones locales:"
echo "  1024^2, 20 iters, diffusive, alpha=0.1875, monomode, ci_p=168"
echo
BASE=(--nx 1024 --ny 1024 --iters 20 --tc both --op-mode diffusive --alpha 0.1875
      --ci-mode monomode --ci-p 168 --fp64-gpu off --cpu-fp64 off)

# $1 etiqueta  $2 flags  $3 esperado FP16  $4 esperado BF16
comparar() {
    local nom="$1" flags="$2" e16="$3" e64="$4"
    local out="${WORK}/${nom}.log"
    "${BIN}" "${BASE[@]}" ${flags} > "${out}" 2>&1
    local vals; vals=$(grep "L2 (estado propagado" "${out}" | awk '{print $NF}')
    local v16 v64; v16=$(echo ${vals} | cut -d' ' -f1); v64=$(echo ${vals} | cut -d' ' -f2)
    echo "--- ${nom} (${flags}) ---"
    python3 - "$v16" "$v64" "$e16" "$e64" <<'PY'
import sys
v16,v64,e16,e64 = map(float, sys.argv[1:5])
ok=True
for lbl,v,e in (("FP16",v16,e16),("BF16",v64,e64)):
    d = abs(v-e)/e
    marca = "OK " if d <= 1e-3 else "X  "
    if d > 1e-3: ok=False
    exacto = "exacto" if v==e else "difiere en %.3g%%"%(100*d)
    print("  %s %s medido=%.6e  local=%.6e  (%s)"%(marca,lbl,v,e,exacto))
sys.exit(0 if ok else 1)
PY
    [[ $? -ne 0 ]] && fallos=$((fallos+1))
}

comparar kahan_off     "--kahan off"        2.271950e-01  1.780251e+00
comparar kahan_on      "--kahan on"         1.679329e-01  1.343621e+00
comparar spatial_on    "--spatial-comp on"  2.017558e-05  2.017665e-05
echo
echo "Nota: la tolerancia es 1e-3 relativo. 'exacto' significa reproduccion"
echo "bit a bit entre toolchains; una desviacion <=0.1% se acepta pero se"
echo "reporta, porque indica codegen distinto y no debe pasar desapercibida."
echo

# --------------------------------------------------------------------------
echo "############ VEREDICTO ############"
if [[ ${fallos} -eq 0 ]]; then
    echo "LIMPIO: ${fallos} fallos. Las cifras de la auditoria quedan validadas"
    echo "en la toolchain real. Se puede proceder al anclaje."
    exit 0
else
    echo "FALLOS: ${fallos}. NO lanzar el anclaje ni ningun piloto."
    exit 1
fi
