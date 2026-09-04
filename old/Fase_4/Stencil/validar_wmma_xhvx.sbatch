#!/bin/bash
#SBATCH --job-name=f4_valida_xhvx
#SBATCH --output=logs/f4_valida_xhvx_%j.out
#SBATCH --error=logs/f4_valida_xhvx_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00

# ===========================================================================
# VALIDACION PEQUENA de la formulacion WMMA Y = X H + V X
#
# Sustituye a la formulacion anterior de cinco mma_sync contra matrices
# identidad. Este job NO mide rendimiento de produccion: comprueba correccion
# numerica, recursos de compilacion y ausencia de accesos ilegales sobre mallas
# minusculas, antes de autorizar cualquier medicion seria.
#
# NO exige igualdad bit a bit con la implementacion eliminada: el orden de
# acumulacion cambio (dos productos matriciales mas cuatro bandas en FP32, en
# vez de cinco productos), asi que se exige equivalencia MATEMATICA y un error
# compatible con FP16/BF16 acumulando en FP32.
#
# Uso:
#   sbatch validar_wmma_xhvx.sbatch
#   sbatch --export=ALL,SANITIZER=0 validar_wmma_xhvx.sbatch
# ===========================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

REPO_ROOT="$(git rev-parse --show-toplevel)"
CUDA_ARCH="${CUDA_ARCH:-80}"
NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"
SANITIZER="${SANITIZER:-1}"

JOB_ID="${SLURM_JOB_ID:-manual}"
WORK="valida_xhvx_${JOB_ID}"
mkdir -p "${WORK}/salidas" "${WORK}/archivo"

echo "Nodo        : $(hostname)"
echo "Fecha       : $(date -Iseconds)"
echo "Commit      : $(git -C "${REPO_ROOT}" rev-parse HEAD)"
echo "Arquitectura: sm_${CUDA_ARCH}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

# --- NVML (mismo bloque que los demas sbatch de la fase) --------------------
NVML_LINK_FLAG="${NVML_LINK_FLAG:--lnvidia-ml}"
if "${NVCC}" -E -x cu - 2>/dev/null <<< '#include <nvml.h>' >/dev/null &&
   (ldconfig -p 2>/dev/null | grep -Eq 'lib(nvidia-ml|nvml)\.so' ||
    [[ -f /usr/lib64/libnvidia-ml.so || -f /usr/lib64/libnvml.so ]]); then
    NVML_FLAGS=(-DUSE_NVML_TELEMETRY "${NVML_LINK_FLAG}" -lpthread)
else
    NVML_FLAGS=(-lpthread)
fi

# ===========================================================================
# ETAPA 1 — Compilacion con reporte de recursos (-Xptxas=-v)
# ===========================================================================
echo "############################################################"
echo "# ETAPA 1: recursos de compilacion (ptxas)"
echo "############################################################"

BIN="${PWD}/${WORK}/stencil_tc"
PTXAS_LOG="${WORK}/salidas/ptxas.txt"

( cd "${REPO_ROOT}/Fase_3/Stencil" && "${NVCC}" -std=c++17 stencil_tensor_activation.cu -o "${BIN}" \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
    -Xptxas=-v \
    --allow-unsupported-compiler "${NVML_FLAGS[@]}" ) 2> "${PTXAS_LOG}"

echo "-- Recursos por kernel (bruto de ptxas) --"
cat "${PTXAS_LOG}"
echo
echo "-- Resumen del kernel WMMA --"
# Las tres instanciaciones (Off/Local/Spatial) x dos tipos (FP16/BF16) salen
# como funciones distintas; se filtran por nombre.
grep -A3 -i "stencil2d_wmma_kernel" "${PTXAS_LOG}" || echo "  (sin lineas de ptxas para el kernel WMMA)"
echo
echo "-- Comprobacion: NO debe haber spills --"
if grep -qi "spill" "${PTXAS_LOG}"; then
    echo "  ATENCION: ptxas reporta spills:"
    grep -i "spill" "${PTXAS_LOG}"
else
    echo "  OK: cero spill stores y cero spill loads."
fi
echo
echo "-- Shared memory esperada por WARP (calculo del codigo) --"
echo "  Off / Local : 256*2 (X) + 256*4 (out) + 4*16*2 (bandas)      = 1664 B"
echo "  Spatial     : 1664 + 256*4 (comp) + 4*16*4 (comp bandas)     = 2944 B"
echo "  Diseno anterior (cinco tiles + salida), en TODOS los modos   = 3584 B"
echo "  Por bloque con kWarpsPerBlock=4: 6656 B / 11776 B / 14336 B"
echo
echo "sha256 binario: $(sha256sum "${BIN}" | awk '{print $1}')"
echo

# ===========================================================================
# ETAPA 2 — Correccion numerica en mallas pequenas
# ===========================================================================
echo "############################################################"
echo "# ETAPA 2: correccion numerica en mallas pequenas"
echo "############################################################"
echo "Mallas: 32x32 (tiles parciales), 34x34 (interior multiplo exacto de 16),"
echo "        63x65 (parcial en ambas dimensiones), 511x509 (parcial, grande)."
echo "El binario compara cada ruta contra la referencia CPU FP64 de la MISMA"
echo "condicion inicial, asi que rel_l2/rel_linf/max_abs salen del propio log."
echo

S="${WORK}/salidas"
fallos=0

correr() {   # $1 etiqueta  $2.. flags
    local etiqueta="$1"; shift
    local log="${S}/${etiqueta}.log"
    if "${BIN}" "$@" --cpu-fp64 off > "${log}" 2>&1; then
        # Posiciones de campo de CSV_SUMMARY, contadas sobre emit_csv_summary_row:
        #   $2 route  $7 t_iter_ms  $12 t_kernel_ms  $15 rel_l2  $16 rel_linf
        #   $17 max_abs  $20 first_nonfinite
        # OJO: NO coinciden con el orden del CSV de --csv (kCsvHeader), que es
        # otro esquema. Verificado contra el codigo, no supuesto.
        grep -E "^CSV_SUMMARY" "${log}" | awk -F, \
            '{printf "    %-14s rel_l2=%-13s rel_linf=%-13s max_abs=%-13s n*=%s\n", $2, $15, $16, $17, $20}'
    else
        echo "    FALLO DE EJECUCION (ver ${log})"
        fallos=$((fallos + 1))
    fi
}

for malla in "32 32" "34 34" "63 65" "511 509"; do
    set -- ${malla}
    NX=$1; NY=$2
    for opmode in stress diffusive; do
        for iters in 1 20; do
            for comp in "off:--kahan off" "local:--kahan on" "spatial:--spatial-comp on"; do
                nombre="${comp%%:*}"; flags="${comp#*:}"
                echo "  ${NX}x${NY}  op=${opmode}  iters=${iters}  comp=${nombre}"
                # shellcheck disable=SC2086
                correr "m${NX}x${NY}_${opmode}_i${iters}_${nombre}" \
                    --nx "${NX}" --ny "${NY}" --iters "${iters}" --tc both \
                    --op-mode "${opmode}" ${flags}
            done
        done
    done
done
echo

# --- Checkpoint pequeno -----------------------------------------------------
echo "-- Checkpoint pequeno (--checkpoint-every 5 sobre 20 iteraciones) --"
correr "checkpoint_63x65" --nx 63 --ny 65 --iters 20 --tc both \
    --op-mode diffusive --spatial-comp on --checkpoint-every 5
grep -cE "^CSV_DRIFT" "${S}/checkpoint_63x65.log" \
    | xargs -I{} echo "    filas CSV_DRIFT emitidas: {}"
echo

# --- Deteccion de no finitos ------------------------------------------------
# El operador de estres amplifica el modo Nyquist por 2 en cada iteracion, asi
# que FP16 desborda pronto. Se exige que first_nonfinite lo DETECTE (n* >= 0),
# no que la ruta sobreviva.
echo "-- Deteccion de no finitos (stress, 60 iteraciones, FP16) --"
correr "nofinitos_stress" --nx 63 --ny 65 --iters 60 --tc fp16 \
    --op-mode stress --kahan off
echo "   (n* >= 0 significa que el contador de overflow disparo, que es lo esperado)"
echo

# ===========================================================================
# ETAPA 3 — Posicion de la maxima diferencia (via campo archivado)
# ===========================================================================
# ErrorMetrics no guarda la POSICION del maximo, y anadirla tocaria una funcion
# de comparacion compartida por todas las rutas. En vez de eso se archivan los
# campos crudos de CPU_FP64 y de las rutas WMMA en la misma iteracion y se
# comparan fuera del binario.
echo "############################################################"
echo "# ETAPA 3: posicion de la maxima diferencia"
echo "############################################################"
"${BIN}" --nx 63 --ny 65 --iters 10 --tc both --op-mode diffusive \
    --spatial-comp on --cpu-fp64 on \
    --checkpoint-iters 10 --archive-iters 10 --archive-dir "${WORK}/archivo" \
    > "${S}/archivo.log" 2>&1 || echo "  FALLO (ver ${S}/archivo.log)"

if [[ -f "${REPO_ROOT}/Fase_4/Stencil/tools/verificar_campo.py" ]]; then
    python3 "${REPO_ROOT}/Fase_4/Stencil/tools/verificar_campo.py" \
        --dir "${WORK}/archivo" --nx 63 --ny 65 || true
else
    echo "  (tools/verificar_campo.py no encontrado; se omite)"
fi
echo

# ===========================================================================
# ETAPA 4 — compute-sanitizer
# ===========================================================================
echo "############################################################"
echo "# ETAPA 4: compute-sanitizer"
echo "############################################################"
if [[ "${SANITIZER}" != "1" ]]; then
    echo "SANITIZER=0, se omite."
elif ! command -v compute-sanitizer >/dev/null 2>&1; then
    echo "compute-sanitizer no esta en el PATH; se omite."
    echo "  (buscar en \$(dirname ${NVCC})/../../cuda/bin/compute-sanitizer)"
else
    # Malla con tiles parciales en ambas dimensiones: es donde un fallo de
    # indexacion en las bandas o en el halo de residuos se manifestaria.
    for cfg in "--kahan off" "--kahan on" "--spatial-comp on"; do
        echo "-- memcheck  ${cfg} --"
        # shellcheck disable=SC2086
        compute-sanitizer --tool memcheck --error-exitcode 1 \
            "${BIN}" --nx 63 --ny 65 --iters 5 --tc both --op-mode diffusive \
            --cpu-fp64 off ${cfg} > "${S}/sanitizer_${cfg// /_}.log" 2>&1 \
            && echo "   OK: sin accesos ilegales" \
            || { echo "   FALLO: ver ${S}/sanitizer_${cfg// /_}.log"; fallos=$((fallos + 1));
                 grep -m5 -E "Invalid|out of bounds|Error" "${S}/sanitizer_${cfg// /_}.log" || true; }
    done
fi
echo

# ===========================================================================
echo "############################################################"
if [[ ${fallos} -eq 0 ]]; then
    echo "# VALIDACION PEQUENA: sin fallos de ejecucion ni de sanitizer."
    echo "# Revisar a mano las columnas rel_l2/rel_linf de la ETAPA 2 antes"
    echo "# de pasar al microbenchmark."
else
    echo "# VALIDACION PEQUENA: ${fallos} fallo(s). NO pasar al microbenchmark."
fi
echo "# Salidas crudas en: ${WORK}/salidas/"
echo "############################################################"
