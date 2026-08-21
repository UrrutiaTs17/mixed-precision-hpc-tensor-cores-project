#!/bin/bash
#SBATCH --job-name=f4_gate1_regresion
#SBATCH --output=logs/f4_gate1_%j.out
#SBATCH --error=logs/f4_gate1_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#
# GATE 1 de la campana de Pareto 3D (Fase 4).
#
# Que verifica
# ------------
# Que parametrizar el operador de 5 puntos (--op-mode / --alpha) y la condicion
# inicial (--ci-mode / --ci-p / --ci-amplitude) NO altero el comportamiento del
# caso historico. Compila DOS binarios -- el del commit anterior al cambio y el
# de HEAD -- y los corre sobre las mismas dos configuraciones, contrastando su
# salida linea a linea.
#
# El binario nuevo se corre DOS veces por configuracion:
#   (a) con --op-mode stress --ci-mode legacy explicitos,
#   (b) sin ningun flag nuevo,
# porque el gate tiene que probar dos cosas distintas: que el operador de estres
# sigue siendo el de antes, y que es el DEFAULT. Si (a) pasa y (b) no, el
# default se rompio; si (b) pasa y (a) no, los flags no hacen lo que dicen.
#
# Sobre "cero diferencias"
# ------------------------
# El stdout mezcla contenido determinista (rel_l2, rel_linf, max_abs, store_*,
# n_star, CSV_DRIFT/CSV_HORIZON/CSV_ONSET) con contenido de medicion (tiempos,
# GFLOP/s, speedups, energia). Dos corridas del MISMO binario ya difieren en el
# segundo grupo por ruido de reloj, estado termico y cuantizacion del contador
# NVML. El veredicto se decide SOLO sobre el primer grupo; el segundo se reporta
# como desviacion relativa. Ver tools/comparar_gate1.py.
#
# Uso:
#   sbatch gate1_regresion.sbatch
#   BASE_COMMIT=<sha> sbatch gate1_regresion.sbatch
#
# Variables:
#   BASE_COMMIT  commit con el codigo ANTERIOR al cambio (def. 2e2dd29, punta de
#                fase4-estadistica-variabilidad antes de esta campana). Su
#                stencil_tensor_activation.cu es identico al de 0f10851, el
#                commit que el enunciado declara como estado verificado, asi que
#                cualquiera de los dos sirve de base.
#   NX, NY       malla del gate (def. 512) -- pequena a proposito: el gate mide
#                identidad numerica, no rendimiento, y tiene que caber en un
#                walltime corto
#   ITERS        iteraciones (def. 20)
#   TOLERANCIA   desviacion relativa admitida en columnas de medicion (def. 0.15)

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

REPO_ROOT="$(git rev-parse --show-toplevel)"
BASE_COMMIT="${BASE_COMMIT:-2e2dd29}"
NX="${NX:-512}"
NY="${NY:-512}"
ITERS="${ITERS:-20}"
TOLERANCIA="${TOLERANCIA:-0.15}"
CUDA_ARCH="${CUDA_ARCH:-80}"
NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"

JOB_ID="${SLURM_JOB_ID:-manual}"
WORK="gate1_${JOB_ID}"
mkdir -p "${WORK}/base" "${WORK}/salidas"

echo "Nodo        : $(hostname)"
echo "Fecha       : $(date -Iseconds)"
echo "REPO_ROOT   : ${REPO_ROOT}"
echo "HEAD        : $(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
echo "BASE_COMMIT : ${BASE_COMMIT} ($(git -C "${REPO_ROOT}" log -1 --format=%s "${BASE_COMMIT}"))"
echo "Malla       : ${NX}x${NY}, ITERS=${ITERS}"
nvidia-smi || true
echo

# --- Deteccion de NVML, identica a run_stencil_tc.sbatch -------------------
# El gate DEBE compilar con los mismos flags que la campana: un binario sin
# telemetria produciria NaN en las columnas de energia y el comparador lo leeria
# como una diferencia de comportamiento en vez de como una diferencia de build.
NVML_FLAGS=()
NVML_LINK_FLAG="${NVML_LINK_FLAG:--lnvidia-ml}"
if [[ "${DISABLE_NVML_TELEMETRY:-0}" != "1" ]] &&
   "${NVCC}" -E -x cu - 2>/dev/null <<< '#include <nvml.h>' >/dev/null &&
   (ldconfig -p 2>/dev/null | grep -Eq 'lib(nvidia-ml|nvml)\.so' ||
    [[ -f /usr/lib64/libnvidia-ml.so || -f /usr/lib64/libnvml.so ||
       -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so ||
       -f /usr/lib/x86_64-linux-gnu/libnvml.so ]]); then
    NVML_FLAGS=(-DUSE_NVML_TELEMETRY "${NVML_LINK_FLAG}" -lpthread)
    echo "NVML habilitado para telemetria GPU."
else
    NVML_FLAGS=(-lpthread)
    echo "NVML no disponible para el enlace; se usara el stub de telemetria GPU."
fi

compilar() {   # $1 = directorio con el .cu   $2 = ruta absoluta del binario
    local dir="$1" salida="$2"
    ( cd "${dir}" && "${NVCC}" -std=c++17 stencil_tensor_activation.cu -o "${salida}" \
        -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
        -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
        --allow-unsupported-compiler "${NVML_FLAGS[@]}" )
}

# --- Binario BASE: arbol completo del commit anterior ----------------------
# Se extrae el arbol ENTERO y no solo el .cu porque el fuente incluye
# "../../Fase_2/common.cuh" (relativo al .cu) y "tools/power_sampling.h": solo
# reproduciendo el layout del repo esos includes resuelven al codigo de ESE
# commit y no al de HEAD, que es justamente lo que invalidaria la comparacion.
echo "== Extrayendo el arbol de ${BASE_COMMIT} =="
git -C "${REPO_ROOT}" archive "${BASE_COMMIT}" | tar -x -C "${WORK}/base"

BIN_BASE="${PWD}/${WORK}/stencil_tc_base"
BIN_NUEVO="${PWD}/${WORK}/stencil_tc_nuevo"

echo "== Compilando binario BASE (${BASE_COMMIT}) =="
compilar "${WORK}/base/Fase_3/Stencil" "${BIN_BASE}"
echo "== Compilando binario NUEVO (HEAD) =="
compilar "${REPO_ROOT}/Fase_3/Stencil" "${BIN_NUEVO}"

echo "sha256 BASE  : $(sha256sum "${BIN_BASE}"  | awk '{print $1}')"
echo "sha256 NUEVO : $(sha256sum "${BIN_NUEVO}" | awk '{print $1}')"
echo

# --- Configuraciones del gate ----------------------------------------------
# Las dos politicas de compensacion se corren por separado porque el binario
# rechaza --kahan on junto con --spatial-comp on (son alternativas, no capas).
CFG_NOMBRES=("kahan_on" "spatial_on")
CFG_FLAGS=("--kahan on" "--spatial-comp on")

S="${WORK}/salidas"
fallos_totales=0

for i in 0 1; do
    nombre="${CFG_NOMBRES[$i]}"
    # shellcheck disable=SC2206
    flags=(${CFG_FLAGS[$i]})
    comun=(--nx "${NX}" --ny "${NY}" --iters "${ITERS}" --tc both "${flags[@]}")

    echo "############################################################"
    echo "# Configuracion: ${nombre}   (${CFG_FLAGS[$i]})"
    echo "############################################################"

    echo "-- corriendo BASE --"
    "${BIN_BASE}"  "${comun[@]}" --csv "${S}/${nombre}_base.csv"        > "${S}/${nombre}_base.log"  2>&1
    echo "-- corriendo NUEVO con --op-mode stress --ci-mode legacy explicitos --"
    "${BIN_NUEVO}" "${comun[@]}" --op-mode stress --ci-mode legacy \
                   --csv "${S}/${nombre}_explicito.csv"                 > "${S}/${nombre}_explicito.log" 2>&1
    echo "-- corriendo NUEVO sin flags nuevos (prueba del default) --"
    "${BIN_NUEVO}" "${comun[@]}" --csv "${S}/${nombre}_default.csv"     > "${S}/${nombre}_default.log"   2>&1

    for variante in explicito default; do
        set +e
        python3 tools/comparar_gate1.py \
            --base  "${S}/${nombre}_base.log" \
            --nuevo "${S}/${nombre}_${variante}.log" \
            --etiqueta "${nombre} / nuevo-${variante}" \
            --tolerancia-medicion "${TOLERANCIA}"
        rc=$?
        set -e
        [[ ${rc} -ne 0 ]] && fallos_totales=$((fallos_totales + 1))
        echo
    done

    # El binario nuevo tiene que ser identico a si mismo entre (a) y (b): si
    # esto falla, el default no es el que dice ser y las dos comparaciones
    # anteriores contra BASE no significan lo mismo.
    echo "-- explicito vs default (deben ser la MISMA configuracion) --"
    set +e
    diff <(grep -E '^(Operador|Coef vecino|Condicion inicial|Alpha|FLOPs por celda)' "${S}/${nombre}_explicito.log") \
         <(grep -E '^(Operador|Coef vecino|Condicion inicial|Alpha|FLOPs por celda)' "${S}/${nombre}_default.log")
    rc=$?
    set -e
    if [[ ${rc} -eq 0 ]]; then
        echo "   OK: --op-mode stress --ci-mode legacy es efectivamente el default."
    else
        echo "   X  El default NO coincide con los flags explicitos."
        fallos_totales=$((fallos_totales + 1))
    fi
    echo

    # Diff literal del stdout, con las lineas de medicion filtradas. No decide
    # el veredicto (para eso esta el comparador); esta para que quede en el log
    # la evidencia cruda de que lo unico que cambia son las lineas nuevas de
    # configuracion y las columnas nuevas del CSV.
    echo "-- diff literal de stdout (sin lineas de tiempo/energia) --"
    filtro='^(.*tiempo|.*rend\.|Speedup|t kernel|t conversion|t no atribuido|t checkpoints|Energy |EDP|Joules/GFLOP|Sobrecosto|CSV_ENERGY|CSV_REGION|CSV_SUMMARY|Fecha|Nodo|BUILD_DATE)'
    set +e
    diff -u <(grep -Ev "${filtro}" "${S}/${nombre}_base.log") \
            <(grep -Ev "${filtro}" "${S}/${nombre}_default.log")
    echo "   (fin del diff literal; vacio = identico fuera de medicion y columnas nuevas)"
    set -e
    echo

    # Cabecera del CSV opcional: BASE trae 23 columnas y NUEVO 30. Se imprime
    # para dejar constancia de que las 7 nuevas van al FINAL y no corrieron
    # ninguna posicion del esquema anterior.
    echo "-- cabecera del CSV opcional --"
    echo "   BASE  : $(head -1 "${S}/${nombre}_base.csv")"
    echo "   NUEVO : $(head -1 "${S}/${nombre}_default.csv")"
    echo
done

echo "############################################################"
if [[ ${fallos_totales} -eq 0 ]]; then
    echo "# GATE 1: PASA. Ninguna diferencia determinista en ninguna"
    echo "# configuracion. Fase 1 queda validada; se puede avanzar."
else
    echo "# GATE 1: FALLA en ${fallos_totales} comparacion(es)."
    echo "# NO avanzar a la campana. Revisar los bloques 'X' de arriba."
fi
echo "# Salidas crudas en: ${S}/"
echo "############################################################"
exit "${fallos_totales}"
