#!/bin/bash
#SBATCH --job-name=f4_micro_xhvx
#SBATCH --output=logs/f4_micro_xhvx_%j.out
#SBATCH --error=logs/f4_micro_xhvx_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --time=03:00:00
# Las etapas 1 y 2 terminan en minutos; el walltime largo es para la etapa 3,
# donde CPU_FP32 y la referencia FP64 iteran junto a la GPU (ver su bloque).

# ===========================================================================
# MICROBENCHMARK de la formulacion WMMA Y = X H + V X
#
# NO es la campana. Son corridas cortas para decidir si la nueva formulacion
# merece una campana. Lanzar SOLO despues de que validar_wmma_xhvx.sbatch salga
# limpio.
#
# --exclusive porque se miden tiempos y un vecino ruidoso los invalida.
#
# Etapas (ETAPA=1|2|3):
#   1  malla 4096x4096, 5 replicas, todas las rutas          (por defecto)
#   2  malla 16384x16384, 1 replica  -- solo si la 1 va bien
#   3  ventana energetica >= 2 s     -- solo si hay mejora temporal
#
# Uso:
#   sbatch microbench_wmma_xhvx.sbatch
#   sbatch --export=ALL,ETAPA=2 microbench_wmma_xhvx.sbatch
#   sbatch --export=ALL,ETAPA=3 microbench_wmma_xhvx.sbatch
#
# NOTA sobre el warm-up: kWarmupIters es una constante de compilacion del
# binario (=3, stencil_tensor_activation.cu:77) y no hay flag para subirla a
# 20 sin recompilar y romper la comparabilidad con todo lo medido antes. En su
# lugar se hace una corrida de CEBADO completa que se descarta, lo que cumple
# el mismo proposito (estabilizar relojes y estado termico) sin tocar el
# binario. Se documenta aqui para que no se lea como si fueran 20 warm-ups.
# ===========================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

REPO_ROOT="$(git rev-parse --show-toplevel)"
CUDA_ARCH="${CUDA_ARCH:-80}"
NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"
ETAPA="${ETAPA:-1}"
REPLICAS="${REPLICAS:-5}"
ITERS="${ITERS:-50}"

JOB_ID="${SLURM_JOB_ID:-manual}"
WORK="micro_xhvx_${JOB_ID}"
mkdir -p "${WORK}/salidas"
S="${WORK}/salidas"

echo "Nodo        : $(hostname)"
echo "Fecha       : $(date -Iseconds)"
echo "Commit      : $(git -C "${REPO_ROOT}" rev-parse HEAD)"
echo "ETAPA       : ${ETAPA}"
nvidia-smi --query-gpu=name,clocks.max.sm,power.limit --format=csv,noheader || true
echo

NVML_LINK_FLAG="${NVML_LINK_FLAG:--lnvidia-ml}"
if "${NVCC}" -E -x cu - 2>/dev/null <<< '#include <nvml.h>' >/dev/null &&
   (ldconfig -p 2>/dev/null | grep -Eq 'lib(nvidia-ml|nvml)\.so' ||
    [[ -f /usr/lib64/libnvidia-ml.so || -f /usr/lib64/libnvml.so ]]); then
    NVML_FLAGS=(-DUSE_NVML_TELEMETRY "${NVML_LINK_FLAG}" -lpthread)
    echo "NVML habilitado."
else
    NVML_FLAGS=(-lpthread)
    echo "NVML NO disponible: la etapa 3 no dara energia fiable."
fi

BIN="${PWD}/${WORK}/stencil_tc"
( cd "${REPO_ROOT}/Fase_3/Stencil" && "${NVCC}" -std=c++17 stencil_tensor_activation.cu -o "${BIN}" \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
    --allow-unsupported-compiler "${NVML_FLAGS[@]}" )
echo "sha256 binario: $(sha256sum "${BIN}" | awk '{print $1}')"
echo

# Campos de CSV_SUMMARY (contados sobre emit_csv_summary_row, NO sobre el CSV
# de --csv, que usa otro esquema):
#   $2 route  $7 t_iter_ms  $12 t_kernel_ms  $15 rel_l2  $25 energy_gpu_j
EXTRAE_TIEMPO='{printf "%s %s %s\n", $2, $7, $12}'

# --- Estadistica de las replicas (mediana, min, max, desviacion) ------------
# El script se escribe a un FICHERO y no se pasa por heredoc a `python3 -`:
# con `python3 - <<PY`, el heredoc ES el stdin del interprete (de ahi lee el
# programa), asi que sys.stdin queda agotado y el fichero de datos redirigido a
# la funcion nunca se lee. Ese fallo dejo la tabla vacia en el job 6683.
RESUMIR_PY="${WORK}/resumir.py"
cat > "${RESUMIR_PY}" <<'PY'
import sys, statistics as st
from collections import defaultdict
d = defaultdict(list)
for linea in sys.stdin:
    p = linea.split()
    if len(p) < 2: continue
    try: d[p[0]].append(float(p[1]))
    except ValueError: continue
if not d:
    print("  (sin datos: ningun CSV_SUMMARY extraido de los logs)")
    sys.exit(0)
print(f"  {'ruta':<22} {'n':>2} {'mediana':>11} {'min':>11} {'max':>11} {'desv':>10} {'CV%':>7}")
print("  " + "-"*78)
for ruta in sorted(d):
    v = d[ruta]
    med = st.median(v)
    sd  = st.stdev(v) if len(v) > 1 else 0.0
    cv  = 100.0*sd/med if med else float('nan')
    print(f"  {ruta:<22} {len(v):>2} {med:>11.6f} {min(v):>11.6f} {max(v):>11.6f} {sd:>10.6f} {cv:>7.2f}")
print()
print("  Unidades: ms por iteracion. CV% = desviacion/mediana.")
PY

resumir() {   # $1 = fichero con lineas "<ruta> <t_iter_ms>"
    python3 "${RESUMIR_PY}" < "$1"
}

correr_matriz() {   # $1 nx  $2 ny  $3 replicas  $4 etiqueta
    local NX="$1" NY="$2" REPS="$3" ETI="$4"
    local acumulado="${S}/${ETI}_tiempos.txt"
    : > "${acumulado}"

    # Cebado descartable: estabiliza relojes y estado termico. Su salida no
    # entra en la estadistica (ver NOTA sobre el warm-up en la cabecera).
    echo "  cebado (descartado)..."
    "${BIN}" --nx "${NX}" --ny "${NY}" --iters "${ITERS}" --tc both \
        --op-mode diffusive --ci-mode monomode --ci-p 5 \
        --cpu-fp64 off --kahan off > "${S}/${ETI}_cebado.log" 2>&1 || true

    for r in $(seq 1 "${REPS}"); do
        for comp in "off:--kahan off" "local:--kahan on" "spatial:--spatial-comp on"; do
            local nombre="${comp%%:*}" flags="${comp#*:}"
            local log="${S}/${ETI}_r${r}_${nombre}.log"
            # shellcheck disable=SC2086
            "${BIN}" --nx "${NX}" --ny "${NY}" --iters "${ITERS}" --tc both \
                --op-mode diffusive --ci-mode monomode --ci-p 5 \
                --cpu-fp64 off ${flags} > "${log}" 2>&1
            # Se etiqueta la ruta con la politica para no mezclar off/local/spatial:
            # la columna `route` de CSV_SUMMARY solo lleva el sufijo _SP en spatial.
            grep -E "^CSV_SUMMARY" "${log}" | awk -F, -v suf="_${nombre}" \
                '{print $2 suf, $7}' >> "${acumulado}"
        done
        echo "    replica ${r}/${REPS} lista"
    done

    echo
    echo "-- ${ETI}: ${NX}x${NY}, ${ITERS} iteraciones medidas, ${REPS} replicas --"
    echo "  (filas extraidas: $(wc -l < "${acumulado}"))"
    resumir "${acumulado}"
}

# ===========================================================================
if [[ "${ETAPA}" == "1" ]]; then
    echo "############################################################"
    echo "# ETAPA 1: 4096x4096, ${REPLICAS} replicas"
    echo "############################################################"
    echo "Sin checkpoints, sin archivado, sin NCU, CPU FP64 desactivado."
    echo "Rutas comparadas: CPU_FP32, GPU_FP32, GPU_FP64, WMMA_FP16, WMMA_BF16"
    echo "en las tres politicas de compensacion."
    echo
    correr_matriz 4096 4096 "${REPLICAS}" "m4096"

elif [[ "${ETAPA}" == "2" ]]; then
    echo "############################################################"
    echo "# ETAPA 2: 16384x16384, 1 replica"
    echo "############################################################"
    echo "Verifica que el comportamiento se mantiene cuando la malla supera"
    echo "ampliamente las caches. NO son 320/640 iteraciones ni 15 replicas."
    echo
    correr_matriz 16384 16384 1 "m16384"

elif [[ "${ETAPA}" == "3" ]]; then
    # La ventana de NVML necesita >= 2 s para que el contador de energia no
    # quede dominado por su cuantizacion (~4-5 J de salto). A 16384^2 una
    # iteracion ronda el milisegundo, asi que hacen falta miles de iteraciones.
    # El operador difusivo es contractivo, de modo que iterar mucho NO diverge
    # y la ventana se puede alargar sin romper la ruta.
    #
    # ADVERTENCIA: estas iteraciones extra son SOLO para alargar la ventana
    # energetica. Su rel_l2 no es comparable con el de las etapas 1 y 2, que
    # usan 50 iteraciones. No mezclar ambas cosas al reportar precision.
    # Dimensionado de ITERS_E: la ventana debe llegar a 2 s en la ruta GPU mas
    # RAPIDA, que es la que antes se queda corta. A 16384^2, GPU_FP32 ronda los
    # 3 ms/iter, luego 700 iteraciones dan ~2.1 s; las rutas WMMA, mas lentas
    # por iteracion, quedan holgadamente por encima.
    #
    # COSTE: no hay flag que apague CPU_FP32 ni la referencia FP64, y ambas
    # iteran junto a la GPU. A 700 iteraciones eso son ~7 min de CPU cada una,
    # por modo de compensacion: la etapa entera ronda los 45 min aunque el
    # trabajo de GPU sean segundos. Es el precio de una ventana NVML fiable, no
    # un error de dimensionado. Bajar ITERS_E acorta el job pero mete la
    # cuantizacion del contador (~4-5 J de salto) dentro de la medida.
    ITERS_E="${ITERS_E:-700}"
    MODOS_E="${MODOS_E:-off local spatial}"
    echo "############################################################"
    echo "# ETAPA 3: ventana energetica (${ITERS_E} iteraciones)"
    echo "############################################################"
    echo "Modos a medir: ${MODOS_E}"
    echo "Para partirlo en jobs mas cortos: --export=ALL,ETAPA=3,MODOS_E=spatial"
    echo
    for comp in "off:--kahan off" "local:--kahan on" "spatial:--spatial-comp on"; do
        nombre="${comp%%:*}"; flags="${comp#*:}"
        [[ " ${MODOS_E} " == *" ${nombre} "* ]] || continue
        log="${S}/energia_${nombre}.log"
        echo "-- ${nombre} --"
        # shellcheck disable=SC2086
        "${BIN}" --nx 16384 --ny 16384 --iters "${ITERS_E}" --tc both \
            --op-mode diffusive --ci-mode monomode --ci-p 5 \
            --cpu-fp64 off ${flags} > "${log}" 2>&1
        # CSV_ENERGY: $2 route, $7 energy_gpu_j, $10 edp_j_s, $12 time_total_s,
        #             $14 energy_gpu_j_per_iter, $15 window_reliable
        grep -E "^CSV_ENERGY" "${log}" | awk -F, \
            '{printf "    %-14s E=%-12s J  ventana=%-10s s  J/iter=%-12s  fiable=%s\n", $2, $7, $12, $14, $15}'
    done
    echo
    echo "Si 'fiable' no es 1, la ventana fue demasiado corta o NVML no estaba"
    echo "disponible: subir ITERS_E y repetir antes de dar la energia por buena."
else
    echo "ETAPA desconocida: ${ETAPA}"; exit 1
fi

echo
echo "############################################################"
echo "# Salidas crudas en: ${WORK}/salidas/"
echo "############################################################"
