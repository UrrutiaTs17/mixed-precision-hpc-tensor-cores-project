#!/bin/bash
#SBATCH --job-name=f4_valida_graphs
#SBATCH --output=logs/f4_valida_graphs_%j.out
#SBATCH --error=logs/f4_valida_graphs_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
# 64G como el microbenchmark: la ETAPA 4 llega a 16384^2 y ahi el patron
# FP64 en host es un vector<double> de 2.1 GB, mas los buffers FP32 de cada
# ruta. Con 32G la ultima fila moriria por OOM tras 20 minutos de job.
#SBATCH --mem=64G
#SBATCH --time=01:30:00

# ===========================================================================
# VALIDACION PEQUENA de la ruta CUDA Graph (--execution-mode graph)
#
# La ruta de grafo NO cambia nada numerico: mismo kernel, mismos argumentos,
# mismo orden de iteraciones, mismo ping-pong. Lo unico que cambia es quien
# paga el costo de CPU de poner cada iteracion en el stream. Por eso el
# criterio de aceptacion aqui es EQUIVALENCIA EXACTA de las columnas
# deterministas contra --execution-mode normal, no una tolerancia.
#
# Este job NO mide rendimiento de produccion ni autoriza la campana: mide
# correccion sobre mallas pequenas y, al final, un contraste de tiempo a las
# mallas reales para decidir si la ruta de grafo merece entrar a la campana.
#
# Uso:
#   sbatch validar_cuda_graphs.sbatch
#   sbatch --export=ALL,SANITIZER=0 validar_cuda_graphs.sbatch
#   sbatch --export=ALL,ETAPA_TIEMPO=0 validar_cuda_graphs.sbatch
# ===========================================================================

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

REPO_ROOT="$(git rev-parse --show-toplevel)"
CUDA_ARCH="${CUDA_ARCH:-80}"
NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"
SANITIZER="${SANITIZER:-1}"
ETAPA_TIEMPO="${ETAPA_TIEMPO:-1}"
# Iteraciones por grafo para la ETAPA 2. Deliberadamente PEQUENO (4) para que
# 20 iteraciones formen varios grafos y queden ejercitadas las colas, los
# cortes de checkpoint y el reinicio del contador global; con el defecto del
# binario (32) casi ningun caso pequeno llegaria a formar un solo grafo. La
# ETAPA 4 usa tamanos realistas aparte.
GRAPH_BLOCK_TEST="${GRAPH_BLOCK_TEST:-4}"

JOB_ID="${SLURM_JOB_ID:-manual}"
WORK="valida_graphs_${JOB_ID}"
mkdir -p "${WORK}/salidas" "${WORK}/archivo"

echo "Nodo        : $(hostname)"
echo "Fecha       : $(date -Iseconds)"
echo "Commit      : $(git -C "${REPO_ROOT}" rev-parse HEAD)"
echo "Arquitectura: sm_${CUDA_ARCH}"
echo "graph-block etapa 2 : ${GRAPH_BLOCK_TEST}"
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
# ETAPA 1 — Compilacion con reporte de recursos
# ===========================================================================
echo "############################################################"
echo "# ETAPA 1: compilacion y recursos (ptxas)"
echo "############################################################"

BIN="${PWD}/${WORK}/stencil_tc"
PTXAS_LOG="${WORK}/salidas/ptxas.txt"

( cd "${REPO_ROOT}/Fase_3/Stencil" && "${NVCC}" -std=c++17 stencil_tensor_activation.cu -o "${BIN}" \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
    -Xptxas=-v \
    --allow-unsupported-compiler "${NVML_FLAGS[@]}" ) 2> "${PTXAS_LOG}"
RC_BUILD=$?
if [[ ${RC_BUILD} -ne 0 ]]; then
    echo "FALLO: la compilacion devolvio ${RC_BUILD}"
    cat "${PTXAS_LOG}"
    exit 1
fi

echo "-- Recursos del kernel WMMA --"
grep -A3 -i "stencil2d_wmma_kernel" "${PTXAS_LOG}" || echo "  (sin lineas de ptxas)"
echo
echo "-- El parametro iter_offset no debe haber introducido spills --"
if grep -qi "spill" "${PTXAS_LOG}"; then
    echo "  ATENCION: ptxas reporta spills:"
    grep -i "spill" "${PTXAS_LOG}"
else
    echo "  OK: cero spill stores y cero spill loads."
fi
echo

# ===========================================================================
# ETAPA 2 — Equivalencia normal vs cuda_graph
# ===========================================================================
echo "############################################################"
echo "# ETAPA 2: equivalencia normal vs cuda_graph (mallas pequenas)"
echo "############################################################"

SAL="${WORK}/salidas"
FALLOS=0

# Columnas deterministas de CSV_SUMMARY: identidad ($2..$6), error y
# first_nonfinite y storage ($15..$24) y reference_role ($36). Se dejan fuera
# tiempos y energia, que no son reproducibles entre corridas. Las demas lineas
# CSV_* (CKPT, NORM, ONSET, DRIFT) se comparan enteras.
deterministas() {
    grep -E '^CSV_(SUMMARY|CKPT|NORM|ONSET|DRIFT)' "$1" \
      | awk -F, '$1=="CSV_SUMMARY"{printf "%s",$1; for(i=2;i<=6;i++)printf ",%s",$i;
                                   for(i=15;i<=24;i++)printf ",%s",$i;
                                   printf ",%s\n",$36; next} {print}'
}

caso() {
    local nombre="$1"; shift
    local n="${SAL}/${nombre}.normal.txt"
    local g="${SAL}/${nombre}.graph.txt"
    "${BIN}" "$@" --execution-mode normal > "${n}" 2>&1
    local rc_n=$?
    "${BIN}" "$@" --execution-mode graph --graph-block "${GRAPH_BLOCK_TEST}" > "${g}" 2>&1
    local rc_g=$?
    if [[ ${rc_n} -ne 0 || ${rc_g} -ne 0 ]]; then
        echo "FALLO ${nombre}: codigo de salida normal=${rc_n} graph=${rc_g}"
        tail -n 5 "${n}"; tail -n 5 "${g}"
        FALLOS=$((FALLOS+1)); return
    fi
    deterministas "${n}" > "${n}.det"
    deterministas "${g}" > "${g}.det"
    if diff -q "${n}.det" "${g}.det" >/dev/null; then
        # first_nonfinite es la columna 12 del fichero .det (columna 20 del
        # CSV_SUMMARY original). Se imprime porque es LA prueba de que el
        # iteration_offset funciona: con GRAPH_BLOCK_TEST=4, una version que
        # reiniciara el contador por bloque no podria reportar un valor > 4.
        local nf etiquetadas
        nf=$(grep '^CSV_SUMMARY,WMMA' "${n}.det" | awk -F, '{printf "%s=%s ",$2,$12}')
        etiquetadas=$(grep -c '^CSV_SUMMARY,WMMA.*,cuda_graph$' "${g}")
        echo "OK   ${nombre}  [first_nf: ${nf}] [filas cuda_graph: ${etiquetadas}]"
    else
        echo "FALLO ${nombre}: difieren columnas deterministas"
        diff "${n}.det" "${g}.det" | head -n 20
        FALLOS=$((FALLOS+1))
    fi
}

COMUN=(--tc both --cpu-fp64 off)

echo "--- A. difusivo estable, 32x32 y 64x64, las tres politicas ---"
for NN in 32 64; do
  for POL in "--kahan off" "--kahan on" "--spatial-comp on"; do
    caso "A_dif_${NN}_$(echo "${POL}" | tr -d ' -')" \
         --nx ${NN} --ny ${NN} --iters 20 \
         --op-mode diffusive --ci-mode monomode --ci-p 5 ${POL} "${COMUN[@]}"
  done
done

echo
echo "--- B. iters no multiplo del bloque (cola por lanzamientos normales) ---"
for POL in "--kahan off" "--spatial-comp on"; do
  caso "B_cola_$(echo "${POL}" | tr -d ' -')" \
       --nx 64 --ny 64 --iters 19 \
       --op-mode diffusive --ci-mode monomode --ci-p 5 ${POL} "${COMUN[@]}"
done

echo
echo "--- C. checkpoints en iteraciones que parten los grafos ---"
for POL in "--kahan off" "--kahan on" "--spatial-comp on"; do
  caso "C_ckpt_$(echo "${POL}" | tr -d ' -')" \
       --nx 64 --ny 64 --iters 20 --checkpoint-iters "3,7,12,20" \
       --archive-dir "${WORK}/archivo" \
       --op-mode diffusive --ci-mode monomode --ci-p 5 ${POL} "${COMUN[@]}"
done

echo
echo "--- D. checkpoint-every (cadencia regular, ruta historica) ---"
caso "D_every5" --nx 64 --ny 64 --iters 20 --checkpoint-every 5 \
     --op-mode diffusive --ci-mode monomode --ci-p 5 --kahan off "${COMUN[@]}"

echo
echo "--- E. divergencia: la primera iteracion no finita debe coincidir ---"
for POL in "--kahan off" "--spatial-comp on"; do
  caso "E_div_$(echo "${POL}" | tr -d ' -')" \
       --nx 64 --ny 64 --iters 200 --op-mode stress ${POL} "${COMUN[@]}"
done

echo
echo "--- F. iters < graph-block: debe degenerar a normal sin romperse ---"
caso "F_degenera" --nx 32 --ny 32 --iters 3 \
     --op-mode diffusive --ci-mode monomode --ci-p 5 --kahan off "${COMUN[@]}"

echo
echo "--- G. validacion de argumentos (deben ser errores, no no-ops) ---"
if "${BIN}" --nx 32 --ny 32 --iters 4 --graph-block 8 >/dev/null 2>&1; then
    echo "FALLO: --graph-block sin --execution-mode graph deberia abortar"; FALLOS=$((FALLOS+1))
else
    echo "OK   --graph-block en modo normal aborta"
fi
if "${BIN}" --nx 32 --ny 32 --iters 4 --cuda-graph --graph-block 7 >/dev/null 2>&1; then
    echo "FALLO: --graph-block impar deberia abortar"; FALLOS=$((FALLOS+1))
else
    echo "OK   --graph-block impar aborta"
fi
echo

# ===========================================================================
# ETAPA 3 — compute-sanitizer sobre la ruta de grafo
# ===========================================================================
if [[ "${SANITIZER}" == "1" ]] && command -v compute-sanitizer >/dev/null 2>&1; then
    echo "############################################################"
    echo "# ETAPA 3: compute-sanitizer (memcheck / racecheck / initcheck)"
    echo "############################################################"
    for POL in "--kahan off" "--kahan on" "--spatial-comp on"; do
        echo "-- memcheck ${POL} (grafo + checkpoints) --"
        compute-sanitizer --tool memcheck --error-exitcode 9 "${BIN}" \
            --nx 64 --ny 64 --iters 14 --tc both --cpu-fp64 off \
            --checkpoint-iters "3,7,14" --archive-dir "${WORK}/archivo" \
            --op-mode diffusive --ci-mode monomode --ci-p 5 ${POL} \
            --execution-mode graph --graph-block 4 2>&1 | grep -E "ERROR SUMMARY" \
            || { echo "   FALLO memcheck"; FALLOS=$((FALLOS+1)); }
    done
    echo "-- racecheck (spatial: unica politica que lee residuos de vecinas) --"
    compute-sanitizer --tool racecheck "${BIN}" \
        --nx 64 --ny 64 --iters 8 --tc fp16 --cpu-fp64 off \
        --op-mode diffusive --ci-mode monomode --ci-p 5 --spatial-comp on \
        --execution-mode graph --graph-block 4 2>&1 | grep -E "RACECHECK SUMMARY"
    echo "-- initcheck --"
    compute-sanitizer --tool initcheck "${BIN}" \
        --nx 64 --ny 64 --iters 8 --tc fp16 --cpu-fp64 off \
        --op-mode diffusive --ci-mode monomode --ci-p 5 \
        --execution-mode graph --graph-block 4 2>&1 | grep -E "ERROR SUMMARY"
    echo
else
    echo "(ETAPA 3 omitida: SANITIZER=${SANITIZER} o compute-sanitizer no disponible)"
    echo
fi

# ===========================================================================
# ETAPA 4 — Contraste de tiempo a las mallas reales
# ===========================================================================
# NO es la campana: son corridas cortas para decidir si el ahorro de
# lanzamiento se nota a las mallas reales. En la GPU local el ahorro medido fue
# de ~3.5 us por iteracion, CONSTANTE con el tamano de malla -- es coste de CPU
# por lanzamiento, no de GPU. Lo que decide es el VALOR ABSOLUTO del ahorro en
# ESTE nodo, no el porcentaje local: por eso la tabla publica ahorro_us.
#
# PRESUPUESTO. Cada corrida de esta etapa paga, ademas del kernel, el patron
# FP64 en CPU y la ruta CPU_FP32, ambos en serie (--cpu-fp64 off solo quita la
# ruta CPU_FP64 CRONOMETRADA, no el ground truth). A 4096^2 x 100 iteraciones
# eso son ~1.7 G actualizaciones de celda por pasada. Por eso la matriz esta
# acotada a 8 corridas a 4096^2 (2 politicas x 2 bloques x 2 modos) y UN solo
# par a 16384^2: barrer bloques a 16384^2 no aporta -- el ahorro por iteracion
# no depende de la malla -- y multiplicaria el coste del ground truth por seis.
if [[ "${ETAPA_TIEMPO}" == "1" ]]; then
    echo "############################################################"
    echo "# ETAPA 4: normal vs cuda_graph a mallas reales (corridas cortas)"
    echo "############################################################"

    # Una fila = un par normal/grafo. $7 de CSV_SUMMARY es t_iter_ms.
    fila_tiempo() {
        local NN=$1 IT=$2 POL=$3 GB=$4
        local N_MS G_MS
        N_MS=$("${BIN}" --nx ${NN} --ny ${NN} --iters ${IT} --tc fp16 --cpu-fp64 off \
               --op-mode diffusive --ci-mode monomode --ci-p 168 ${POL} \
               --execution-mode normal 2>/dev/null \
               | grep '^CSV_SUMMARY,WMMA' | awk -F, '{print $7}')
        G_MS=$("${BIN}" --nx ${NN} --ny ${NN} --iters ${IT} --tc fp16 --cpu-fp64 off \
               --op-mode diffusive --ci-mode monomode --ci-p 168 ${POL} \
               --execution-mode graph --graph-block ${GB} 2>/dev/null \
               | grep '^CSV_SUMMARY,WMMA' | awk -F, '{print $7}')
        printf "%-10s %-6s %-16s %-7s %14s %14s %10s %11s\n" \
            "${NN}^2" "${IT}" "$(echo "${POL}" | tr -d ' -')" "${GB}" \
            "${N_MS:-ERROR}" "${G_MS:-ERROR}" \
            "$(awk -v n="${N_MS:-0}" -v g="${G_MS:-0}" 'BEGIN{if(n>0)printf "%+.1f%%",(1-g/n)*100; else print "NA"}')" \
            "$(awk -v n="${N_MS:-0}" -v g="${G_MS:-0}" 'BEGIN{if(n>0)printf "%+.2f",(n-g)*1000; else print "NA"}')"
    }

    printf "%-10s %-6s %-16s %-7s %14s %14s %10s %11s\n" \
           malla iters politica bloque normal_ms grafo_ms mejora ahorro_us
    for POL in "--kahan off" "--spatial-comp on"; do
        for GB in 8 32; do
            fila_tiempo 4096 100 "${POL}" "${GB}"
        done
    done
    # Un unico par a 16384^2, con el bloque por defecto: confirma que el ahorro
    # absoluto por iteracion es el mismo que a 4096^2 (y por tanto que su peso
    # relativo cae). Barrer bloques aqui seria pagar el ground truth de balde.
    fila_tiempo 16384 40 "--kahan off" 32
    echo
    echo "Lectura: si ahorro_us es del mismo orden en 4096^2 y en 16384^2, el"
    echo "ahorro es coste de CPU por lanzamiento y su PESO RELATIVO cae al crecer"
    echo "la malla. La decision de llevar la ruta de grafo a la campana se toma"
    echo "sobre la columna mejora a la malla de la campana, no sobre ahorro_us."
    echo
else
    echo "(ETAPA 4 omitida: ETAPA_TIEMPO=${ETAPA_TIEMPO})"
    echo
fi

echo "############################################################"
if [[ ${FALLOS} -eq 0 ]]; then
    echo "# VEREDICTO: normal y cuda_graph son EQUIVALENTES en todos los casos"
else
    echo "# VEREDICTO: ${FALLOS} caso(s) con fallo -- NO usar la ruta de grafo"
fi
echo "# Salidas completas en ${WORK}/salidas"
echo "############################################################"
exit ${FALLOS}
