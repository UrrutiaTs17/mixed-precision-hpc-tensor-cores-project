#!/bin/bash
#SBATCH --job-name=mixed_precision_stencil_horizon_f3
#SBATCH --output=logs/mixed_precision_stencil_horizon_f3_%j.out
#SBATCH --error=logs/mixed_precision_stencil_horizon_f3_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --exclusive
# Subido de 01:00:00 a 02:00:00 (historico): con KAHAN_LIST="off on" la
# corrida se duplica; Kahan no toca CPU FP32/FP64 (las referencias, el costo
# dominante de esta corrida) asi que la segunda pasada las recalcula de forma
# redundante -- el tiempo total sube aprox. al doble del de una sola pasada.
# El limite se CONSERVA aunque el default actual (SPATIAL_COMP=on) fuerce
# KAHAN_LIST=off y por tanto una sola pasada: SPATIAL_COMP=off restaura las
# dos pasadas, y es esa la configuracion que necesita las 2 horas.
#SBATCH --time=02:00:00

set -euo pipefail

# No forzar cd a SLURM_SUBMIT_DIR: cuando este script se invoca ANIDADO
# (`bash` dentro de un job ya en marcha -- ver tools/validacion_preliminar.sbatch,
# que hace `cd` al directorio correcto ANTES de llamar `bash <este script>`),
# SLURM_SUBMIT_DIR es el del job PADRE (la raiz del repo), no el de este
# kernel, y pisaria un pwd que el llamador ya dejo correcto. Se prueba pwd
# primero; solo se cae a SLURM_SUBMIT_DIR cuando pwd no trae el fuente
# (invocacion de primer nivel via `sbatch` desde la raiz del repo).
if [[ ! -f stencil_tensor_activation.cu ]]; then
    if [[ -f Stencil/stencil_tensor_activation.cu ]]; then
        cd Stencil
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/stencil_tensor_activation.cu" ]]; then
        cd "${SLURM_SUBMIT_DIR}"
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/Stencil/stencil_tensor_activation.cu" ]]; then
        cd "${SLURM_SUBMIT_DIR}/Stencil"
    fi
fi

# Directorios de salida, parametrizados (mismo patron que run_stencil_tc.sbatch).
LOGS_DIR="${LOGS_DIR:-logs}"
RESULTS_DIR="${RESULTS_DIR:-results}"
mkdir -p "${LOGS_DIR}" "${RESULTS_DIR}"
JOB_ID="${SLURM_JOB_ID:-manual}"
RESULTS_LOG="${RESULTS_DIR}/run_${JOB_ID}.log"

# Job separado del barrido de metricas (run_stencil_tc.sbatch): ese barrido
# corre dentro de la ventana finita (ITERS_LIST <= 120 por defecto, todas las
# metricas L2/Linf/etc. comparables). Este job mide el horizonte de overflow
# REAL por formato (n*, impreso via "Horizonte de overflow (predicho /
# medido)" en stencil_tensor_activation.cu) con ITERS grande, suficiente para
# capturar incluso FP64.
# No corre ncu: el objetivo es el texto de n* por stdout (primera iteracion
# no finita de cada ruta), no perfilar Tensor Cores -- para eso esta
# run_stencil_tc.sbatch. TODO el perfilado NCU del stencil vive alli
# (variables RUN_NCU / ITERS_NCU / NCU_COMP_LIST); este script no acepta
# RUN_NCU y exportarlo aqui no tiene efecto alguno.

# Arquitectura por defecto para A100.
# Puedes sobrescribirla con:
#   sbatch --export=ALL,CUDA_ARCH=86 run_stencil_horizon.sbatch
# --- Deteccion portable del toolchain (tools/detect_toolchain.sh) --------
# Corre igual bajo SLURM en PACCA o con `bash` directo en cualquier otra
# maquina con el entorno conda de environment.yml activo -- ver el
# comentario de cabecera de ese script.
for _toolchain in "../../tools/detect_toolchain.sh" "./detect_toolchain.sh"; do
    if [[ -f "${_toolchain}" ]]; then
        # shellcheck source=/dev/null
        source "${_toolchain}"
        break
    fi
done
CUDA_ARCH="$(detect_cuda_arch)"  # A100=80, RTX 3050=86, V100=70, Ada=89, Hopper=90

# Ruta del toolchain: especifica del cluster historico (PACCA/HPC SDK).
# Sobrescribir con --export si el cluster de destino tiene otro layout.
NVCC="$(detect_nvcc)"

# NVML es opcional: cuando nvml.h y la biblioteca del driver estan visibles,
# se habilita la telemetria y se enlaza con libnvidia-ml. Si el cluster no
# expone alguno de los dos, el mismo fuente compila con el stub sin
# telemetria (ver common/power_sampling.h).
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

# --- Tamano de malla, formato e iteraciones ---------------------------------
# NX/NY 4x mas chicos que el default de run_stencil_tc.sbatch para que la
# referencia CPU FP64 encadenada sea viable a ITERS grande; TC=both corre
# FP16 y BF16 en la misma invocacion. SIN limite superior/inferior impuesto
# por este script: al igual que en run_stencil_tc.sbatch, NX/NY/ITERS son
# enteramente configurables por entorno. AUDITORIA (ver
# Fase_3/Stencil/README.md, "Hallazgos de la auditoria"): el rango
# 512^2-2048^2 nunca se ejercito historicamente para medir horizonte de
# overflow -- si se usa aqui, conviene revisar tambien CHECKPOINT_EVERY y el
# presupuesto de memoria de fp64_checkpoints (ver mas abajo), calibrados para
# el default de 4096^2.
NX="${NX:-4096}"
NY="${NY:-4096}"
ITERS="${ITERS:-1200}"
TC="${TC:-both}"

# --checkpoint-every: unico modo de que storage_rel sea evaluable en esta
# malla (sin checkpoints, si la ruta diverge no queda estado finito
# recuperable, ver stencil_tensor_activation.cu). El default (20) fue
# calibrado para que un checkpoint capture con margen la divergencia de FP16
# (n*~28 a NX/NY=4096 con la condicion inicial y el operador por defecto).
#
# Presupuesto de memoria de fp64_checkpoints (el consumo dominante que escala
# con ITERS/CHECKPOINT_EVERY): cada snapshot son NX*NY doubles. A 4096^2 son
# ~134 MB/snapshot; con CHECKPOINT_EVERY=20 e ITERS=1200 son hasta 60
# snapshots (~8 GiB), dentro de los --mem=32G que pide el job. Si se cambian
# NX/NY/ITERS/CHECKPOINT_EVERY, reescalar esta cuenta (snapshots ~=
# min(ITERS, n*_FP64) / CHECKPOINT_EVERY) antes de lanzar, o el job puede
# quedarse sin memoria a mitad de corrida.
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-20}"

# Diseno factorial {fp16,bf16} x {kahan off,on}: la corrida donde SI importa
# si Kahan mueve el horizonte n* de cada formato. --kahan off primero
# preserva el orden de no-regresion de las cifras ya validadas.
# OJO: con el default actual (SPATIAL_COMP=on, ver abajo) esta lista se fuerza
# a "off" y el barrido factorial de Kahan NO corre. Para recuperarlo hay que
# pedir explicitamente SPATIAL_COMP=off.
KAHAN_LIST="${KAHAN_LIST:-off on}"

# --- Politica de compensacion (none / kahan_local / spatial) ----------------
# Mismo mecanismo de flags que run_stencil_tc.sbatch (ver ese script y
# Fase_3/Stencil/README.md, "Compensacion del redondeo de almacenamiento"):
# el binario expone --kahan off|on y --spatial-comp off|on como politicas
# alternativas, mutuamente excluyentes; "none" es --kahan off --spatial-comp
# off. El default de ESTE SCRIPT es "on" (espacial) por ser la unica politica
# que efectivamente reduce el error medido.
#
# Se expone aqui (y no solo en el barrido de metricas) porque medir si una
# politica mueve el horizonte n* exige ITERS grande, que es justo lo que este
# barrido hace y el de metricas no.
#
# COMO REPRODUCIR LA COMPARACION LOCAL-vs-ESPACIAL: SPATIAL_COMP=off es la
# unica via para obtener las rutas de Kahan local con kahan off y on, porque
# SPATIAL_COMP=on fuerza KAHAN_LIST=off (el binario rechaza --kahan on junto
# con --spatial-comp on: politicas alternativas, no capas acumulables, y con
# `set -e` esa combinacion mataria el job):
#   sbatch run_stencil_horizon.sbatch                          -> WMMA_*_SP
#   sbatch --export=ALL,SPATIAL_COMP=off run_stencil_horizon.sbatch
#                                        -> WMMA_FP16/BF16 con kahan off y on
SPATIAL_COMP="${SPATIAL_COMP:-on}"
if [[ "${SPATIAL_COMP}" == "on" ]]; then
    KAHAN_LIST="off"
fi

# --- Operador y condicion inicial -------------------------------------------
# Mismo mecanismo que run_stencil_tc.sbatch (portado aqui: el binario
# migrado soporta --op-mode/--alpha/--ci-mode/--ci-p/--ci-amplitude, pero la
# version historica de este script -old/Fase_3/Stencil/run_stencil_horizon.sbatch-
# nunca los exponia). Los defaults reproducen el operador de estres +
# condicion inicial legacy, el unico comportamiento que este script corria
# antes de este cambio.
#   sbatch --export=ALL,OP_MODE=diffusive,CI_MODE=monomode run_stencil_horizon.sbatch
#
# ALPHA solo se pasa con --op-mode diffusive y CI_P/CI_AMPLITUDE solo con
# --ci-mode monomode: el binario RECHAZA --alpha junto a stress (es un error,
# no un no-op silencioso), asi que con `set -e` pasarlo siempre mataria el
# job. Nota: bajo --op-mode diffusive el horizonte de overflow no esta
# definido (el operador no amplifica el modo de Nyquist, ver
# Fase_3/Stencil/README.md) -- este script sigue siendo util alli para medir
# tiempo/energia/drift a ITERS grande, pero CSV_HORIZON saldra con estado
# "contractive_operator" en vez de una prediccion.
OP_MODE="${OP_MODE:-stress}"
ALPHA="${ALPHA:-0.1875}"
CI_MODE="${CI_MODE:-legacy}"
CI_P="${CI_P:-168}"
CI_AMPLITUDE="${CI_AMPLITUDE:-1.0}"

OP_ARGS=(--op-mode "${OP_MODE}" --ci-mode "${CI_MODE}")
[[ "${OP_MODE}" == "diffusive" ]] && OP_ARGS+=(--alpha "${ALPHA}")
[[ "${CI_MODE}" == "monomode" ]] && OP_ARGS+=(--ci-p "${CI_P}" --ci-amplitude "${CI_AMPLITUDE}")

# Ruta GPU_FP64 (referencia de maxima precision EN GPU): ON por defecto, con
# los MISMOS NX/NY/ITERS que las rutas WMMA de este barrido, para que sus
# filas se puedan unir por (nx, ny, iters) igual que en run_stencil_tc.sbatch.
# Aqui aporta ademas algo que el barrido de metricas no puede dar: con ITERS
# grande esta es la unica corrida donde el horizonte n* de la ruta FP64 de
# GPU es observable, y por tanto donde se puede contrastar contra el n* de la
# referencia FP64 de CPU (fila CSV_HORIZON "FP64" vs fila "GPU_FP64").
FP64_GPU="${FP64_GPU:-on}"

# Ruta CPU_FP64: cronometra el patron de oro IEEE 754 (double serial en CPU),
# referencia de COSTO del proyecto. Implica una segunda pasada FP64 sobre la
# malla, separada del ground truth para no heredar su instrumentacion; en
# este barrido de horizonte los ITERS son altos, asi que es la parte mas cara.
# CPU_FP64=off la omite.
CPU_FP64="${CPU_FP64:-on}"

echo "Nodo: $(hostname)"
echo "Fecha: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-no definido}"
echo "Arquitectura CUDA: sm_${CUDA_ARCH}"
echo "NVCC=${NVCC}"
echo "Operador         : OP_MODE=${OP_MODE} CI_MODE=${CI_MODE} (args: ${OP_ARGS[*]})"
echo "RUN_PARAMS=NX=${NX} NY=${NY} ITERS=${ITERS} TC=${TC} CHECKPOINT_EVERY=${CHECKPOINT_EVERY} KAHAN_LIST=${KAHAN_LIST} SPATIAL_COMP=${SPATIAL_COMP} FP64_GPU=${FP64_GPU} CPU_FP64=${CPU_FP64}"

nvidia-smi || true

echo "Compilando stencil_tc (Fase_3/Stencil, incluye common/cuda_checks.cuh," \
     "common/metrics.cuh y common/power_sampling.h por ruta relativa)..."
"${NVCC}" -std=c++17 stencil_tensor_activation.cu -o stencil_tc \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
    --allow-unsupported-compiler "${NVML_FLAGS[@]}"

TIMEFORMAT=">>> Tiempo real de la corrida: %R s (user %U, sys %S)"
# Una sola invocacion de tee envuelve TODO el bucle de KAHAN_LIST, en vez de
# reabrirse por cada pasada: ninguna pasada individual puede perderse sin que
# se pierdan todas.
{
for KAHAN in ${KAHAN_LIST}; do
    echo
    echo "============================================================"
    echo "Barrido de horizonte: NX=${NX}, NY=${NY}, ITERS=${ITERS}, TC=${TC}, KAHAN=${KAHAN}, FP64_GPU=${FP64_GPU}, CPU_FP64=${CPU_FP64}"
    echo "============================================================"
    time ./stencil_tc --nx "${NX}" --ny "${NY}" --iters "${ITERS}" --tc "${TC}" \
          --checkpoint-every "${CHECKPOINT_EVERY}" --kahan "${KAHAN}" \
          --spatial-comp "${SPATIAL_COMP}" --fp64-gpu "${FP64_GPU}" \
          --cpu-fp64 "${CPU_FP64}" "${OP_ARGS[@]}"
done
} 2>&1 | tee -a "${RESULTS_LOG}"

# tools/extract_csv.py migra por separado, en Fase_3/tools/ (no es parte de
# este entregable; ver README.md de la raiz del repo). Mismo mecanismo de
# deteccion robusta que run_stencil_tc.sbatch: se prueban varias rutas
# candidatas y, si ninguna existe, se avisa y se omite la extraccion en vez
# de fallar el job -- el log crudo (${RESULTS_LOG}) queda disponible para
# post-procesar manualmente despues.
EXTRACT_CSV_SCRIPT="${EXTRACT_CSV_SCRIPT:-}"
if [[ -z "${EXTRACT_CSV_SCRIPT}" ]]; then
    for _cand in "../tools/extract_csv.py" "tools/extract_csv.py" "../../tools/extract_csv.py"; do
        if [[ -f "${_cand}" ]]; then
            EXTRACT_CSV_SCRIPT="${_cand}"
            break
        fi
    done
fi

if [[ -n "${EXTRACT_CSV_SCRIPT}" ]]; then
    echo
    echo "Extrayendo CSVs desde ${RESULTS_LOG} (via ${EXTRACT_CSV_SCRIPT})..."
    python3 "${EXTRACT_CSV_SCRIPT}" \
        --input  "${RESULTS_LOG}" \
        --outdir "${RESULTS_DIR}" \
        --job-id "${JOB_ID}" \
        --kernel stencil

    echo "CSVs generados:"
    for csv_path in \
        "${RESULTS_DIR}/drift_stencil_${JOB_ID}.csv" \
        "${RESULTS_DIR}/summary_stencil_${JOB_ID}.csv" \
        "${RESULTS_DIR}/horizon_stencil_${JOB_ID}.csv" \
        "${RESULTS_DIR}/store_stencil_${JOB_ID}.csv" \
        "${RESULTS_DIR}/energy_stencil_${JOB_ID}.csv"; do
        rows=$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' "${csv_path}" 2>/dev/null || echo 0)
        echo "  ${csv_path}: ${rows} filas"
    done
else
    echo
    echo "AVISO: extract_csv.py no encontrado (se probo ../tools/, tools/ y" \
         "../../tools/ relativos a esta carpeta). Se omite la extraccion de" \
         "CSVs; el log crudo con los tokens CSV_* queda en ${RESULTS_LOG}." \
         "Defina EXTRACT_CSV_SCRIPT=/ruta/a/extract_csv.py para forzar una" \
         "ubicacion." >&2
fi
