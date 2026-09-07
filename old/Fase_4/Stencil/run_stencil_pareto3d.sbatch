#!/bin/bash
#SBATCH --job-name=f4_pareto3d
#SBATCH --output=logs/f4_pareto3d_%j.out
#SBATCH --error=logs/f4_pareto3d_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=04:00:00
#
# ===========================================================================
# Campana Fase 4 -- frente de Pareto 3D (precision / tiempo / energia)
# ===========================================================================
# Derivado de Fase_3/Stencil/run_stencil_tc.sbatch, que NO se modifica: aquel
# sigue siendo el lanzador reproducible de la Fase 3 y varias campanas cerradas
# dependen de su comportamiento exacto.
#
# Diferencias respecto al de Fase 3:
#   - NO perfila. RUN_NCU se fija a 0 aqui y se ignora lo que traiga el entorno
#     (ver el bloque que lo fuerza): bajo ncu los tiempos y la energia quedan
#     inflados, y esta campana existe precisamente para medir esas dos cosas.
#   - Variables nuevas: OP_MODE, ALPHA, CI_MODE, CI_P, CHECKPOINT_ITERS,
#     ARCHIVE_ITERS, RUN_KIND.
#   - RUN_KIND separa las dos corridas del diseno:
#       energy  -> sin checkpoints y sin archivado. Es la que se replica 15
#                  veces: cualquier instrumentacion contamina t/iter y la
#                  ventana de energia.
#       numeric -> con checkpoints y archivado. Una sola ejecucion, porque el
#                  error es DETERMINISTA (mismo binario, misma CI, mismo
#                  hardware -> mismo resultado bit a bit): n=15 aplica a tiempo
#                  y energia, n=1 al error.
#   - CSV propio bajo results/<CAMPAIGN_ID>/. Nunca appendea a un CSV de otra
#     campana: el binario aborta si la cabecera no coincide, y la de Fase 4 trae
#     7 columnas mas.
#   - Registra procedencia completa: commit, sha256 de las TRES fuentes, sha256
#     del binario, hostname, UUID de GPU y power limit.
#
# NO se ejecuta tools/extract_csv.py. Ese extractor sigue con el esquema de 29
# campos de Fase 3 y truncaria por la derecha las 7 columnas nuevas de
# CSV_SUMMARY sin avisar. Se actualizara en una tarea posterior, cuando el
# piloto congele el esquema. Mientras tanto los artefactos autoritativos son el
# log crudo (results/<CAMPAIGN_ID>/run_<job>.log, que trae todos los tokens
# CSV_*) y el CSV del binario (--csv), que si escribe las 30 columnas.
#
# Uso tipico (a traves del orquestador, no a mano):
#   sbatch --export=ALL,RUN_KIND=energy,ITERS_LIST=320,... run_stencil_pareto3d.sbatch

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
JOB_ID="${SLURM_JOB_ID:-manual}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
SRC_DIR="${REPO_ROOT}/Fase_4/Stencil"

# --- Perfilado: apagado y no negociable ------------------------------------
# Se sobreescribe DESPUES de leer el entorno, a proposito. Un --export=ALL
# arrastra el RUN_NCU=1 que el usuario tenga exportado de una sesion de Fase 3,
# y una sola corrida perfilada dentro de la campana energetica arruinaria la
# replica sin dejar rastro evidente en el CSV.
if [[ "${RUN_NCU:-0}" != "0" ]]; then
    echo "AVISO: RUN_NCU=${RUN_NCU} en el entorno; esta campana lo ignora y perfila nunca."
fi
RUN_NCU=0

# --- Parametros del operador y la condicion inicial -------------------------
OP_MODE="${OP_MODE:-diffusive}"
ALPHA="${ALPHA:-0.1875}"
CI_MODE="${CI_MODE:-monomode}"
CI_P="${CI_P:-168}"
CI_AMPLITUDE="${CI_AMPLITUDE:-1.0}"

# --- Diseno experimental ----------------------------------------------------
NX="${NX:-16384}"
NY="${NY:-16384}"
# Las listas viajan con ':' como separador, no con ',' ni con espacios.
# Motivo: SLURM parsea --export=... como una lista separada por COMAS, asi que
# un valor con comas (KAHAN_LIST=off,on) se romperia en variables sueltas -- y
# uno con espacios rompe la construccion del entorno de la tarea. Se normaliza
# aqui; se admite tambien ',' para poder invocar este sbatch a mano.
norm_espacios() { local v="$1"; echo "${v//[:,]/ }"; }
norm_comas()    { local v="$1"; echo "${v//:/,}"; }

ITERS_LIST="$(norm_espacios "${ITERS_LIST:-320:640}")"
TC="${TC:-both}"
FP64_GPU="${FP64_GPU:-on}"
CPU_FP64="${CPU_FP64:-on}"

# Tratamientos. Igual que en Fase 3, el binario RECHAZA --kahan on junto con
# --spatial-comp on: son dos politicas alternativas de la misma compensacion, no
# dos capas. Con `set -e` una combinacion invalida mataria el job a mitad del
# barrido, asi que SPATIAL_COMP=on fuerza KAHAN_LIST=off aqui.
#
# Consecuencia de diseno, y por eso el orquestador lanza DOS jobs por replica:
#   bloque A: SPATIAL_COMP=off KAHAN_LIST="off on" -> tratamientos none y kahan_local
#   bloque B: SPATIAL_COMP=on                      -> tratamiento spatial
# Las filas de ambos se unen despues por (nx, ny, iters); las rutas espaciales
# se reportan como WMMA_*_SP, asi que no se confunden en un CSV mezclado.
KAHAN_LIST="$(norm_espacios "${KAHAN_LIST:-off:on}")"
SPATIAL_COMP="${SPATIAL_COMP:-off}"
if [[ "${SPATIAL_COMP}" == "on" ]]; then
    KAHAN_LIST="off"
fi

# --- Tipo de corrida --------------------------------------------------------
RUN_KIND="${RUN_KIND:-energy}"
# El binario espera estas dos con COMAS; se aceptan con ':' por la razon de
# arriba y se convierten.
CHECKPOINT_ITERS="$(norm_comas "${CHECKPOINT_ITERS:-1:2:5:10:20:50:100:200:320:640}")"
ARCHIVE_ITERS="$(norm_comas "${ARCHIVE_ITERS:-320:640}")"

case "${RUN_KIND}" in
    energy)
        # Sin instrumentacion: es lo unico que hace comparables las 15 replicas.
        CKPT_ARGS=()
        ;;
    numeric)
        CKPT_ARGS=(--checkpoint-iters "${CHECKPOINT_ITERS}")
        ;;
    *)
        echo "ERROR: RUN_KIND debe ser 'energy' o 'numeric' (recibido: ${RUN_KIND})." >&2
        exit 2
        ;;
esac

CAMPAIGN_ID="${CAMPAIGN_ID:-f4_pareto3d_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="results/${CAMPAIGN_ID}"
mkdir -p "${OUT_DIR}"
RESULTS_LOG="${OUT_DIR}/run_${JOB_ID}.log"
CSV_PATH="${OUT_DIR}/stencil_${CAMPAIGN_ID}_${JOB_ID}.csv"
ARCHIVE_DIR="${OUT_DIR}/archive_${JOB_ID}"
mkdir -p "${ARCHIVE_DIR}"

CUDA_ARCH="${CUDA_ARCH:-80}"
NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"

# --- Deteccion de NVML: identica a la de Fase 3, deliberadamente ------------
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

# --- Compilacion ------------------------------------------------------------
# Se compila con cwd = Fase_4/Stencil porque el fuente incluye
# "../../Fase_2/common.cuh" y "tools/power_sampling.h" por rutas relativas a si
# mismo; el binario sale a este job, no al arbol de fuentes, para que dos jobs
# concurrentes no se pisen el ejecutable.
#
# Fase_4/Stencil esta a la misma profundidad que Fase_3/Stencil, asi que
# "../../Fase_2/common.cuh" resuelve igual. "tools/power_sampling.h" resuelve
# por el SYMLINK Fase_4/Stencil/tools/power_sampling.h -> el header de Fase 3:
# hay un solo header en el repo, no una copia que pueda derivar.
BIN="${PWD}/${OUT_DIR}/stencil_tc_${JOB_ID}"
echo "Compilando stencil_tc desde ${SRC_DIR}..."
( cd "${SRC_DIR}" && "${NVCC}" -std=c++17 stencil_tensor_activation.cu -o "${BIN}" \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
    --allow-unsupported-compiler "${NVML_FLAGS[@]}" )

# --- Procedencia ------------------------------------------------------------
# Se hashean las TRES fuentes, no solo el .cu: common.cuh y power_sampling.h
# entran por include y un cambio en cualquiera de ellos produce un binario
# distinto que el sha256 del .cu no delataria.
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
if git -C "${REPO_ROOT}" diff --quiet 2>/dev/null; then GIT_STATUS=limpio; else GIT_STATUS=sucio; fi
{
echo "=============== PROCEDENCIA DE LA CORRIDA ==============="
echo "campaign_id     : ${CAMPAIGN_ID}"
echo "job_id          : ${JOB_ID}"
echo "run_kind        : ${RUN_KIND}"
echo "fecha           : $(date -Iseconds)"
echo "hostname        : $(hostname)"
echo "git_commit      : ${GIT_COMMIT}"
echo "git_status      : ${GIT_STATUS}"
for f in "${SRC_DIR}/stencil_tensor_activation.cu" \
         "${REPO_ROOT}/Fase_2/common.cuh" \
         "${SRC_DIR}/tools/power_sampling.h"; do
    echo "sha256 fuente   : $(sha256sum "${f}" | awk '{print $1}')  ${f#${REPO_ROOT}/}"
done
echo "sha256 binario  : $(sha256sum "${BIN}" | awk '{print $1}')"
echo "nvcc            : ${NVCC}"
echo "cuda_arch       : sm_${CUDA_ARCH}"
echo "nvml_flags      : ${NVML_FLAGS[*]}"
echo "gpu             : $(nvidia-smi --query-gpu=name,uuid,power.limit,memory.total \
                            --format=csv,noheader 2>/dev/null || echo 'nvidia-smi no disponible')"
echo "cuda_visible    : ${CUDA_VISIBLE_DEVICES:-no definido}"
echo "slurm_nodelist  : ${SLURM_JOB_NODELIST:-no definido}"
echo "run_ncu         : ${RUN_NCU} (fijo: esta campana no perfila)"
echo "params          : NX=${NX} NY=${NY} ITERS_LIST=${ITERS_LIST} TC=${TC}"
echo "                  OP_MODE=${OP_MODE} ALPHA=${ALPHA} CI_MODE=${CI_MODE} CI_P=${CI_P}"
echo "                  KAHAN_LIST=${KAHAN_LIST} SPATIAL_COMP=${SPATIAL_COMP}"
echo "                  FP64_GPU=${FP64_GPU} CPU_FP64=${CPU_FP64}"
if [[ "${RUN_KIND}" == "numeric" ]]; then
    echo "                  CHECKPOINT_ITERS=${CHECKPOINT_ITERS} ARCHIVE_ITERS=${ARCHIVE_ITERS}"
fi
echo "csv             : ${CSV_PATH}"
echo "========================================================="
} | tee "${RESULTS_LOG}"

# --- Barrido ----------------------------------------------------------------
# Una sola invocacion del bloque bajo tee: si cada corrida redirigiera por su
# cuenta, perder una invocacion puntual dejaria un hueco silencioso en el log.
{
for ITERS in ${ITERS_LIST}; do
    for KAHAN in ${KAHAN_LIST}; do
        echo
        echo "============================================================"
        echo "Corrida: NX=${NX}, NY=${NY}, ITERS=${ITERS}, TC=${TC}, KAHAN=${KAHAN}"
        echo "============================================================"

        RUN_ARGS=(--nx "${NX}" --ny "${NY}" --iters "${ITERS}" --tc "${TC}"
                  --kahan "${KAHAN}" --spatial-comp "${SPATIAL_COMP}"
                  --fp64-gpu "${FP64_GPU}" --cpu-fp64 "${CPU_FP64}"
                  --op-mode "${OP_MODE}" --ci-mode "${CI_MODE}"
                  --csv "${CSV_PATH}")
        [[ "${OP_MODE}" == "diffusive" ]] && RUN_ARGS+=(--alpha "${ALPHA}")
        [[ "${CI_MODE}" == "monomode" ]] && RUN_ARGS+=(--ci-p "${CI_P}" --ci-amplitude "${CI_AMPLITUDE}")

        if [[ "${RUN_KIND}" == "numeric" ]]; then
            # La lista de checkpoints se recorta a las iteraciones que este
            # horizonte alcanza: pedir 640 en una corrida de 320 es un error del
            # binario, y con `set -e` mataria el barrido a mitad.
            CK=$(python3 -c "import sys; print(','.join(x for x in sys.argv[1].split(',') if int(x) <= ${ITERS}))" "${CHECKPOINT_ITERS}")
            AR=$(python3 -c "import sys; print(','.join(x for x in sys.argv[1].split(',') if int(x) <= ${ITERS}))" "${ARCHIVE_ITERS}")
            RUN_ARGS+=(--checkpoint-iters "${CK}" --archive-dir "${ARCHIVE_DIR}/iters${ITERS}_kahan${KAHAN}")
            mkdir -p "${ARCHIVE_DIR}/iters${ITERS}_kahan${KAHAN}"
            [[ -n "${AR}" ]] && RUN_ARGS+=(--archive-iters "${AR}")
        fi

        "${BIN}" "${RUN_ARGS[@]}"
    done
done
} 2>&1 | tee -a "${RESULTS_LOG}"

echo
echo "=============== ARTEFACTOS ==============="
echo "log crudo : ${RESULTS_LOG}"
echo "csv       : ${CSV_PATH} ($(( $(wc -l < "${CSV_PATH}") - 1 )) filas)"
if [[ "${RUN_KIND}" == "numeric" ]]; then
    echo "archivo   : ${ARCHIVE_DIR} ($(du -sh "${ARCHIVE_DIR}" 2>/dev/null | cut -f1))"
fi
echo "NOTA: no se corre tools/extract_csv.py -- su esquema es el de 29 campos de"
echo "      Fase 3 y truncaria las 7 columnas nuevas de CSV_SUMMARY. Se actualiza"
echo "      en una tarea posterior, cuando el piloto congele el esquema."
echo "=========================================="
