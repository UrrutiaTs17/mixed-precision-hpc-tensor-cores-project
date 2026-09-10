#!/bin/bash
#SBATCH --job-name=mixed_precision_gemm_chained_f3
#SBATCH --output=logs/mixed_precision_gemm_chained_f3_%j.out
#SBATCH --error=logs/mixed_precision_gemm_chained_f3_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=24:00:00

# Lanzador parametrizado de gemm_chained.cu (Fase 3: encadenamiento genuino
# X(n+1)=X(n)*A, ver README.md de esta carpeta). Ningun tamano, formato o
# politica de compensacion queda hardcodeado -- todo se controla con
# variables de entorno via --export, con los mismos defaults documentados
# abajo si no se pasa nada.
#
#   sbatch run_gemm_chained.sbatch                       # CAMPANA COMPLETA
#   sbatch --export=ALL,SMOKE_TEST=1 run_gemm_chained.sbatch   # prueba de humo
#   sbatch --export=ALL,N_LIST="1024 2048",COMP_LIST="off on" run_gemm_chained.sbatch
#   sbatch --export=ALL,TC_FORMAT=fp16,CHECKPOINT_EVERY=1 run_gemm_chained.sbatch
#
# OJO: el default de N_LIST/ITERS_LIST es la CAMPANA COMPLETA (ver la seccion
# de barrido mas abajo), no una prueba rapida. Para eso esta SMOKE_TEST=1.
#
# Mismo patron de deteccion de NVML/toolchain que Fase_4/Stencil/run_stencil_tc.sbatch
# -- ver ese archivo si algo de esta seccion inicial no es obvio.

set -euo pipefail

# No forzar cd a SLURM_SUBMIT_DIR: cuando este script se invoca ANIDADO
# (`bash` dentro de un job ya en marcha -- ver tools/validacion_preliminar.sbatch,
# que hace `cd` al directorio correcto ANTES de llamar `bash <este script>`),
# SLURM_SUBMIT_DIR es el del job PADRE (la raiz del repo), no el de este
# kernel, y pisaria un pwd que el llamador ya dejo correcto. Se prueba pwd
# primero; solo se cae a SLURM_SUBMIT_DIR cuando pwd no trae el fuente
# (invocacion de primer nivel via `sbatch` desde la raiz del repo).
if [[ ! -f gemm_chained.cu ]]; then
    if [[ -f GEMM/gemm_chained.cu ]]; then
        cd GEMM
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/gemm_chained.cu" ]]; then
        cd "${SLURM_SUBMIT_DIR}"
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/GEMM/gemm_chained.cu" ]]; then
        cd "${SLURM_SUBMIT_DIR}/GEMM"
    fi
fi

LOGS_DIR="${LOGS_DIR:-logs}"
RESULTS_DIR="${RESULTS_DIR:-results}"
mkdir -p "${LOGS_DIR}" "${RESULTS_DIR}"
JOB_ID="${SLURM_JOB_ID:-manual}"
RESULTS_LOG="${RESULTS_DIR}/run_${JOB_ID}.log"

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
NVCC="$(detect_nvcc)"
NCU="$(detect_ncu || true)"
if [[ -z "${NCU}" ]]; then
    echo "AVISO: Nsight Compute (ncu) no encontrado -- se deshabilita el perfilado (RUN_NCU forzado a 0)." >&2
    RUN_NCU=0
fi

# NVML es opcional -- ver common/power_sampling.h. Mismo bloque de deteccion
# que Fase_4/Stencil/run_stencil_tc.sbatch, copiado tal cual (no depende de
# que kernel se compile).
NVML_FLAGS=()
NVML_LINK_FLAG="${NVML_LINK_FLAG:--lnvidia-ml}"
# Deteccion por PRUEBA DE ENLACE REAL, no por heuristica de rutas/ldconfig:
# en PACCA el driver deja libnvidia-ml.so.1 (versionado, en /usr/lib64) pero
# NUNCA el symlink de desarrollo libnvidia-ml.so sin version -- haria falta
# un paquete -devel que no esta instalado. "ldconfig -p" (puede no estar en
# PATH dentro del entorno del job de SLURM) y el chequeo de ruta exacta de
# abajo no detectaban ese caso, aunque "-lnvidia-ml" SI enlaza bien contra
# el .so versionado (verificado a mano en paccaA100, el compute node real).
# Sin este fix la campana entera queda con energy_gpu_j=NaN en TODAS las
# filas, aunque nvidia-smi funcione perfecto en el mismo nodo.
NVML_PROBE_BIN="$(mktemp -u)"
if [[ "${DISABLE_NVML_TELEMETRY:-0}" != "1" ]] &&
   "${NVCC}" -E -x cu - 2>/dev/null <<< '#include <nvml.h>' >/dev/null &&
   echo 'int main(){return 0;}' | "${NVCC}" -x cu - -o "${NVML_PROBE_BIN}" \
       ${NVML_LINK_FLAG} -lpthread >/dev/null 2>&1; then
    NVML_FLAGS=(-DUSE_NVML_TELEMETRY "${NVML_LINK_FLAG}" -lpthread)
    echo "NVML habilitado para telemetria GPU (verificado con enlace de prueba)."
else
    NVML_FLAGS=(-lpthread)
    echo "NVML no disponible para el enlace; se usara el stub de telemetria GPU."
fi
rm -f "${NVML_PROBE_BIN}"

# --- Prueba de humo ----------------------------------------------------------
# SMOKE_TEST=1 recorta el barrido al punto mas chico de cada lista y a 3
# iteraciones: sirve para confirmar que el binario compila, corre y emite
# CSV_SUMMARY/CSV_DRIFT bien formados ANTES de comprometer horas de cola. NO es
# un punto de la campana; sus numeros no se reportan.
SMOKE_TEST="${SMOKE_TEST:-0}"

# --- Barrido: tamanos, iteraciones, formato, compensacion --------------------
# N_LIST: cada valor debe ser potencia de 2 (construccion de Sylvester-Hadamard,
# ver el aviso de cabecera del .cu).
#
# PRESUPUESTO DE MEMORIA (A100-PCIE-40GB, 39.49 GiB) -- no subir N_LIST sin
# rehacer esta cuenta. Por elemento de la matriz NxN, con --tc both, --comp on
# y el ancla activa (el caso mas caro, Fase 4):
#     referencia FP64 (d_x64_in/out)          16 B
#     operador A en fp64 + fp16 + bf16        12 B
#     ruta sin comp  (2xT + t_raw)             8 B
#     ruta con comp  (2xT + t_raw + 2xcomp
#                     + comp_tc_raw + comp_as_tc)  22 B
#     ancla FP64 (4 buffers double)           32 B
#                                            -------
#                                             90 B/elemento
#   N=4096 ->  1.51 GB     N=8192  ->  6.04 GB   (15 % de la tarjeta)
#   N=16384 -> 24.2 GB (61 %: DESCARTADO, sin margen de seguridad)
#   N=32768 -> 96.6 GB (imposible)
# El limite superior 8192 deja un margen de 6.5x, muy por encima del 2x que
# exige el criterio de dimensionamiento del proyecto.
#
# ITERS_LIST reemplaza al antiguo escalar ITERS. La variable ITERS SIGUE
# funcionando y GANA si se exporta: `ITERS=20 bash run_gemm_chained.sbatch`
# corre una sola pasada a 20 iteraciones, exactamente como antes de este
# cambio. Solo cuando NO se exporta se usa el barrido de campana.
if [[ "${SMOKE_TEST}" == "1" ]]; then
    N_LIST="${N_LIST:-1024}"
    ITERS_LIST="${ITERS_LIST:-${ITERS:-3}}"
    RUN_NCU="${RUN_NCU:-0}"
else
    N_LIST="${N_LIST:-1024 2048 4096 8192}"
    ITERS_LIST="${ITERS_LIST:-${ITERS:-20 40 80}}"
fi
TC_FORMAT="${TC_FORMAT:-both}"           # fp16 | bf16 | both
COMP_LIST="${COMP_LIST:-off on}"         # compensacion por linealidad
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
SEED="${SEED:-42}"
TARGET_LAMBDA="${TARGET_LAMBDA:-1.1}"

# --- Perfilado NCU opcional (un solo kernel de interes: wmma_gemm_kernel) ---
RUN_NCU="${RUN_NCU:-0}"
NCU_MODE="${NCU_MODE:-quick}"
NCU_SET="${NCU_SET:-full}"
NCU_KERNEL_REGEX="${NCU_KERNEL_REGEX:-.*wmma_gemm_kernel.*}"

echo "Nodo: $(hostname)"
echo "Fecha: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-no definido}"
echo "Arquitectura CUDA: sm_${CUDA_ARCH}"
echo "NVCC=${NVCC}"
echo "RUN_NCU=${RUN_NCU}"

nvidia-smi || true

echo "Compilando gemm_chained (Fase_3/GEMM, incluye common/ por ruta relativa)..."
"${NVCC}" -std=c++17 gemm_chained.cu -o gemm_chained \
    -lcublas -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
    --allow-unsupported-compiler "${NVML_FLAGS[@]}"

# Procedencia del binario/corrida -- mismo patron que Fase_4/Stencil.
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo desconocido)"
if git diff --quiet 2>/dev/null; then GIT_STATUS="limpio"; else GIT_STATUS="sucio"; fi
BUILD_DATE="$(date -Iseconds)"
BIN_SHA256="$(sha256sum gemm_chained | awk '{print $1}')"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "GIT_STATUS=${GIT_STATUS}"
echo "BUILD_DATE=${BUILD_DATE}"
echo "BIN_SHA256=${BIN_SHA256}"
echo "RUN_PARAMS=N_LIST=${N_LIST} ITERS_LIST=${ITERS_LIST} TC_FORMAT=${TC_FORMAT} COMP_LIST=${COMP_LIST} CHECKPOINT_EVERY=${CHECKPOINT_EVERY} SEED=${SEED} TARGET_LAMBDA=${TARGET_LAMBDA} SMOKE_TEST=${SMOKE_TEST}"

# Validacion SASS pre-perfilado -- mismo patron que Fase_4/Stencil.
_nvcc_dir="$(dirname "${NVCC}")"
if [[ -z "${CUOBJDUMP:-}" ]]; then
    if command -v cuobjdump >/dev/null 2>&1; then
        CUOBJDUMP="$(command -v cuobjdump)"
    else
        for _cand in "${_nvcc_dir}/cuobjdump" "${_nvcc_dir}/../../cuda/bin/cuobjdump" \
                     "${_nvcc_dir}"/../../cuda/*/bin/cuobjdump; do
            [[ -x "${_cand}" ]] && { CUOBJDUMP="${_cand}"; break; }
        done
    fi
fi
if [[ -x "${CUOBJDUMP:-}" ]] || command -v "${CUOBJDUMP:-}" >/dev/null 2>&1; then
    HMMA_COUNT=$("${CUOBJDUMP}" --dump-sass gemm_chained | grep -c HMMA || true)
    echo "Instrucciones HMMA detectadas en binario: ${HMMA_COUNT} (cuobjdump: ${CUOBJDUMP})"
    if [[ "${RUN_NCU}" == "1" && "${HMMA_COUNT}" -eq 0 ]]; then
        echo "ERROR: binario sin instrucciones HMMA; revisar wmma_gemm_kernel antes de perfilar." >&2
        exit 1
    fi
else
    echo "AVISO: no se encontro cuobjdump; se omite la validacion HMMA." >&2
fi

for _common_ncu in "../../tools/common_ncu.sh" "./common_ncu.sh"; do
    if [[ -f "${_common_ncu}" ]]; then
        # shellcheck source=/dev/null
        source "${_common_ncu}"
        break
    fi
done

echo "Ejecutando gemm_chained (Fase 3: encadenamiento genuino, sin ancla)..."

# El perfilado NCU se restringe a la PRIMERA pasada de ITERS_LIST: ncu mide
# wmma_gemm_kernel en un solo lanzamiento (--launch-count 1) y ese kernel es
# identico para cualquier valor de --iters, asi que repetirlo por cada
# iteracion de la lista multiplicaria el tiempo de cola sin producir un dato
# nuevo. Mismo criterio que ya usa Fase_4/Stencil con KAHAN_LIST/ANCHOR_LIST.
read -r _iters_first _ <<< "${ITERS_LIST}"

{
for N in ${N_LIST}; do
  for ITERS in ${ITERS_LIST}; do
    for COMP in ${COMP_LIST}; do
        echo
        echo "============================================================"
        echo "Corrida: N=${N}, ITERS=${ITERS}, TC=${TC_FORMAT}, COMP=${COMP}"
        echo "============================================================"
        ./gemm_chained --n "${N}" --iters "${ITERS}" --tc "${TC_FORMAT}" \
            --comp "${COMP}" --checkpoint-every "${CHECKPOINT_EVERY}" \
            --seed "${SEED}" --target-lambda "${TARGET_LAMBDA}"

        if [[ "${ITERS}" != "${_iters_first}" ]]; then
            continue
        fi

        if [[ "${RUN_NCU}" == "1" ]]; then
            if [[ "${NCU_MODE}" == "full" ]]; then
                NCU_CAPTURE_ARGS=(--set "${NCU_SET}")
            else
                NCU_CAPTURE_ARGS=(--metrics "${NCU_QUICK_METRICS:-}")
            fi
            REPORT="${LOGS_DIR}/ncu_gemm_n${N}_comp${COMP}_job${SLURM_JOB_ID:-manual}"
            echo "Perfilando wmma_gemm_kernel con Nsight Compute (N=${N}, COMP=${COMP})..."
            ncu_run "${NCU}" "${NCU_CAPTURE_ARGS[@]}" --target-processes all \
                --kernel-name "regex:${NCU_KERNEL_REGEX}" \
                --launch-count 1 --export "${REPORT}" --force-overwrite \
                ./gemm_chained --n "${N}" --iters 1 --tc "${TC_FORMAT}" --comp "${COMP}"
        fi
    done
  done
done
} 2>&1 | tee -a "${RESULTS_LOG}"

# Post-proceso de CSV: extract_csv_chained.py (esquema propio de gemm_chained/
# conv_chained, no el de Stencil -- ver Fase_3/tools/README.md). Se busca en
# varias rutas candidatas y, si no se encuentra, se avisa y se omite en vez de
# fallar el job -- el log crudo queda disponible para post-procesar despues.
EXTRACT_CSV_SCRIPT="${EXTRACT_CSV_SCRIPT:-}"
if [[ -z "${EXTRACT_CSV_SCRIPT}" ]]; then
    for _cand in "../tools/extract_csv_chained.py" "tools/extract_csv_chained.py" \
                 "../../tools/extract_csv_chained.py"; do
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
        --input "${RESULTS_LOG}" --outdir "${RESULTS_DIR}" --job-id "${JOB_ID}" --kernel gemm
else
    echo
    echo "AVISO: extract_csv_chained.py no encontrado (se probo ../tools/, tools/ y" \
         "../../tools/ relativos a esta carpeta). Se omite la extraccion; el log crudo" \
         "con los tokens CSV_* queda en ${RESULTS_LOG}." >&2
fi
