#!/bin/bash
#SBATCH --job-name=mixed_precision_gemm_chained_f4
#SBATCH --output=logs/mixed_precision_gemm_chained_f4_%j.out
#SBATCH --error=logs/mixed_precision_gemm_chained_f4_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=24:00:00

# Fase 4: mismo binario y mismos parametros que Fase_3/GEMM/run_gemm_chained.sbatch
# (del que este script parte, ver Fase_4/GEMM/README.md), mas ANCHOR_LIST para
# barrer --anchor-every (ancla FP64). Con ANCHOR_LIST="0" (el default) el
# comportamiento es identico al de Fase 3.
#
#   sbatch run_gemm_chained.sbatch                                     # CAMPANA COMPLETA
#   sbatch --export=ALL,SMOKE_TEST=1 run_gemm_chained.sbatch           # prueba de humo
#   sbatch --export=ALL,ANCHOR_LIST="0 1" run_gemm_chained.sbatch      # gates K=0 / K=1
#   sbatch --export=ALL,ANCHOR_LIST="2 5 10 20" run_gemm_chained.sbatch  # barrido
#
# OJO: el default de N_LIST/ITERS_LIST/ANCHOR_LIST es la CAMPANA COMPLETA (ver
# las secciones de barrido mas abajo), no una prueba rapida. Para eso esta
# SMOKE_TEST=1. Los gates K=0/K=1 se lanzan con gate3_ancla.sbatch de esta
# misma carpeta, que no necesita el barrido completo.
#
# El ancla solo aplica a la ruta CON compensacion (el binario rechaza
# --anchor-every>0 sin --comp on) -- este script, a diferencia de
# Fase_4/Stencil/run_stencil_tc.sbatch, no necesita forzar COMP_LIST: en vez
# de eso, deriva por cada valor de COMP que ANCHOR_LIST solo se recorre
# cuando COMP=on (ver el bucle principal, mas abajo) y valida por adelantado
# que no haya una combinacion imposible antes de compilar nada.

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

# --- Prueba de humo ----------------------------------------------------------
# SMOKE_TEST=1 recorta el barrido al punto mas chico de cada lista y a 3
# iteraciones: confirma que el binario compila, corre y emite CSV_SUMMARY/
# CSV_DRIFT bien formados (con la columna anchor_every) ANTES de comprometer
# horas de cola. NO es un punto de la campana; sus numeros no se reportan.
SMOKE_TEST="${SMOKE_TEST:-0}"

# --- Barrido: tamanos, iteraciones, formato, compensacion --------------------
# N_LIST: cada valor debe ser potencia de 2 (Sylvester-Hadamard).
#
# PRESUPUESTO DE MEMORIA (A100-PCIE-40GB, 39.49 GiB) -- no subir N_LIST sin
# rehacer esta cuenta. Por elemento de la matriz NxN, con --tc both, --comp on
# y el ancla activa (el caso mas caro):
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
# El limite superior 8192 deja un margen de 6.5x sobre la memoria fisica.
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
TC_FORMAT="${TC_FORMAT:-both}"
COMP_LIST="${COMP_LIST:-off on}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
SEED="${SEED:-42}"
TARGET_LAMBDA="${TARGET_LAMBDA:-1.1}"

# --- ANCLA FP64 (Fase 4) -----------------------------------------------------
# Ver el comentario de cabecera del archivo. Aviso temprano si se pide un K>0
# sin que COMP_LIST incluya "on" en absoluto (si lo incluye, el bucle
# principal ya se encarga de nunca combinar COMP=off con ANCHOR>0).
# Escala de K de la campana (Tarea 5 del prompt de auditoria): 0 (control, sin
# ancla), 1 (gate de exactitud), 5 y 20 (los dos K intermedios que responden la
# pregunta real: cuanto se recupera y a que costo). En SMOKE_TEST solo K=0 y
# K=1, que son los dos que hay que validar antes de cualquier campana.
if [[ "${SMOKE_TEST}" == "1" ]]; then
    ANCHOR_LIST="${ANCHOR_LIST:-0 1}"
else
    ANCHOR_LIST="${ANCHOR_LIST:-0 1 5 20}"
fi
_comp_has_on=0
for _c in ${COMP_LIST}; do [[ "${_c}" == "on" ]] && _comp_has_on=1; done
if [[ "${_comp_has_on}" == "0" ]]; then
    for _a in ${ANCHOR_LIST}; do
        if [[ "${_a}" != "0" ]]; then
            echo "ERROR: ANCHOR_LIST='${ANCHOR_LIST}' incluye K=${_a} > 0, pero" \
                 "COMP_LIST='${COMP_LIST}' no incluye 'on': el ancla FP64 requiere" \
                 "--comp on. Agregue 'on' a COMP_LIST o quite los valores distintos" \
                 "de 0 de ANCHOR_LIST." >&2
            exit 1
        fi
    done
fi

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

echo "Compilando gemm_chained (Fase_4/GEMM, incluye common/ por ruta relativa)..."
"${NVCC}" -std=c++17 gemm_chained.cu -o gemm_chained \
    -lcublas -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
    --allow-unsupported-compiler "${NVML_FLAGS[@]}"

GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo desconocido)"
if git diff --quiet 2>/dev/null; then GIT_STATUS="limpio"; else GIT_STATUS="sucio"; fi
BUILD_DATE="$(date -Iseconds)"
BIN_SHA256="$(sha256sum gemm_chained | awk '{print $1}')"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "GIT_STATUS=${GIT_STATUS}"
echo "BUILD_DATE=${BUILD_DATE}"
echo "BIN_SHA256=${BIN_SHA256}"
echo "RUN_PARAMS=N_LIST=${N_LIST} ITERS_LIST=${ITERS_LIST} TC_FORMAT=${TC_FORMAT} COMP_LIST=${COMP_LIST} CHECKPOINT_EVERY=${CHECKPOINT_EVERY} SEED=${SEED} TARGET_LAMBDA=${TARGET_LAMBDA} ANCHOR_LIST=${ANCHOR_LIST} SMOKE_TEST=${SMOKE_TEST}"

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

echo "Ejecutando gemm_chained (Fase 4: encadenamiento + ancla FP64)..."

# El perfilado NCU se restringe a la PRIMERA pasada de ITERS_LIST (ademas de
# la primera de ANCHOR_LIST, ver abajo): ncu mide wmma_gemm_kernel en un solo
# lanzamiento y ese kernel es identico para cualquier --iters, asi que
# repetirlo por cada valor de la lista multiplicaria el tiempo de cola sin
# producir un dato nuevo.
read -r _iters_first _ <<< "${ITERS_LIST}"

{
for N in ${N_LIST}; do
  for ITERS in ${ITERS_LIST}; do
    for COMP in ${COMP_LIST}; do
        # ANCLA FP64 (Fase 4): solo se barre ANCHOR_LIST cuando COMP=on -- con
        # COMP=off la unica pasada valida es K=0 (ver el aviso temprano arriba
        # para el caso en que ni siquiera exista un COMP=on en la lista).
        if [[ "${COMP}" == "on" ]]; then
            _anchor_values="${ANCHOR_LIST}"
        else
            _anchor_values="0"
        fi
        for ANCHOR in ${_anchor_values}; do
            echo
            echo "============================================================"
            echo "Corrida: N=${N}, ITERS=${ITERS}, TC=${TC_FORMAT}, COMP=${COMP}, ANCHOR_EVERY=${ANCHOR}"
            echo "============================================================"
            ./gemm_chained --n "${N}" --iters "${ITERS}" --tc "${TC_FORMAT}" \
                --comp "${COMP}" --checkpoint-every "${CHECKPOINT_EVERY}" \
                --seed "${SEED}" --target-lambda "${TARGET_LAMBDA}" \
                --anchor-every "${ANCHOR}"

            # El perfilado NCU no depende de ANCHOR ni de ITERS: mide
            # wmma_gemm_kernel en un paso normal, que el ancla no modifica (ver
            # Fase_4/GEMM/README.md -- el ancla sustituye el paso, no cambia el
            # kernel). Se restringe a la primera pasada de ambas listas.
            read -r _anchor_first _ <<< "${_anchor_values}"
            if [[ "${ANCHOR}" != "${_anchor_first}" || "${ITERS}" != "${_iters_first}" ]]; then
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
done
} 2>&1 | tee -a "${RESULTS_LOG}"

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
