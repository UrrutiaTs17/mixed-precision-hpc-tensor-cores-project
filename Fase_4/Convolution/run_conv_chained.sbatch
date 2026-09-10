#!/bin/bash
#SBATCH --job-name=mixed_precision_conv_chained_f4
#SBATCH --output=logs/mixed_precision_conv_chained_f4_%j.out
#SBATCH --error=logs/mixed_precision_conv_chained_f4_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=24:00:00

# Fase 4: mismo binario y mismos parametros que
# Fase_3/Convolution/run_conv_chained.sbatch (del que este script parte, ver
# Fase_4/Convolution/README.md), mas ANCHOR_LIST para barrer --anchor-every
# (ancla FP64). Con ANCHOR_LIST="0" (el default) el comportamiento es
# identico al de Fase 3. Ver el comentario de cabecera de
# Fase_4/GEMM/run_gemm_chained.sbatch para el razonamiento completo sobre
# como se combina ANCHOR_LIST con COMP_LIST (identico aqui).
#
#   sbatch run_conv_chained.sbatch                                     # CAMPANA COMPLETA
#   sbatch --export=ALL,SMOKE_TEST=1 run_conv_chained.sbatch           # prueba de humo
#   sbatch --export=ALL,ANCHOR_LIST="0 1" run_conv_chained.sbatch      # gates K=0 / K=1
#   sbatch --export=ALL,ANCHOR_LIST="2 5 10 20" run_conv_chained.sbatch  # barrido
#
# OJO: el default de HW_LIST/ITERS_LIST/ANCHOR_LIST es la CAMPANA COMPLETA, no
# una prueba rapida. Para eso esta SMOKE_TEST=1. Los gates K=0/K=1 se lanzan
# con gate3_ancla.sbatch de esta misma carpeta.

set -euo pipefail

# No forzar cd a SLURM_SUBMIT_DIR: cuando este script se invoca ANIDADO
# (`bash` dentro de un job ya en marcha -- ver tools/validacion_preliminar.sbatch,
# que hace `cd` al directorio correcto ANTES de llamar `bash <este script>`),
# SLURM_SUBMIT_DIR es el del job PADRE (la raiz del repo), no el de este
# kernel, y pisaria un pwd que el llamador ya dejo correcto. Se prueba pwd
# primero; solo se cae a SLURM_SUBMIT_DIR cuando pwd no trae el fuente
# (invocacion de primer nivel via `sbatch` desde la raiz del repo).
if [[ ! -f conv_chained.cu ]]; then
    if [[ -f Convolution/conv_chained.cu ]]; then
        cd Convolution
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/conv_chained.cu" ]]; then
        cd "${SLURM_SUBMIT_DIR}"
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/Convolution/conv_chained.cu" ]]; then
        cd "${SLURM_SUBMIT_DIR}/Convolution"
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
# SMOKE_TEST=1 recorta el barrido al punto mas chico y a 3 iteraciones: sirve
# para confirmar que el binario compila, corre y emite CSV_SUMMARY/CSV_DRIFT
# bien formados (con la columna anchor_every) ANTES de comprometer horas de
# cola. No es un punto de campana.
SMOKE_TEST="${SMOKE_TEST:-0}"

# --- Barrido: tamano espacial, iteraciones, formato, compensacion ------------
# PRESUPUESTO DE MEMORIA (A100-PCIE-40GB, 39.49 GiB) -- CALCULO CORREGIDO.
# conv_chained.cu NO usa cuDNN ni CUTLASS (eso es Fase 2): reescribe la
# convolucion como GEMM con un im2col propio y la resuelve con el mismo kernel
# WMMA y las mismas llamadas cublasDgemm que GEMM, asi que todos los buffers
# son cudaMalloc de tamano conocido -- se calcula con la misma precision que el
# de GEMM, sin el workspace impredecible de una libreria.
#
# Por elemento de campo (kChannels*hw^2 = 64*hw^2), con --tc both, --comp on y
# ancla activa; el im2col en double (kCRS*hw^2 = 576*hw^2, 9x el campo) domina:
#     referencia FP64 16 B | im2col double 72 B | ruta sin comp 26 B
#     ruta con comp   56 B | ancla FP64    32 B   -> 202 B/elemento de campo
#   = 12.9 KB por celda espacial
#   hw=256  -> 0.85 GB    hw=512  -> 3.39 GB (12x de margen)
#   hw=1024 -> 13.6 GB (2.9x: techo bajo el criterio de 2x)
#   hw=2048 -> 54.2 GB (imposible)
# Ver el comentario extenso equivalente en Fase_3/Convolution/run_conv_chained.sbatch
# para la referencia cruzada con GEMM y por que el limite queda en 512.
#
# ITERS_LIST reemplaza al antiguo escalar ITERS, que SIGUE funcionando y GANA
# si se exporta (`ITERS=20 bash run_conv_chained.sbatch` = una sola pasada).
if [[ "${SMOKE_TEST}" == "1" ]]; then
    HW_LIST="${HW_LIST:-64}"
    ITERS_LIST="${ITERS_LIST:-${ITERS:-3}}"
    RUN_NCU="${RUN_NCU:-0}"
else
    HW_LIST="${HW_LIST:-64 128 256 512}"
    ITERS_LIST="${ITERS_LIST:-${ITERS:-20 40 80}}"
fi
TC_FORMAT="${TC_FORMAT:-both}"
COMP_LIST="${COMP_LIST:-off on}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
SEED="${SEED:-42}"

# --- ANCLA FP64 (Fase 4) -----------------------------------------------------
# Misma escala de K que GEMM (0 control, 1 gate, 5 y 20 intermedios). En
# SMOKE_TEST solo K=0 y K=1.
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

echo "Compilando conv_chained (Fase_4/Convolution, incluye common/ por ruta relativa)..."
"${NVCC}" -std=c++17 conv_chained.cu -o conv_chained \
    -lcublas -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
    --allow-unsupported-compiler "${NVML_FLAGS[@]}"

GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo desconocido)"
if git diff --quiet 2>/dev/null; then GIT_STATUS="limpio"; else GIT_STATUS="sucio"; fi
BUILD_DATE="$(date -Iseconds)"
BIN_SHA256="$(sha256sum conv_chained | awk '{print $1}')"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "GIT_STATUS=${GIT_STATUS}"
echo "BUILD_DATE=${BUILD_DATE}"
echo "BIN_SHA256=${BIN_SHA256}"
echo "RUN_PARAMS=HW_LIST=${HW_LIST} ITERS_LIST=${ITERS_LIST} TC_FORMAT=${TC_FORMAT} COMP_LIST=${COMP_LIST} CHECKPOINT_EVERY=${CHECKPOINT_EVERY} SEED=${SEED} ANCHOR_LIST=${ANCHOR_LIST} SMOKE_TEST=${SMOKE_TEST}"

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
    HMMA_COUNT=$("${CUOBJDUMP}" --dump-sass conv_chained | grep -c HMMA || true)
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

echo "Ejecutando conv_chained (Fase 4: encadenamiento + ancla FP64)..."

# El perfilado NCU se restringe a la PRIMERA pasada de ITERS_LIST y de
# ANCHOR_LIST: ncu mide wmma_gemm_kernel en un solo lanzamiento y ese kernel es
# identico para cualquier --iters y cualquier K (el ancla sustituye el paso, no
# cambia el kernel), asi que repetirlo multiplicaria el tiempo de cola sin
# producir un dato nuevo.
read -r _iters_first _ <<< "${ITERS_LIST}"

{
for HW in ${HW_LIST}; do
  for ITERS in ${ITERS_LIST}; do
    for COMP in ${COMP_LIST}; do
        if [[ "${COMP}" == "on" ]]; then
            _anchor_values="${ANCHOR_LIST}"
        else
            _anchor_values="0"
        fi
        for ANCHOR in ${_anchor_values}; do
            echo
            echo "============================================================"
            echo "Corrida: HW=${HW}, ITERS=${ITERS}, TC=${TC_FORMAT}, COMP=${COMP}, ANCHOR_EVERY=${ANCHOR}"
            echo "============================================================"
            ./conv_chained --hw "${HW}" --iters "${ITERS}" --tc "${TC_FORMAT}" \
                --comp "${COMP}" --checkpoint-every "${CHECKPOINT_EVERY}" \
                --seed "${SEED}" --anchor-every "${ANCHOR}"

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
                REPORT="${LOGS_DIR}/ncu_conv_hw${HW}_comp${COMP}_job${SLURM_JOB_ID:-manual}"
                echo "Perfilando wmma_gemm_kernel con Nsight Compute (HW=${HW}, COMP=${COMP})..."
                ncu_run "${NCU}" "${NCU_CAPTURE_ARGS[@]}" --target-processes all \
                    --kernel-name "regex:${NCU_KERNEL_REGEX}" \
                    --launch-count 1 --export "${REPORT}" --force-overwrite \
                    ./conv_chained --hw "${HW}" --iters 1 --tc "${TC_FORMAT}" --comp "${COMP}"
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
        --input "${RESULTS_LOG}" --outdir "${RESULTS_DIR}" --job-id "${JOB_ID}" --kernel conv
else
    echo
    echo "AVISO: extract_csv_chained.py no encontrado (se probo ../tools/, tools/ y" \
         "../../tools/ relativos a esta carpeta). Se omite la extraccion; el log crudo" \
         "con los tokens CSV_* queda en ${RESULTS_LOG}." >&2
fi
