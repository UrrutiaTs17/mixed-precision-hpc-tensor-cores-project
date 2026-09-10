#!/bin/bash
#SBATCH --job-name=mixed_precision_conv_tc
#SBATCH --output=logs/mixed_precision_conv_tc_%j.out
#SBATCH --error=logs/mixed_precision_conv_tc_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=24:00:00

# Corre conv_tensor_activation.cu (las 4 rutas de convolucion 2D: CPU,
# cuDNN FP32, cuDNN Tensor Core FP16/BF16, WMMA propio) y, salvo que se pida
# lo contrario, perfila el kernel WMMA con Nsight Compute.
#
# Nada esta hardcodeado en el cuerpo del script: el formato Tensor Core, la
# forma del problema (entrada, filtro, batch, padding/stride/dilation), las
# iteraciones y el directorio de salida son variables de entorno con un
# default documentado mas abajo (patron VAR="${VAR:-default}"). El unico
# hardcodeo real que queda son los paths de #SBATCH --output/--error de
# arriba: SLURM los procesa antes de que este script exista, asi que no
# pueden depender de una variable definida aqui adentro; para cambiarlos,
# pasa --output=/--error= en la propia linea de sbatch.
#
# TC_FORMAT por defecto es "both" (corre FP16 y BF16 en la misma invocacion,
# el binario ya soporta --tc-format both nativamente): el .sbatch historico
# (old/Fase_2/Convolution/run_conv_tc.sbatch) solo pasaba --iters sin
# --tc-format, es decir siempre corria unicamente FP16 (el default del
# binario) pese a que la ruta BF16 (benchmark_gpu_tensor_cores_conv_bf16)
# siempre existio -- ver Fase_2/Convolution/README.md, seccion "Gap
# historico: BF16 nunca se corria por defecto".
#
#   sbatch run_conv_tc.sbatch
#   sbatch --export=ALL,SMOKE_TEST=1 run_conv_tc.sbatch
#   sbatch --export=ALL,TC_FORMAT=fp16 run_conv_tc.sbatch
#   sbatch --export=ALL,HW_LIST="64 128 256 512" run_conv_tc.sbatch
#   sbatch --export=ALL,CUDA_ARCH=86 run_conv_tc.sbatch
#   sbatch --export=ALL,CUDNN_ROOT=/otra/ruta run_conv_tc.sbatch
#   sbatch --export=ALL,RUN_CUTLASS=0 run_conv_tc.sbatch
#   sbatch --export=ALL,CUTLASS_DIR=/ruta/a/otro/cutlass/include run_conv_tc.sbatch

set -euo pipefail

# No forzar cd a SLURM_SUBMIT_DIR: cuando este script se invoca ANIDADO
# (`bash` dentro de un job ya en marcha -- ver tools/validacion_preliminar.sbatch,
# que hace `cd` al directorio correcto ANTES de llamar `bash <este script>`),
# SLURM_SUBMIT_DIR es el del job PADRE (la raiz del repo), no el de este
# kernel, y pisaria un pwd que el llamador ya dejo correcto. Se prueba pwd
# primero; solo se cae a SLURM_SUBMIT_DIR cuando pwd no trae el fuente
# (invocacion de primer nivel via `sbatch` desde la raiz del repo).
if [[ ! -f conv_tensor_activation.cu ]]; then
    if [[ -f Convolution/conv_tensor_activation.cu ]]; then
        cd Convolution
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/conv_tensor_activation.cu" ]]; then
        cd "${SLURM_SUBMIT_DIR}"
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/Convolution/conv_tensor_activation.cu" ]]; then
        cd "${SLURM_SUBMIT_DIR}/Convolution"
    fi
fi

# Directorio para binarios/reportes NCU/artefactos auxiliares de esta
# corrida (no los logs de SLURM en si, ver la nota sobre #SBATCH
# --output/--error arriba).
OUTPUT_DIR="${OUTPUT_DIR:-logs}"
mkdir -p "${OUTPUT_DIR}"

# Carga de modulos: en PACCA no hay modulo "cuda"; nvcc/ncu se toman por ruta.
if command -v module >/dev/null 2>&1; then
    # En PACCA (OpenHPC/Lmod) "openblas" vive bajo la familia de
    # compilador "gnu12" -- cargarlo sin eso primero falla en silencio
    # (el "|| true" se traga el error de Lmod) y deja OPENBLAS_DIR sin
    # definir, cayendo al fallback generico de abajo que no encuentra
    # cblas.h en este sistema. module load es un no-op inofensivo si
    # gnu12 ya esta activo o si el cluster no usa esta jerarquia.
    module load gnu12 || true
    module load openblas || true
fi

# Arquitectura por defecto para A100.
# Puedes sobrescribirla con:
#   sbatch --export=ALL,CUDA_ARCH=86 run_conv_tc.sbatch
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
normalize_include_flags() {
    local raw="${1:-}"
    local item
    local normalized=()

    for item in ${raw}; do
        if [[ "${item}" == -I* || "${item}" == -* ]]; then
            normalized+=("${item}")
        else
            normalized+=("-I${item}")
        fi
    done

    if [[ "${#normalized[@]}" -gt 0 ]]; then
        printf '%s ' "${normalized[@]}"
    fi
}

# On OpenHPC, OPENBLAS_DIR is reliably set by the module; derive flags from it
# to avoid getting the raw path passed to nvcc without -I.
if [[ -n "${OPENBLAS_DIR:-}" ]]; then
    OPENBLAS_INC="-I${OPENBLAS_DIR}/include"
    # -Xlinker -rpath -Xlinker hornea la ruta en el binario (nvcc del HPC SDK
    # 23.1 de PACCA no acepta "-Wl,-rpath,...", ver commit efd220e): el modulo de Lmod en este
    # cluster no deja LD_LIBRARY_PATH confiable en todos los casos (ver
    # el comentario junto a "module load gnu12" mas arriba), y sin rpath
    # el binario compila pero falla en RUNTIME con
    # "libopenblas.so.0: cannot open shared object file".
    OPENBLAS_LIBS="-L${OPENBLAS_DIR}/lib -Xlinker -rpath -Xlinker ${OPENBLAS_DIR}/lib -lopenblas"
elif [[ -z "${OPENBLAS_INC:-}" && -f "/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/include/cblas.h" ]]; then
    # Ruta historica de PACCA (mismo patron que detect_nvcc en
    # tools/detect_toolchain.sh): cubre el caso en que el modulo no
    # cargo (Lmod ausente o jerarquia distinta) pero el paquete OpenHPC
    # si esta instalado.
    OPENBLAS_INC="-I/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/include"
    OPENBLAS_LIBS="${OPENBLAS_LIBS:--L/opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib -Xlinker -rpath -Xlinker /opt/ohpc/pub/libs/gnu12/openblas/0.3.21/lib -lopenblas}"
else
    OPENBLAS_INC="$(normalize_include_flags "${OPENBLAS_INC:--I/usr/include/openblas}")"
    OPENBLAS_LIBS="${OPENBLAS_LIBS:--lopenblas}"
fi
# Resolucion de cuDNN. En PACCA no hay cuDNN del sistema ni modulo: la unica
# instalacion compatible con CUDA 12 y con la API v7 que usa este codigo es la
# cuDNN 9.x (wheel cu12) del entorno Analytics. Se auto-detecta desde CUDNN_ROOT
# para no tener que exportar nada en cada corrida. Sobrescribible con:
#   sbatch --export=ALL,CUDNN_ROOT=/otra/ruta        run_conv_tc.sbatch
#   sbatch --export=ALL,CUDNN_INC=...,CUDNN_LIBS=...  run_conv_tc.sbatch
CUDNN_ROOT="${CUDNN_ROOT:-/opt/ohpc/pub/Analytics/anaconda3/envs/hybrid-profiler/lib/python3.10/site-packages/nvidia/cudnn}"

if [[ -z "${CUDNN_INC:-}" && -f "${CUDNN_ROOT}/include/cudnn.h" ]]; then
    CUDNN_INC="-I${CUDNN_ROOT}/include"
else
    CUDNN_INC="$(normalize_include_flags "${CUDNN_INC:-}")"
fi

if [[ -z "${CUDNN_LIBS:-}" ]]; then
    if [[ -e "${CUDNN_ROOT}/lib/libcudnn.so" ]]; then
        CUDNN_LIBS="-L${CUDNN_ROOT}/lib -lcudnn"
    elif compgen -G "${CUDNN_ROOT}/lib/libcudnn.so.*" >/dev/null; then
        # El wheel de cuDNN 9 solo trae libcudnn.so.9 (sin el symlink .so), por
        # lo que -lcudnn fallaria: se enlaza el soname exacto con -l:.
        _cudnn_soname="$(basename "$(ls -1 "${CUDNN_ROOT}"/lib/libcudnn.so.* | sort -V | tail -1)")"
        CUDNN_LIBS="-L${CUDNN_ROOT}/lib -l:${_cudnn_soname}"
    else
        CUDNN_LIBS="-lcudnn"
    fi
fi

# Runtime: cuDNN 9 despacha a sub-librerias (cnn/graph/ops...) y se apoya en
# cublas/cudart del mismo stack cu12. Se agregan al LD_LIBRARY_PATH los lib/ de
# cudnn y de sus paquetes hermanos (../<pkg>/lib) para que el loader los halle.
if [[ -d "${CUDNN_ROOT}/lib" ]]; then
    LD_LIBRARY_PATH="${CUDNN_ROOT}/lib:${LD_LIBRARY_PATH:-}"
    for _libdir in "${CUDNN_ROOT}"/../*/lib; do
        [[ -d "${_libdir}" ]] && LD_LIBRARY_PATH="${_libdir}:${LD_LIBRARY_PATH}"
    done
    export LD_LIBRARY_PATH
fi
SMOKE_TEST="${SMOKE_TEST:-0}"
if [[ "${SMOKE_TEST}" == "1" ]]; then
    RUN_NCU="${RUN_NCU:-0}"
else
    RUN_NCU="${RUN_NCU:-1}"
fi
NCU_SET="${NCU_SET:-full}"
NCU_KERNEL_REGEX="${NCU_KERNEL_REGEX:-.*wmma_gemm_kernel.*}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-3}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-1}"

# Definiciones compartidas de perfilado (NCU_QUICK_METRICS). Se declara una
# sola vez en tools/common_ncu.sh (fuera de cualquier Fase_N) y se reutiliza
# entre los .sbatch de todas las fases; este script corre desde
# Fase_2/Convolution, de ahi "../../tools/common_ncu.sh" como ruta principal.
for _common_ncu in "../../tools/common_ncu.sh" "./common_ncu.sh"; do
    if [[ -f "${_common_ncu}" ]]; then
        # shellcheck source=/dev/null
        source "${_common_ncu}"
        break
    fi
done

# Separacion validacion/caracterizacion en NCU:
#   quick -> --metrics "${NCU_QUICK_METRICS}" (rapido, valida Tensor Cores)
#   full  -> --set "${NCU_SET}"               (caracterizacion completa)
NCU_MODE="${NCU_MODE:-quick}"
TIMEFORMAT=">>> Tiempo real de la llamada ncu (NCU_MODE=${NCU_MODE}): %R s (user %U, sys %S)"

# Formato(s) de la ruta Tensor Core (3): fp16 | bf16 | both. El binario ya
# soporta "both" nativamente (una sola invocacion corre las dos rutas), asi
# que no hace falta invocarlo dos veces como en Fase_1 (donde fp32/fp64 son
# dos binarios logicos distintos, no un solo flag).
TC_FORMAT="${TC_FORMAT:-both}"

# RUN_CUTLASS: activa la ruta 5 (CUTLASS ImplicitGemmConvolution) ademas de
# las cuatro rutas historicas (CPU, cuDNN clasico, cuDNN TC, WMMA propio).
# Mismo mecanismo que Fase_2/GEMM/run_gemm_tc.sbatch: CUTLASS es header-only,
# solo hace falta un checkout de github.com/NVIDIA/cutlass (serie 2.x, ver
# REQUIREMENTS.md) cuyo include/ se agrega al -I de nvcc via CUTLASS_DIR.
#
# Por defecto CUTLASS_DIR apunta a $HOME/cutlass/include: clona CUTLASS ahi
# antes de lanzar el job --
#   git clone --branch v2.11.0 https://github.com/NVIDIA/cutlass "$HOME/cutlass"
# -- o exporta CUTLASS_DIR apuntando a otro checkout ya existente en PACCA.
# Si RUN_CUTLASS=1 pero CUTLASS_DIR no contiene un checkout valido, la
# compilacion de conv_tc NO falla (conv_tensor_activation.cu detecta la
# ausencia de CUTLASS con __has_include y compila igual las 4 rutas
# restantes), pero cada corrida con --cutlass terminara en tiempo de
# ejecucion con un mensaje de error explicando como habilitarlo.
RUN_CUTLASS="${RUN_CUTLASS:-1}"
CUTLASS_DIR="${CUTLASS_DIR:-$HOME/cutlass/include}"
if [[ "${RUN_CUTLASS}" == "1" ]]; then
    CUTLASS_INC="-I${CUTLASS_DIR}"
    # VERIFICAR EN PACCA: ver la misma nota en run_gemm_tc.sbatch -- no se
    # pudo confirmar si esta version puntual de CUTLASS necesita
    # --expt-relaxed-constexpr (no hay nvcc ni GPU en el entorno donde se
    # escribio esto).
    CUTLASS_EXTRA_NVCC_FLAGS="--expt-relaxed-constexpr"
else
    CUTLASS_INC=""
    CUTLASS_EXTRA_NVCC_FLAGS=""
fi

# Forma del problema: misma campana que Fase 3/4, con 64 canales y
# HW_LIST=64,128,256,512. H/W escalares siguen disponibles para una corrida
# manual y, si se exportan, reemplazan el barrido por una sola malla cuadrada.
# SMOKE_TEST conserva solo el menor tamano y cumple
# las condiciones de divisibilidad de la ruta WMMA (K multiplo de 64,
# outH*outW multiplo de 64, C*R*S multiplo de 32): con C=H=W=K=64, R=S=3,
# outH*outW=4096 (mult. 64), K=64 (mult. 64), C*R*S=576 (mult. 32).
if [[ "${SMOKE_TEST}" == "1" ]]; then
    _default_hw_list="64"
    _default_iters=2
else
    _default_hw_list="64 128 256 512"
    _default_iters=10
fi
N="${N:-1}"
C="${C:-64}"
K="${K:-64}"
R="${R:-3}"
S="${S:-3}"
ITERS="${ITERS:-${_default_iters}}"
if [[ -n "${H:-}" || -n "${W:-}" ]]; then
    H="${H:-${W}}"
    W="${W:-${H}}"
    if [[ "${H}" != "${W}" ]]; then
        echo "ERROR: las campanas comparables exigen H=W (recibido H=${H}, W=${W})." >&2
        exit 2
    fi
    HW_LIST="${H}"
else
    HW_LIST="${HW_LIST:-${_default_hw_list}}"
fi
PAD_H="${PAD_H:-1}"
PAD_W="${PAD_W:-1}"
STRIDE_H="${STRIDE_H:-1}"
STRIDE_W="${STRIDE_W:-1}"
DILATION_H="${DILATION_H:-1}"
DILATION_W="${DILATION_W:-1}"

echo "Nodo: $(hostname)"
echo "Fecha: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-no definido}"
echo "Arquitectura CUDA: sm_${CUDA_ARCH}"
echo "NVCC=${NVCC}"
echo "NCU=${NCU}"
echo "SMOKE_TEST=${SMOKE_TEST}"
echo "RUN_NCU=${RUN_NCU}"
echo "TC_FORMAT=${TC_FORMAT}"
echo "HW_LIST=${HW_LIST}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RUN_CUTLASS=${RUN_CUTLASS}"
echo "CUTLASS_DIR=${CUTLASS_DIR}"

nvidia-smi || true

echo "Compilando conv_tc..."
"${NVCC}" -std=c++17 -O3 conv_tensor_activation.cu -o conv_tc \
    ${OPENBLAS_INC} ${CUDNN_INC} ${CUDNN_LIBS} ${OPENBLAS_LIBS} \
    ${CUTLASS_INC} ${CUTLASS_EXTRA_NVCC_FLAGS} \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    --allow-unsupported-compiler

# Validacion SASS pre-perfilado: si el binario no contiene instrucciones HMMA,
# el kernel WMMA no se activo (tipicamente por dimensiones no divisibles) y
# perfilar seria inutil. CUOBJDUMP se busca por PATH, junto a NVCC, o en el
# arbol cuda/ del HPC SDK (layout tipico .../<version>/cuda/bin/cuobjdump).
_nvcc_dir="$(dirname "${NVCC}")"
if [[ -z "${CUOBJDUMP:-}" ]]; then
    if command -v cuobjdump >/dev/null 2>&1; then
        CUOBJDUMP="$(command -v cuobjdump)"
    else
        for _cand in \
            "${_nvcc_dir}/cuobjdump" \
            "${_nvcc_dir}/../../cuda/bin/cuobjdump" \
            "${_nvcc_dir}"/../../cuda/*/bin/cuobjdump; do
            if [[ -x "${_cand}" ]]; then
                CUOBJDUMP="${_cand}"
                break
            fi
        done
    fi
fi

if [[ -x "${CUOBJDUMP:-}" ]] || command -v "${CUOBJDUMP:-}" >/dev/null 2>&1; then
    HMMA_COUNT=$("${CUOBJDUMP}" --dump-sass conv_tc | grep -c HMMA || true)
    echo "Instrucciones HMMA detectadas en binario: ${HMMA_COUNT} (cuobjdump: ${CUOBJDUMP})"
    if [[ "${RUN_NCU}" == "1" && "${HMMA_COUNT}" -eq 0 ]]; then
        echo "ERROR: binario sin instrucciones HMMA; revisar condiciones" \
             "de divisibilidad M/N/K antes de perfilar." >&2
        exit 1
    fi
else
    echo "AVISO: no se encontro cuobjdump; se omite la validacion HMMA." \
         "Exporta CUOBJDUMP=/ruta/a/cuobjdump para habilitarla." >&2
fi

echo "Ejecutando benchmarks de convolucion 2D con Tensor Cores..."

# Ruta 5 (CUTLASS): un solo flag adicional, valido para las mismas formas
# que TC_FORMAT (fp16/bf16/both) -- ver conv_tensor_activation.cu y
# Fase_2/Convolution/README.md, seccion "Layout NHWC, no NCHW".
declare -a CUTLASS_RUN_FLAGS=()
if [[ "${RUN_CUTLASS}" == "1" ]]; then
    CUTLASS_RUN_FLAGS+=("--cutlass")
fi

# La ruta 2 (cuDNN "clasico") corre con CUDNN_FMA_MATH desde el codigo, de
# modo que el speedup TC se mide contra FP32 escalar y no contra TF32, que
# es lo que cuDNN elige por defecto en Ampere (ver Fase_2/Convolution/README.md).
for HW in ${HW_LIST}; do
    echo
    echo "============================================================"
    echo "Corrida: N=${N}, C=${C}, H=W=${HW}, K=${K}, R=${R}, S=${S}, ITERS=${ITERS}"
    echo "Padding: (${PAD_H}, ${PAD_W}), stride: (${STRIDE_H}, ${STRIDE_W}), dilation: (${DILATION_H}, ${DILATION_W})"
    echo "Formato Tensor Core: ${TC_FORMAT}"
    echo "============================================================"
    ./conv_tc \
        --N "${N}" --C "${C}" --H "${HW}" --W "${HW}" \
        --K "${K}" --R "${R}" --S "${S}" \
        --pad_h "${PAD_H}" --pad_w "${PAD_W}" \
        --stride_h "${STRIDE_H}" --stride_w "${STRIDE_W}" \
        --dilation_h "${DILATION_H}" --dilation_w "${DILATION_W}" \
        --iters "${ITERS}" \
        --tc-format "${TC_FORMAT}" \
        "${CUTLASS_RUN_FLAGS[@]}"
done

if [[ "${RUN_NCU}" == "1" ]]; then
    read -r _first_hw _ <<< "${HW_LIST}"
    NCU_HW="${NCU_HW:-${_first_hw}}"
    REPORT="${OUTPUT_DIR}/ncu_wmma_conv_N${N}_C${C}_H${NCU_HW}_W${NCU_HW}_K${K}_R${R}_S${S}_job${SLURM_JOB_ID:-manual}"
    # Con batch N>1 el benchmark lanza N invocaciones del kernel WMMA por
    # iteracion, por lo que hay que saltar WARMUP_ITERS*N lanzamientos para
    # no perfilar los warmups. WARMUP_ITERS debe coincidir con kWarmupIters
    # en conv_tensor_activation.cu -- si cambias uno, cambia el otro; no hay
    # forma de derivarlo automaticamente del binario sin ejecutarlo primero.
    WARMUP_ITERS="${WARMUP_ITERS:-3}"
    NCU_LAUNCH_SKIP_RUN=$((WARMUP_ITERS * N))
    if [[ "${NCU_MODE}" == "full" ]]; then
        NCU_CAPTURE_ARGS=(--set "${NCU_SET}")
    else
        NCU_CAPTURE_ARGS=(--metrics "${NCU_QUICK_METRICS}")
    fi
    echo
    echo "Perfilando WMMA con Nsight Compute (NCU_MODE=${NCU_MODE}, launch-skip=${NCU_LAUNCH_SKIP_RUN})..."
    echo "Reporte: ${REPORT}.ncu-rep"
    # `time` (y stderr de ncu) se redirigen al .out para correlacionar
    # duracion vs #SBATCH --time en corridas con --set full.
    { time "${NCU}" "${NCU_CAPTURE_ARGS[@]}" \
        --target-processes all \
        --kernel-name "regex:${NCU_KERNEL_REGEX}" \
        --launch-skip "${NCU_LAUNCH_SKIP_RUN}" \
        --launch-count "${NCU_LAUNCH_COUNT}" \
        --export "${REPORT}" \
        --force-overwrite \
        ./conv_tc \
            --N "${N}" --C "${C}" --H "${NCU_HW}" --W "${NCU_HW}" \
            --K "${K}" --R "${R}" --S "${S}" \
            --pad_h "${PAD_H}" --pad_w "${PAD_W}" \
            --stride_h "${STRIDE_H}" --stride_w "${STRIDE_W}" \
            --dilation_h "${DILATION_H}" --dilation_w "${DILATION_W}" \
            --iters "${ITERS}" \
            --tc-format "${TC_FORMAT}" \
            "${CUTLASS_RUN_FLAGS[@]}"; } 2>&1
fi
