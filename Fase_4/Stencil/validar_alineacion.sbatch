#!/bin/bash
#SBATCH --job-name=f4_valida_align
#SBATCH --output=logs/f4_valida_align_%j.out
#SBATCH --error=logs/f4_valida_align_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00

# ===========================================================================
# ETAPA 3 de la alineacion de buffers: IGUALDAD BIT A BIT contra la revision
# anterior.
#
# El cambio bajo prueba (padding de cudaMalloc + punteros *_aligned en
# benchmark_gpu_tensor_core_stencil) es un desplazamiento PURO de direcciones:
# no anade, quita ni reordena ninguna operacion de punto flotante, no cambia
# casts ni cuantizacion y no toca los kernels. Por tanto la exigencia NO es
# "error dentro de tolerancia" sino IDENTIDAD BYTE A BYTE de los campos
# archivados. Cualquier diferencia es un puntero mal sustituido, no redondeo.
#
# Por que se comparan campos archivados y no los logs: los logs llevan tiempos,
# que varian entre corridas. Los .bin de --archive-iters son el estado numerico
# crudo y son inmunes al ruido del cronometro.
#
# BASE_COMMIT debe ser la revision ANTERIOR al commit de alineacion.
#
# Uso:
#   sbatch validar_alineacion.sbatch
#   sbatch --export=ALL,BASE_COMMIT=26e35ab validar_alineacion.sbatch
# ===========================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

REPO_ROOT="$(git rev-parse --show-toplevel)"
CUDA_ARCH="${CUDA_ARCH:-80}"
NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"
BASE_COMMIT="${BASE_COMMIT:-HEAD~1}"

JOB_ID="${SLURM_JOB_ID:-manual}"
WORK="valida_align_${JOB_ID}"
mkdir -p "${WORK}/salidas"

echo "Nodo        : $(hostname)"
echo "Fecha       : $(date -Iseconds)"
echo "Commit NUEVO: $(git -C "${REPO_ROOT}" rev-parse HEAD)"
echo "Commit BASE : $(git -C "${REPO_ROOT}" rev-parse "${BASE_COMMIT}")"
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
# ETAPA 1 — Dos binarios: BASE y NUEVO
# ===========================================================================
echo "############################################################"
echo "# ETAPA 1: compilacion de las dos revisiones"
echo "############################################################"

# El .cu incluye ../../Fase_2/common.cuh y tools/power_sampling.h por ruta
# RELATIVA, asi que la revision base necesita un arbol completo: worktree, no
# un 'git show' a un fichero suelto.
BASE_TREE="${PWD}/${WORK}/arbol_base"
git -C "${REPO_ROOT}" worktree add --detach "${BASE_TREE}" "${BASE_COMMIT}"

BIN_BASE="${PWD}/${WORK}/stencil_base"
BIN_NEW="${PWD}/${WORK}/stencil_nuevo"

compilar() {   # $1 = raiz del arbol, $2 = binario de salida
    ( cd "$1/Fase_3/Stencil" && "${NVCC}" -std=c++17 stencil_tensor_activation.cu -o "$2" \
        -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
        -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
        --allow-unsupported-compiler "${NVML_FLAGS[@]}" )
}

compilar "${BASE_TREE}" "${BIN_BASE}"
echo "  BASE  compilado: $(sha256sum "${BIN_BASE}" | awk '{print $1}')"
compilar "${REPO_ROOT}" "${BIN_NEW}"
echo "  NUEVO compilado: $(sha256sum "${BIN_NEW}" | awk '{print $1}')"
echo "  (los sha256 DEBEN diferir: son revisiones distintas)"
echo

# ===========================================================================
# ETAPA 2 — Igualdad bit a bit de los campos archivados
# ===========================================================================
echo "############################################################"
echo "# ETAPA 2: igualdad bit a bit BASE vs NUEVO"
echo "############################################################"
echo "Mallas: 64 y 512 tienen nx multiplo de 8 -> el desplazamiento ALINEA."
echo "        63 y 511 no lo son -> el desplazamiento NO alinea, pero debe"
echo "        seguir siendo CORRECTO. Ese es justamente el caso que atrapa un"
echo "        error de aritmetica de punteros."
echo

S="${PWD}/${WORK}/salidas"
fallos=0
casos=0

correr_par() {   # $1 = etiqueta, $2.. = flags comunes a las dos revisiones
    local etiqueta="$1"; shift
    local dir_base="${S}/${etiqueta}/base"
    local dir_new="${S}/${etiqueta}/nuevo"
    mkdir -p "${dir_base}" "${dir_new}"
    casos=$((casos + 1))

    "${BIN_BASE}" "$@" --archive-dir "${dir_base}" \
        > "${S}/${etiqueta}/base.log" 2>&1 || { echo "  ERROR: BASE fallo en ${etiqueta}"; fallos=$((fallos+1)); return; }
    "${BIN_NEW}"  "$@" --archive-dir "${dir_new}"  \
        > "${S}/${etiqueta}/nuevo.log" 2>&1 || { echo "  ERROR: NUEVO fallo en ${etiqueta}"; fallos=$((fallos+1)); return; }

    # sha256 de cada campo archivado, ordenado por nombre de fichero.
    ( cd "${dir_base}" && sha256sum *.bin | sort -k2 ) > "${S}/${etiqueta}/base.sha"
    ( cd "${dir_new}"  && sha256sum *.bin | sort -k2 ) > "${S}/${etiqueta}/nuevo.sha"

    local n_base; n_base=$(wc -l < "${S}/${etiqueta}/base.sha")
    if [[ "${n_base}" -eq 0 ]]; then
        echo "  FALLO ${etiqueta}: no se archivo ningun campo (revisa --checkpoint-iters)"
        fallos=$((fallos + 1)); return
    fi

    if diff -q "${S}/${etiqueta}/base.sha" "${S}/${etiqueta}/nuevo.sha" > /dev/null; then
        echo "  OK    ${etiqueta}  (${n_base} campos identicos byte a byte)"
    else
        echo "  FALLO ${etiqueta}: los campos DIFIEREN"
        diff "${S}/${etiqueta}/base.sha" "${S}/${etiqueta}/nuevo.sha" | sed 's/^/        /'
        fallos=$((fallos + 1))
    fi
}

COMUN=(--iters 20 --tc both --fp64-gpu off --cpu-fp64 off
       --checkpoint-iters 10,20 --archive-iters 10,20)

for NX in 64 512 63 511; do
    correr_par "nx${NX}_off"     --nx "${NX}" --ny "${NX}" "${COMUN[@]}"
    correr_par "nx${NX}_local"   --nx "${NX}" --ny "${NX}" "${COMUN[@]}" --kahan on
    correr_par "nx${NX}_spatial" --nx "${NX}" --ny "${NX}" "${COMUN[@]}" --spatial-comp on
done

# Operador difusivo + CI monomodo: la configuracion real de la campana de Fase 4.
for NX in 64 512; do
    correr_par "nx${NX}_dif_spatial" --nx "${NX}" --ny "${NX}" "${COMUN[@]}" \
        --spatial-comp on --op-mode diffusive --alpha 0.1875 --ci-mode monomode --ci-p 5
done

# CUDA Graphs (requisito 17): los grafos hornean punteros en sus nodos, asi que
# esta es la comprobacion de que la captura recibio los desplazados.
for NX in 64 512; do
    correr_par "nx${NX}_graph_off"     --nx "${NX}" --ny "${NX}" "${COMUN[@]}" \
        --execution-mode graph --graph-block 2
    correr_par "nx${NX}_graph_spatial" --nx "${NX}" --ny "${NX}" "${COMUN[@]}" \
        --spatial-comp on --execution-mode graph --graph-block 2
done

echo
# ===========================================================================
# ETAPA 3 — Veredicto
# ===========================================================================
echo "############################################################"
echo "# ETAPA 3: veredicto"
echo "############################################################"
echo "Casos ejecutados : ${casos}"
echo "Casos con FALLO  : ${fallos}"
git -C "${REPO_ROOT}" worktree remove --force "${BASE_TREE}" || true
if [[ "${fallos}" -eq 0 ]]; then
    echo
    echo "VEREDICTO: el desplazamiento de punteros NO altera un solo bit."
    echo "Autorizada la Etapa 4 (vectorizacion de los accesos globales)."
    exit 0
else
    echo
    echo "VEREDICTO: HAY DIFERENCIAS. No continuar con la vectorizacion."
    echo "Revisa ${WORK}/salidas/<caso>/{base,nuevo}.sha y los .log del caso."
    exit 1
fi
