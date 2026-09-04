#!/bin/bash
# tools/detect_toolchain.sh
#
# Deteccion PORTABLE del toolchain CUDA (nvcc, ncu) y de la arquitectura de
# GPU, para que cada .sbatch de este proyecto corra tanto bajo SLURM en
# PACCA (via `sbatch archivo.sbatch`) como en CUALQUIER OTRA maquina con GPU
# Ampere o mas nueva (compute capability >= 8.0 -- ver la nota sobre
# cp.async en REQUIREMENTS.md, es un requisito real de hardware, no solo de
# software) y el entorno conda de environment.yml activo, corriendo
# `bash archivo.sbatch` directamente: un .sbatch es un script bash normal, y
# las lineas `#SBATCH` son simples comentarios fuera de SLURM. No hace falta
# copiar ni modificar nada mas para correr fuera de PACCA -- basta con que
# `nvcc` este en el PATH (environment.yml lo garantiza).
#
# Se source-ea, no se ejecuta: define detect_nvcc/detect_ncu/detect_cuda_arch
# como funciones y no toca ninguna variable por su cuenta -- cada .sbatch
# decide cuando llamarlas y en que variable guardar el resultado.

# --- nvcc --------------------------------------------------------------
# Orden de resolucion:
#   1. $NVCC ya exportado (maxima prioridad -- override explicito).
#   2. `nvcc` en el PATH (cubre el entorno conda de environment.yml y
#      cualquier instalacion manual del CUDA Toolkit).
#   3. Ruta historica de PACCA (compatibilidad retroactiva, sigue
#      funcionando en el cluster original sin exportar nada).
# Si ninguna existe, aborta con un mensaje claro -- sin nvcc no hay nada que
# compilar, mejor fallar aqui que 200 lineas mas abajo con un error críptico.
detect_nvcc() {
    if [[ -n "${NVCC:-}" ]]; then
        echo "${NVCC}"
        return 0
    fi
    if command -v nvcc >/dev/null 2>&1; then
        command -v nvcc
        return 0
    fi
    local pacca_default="/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc"
    if [[ -x "${pacca_default}" ]]; then
        echo "${pacca_default}"
        return 0
    fi
    echo "ERROR: no se encontro nvcc (se probo \$NVCC, el PATH, y la ruta" >&2
    echo "       historica de PACCA). Instala el CUDA Toolkit -- ver" >&2
    echo "       REQUIREMENTS.md y environment.yml en la raiz del repo -- o" >&2
    echo "       exporta NVCC=/ruta/a/nvcc antes de lanzar este script." >&2
    exit 1
}

# --- Nsight Compute (ncu) -- OPCIONAL ------------------------------------
# Mismo orden que nvcc, pero NUNCA aborta: el perfilado nunca es obligatorio
# para producir el CSV de una campaña. Devuelve la ruta por stdout y 0 si la
# encuentra; si no, devuelve 1 y no imprime nada -- el llamador decide como
# avisar (ver el patron ya usado en cada .sbatch: RUN_NCU se fuerza a 0).
detect_ncu() {
    if [[ -n "${NCU:-}" ]]; then
        echo "${NCU}"
        return 0
    fi
    if command -v ncu >/dev/null 2>&1; then
        command -v ncu
        return 0
    fi
    local pacca_default="${HOME}/fcrojasv/nsight_compute/ncu"
    if [[ -x "${pacca_default}" ]]; then
        echo "${pacca_default}"
        return 0
    fi
    return 1
}

# --- Arquitectura CUDA objetivo ------------------------------------------
# Si $CUDA_ARCH ya viene fijado (por el usuario o por --export=ALL en
# SLURM), se respeta tal cual. Si no, se consulta la capacidad de computo
# REAL de la GPU 0 via nvidia-smi -- evita que alguien con una GPU Ada
# (sm_89) o Hopper (sm_90) siga compilando para sm_80 solo porque ese era el
# default historico de PACCA (A100). Si nvidia-smi no esta disponible (poco
# probable si hay una GPU NVIDIA visible, pero posible en un entorno de
# compilacion sin acceso al dispositivo), cae al default historico.
detect_cuda_arch() {
    if [[ -n "${CUDA_ARCH:-}" ]]; then
        echo "${CUDA_ARCH}"
        return 0
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local cc
        cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d '[:space:]')"
        if [[ "${cc}" =~ ^[0-9]+\.[0-9]+$ ]]; then
            echo "${cc}" | tr -d '.'
            return 0
        fi
    fi
    echo "80"  # A100, default historico de PACCA
}
