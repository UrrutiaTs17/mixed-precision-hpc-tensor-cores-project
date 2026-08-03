#!/bin/bash
#SBATCH --job-name=mixed_precision_stencil_horizon_f3
#SBATCH --output=logs/mixed_precision_stencil_horizon_f3_%j.out
#SBATCH --error=logs/mixed_precision_stencil_horizon_f3_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --exclusive
# Subido de 01:00:00 a 02:00:00 (bloque 5): la corrida ahora se duplica
# (--kahan off y --kahan on, ver KAHAN_LIST mas abajo); Kahan no toca CPU
# FP32/FP64 (las referencias, el costo dominante de esta corrida) asi que la
# segunda pasada las recalcula de forma redundante -- el tiempo total sube
# aprox. al doble del de una sola pasada. Sin datos de wall-time medidos de
# esta corrida en PACCA para calibrar con precision, 02:00:00 da margen frente
# a duplicar un limite que ya se asumia ajustado a una sola pasada.
#SBATCH --time=02:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f stencil_tensor_activation.cu && -f Stencil/stencil_tensor_activation.cu ]]; then
    cd Stencil
fi
mkdir -p logs results
JOB_ID="${SLURM_JOB_ID:-manual}"
RESULTS_LOG="results/run_${JOB_ID}.log"

# Job separado del barrido de metricas (run_stencil_tc.sbatch): ese barrido
# corre a 16384^2 dentro de la ventana finita (ITERS_LIST <= 120, todas las
# metricas L2/Linf/etc. comparables). Este job mide el horizonte de overflow
# REAL por formato (n*, impreso via "Horizonte de overflow (predicho /
# medido)" en stencil_tensor_activation.cu) a 4096^2 con iters=1200,
# suficiente para capturar incluso FP64 (prediccion n* ~1038 dada la
# componente Nyquist de la condicion inicial). A 16384^2 el mismo barrido
# costaria ~1038 x 7.4s de CPU FP64 encadenada (inviable); a 4096^2 (1/16 de
# los puntos) ese costo cabe holgado en la ventana de este job.
# No corre ncu: el objetivo es el texto de n* por stdout (primera iteracion
# no finita de cada ruta), no perfilar Tensor Cores -- para eso esta
# run_stencil_tc.sbatch.

# Arquitectura por defecto para A100.
# Puedes sobrescribirla con:
#   sbatch --export=ALL,CUDA_ARCH=86 run_stencil_horizon.sbatch
CUDA_ARCH="${CUDA_ARCH:-80}"  # A100=80, RTX 3050=86, V100=70

NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"

# NVML es opcional: cuando nvml.h y la biblioteca del driver estan visibles,
# se habilita la telemetria y se enlaza con libnvidia-ml. Si PACCA no expone
# alguno de los dos, el mismo fuente compila con el stub sin telemetria.
NVML_FLAGS=()
# El driver NVIDIA suele instalarse como -lnvidia-ml; PACCA puede sobreescribir
# este valor con -lnvml si su modulo ofrece ese alias.
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

# Dimensiones y horizonte del barrido: NX/NY 16x mas chicos que el barrido de
# metricas para que la referencia CPU FP64 encadenada sea viable a
# ITERS=1200; TC=both corre FP16 y BF16 en la misma invocacion.
NX="${NX:-4096}"
NY="${NY:-4096}"
ITERS="${ITERS:-1200}"
TC="${TC:-both}"

# --checkpoint-every 20: unico modo de que storage_rel sea evaluable en esta
# malla (sin checkpoints, si la ruta diverge no queda estado finito
# recuperable, ver stencil_tensor_activation.cu). FP16 diverge en ~28, asi
# que un checkpoint en la iteracion 20 lo captura con margen.
#
# Presupuesto de memoria de fp64_checkpoints (el unico consumo que escala con
# ITERS/CHECKPOINT_EVERY): cada snapshot es NX*NY doubles = 4096*4096*8 =
# 134217728 bytes (~134 MB / 128 MiB). La referencia FP64 se mantiene finita
# hasta iter ~1045 (n* FP64 predicho 1040.6, ver bloque 3 de la correccion de
# 2026-07-25), asi que con checkpoint_every=20 los snapshots validos van de
# 20 a 1040 => 1040/20 = 52 checkpoints => 52 * 134217728 ~= 6.98 GB (~6.5
# GiB). El job pide --mem=32G: 52 checkpoints caben con margen (~22% del
# presupuesto), dejando espacio de sobra para los buffers O(NX*NY) de cada
# ruta (unas pocas decenas de MB cada uno, no escalan con ITERS).
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-20}"

# Diseno factorial {fp16,bf16} x {kahan off,on} (bloque 5): esta es la corrida
# donde SI importa si Kahan extiende n* de FP16 mas alla de 28 (pregunta
# explicita del bloque). --kahan off primero preserva el orden de no-regresion
# de las cifras ya validadas (bloque de gates).
KAHAN_LIST="${KAHAN_LIST:-off on}"

# Tercera politica de compensacion, alternativa a --kahan off|on: error
# feedback ESPACIAL (ver CompMode en stencil_tensor_activation.cu). Mismo
# mecanismo y mismo default que en run_stencil_tc.sbatch; "off" no cambia una
# sola operacion respecto de las corridas ya validadas. Se expone tambien aqui
# porque esta es la corrida donde la pregunta tiene sentido: si la compensacion
# espacial mueve o no el horizonte n* (FP16~28, BF16~138) hace falta ITERS
# grande, que es justo lo que este barrido hace y el de metricas no.
# SPATIAL_COMP=on fuerza KAHAN_LIST=off: el binario rechaza la combinacion y
# con `set -e` eso mataria el job.
SPATIAL_COMP="${SPATIAL_COMP:-off}"
if [[ "${SPATIAL_COMP}" == "on" ]]; then
    KAHAN_LIST="off"
fi

echo "Nodo: $(hostname)"
echo "Fecha: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-no definido}"
echo "Arquitectura CUDA: sm_${CUDA_ARCH}"
echo "NVCC=${NVCC}"

nvidia-smi || true

echo "Compilando stencil_tc (Fase_3/Stencil, incluye Fase_2/common.cuh por ruta relativa)..."
"${NVCC}" -std=c++17 stencil_tensor_activation.cu -o stencil_tc \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
    --allow-unsupported-compiler "${NVML_FLAGS[@]}"

TIMEFORMAT=">>> Tiempo real de la corrida: %R s (user %U, sys %S)"
# Una sola invocacion de tee envuelve TODO el bucle de KAHAN_LIST, en vez de
# reabrirse por cada pasada: mismo fix aplicado en run_stencil_tc.sbatch (ver
# comentario ahi) para que ninguna pasada dependa de que su propia invocacion
# de tee sobreviva intacta.
{
for KAHAN in ${KAHAN_LIST}; do
    echo
    echo "============================================================"
    echo "Barrido de horizonte: NX=${NX}, NY=${NY}, ITERS=${ITERS}, TC=${TC}, KAHAN=${KAHAN}"
    echo "============================================================"
    time ./stencil_tc --nx "${NX}" --ny "${NY}" --iters "${ITERS}" --tc "${TC}" \
          --checkpoint-every "${CHECKPOINT_EVERY}" --kahan "${KAHAN}" \
          --spatial-comp "${SPATIAL_COMP}"
done
} 2>&1 | tee -a "${RESULTS_LOG}"

echo
echo "Extrayendo CSVs desde ${RESULTS_LOG}..."
python3 tools/extract_csv.py \
    --input  "${RESULTS_LOG}" \
    --outdir "results" \
    --job-id "${JOB_ID}" \
    --kernel stencil

echo "CSVs generados:"
for csv_path in \
    "results/drift_stencil_${JOB_ID}.csv" \
    "results/summary_stencil_${JOB_ID}.csv" \
    "results/horizon_stencil_${JOB_ID}.csv" \
    "results/store_stencil_${JOB_ID}.csv" \
    "results/energy_stencil_${JOB_ID}.csv"; do
    rows=$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' "${csv_path}")
    echo "  ${csv_path}: ${rows} filas"
done
