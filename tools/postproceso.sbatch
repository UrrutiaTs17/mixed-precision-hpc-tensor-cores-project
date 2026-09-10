#!/bin/bash
#SBATCH --job-name=postproceso_stats_pareto
#SBATCH --output=logs/postproceso_%j.out
#SBATCH --error=logs/postproceso_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Post-proceso final: estadistica inferencial (Etapa 7) y Frente de Pareto 3D
# (Etapa 9) sobre TODOS los results/ de Fase 3 y Fase 4 de los tres kernels.
#
# NO PIDE GPU a proposito: no hay ni una linea de CUDA aqui, solo pandas/
# statsmodels/matplotlib sobre los CSV ya extraidos. Ocupar una GPU de una
# particion compartida para correr un ANOVA seria desperdiciarla durante la
# hora larga que puede tardar el Pareto. Por eso este archivo NO lleva
# `#SBATCH --gres=gpu:1` ni `--partition`: la particion la elige quien lo
# envia (run_full_pipeline_pacca.sh lo hace con POST_PARTITION), y sin ese
# flag cae en la particion por defecto del cluster.
#
# Se envia normalmente como el ultimo eslabon de una cadena de dependencias:
#   sbatch --dependency=afterok:J1:J2:...:J6 tools/postproceso.sbatch
# pero tambien corre solo, sobre lo que ya haya en results/:
#   bash tools/postproceso.sbatch

set -euo pipefail

# Ver la nota extensa en tools/validacion_preliminar.sbatch: SLURM copia
# este script a un spool del nodo y ejecuta la copia, asi que BASH_SOURCE[0]
# no sirve para ubicar el repo bajo SLURM. SLURM_SUBMIT_DIR si apunta al
# directorio real de envio; BASH_SOURCE solo se usa como fallback fuera de
# SLURM (`bash tools/postproceso.sbatch` directo).
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

STATS_OUTDIR="${STATS_OUTDIR:-${REPO_ROOT}/stats_out}"
PARETO_OUTDIR="${PARETO_OUTDIR:-${REPO_ROOT}/pareto_out}"
RUN_STATS="${RUN_STATS:-1}"
RUN_PARETO="${RUN_PARETO:-1}"

# Los patrones que no matchean ningun archivo se ignoran solos (ver
# _expand_globs en Fase_4/tools/common_analysis.py): si una fase no llego a
# correr, sus patrones simplemente no aportan filas, sin error.
STENCIL_SUMMARY=(Fase_3/Stencil/results/summary_stencil_*.csv Fase_4/Stencil/results/summary_stencil_*.csv)
STENCIL_ENERGY=(Fase_3/Stencil/results/energy_stencil_*.csv Fase_4/Stencil/results/energy_stencil_*.csv)
GEMM_SUMMARY=(Fase_3/GEMM/results/summary_gemm_*.csv Fase_4/GEMM/results/summary_gemm_*.csv)
GEMM_DRIFT=(Fase_3/GEMM/results/drift_gemm_*.csv Fase_4/GEMM/results/drift_gemm_*.csv)
CONV_SUMMARY=(Fase_3/Convolution/results/summary_conv_*.csv Fase_4/Convolution/results/summary_conv_*.csv)
CONV_DRIFT=(Fase_3/Convolution/results/drift_conv_*.csv Fase_4/Convolution/results/drift_conv_*.csv)

echo "Nodo: $(hostname)"
echo "Fecha: $(date)"
echo "Repositorio: ${REPO_ROOT}"

if [[ "${RUN_STATS}" == "1" ]]; then
    echo
    echo "################################################################"
    echo "# Estadistica inferencial (Etapa 7)"
    echo "################################################################"
    mkdir -p "${STATS_OUTDIR}"
    python3 Fase_4/tools/run_statistics.py \
        --stencil-summary "${STENCIL_SUMMARY[@]}" --stencil-energy "${STENCIL_ENERGY[@]}" \
        --gemm-summary "${GEMM_SUMMARY[@]}" --gemm-drift "${GEMM_DRIFT[@]}" \
        --conv-summary "${CONV_SUMMARY[@]}" --conv-drift "${CONV_DRIFT[@]}" \
        --outdir "${STATS_OUTDIR}"
fi

if [[ "${RUN_PARETO}" == "1" ]]; then
    echo
    echo "################################################################"
    echo "# Frente de Pareto 3D (Etapa 9)"
    echo "################################################################"
    mkdir -p "${PARETO_OUTDIR}"
    python3 Fase_4/tools/pareto_front.py \
        --stencil-summary "${STENCIL_SUMMARY[@]}" --stencil-energy "${STENCIL_ENERGY[@]}" \
        --gemm-summary "${GEMM_SUMMARY[@]}" --gemm-drift "${GEMM_DRIFT[@]}" \
        --conv-summary "${CONV_SUMMARY[@]}" --conv-drift "${CONV_DRIFT[@]}" \
        --outdir "${PARETO_OUTDIR}"
fi

echo
echo "Post-proceso completo. Estadistica en ${STATS_OUTDIR}, Pareto en ${PARETO_OUTDIR}."
