#!/bin/bash
#SBATCH --job-name=postproceso_variabilidad
#SBATCH --output=logs/postproceso_variabilidad_%j.out
#SBATCH --error=logs/postproceso_variabilidad_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Estadistica inferencial (Etapa 7) SOLO sobre la campana de variabilidad
# (replicas), separada del post-proceso principal (tools/postproceso.sbatch,
# que apunta a Fase_3/Fase_4/*/results/ -- el barrido de un solo punto por
# celda, sin replicas).
#
# POR QUE UN SCRIPT APARTE Y NO REUTILIZAR postproceso.sbatch
# -------------------------------------------------------------
# Los globs de resultados estan hardcodeados ahi, apuntando a results/ (la
# campana principal). Mezclar esos CSV (1 replica por celda, muchos tamanos)
# con los de variabilidad (pocos tamanos, muchas replicas) en el mismo ANOVA
# complicaria la lectura sin aportar nada -- run_statistics.py ya avisa por
# celda si faltan replicas (Dataset.warn_low_replicas), asi que lo util es
# poder correr el ANOVA SOLO sobre el dataset diseñado para tener replicas,
# y verificar ahi mismo que el aviso desaparecio.
#
# NO corre pareto_front.py: el Frente de Pareto quiere DIVERSIDAD de puntos
# (tamanos/formatos distintos), no replicas del mismo punto -- no tiene
# sentido sobre este dataset. Si hace falta, RUN_PARETO=1 lo activa igual.
#
# Se envia normalmente encadenado a la cola de replicas que arma
# tools/lanzar_campana_variabilidad.sh (que ya pasa --dependency=afterok con
# todos los job id de replicas), pero corre solo igual:
#   bash tools/postproceso_variabilidad.sbatch

set -euo pipefail

# Ver la nota extensa en tools/validacion_preliminar.sbatch sobre por que
# SLURM_SUBMIT_DIR y no BASH_SOURCE.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

# Mismo directorio relativo que usa lanzar_campana_variabilidad.sh al fijar
# RESULTS_DIR en cada replica -- si se cambia alli, cambiar tambien aqui.
VARDIR="${VARDIR:-results/variabilidad}"
STATS_OUTDIR="${STATS_OUTDIR:-${REPO_ROOT}/stats_out_variabilidad}"
PARETO_OUTDIR="${PARETO_OUTDIR:-${REPO_ROOT}/pareto_out_variabilidad}"
RUN_STATS="${RUN_STATS:-1}"
RUN_PARETO="${RUN_PARETO:-0}"

STENCIL_SUMMARY=(Fase_4/Stencil/${VARDIR}/summary_stencil_*.csv)
STENCIL_ENERGY=(Fase_4/Stencil/${VARDIR}/energy_stencil_*.csv)
GEMM_SUMMARY=(Fase_4/GEMM/${VARDIR}/summary_gemm_*.csv)
GEMM_DRIFT=(Fase_4/GEMM/${VARDIR}/drift_gemm_*.csv)
CONV_SUMMARY=(Fase_4/Convolution/${VARDIR}/summary_conv_*.csv)
CONV_DRIFT=(Fase_4/Convolution/${VARDIR}/drift_conv_*.csv)

echo "Nodo: $(hostname)"
echo "Fecha: $(date)"
echo "Repositorio: ${REPO_ROOT}"
echo "Directorio de variabilidad: ${VARDIR}"

if [[ "${RUN_STATS}" == "1" ]]; then
    echo
    echo "################################################################"
    echo "# Estadistica inferencial sobre la campana de VARIABILIDAD (Etapa 7)"
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
    echo "# Frente de Pareto 3D sobre variabilidad (opt-in, RUN_PARETO=1)"
    echo "################################################################"
    mkdir -p "${PARETO_OUTDIR}"
    python3 Fase_4/tools/pareto_front.py \
        --stencil-summary "${STENCIL_SUMMARY[@]}" --stencil-energy "${STENCIL_ENERGY[@]}" \
        --gemm-summary "${GEMM_SUMMARY[@]}" --gemm-drift "${GEMM_DRIFT[@]}" \
        --conv-summary "${CONV_SUMMARY[@]}" --conv-drift "${CONV_DRIFT[@]}" \
        --outdir "${PARETO_OUTDIR}"
fi

echo
echo "Post-proceso de variabilidad completo. Estadistica en ${STATS_OUTDIR}."
echo "Revisa el .err de este job: si SIGUE apareciendo el aviso de 'menos de 2"
echo "replicas', REPLICAS (o los *_LIST fijados) en lanzar_campana_variabilidad.sh"
echo "no alcanzaron -- subir REPLICAS y volver a lanzar, no reinterpretar el ANOVA."
