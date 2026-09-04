#!/bin/bash
# run_full_pipeline.sh
#
# Orquestador de PRINCIPIO A FIN: compila y corre las cuatro fases (los tres
# kernels en cada una que aplique), y al terminar corre el post-proceso
# estadistico (Fase_4/tools/run_statistics.py, Etapa 7 del plan) y el Frente
# de Pareto 3D (Fase_4/tools/pareto_front.py, Etapa 9) sobre TODO lo que se
# haya generado. Pensado para correr en CUALQUIER maquina con GPU Ampere+ y
# el entorno conda de environment.yml activo -- ver REQUIREMENTS.md. No
# depende de SLURM: cada fase se invoca con `bash`, no `sbatch` (aunque
# corre exactamente igual si lo lanzas tu mismo con `sbatch` en PACCA -- ver
# la nota de cada .sbatch).
#
# Uso basico (defaults razonables, campana chica de humo):
#   bash run_full_pipeline.sh
#
# Saltar fases (por si ya corriste algunas, o para iterar rapido):
#   RUN_FASE1=0 RUN_FASE2=0 bash run_full_pipeline.sh
#
# Parametrizar una fase especifica: exporta las mismas variables que acepta
# su .sbatch ANTES de llamar a este script -- se propagan tal cual (este
# script no las redeclara ni las intercepta):
#   N_LIST="512 1024 2048" COMP_LIST="off on" ANCHOR_LIST="0 1 5 20" \
#       bash run_full_pipeline.sh
#
# Solo el post-proceso (si ya tienes results/ de una corrida anterior):
#   RUN_FASE1=0 RUN_FASE2=0 RUN_FASE3=0 RUN_FASE4=0 bash run_full_pipeline.sh
#
# SIN GPU: este script en si no compila nada directamente -- delega en los
# .sbatch de cada carpeta, que si necesitan GPU+nvcc para compilar y correr
# (ver REQUIREMENTS.md, "Requisito de hardware"). Si no hay GPU, cada fase
# falla en su propio paso de compilacion/ejecucion (no en este script) y el
# pipeline se detiene ahi (set -e) -- no hay una ruta "simulada" sin GPU.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

RUN_FASE1="${RUN_FASE1:-1}"
RUN_FASE2="${RUN_FASE2:-1}"
RUN_FASE3="${RUN_FASE3:-1}"
RUN_FASE4="${RUN_FASE4:-1}"
RUN_STATS="${RUN_STATS:-1}"
RUN_PARETO="${RUN_PARETO:-1}"
STATS_OUTDIR="${STATS_OUTDIR:-${REPO_ROOT}/stats_out}"
PARETO_OUTDIR="${PARETO_OUTDIR:-${REPO_ROOT}/pareto_out}"

PIPELINE_LOG="${REPO_ROOT}/run_full_pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "Log completo de esta corrida: ${PIPELINE_LOG}"

# Todo el cuerpo del pipeline vive en esta funcion, para poder envolver UNA
# sola vez toda la salida (stdout+stderr de las cuatro fases y del
# post-proceso) con `tee` hacia PIPELINE_LOG -- ver la llamada a `main` al
# final del archivo. `set -euo pipefail` ya esta activo desde arriba, asi
# que un fallo en cualquier fase interrumpe main() y, via pipefail, tambien
# el exit code de este script (no lo enmascara el `tee`).
main() {

# Corre un .sbatch de una carpeta como script de bash normal (sin SLURM) --
# ver tools/detect_toolchain.sh, que cada .sbatch source-ea, sobre por que
# esto es seguro fuera de un cluster. Si alguna fase falla, el pipeline
# entero se detiene aqui (set -e) -- no tiene sentido seguir con Fase 3 si
# Fase 1 no compilo, el toolchain esta roto para todas por igual.
run_phase() {
    local label="$1" dir="$2" script="$3"
    echo
    echo "################################################################"
    echo "# ${label}"
    echo "################################################################"
    ( cd "${REPO_ROOT}/${dir}" && bash "${script}" )
}

if [[ "${RUN_FASE1}" == "1" ]]; then
    run_phase "Fase 1 / GEMM"        Fase_1/GEMM        run_gemm_fase1.sbatch
    run_phase "Fase 1 / Convolucion" Fase_1/Convolution run_conv_fase1.sbatch
    run_phase "Fase 1 / Stencil"     Fase_1/Stencil     run_stencil_fase1.sbatch
else
    echo "RUN_FASE1=0 -- se omite Fase 1."
fi

if [[ "${RUN_FASE2}" == "1" ]]; then
    run_phase "Fase 2 / GEMM"        Fase_2/GEMM        run_gemm_tc.sbatch
    run_phase "Fase 2 / Convolucion" Fase_2/Convolution run_conv_tc.sbatch
    run_phase "Fase 2 / Stencil"     Fase_2/Stencil     run_stencil_tc.sbatch
else
    echo "RUN_FASE2=0 -- se omite Fase 2."
fi

if [[ "${RUN_FASE3}" == "1" ]]; then
    run_phase "Fase 3 / Stencil (tc)"      Fase_3/Stencil     run_stencil_tc.sbatch
    run_phase "Fase 3 / GEMM (chained)"    Fase_3/GEMM        run_gemm_chained.sbatch
    run_phase "Fase 3 / Convolucion (chained)" Fase_3/Convolution run_conv_chained.sbatch
else
    echo "RUN_FASE3=0 -- se omite Fase 3."
fi

if [[ "${RUN_FASE4}" == "1" ]]; then
    run_phase "Fase 4 / Stencil (ancla FP64)"      Fase_4/Stencil     run_stencil_tc.sbatch
    run_phase "Fase 4 / GEMM (ancla FP64)"         Fase_4/GEMM        run_gemm_chained.sbatch
    run_phase "Fase 4 / Convolucion (ancla FP64)"  Fase_4/Convolution run_conv_chained.sbatch
else
    echo "RUN_FASE4=0 -- se omite Fase 4."
fi

# --- Post-proceso: estadistica (Etapa 7) y Frente de Pareto (Etapa 9) ------
# Apunta a los results/ de Fase 3 Y Fase 4 (Fase 1/2 no tienen CSV_* que
# extraer todavia -- ver docs/MANUAL.md, seccion Fase 1). Los patrones que
# no matchean ningun archivo se ignoran solos (ver _expand_globs en
# common_analysis.py) -- si solo corriste Fase 4, por ejemplo, los patrones
# de Fase 3 simplemente no aportan filas, sin error.
if [[ "${RUN_STATS}" == "1" || "${RUN_PARETO}" == "1" ]]; then
    STENCIL_SUMMARY=(Fase_3/Stencil/results/summary_stencil_*.csv Fase_4/Stencil/results/summary_stencil_*.csv)
    STENCIL_ENERGY=(Fase_3/Stencil/results/energy_stencil_*.csv Fase_4/Stencil/results/energy_stencil_*.csv)
    GEMM_SUMMARY=(Fase_3/GEMM/results/summary_gemm_*.csv Fase_4/GEMM/results/summary_gemm_*.csv)
    GEMM_DRIFT=(Fase_3/GEMM/results/drift_gemm_*.csv Fase_4/GEMM/results/drift_gemm_*.csv)
    CONV_SUMMARY=(Fase_3/Convolution/results/summary_conv_*.csv Fase_4/Convolution/results/summary_conv_*.csv)
    CONV_DRIFT=(Fase_3/Convolution/results/drift_conv_*.csv Fase_4/Convolution/results/drift_conv_*.csv)

    if [[ "${RUN_STATS}" == "1" ]]; then
        echo
        echo "################################################################"
        echo "# Post-proceso: estadistica inferencial (Etapa 7)"
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
        echo "# Post-proceso: Frente de Pareto 3D (Etapa 9)"
        echo "################################################################"
        mkdir -p "${PARETO_OUTDIR}"
        python3 Fase_4/tools/pareto_front.py \
            --stencil-summary "${STENCIL_SUMMARY[@]}" --stencil-energy "${STENCIL_ENERGY[@]}" \
            --gemm-summary "${GEMM_SUMMARY[@]}" --gemm-drift "${GEMM_DRIFT[@]}" \
            --conv-summary "${CONV_SUMMARY[@]}" --conv-drift "${CONV_DRIFT[@]}" \
            --outdir "${PARETO_OUTDIR}"
    fi
fi

echo
echo "Pipeline completo. Resultados de estadistica en ${STATS_OUTDIR}, Pareto en ${PARETO_OUTDIR}."

}  # fin de main()

main 2>&1 | tee "${PIPELINE_LOG}"
