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
# AVISO -- el default NO es una prueba de humo:
#   bash run_full_pipeline.sh
# corre la CAMPANA COMPLETA de las cuatro fases. Cada .sbatch usa su propio
# default (SMOKE_TEST=0), que es el barrido entero de tamanos, iteraciones y
# valores de K, con RUN_NCU=1 en Fase 2 y en Fase 3/4 de Stencil. Son decenas
# de horas de GPU sumadas.
#
# (Este comentario decia "campana chica de humo". Era falso: SMOKE_TEST nunca
# se exportaba desde aqui, asi que cada .sbatch caia en su propio default de
# campana completa. Se corrigieron LOS DOS lados -- el texto, que ahora
# describe lo que pasa de verdad, y el comportamiento, que ahora SI tiene una
# via de humo explicita. El default sigue siendo `full` para no cambiarle el
# significado a nadie que ya lo estuviera usando en serio.)
#
# Prueba de humo real (minutos, no horas -- exporta SMOKE_TEST=1 y RUN_NCU=0 a
# las cuatro fases):
#   PIPELINE_MODE=smoke bash run_full_pipeline.sh
#
# Antes de una campana de verdad, la puerta previa de la Tarea 8 corre las
# verificaciones baratas (orden de operandos, humo, gates K=0/K=1):
#   bash tools/validacion_preliminar.sbatch
#
# En un cluster con SLURM, este script NO es el camino recomendado: exige la
# GPU reservada de principio a fin en una sola sesion. Use
# run_full_pipeline_pacca.sh, que envia cada fase como un job con
# dependencias.
#
# Saltar fases (por si ya corriste algunas, o para iterar rapido):
#   RUN_FASE1=0 RUN_FASE2=0 bash run_full_pipeline.sh
#
# Cada Fase 3/Fase 4 corre DOS VECES por kernel si se activa RUN_ENERGY_PASS=1
# (opt-in, default 0):
# la pasada normal y una pasada SOLO de energia (RUN_KIND=energy, ITERS_LIST
# grande) -- sin esto, energy_window_reliable sale en 0 casi siempre (ver la
# nota de cabecera junto a RUN_ENERGY_PASS mas abajo). Para desactivarla:
#   RUN_ENERGY_PASS=0 bash run_full_pipeline.sh
#
# La campana de VARIABILIDAD (replicas para el ANOVA, Etapa 7) NO corre desde
# aqui -- requiere SLURM por diseno (cada replica es un job independiente).
# Usar run_full_pipeline_pacca.sh en un cluster, o tools/lanzar_campana_
# variabilidad.sh directo.
#
# Parametrizar una fase especifica: exporta las mismas variables que acepta
# su .sbatch ANTES de llamar a este script -- se propagan tal cual (este
# script no las redeclara ni las intercepta):
#   N_LIST="1024 2048 4096 8192" COMP_LIST="off on" ANCHOR_LIST="0 1 5 20" \
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

# --- Pase de energia (RUN_KIND=energy), Fase 3/4 de los 3 kernels ------------
# Mismo mecanismo y mismos defaults que run_full_pipeline_pacca.sh -- ver la
# nota de cabecera de ese script para el porque completo, la verificacion con
# datos reales (gpu_segments=2 en el 100% de las filas de GEMM/Conv, umbral
# 1000 ms) y por que los ITERS_LIST se calcularon con el MINIMO t_iter_ms
# observado (reloj de GPU en estado estable) y no con la mediana -- la
# mediana subestima el caso real, el mismo error que dejo corta la tanda D
# de Stencil. En resumen: una sola pasada con los ITERS_LIST default (20-80
# en GEMM/Conv) SIEMPRE da energy_window_reliable=0 en casi todas las filas
# (confirmado en produccion: jobs 6925/6926/6927/6928/6866/6890, 96-276 de
# 96-276 filas GPU cada uno). Causa en dos capas: (1) CHECKPOINT_EVERY>0
# fragmenta la ventana NVML en tramos -- RUN_KIND=energy fuerza
# CHECKPOINT_EVERY=0 y RUN_NCU=0; (2) con CHECKPOINT_EVERY=0, TODAS las rutas
# de GEMM/Conv (no solo _comp) igual cierran 2 tramos -- umbral real 1000 ms,
# no 500. Los ITERS_LIST default no se acercan. La correccion es una SEGUNDA
# pasada solo de energia, con ITERS_LIST grande (dimensionado con la ruta MAS
# RAPIDA de cada barrido, que es la que manda porque todas las rutas de una
# invocacion comparten --iters) y sin checkpoints intermedios; el job vive en
# el MISMO results/ que la pasada numerica (job_id/PID nuevo, no se pisan) y
# el post-proceso los toma a ambos como replicas del mismo tamano/formato.
#
# Default 0, OPT-IN: no tiene sentido que una corrida exploratoria (solo
# exactitud, gates, smoke) dispare de oficio una segunda invocacion pesada
# por kernel. Activar solo cuando el objetivo de la corrida incluye
# energia/Frente de Pareto: RUN_ENERGY_PASS=1 bash run_full_pipeline.sh
RUN_ENERGY_PASS="${RUN_ENERGY_PASS:-0}"
ENERGY_ITERS_GEMM="${ENERGY_ITERS_GEMM:-24000}"
ENERGY_ITERS_CONV="${ENERGY_ITERS_CONV:-37000}"
ENERGY_ITERS_STENCIL="${ENERGY_ITERS_STENCIL:-4000}"

# La campana de VARIABILIDAD (replicas independientes para el ANOVA/Tukey,
# Etapa 7 -- tools/lanzar_campana_variabilidad.sh) NO se puede correr desde
# este script: por diseno, cada replica es un job de SLURM INDEPENDIENTE
# (job_id propio, que es justo lo que hace a dos corridas "observaciones
# independientes" para el ANOVA), y el propio lanzador exige `sbatch` y sale
# con error si no lo encuentra. Este orquestador corre todo con `bash`, sin
# SLURM -- no hay forma honesta de fingir job_ids independientes en un solo
# proceso secuencial. En un cluster con SLURM, correr la campana de
# variabilidad con:
#   bash tools/lanzar_campana_variabilidad.sh
# (o usar run_full_pipeline_pacca.sh, que ya la integra como parte del
# pipeline completo).

# PIPELINE_MODE=smoke|full (default full).
#
#   full  -> no se toca nada: cada .sbatch usa sus propios defaults, que son la
#            campana completa. Es el comportamiento historico de este script,
#            y sigue siendo el default para no cambiarle el significado a quien
#            ya lo estuviera usando en serio.
#   smoke -> exporta SMOKE_TEST=1 y RUN_NCU=0 a TODAS las fases. Los ocho
#            .sbatch que participan lo entienden (Fase 1 y 2 desde siempre;
#            los encadenados de GEMM/Convolucion de Fase 3/4 lo aceptan desde
#            la auditoria que agrego esta bandera, que hasta entonces era la
#            unica asimetria: Stencil si lo tenia y ellos no).
#
# Las variables se exportan solo si el usuario NO las fijo ya: `SMOKE_TEST=0
# PIPELINE_MODE=smoke ...` deja ganar al valor explicito, no al modo.
PIPELINE_MODE="${PIPELINE_MODE:-full}"
case "${PIPELINE_MODE}" in
    smoke)
        export SMOKE_TEST="${SMOKE_TEST:-1}"
        export RUN_NCU="${RUN_NCU:-0}"
        echo "PIPELINE_MODE=smoke -- SMOKE_TEST=${SMOKE_TEST}, RUN_NCU=${RUN_NCU}." \
             "Campana chica de verificacion, NO datos reportables."
        ;;
    full)
        echo "PIPELINE_MODE=full -- campana COMPLETA (cada .sbatch con sus propios" \
             "defaults). Para una prueba rapida: PIPELINE_MODE=smoke."
        ;;
    *)
        echo "ERROR: PIPELINE_MODE debe ser 'smoke' o 'full' (recibido:" \
             "'${PIPELINE_MODE}')." >&2
        exit 2
        ;;
esac
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

# Corre una fase dos veces si RUN_ENERGY_PASS=1: la normal (defaults del
# .sbatch, numerica) y una segunda SOLO con RUN_KIND=energy/ITERS_LIST
# grande (ver la nota de cabecera sobre por que hace falta). RUN_KIND e
# ITERS_LIST se exportan solo para la segunda llamada y se limpian despues,
# para no dejarlos pegados en el resto del pipeline (p.ej. Fase 1/2, que no
# entienden RUN_KIND).
run_phase_con_energia() {
    local label="$1" dir="$2" script="$3" energy_iters="$4"
    run_phase "${label}" "${dir}" "${script}"
    if [[ "${RUN_ENERGY_PASS}" == "1" && "${SMOKE_TEST:-0}" != "1" ]]; then
        RUN_KIND=energy ITERS_LIST="${energy_iters}" \
            run_phase "${label} (energia)" "${dir}" "${script}"
    elif [[ "${RUN_ENERGY_PASS}" == "1" ]]; then
        # SMOKE_TEST=1 ya deja ITERS_LIST en 3 dentro del .sbatch -- si aqui
        # se exportara igual ENERGY_ITERS_* (miles de iters) lo pisaria y el
        # "humo" dejaria de ser rapido. El pase de energia no aporta nada en
        # modo humo (solo valida que compile y corra, no que la energia sea
        # fiable), asi que se omite entero.
        echo "PIPELINE_MODE=smoke -- se omite el pase de energia de ${label}" \
             "(no tiene sentido con ITERS_LIST de humo)."
    fi
}

if [[ "${RUN_FASE3}" == "1" ]]; then
    run_phase_con_energia "Fase 3 / Stencil (tc)"      Fase_3/Stencil     run_stencil_tc.sbatch "${ENERGY_ITERS_STENCIL}"
    run_phase_con_energia "Fase 3 / GEMM (chained)"    Fase_3/GEMM        run_gemm_chained.sbatch "${ENERGY_ITERS_GEMM}"
    run_phase_con_energia "Fase 3 / Convolucion (chained)" Fase_3/Convolution run_conv_chained.sbatch "${ENERGY_ITERS_CONV}"
else
    echo "RUN_FASE3=0 -- se omite Fase 3."
fi

if [[ "${RUN_FASE4}" == "1" ]]; then
    run_phase_con_energia "Fase 4 / Stencil (ancla FP64)"      Fase_4/Stencil     run_stencil_tc.sbatch "${ENERGY_ITERS_STENCIL}"
    run_phase_con_energia "Fase 4 / GEMM (ancla FP64)"         Fase_4/GEMM        run_gemm_chained.sbatch "${ENERGY_ITERS_GEMM}"
    run_phase_con_energia "Fase 4 / Convolucion (ancla FP64)"  Fase_4/Convolution run_conv_chained.sbatch "${ENERGY_ITERS_CONV}"
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
