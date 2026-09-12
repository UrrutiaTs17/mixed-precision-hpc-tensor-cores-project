#!/bin/bash
# tools/lanzar_campana_variabilidad.sh
#
# CAMPANA DE VARIABILIDAD (replicas) para el ANOVA/Tukey de la Etapa 7 del
# plan -- Fase_4/tools/run_statistics.py.
#
# POR QUE HACE FALTA, APARTE DE run_full_pipeline_pacca.sh
# ----------------------------------------------------------
# La campana principal corre cada celda del diseno (formato x tratamiento x
# tamano x K) UNA sola vez: barre muchos tamanos, pero no repite ningun punto.
# Un ANOVA necesita, para CADA celda que se va a comparar, al menos 2
# observaciones INDEPENDIENTES (mismo formato/tratamiento/iters, job_id
# distinto) para estimar varianza de error -- sin eso, statsmodels ajusta
# igual (no explota) pero el resultado es basura: sumas de cuadrados
# negativas, efectos que deberian ser obvios saliendo en cero. Asi salio la
# primera corrida completa en PACCA (2026-09-10): run_statistics.py avisa por
# stderr "N celda(s) del diseno tienen menos de 2 replica(s)" (ver
# Dataset.warn_low_replicas en common_analysis.py) pero eso no bastaba, el
# ANOVA ya habia salido mal.
#
# DISENO: MENOS DIVERSIDAD, MAS REPLICAS
# ---------------------------------------
# Ni run_statistics.py necesita "tamano" como factor (revisa sus formulas:
# anova_treatment_stencil usa format*treatment*iters, anchor_k_analysis usa
# anchor_every*kernel -- "size" no aparece en ninguna) ni el presupuesto de
# cola de un solo GPU compartido aguanta repetir el barrido COMPLETO (hasta
# N=8192 en GEMM, NX=16384 en Stencil) muchas veces. Esta campana FIJA un
# tamano chico por kernel (barato, ver los *_LIST de abajo) y en cambio
# REPITE esa misma configuracion REPLICAS veces como jobs de SLURM
# independientes -- cada uno con su propio SLURM_JOB_ID, que es exactamente
# la columna que Dataset.warn_low_replicas cuenta como "replica".
#
# ALEATORIZACION DEL ORDEN DE ENVIO (Etapa 5, Aislamiento termico)
# -------------------------------------------------------------------
# docs/MANUAL.md ya documenta el riesgo: si las replicas de una celda se
# envian en bloque, la deriva termica del nodo (un GPU compartido, sin
# nvidia-smi -lgc disponible en PACCA) queda confundida con la celda. Este
# script arma la lista completa de trabajos (kernel x replica) y la
# BARAJA antes de enviar -- ni todas las replicas de GEMM seguidas, ni todas
# las de una fase antes que las de otra.
#
# USO
# ---
#   bash tools/lanzar_campana_variabilidad.sh                 # todo, REPLICAS=8
#   REPLICAS=15 bash tools/lanzar_campana_variabilidad.sh      # mas potencia estadistica
#   DRY_RUN=1 bash tools/lanzar_campana_variabilidad.sh        # imprime, no envia
#   RUN_GEMM=0 bash tools/lanzar_campana_variabilidad.sh       # salta un kernel
#
# Parametrizar el punto fijo de cada kernel: exporta REPL_<KERNEL>_<VAR> antes
# de llamar (mismo nombre de variable que el .sbatch correspondiente, con el
# prefijo REPL_<KERNEL>_). Ver los defaults abajo para la lista completa.
#
# Al final imprime la tabla de job ids y encadena UN job de estadistica
# (tools/postproceso_variabilidad.sbatch) con --dependency=afterok sobre
# TODAS las replicas -- si alguna falla, el ANOVA no corre sobre un dataset
# incompleto sin avisar.
#
# QUE HACER SI, TRAS CORRER, SIGUE FALTANDO REPLICAS
# -----------------------------------------------------
# Revisa el .err de postproceso_variabilidad (job final): si el aviso de
# "menos de 2 replicas" persiste, sube REPLICAS y vuelve a lanzar -- este
# script es aditivo, no borra resultados anteriores en results/variabilidad/
# (job_id nuevo cada vez), asi que relanzar con REPLICAS mas alto y volver a
# correr el post-proceso ya suma a lo que hay, no repite desde cero.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-0}"
REPLICAS="${REPLICAS:-8}"
RUN_GEMM="${RUN_GEMM:-1}"
RUN_CONV="${RUN_CONV:-1}"
RUN_STENCIL="${RUN_STENCIL:-1}"
RUN_STATS_FINAL="${RUN_STATS_FINAL:-1}"

# Mismo VARDIR que tools/postproceso_variabilidad.sbatch espera por defecto
# (relativo a cada carpeta de kernel) -- si se cambia aqui, cambiar alli.
VARDIR="${VARDIR:-results/variabilidad}"

if [[ "${DRY_RUN}" != "1" ]] && ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: no se encontro sbatch. Este script es especifico de un" >&2
    echo "cluster con SLURM (PACCA)." >&2
    exit 1
fi

# --- Punto fijo por kernel (chico y barato, deliberado -- ver cabecera) -----
# GEMM: N=1024 (el mas chico "real" del barrido principal), 2 niveles de
# iters, K=0/1/5 (el ancla solo aplica con comp=on).
REPL_GEMM_N_LIST="${REPL_GEMM_N_LIST:-1024}"
REPL_GEMM_ITERS_LIST="${REPL_GEMM_ITERS_LIST:-20 40}"
REPL_GEMM_ANCHOR_LIST="${REPL_GEMM_ANCHOR_LIST:-0 1 5}"
REPL_GEMM_COMP_LIST="${REPL_GEMM_COMP_LIST:-off on}"
REPL_GEMM_TC_FORMAT="${REPL_GEMM_TC_FORMAT:-both}"

# Convolucion: mismo criterio, HW=64 (el minimo del binario).
REPL_CONV_HW_LIST="${REPL_CONV_HW_LIST:-64}"
REPL_CONV_ITERS_LIST="${REPL_CONV_ITERS_LIST:-20 40}"
REPL_CONV_ANCHOR_LIST="${REPL_CONV_ANCHOR_LIST:-0 1 5}"
REPL_CONV_COMP_LIST="${REPL_CONV_COMP_LIST:-off on}"
REPL_CONV_TC_FORMAT="${REPL_CONV_TC_FORMAT:-both}"

# Stencil: NX=NY=1024 (muy por debajo del 4096-16384 del barrido principal --
# ahi es donde vive el costo real). KAHAN_LIST/SPATIAL_COMP en su default
# normal (el mismo que ya uso la campana principal y SI produjo los tres
# niveles de tratamiento none/kahan_local/spatial) para no arriesgar perder
# un nivel del factor por reinventar la combinacion de flags.
REPL_STENCIL_NX_LIST="${REPL_STENCIL_NX_LIST:-1024}"
REPL_STENCIL_NY_LIST="${REPL_STENCIL_NY_LIST:-1024}"
REPL_STENCIL_ITERS_LIST="${REPL_STENCIL_ITERS_LIST:-20 40}"
REPL_STENCIL_KAHAN_LIST="${REPL_STENCIL_KAHAN_LIST:-off on}"
REPL_STENCIL_SPATIAL_COMP="${REPL_STENCIL_SPATIAL_COMP:-on}"
REPL_STENCIL_TC_FORMAT="${REPL_STENCIL_TC_FORMAT:-both}"

# El ancla FP64 (K>0 en ANCHOR_LIST) exige --spatial-comp on en el binario;
# si esta pasada corre con SPATIAL_COMP=off (para ejercitar kahan_local/none)
# hay que anular los anclajes o run_stencil_tc.sbatch rechaza el job entero.
if [[ "${REPL_STENCIL_SPATIAL_COMP}" == "off" ]]; then
    REPL_STENCIL_ANCHOR_LIST="${REPL_STENCIL_ANCHOR_LIST:-0}"
else
    REPL_STENCIL_ANCHOR_LIST="${REPL_STENCIL_ANCHOR_LIST:-0 1 5}"
fi

# --- Arma la lista de trabajos (kernel, indice de replica) y la baraja -----
declare -a WORKLIST=()
for ((r = 1; r <= REPLICAS; r++)); do
    [[ "${RUN_GEMM}" == "1" ]]    && WORKLIST+=("gemm:${r}")
    [[ "${RUN_CONV}" == "1" ]]    && WORKLIST+=("conv:${r}")
    [[ "${RUN_STENCIL}" == "1" ]] && WORKLIST+=("stencil:${r}")
done

if [[ "${#WORKLIST[@]}" -eq 0 ]]; then
    echo "ERROR: ningun kernel activo (RUN_GEMM/RUN_CONV/RUN_STENCIL) y/o REPLICAS=0." >&2
    exit 1
fi

# Baraja con shuf si esta disponible; si no (poco comun en un nodo Linux de
# cluster, pero por si acaso), usa el orden tal cual -- degrada a "sin
# aleatorizar" en vez de fallar, con aviso.
if command -v shuf >/dev/null 2>&1; then
    mapfile -t WORKLIST < <(printf '%s\n' "${WORKLIST[@]}" | shuf)
else
    echo "AVISO: no se encontro 'shuf' -- se envia sin barajar el orden" \
         "(el aislamiento termico entre celdas queda mas expuesto)." >&2
fi

echo "################################################################"
echo "# Campana de variabilidad -- REPLICAS=${REPLICAS}, ${#WORKLIST[@]} jobs a enviar"
echo "################################################################"
echo "Orden de envio (barajado):"
printf '  %s\n' "${WORKLIST[@]}"
echo

declare -a RESUMEN=()
declare -a JOBS_GEMM=() JOBS_CONV=() JOBS_STENCIL=()
_DRY_ID=2000

enviar_replica() {
    local kernel="$1" rep="$2" dir="$3" script="$4"
    shift 4
    local -a extra_export=("$@")
    local jid

    if [[ "${DRY_RUN}" == "1" ]]; then
        _DRY_ID=$((_DRY_ID + 1))
        jid="${_DRY_ID}"
        echo "[DRY_RUN] (cd ${dir} && sbatch --export=ALL,RESULTS_DIR=${VARDIR},$(IFS=,; echo "${extra_export[*]}") ${script})  -> ${jid}"
    else
        jid="$(cd "${dir}" && sbatch --parsable \
            --export="ALL,RESULTS_DIR=${VARDIR},$(IFS=,; echo "${extra_export[*]}")" \
            "${script}")"
        echo "Enviada replica ${rep} de ${kernel}: job ${jid}"
    fi
    RESUMEN+=("$(printf '%-10s %-10s replica=%-3s' "${jid}" "${kernel}" "${rep}")")
    case "${kernel}" in
        gemm) JOBS_GEMM+=("${jid}") ;;
        conv) JOBS_CONV+=("${jid}") ;;
        stencil) JOBS_STENCIL+=("${jid}") ;;
    esac
}

for item in "${WORKLIST[@]}"; do
    kernel="${item%%:*}"
    rep="${item##*:}"
    case "${kernel}" in
        gemm)
            enviar_replica gemm "${rep}" Fase_4/GEMM run_gemm_chained.sbatch \
                "N_LIST=${REPL_GEMM_N_LIST}" "ITERS_LIST=${REPL_GEMM_ITERS_LIST}" \
                "ANCHOR_LIST=${REPL_GEMM_ANCHOR_LIST}" "COMP_LIST=${REPL_GEMM_COMP_LIST}" \
                "TC_FORMAT=${REPL_GEMM_TC_FORMAT}"
            ;;
        conv)
            enviar_replica conv "${rep}" Fase_4/Convolution run_conv_chained.sbatch \
                "HW_LIST=${REPL_CONV_HW_LIST}" "ITERS_LIST=${REPL_CONV_ITERS_LIST}" \
                "ANCHOR_LIST=${REPL_CONV_ANCHOR_LIST}" "COMP_LIST=${REPL_CONV_COMP_LIST}" \
                "TC_FORMAT=${REPL_CONV_TC_FORMAT}"
            ;;
        stencil)
            enviar_replica stencil "${rep}" Fase_4/Stencil run_stencil_tc.sbatch \
                "NX_LIST=${REPL_STENCIL_NX_LIST}" "NY_LIST=${REPL_STENCIL_NY_LIST}" \
                "ITERS_LIST=${REPL_STENCIL_ITERS_LIST}" "ANCHOR_LIST=${REPL_STENCIL_ANCHOR_LIST}" \
                "KAHAN_LIST=${REPL_STENCIL_KAHAN_LIST}" "SPATIAL_COMP=${REPL_STENCIL_SPATIAL_COMP}" \
                "TC_FORMAT=${REPL_STENCIL_TC_FORMAT}"
            ;;
    esac
done

echo
echo "################################################################"
echo "# Jobs de replica enviados"
echo "################################################################"
printf '  %s\n' "${RESUMEN[@]}"

if [[ "${RUN_STATS_FINAL}" == "1" ]]; then
    ALL_JOBS=("${JOBS_GEMM[@]}" "${JOBS_CONV[@]}" "${JOBS_STENCIL[@]}")
    if [[ "${#ALL_JOBS[@]}" -gt 0 ]]; then
        DEP="afterok:$(IFS=:; echo "${ALL_JOBS[*]}")"
        if [[ "${DRY_RUN}" == "1" ]]; then
            _DRY_ID=$((_DRY_ID + 1))
            echo
            echo "[DRY_RUN] sbatch --export=ALL,VARDIR=${VARDIR} --dependency=${DEP} tools/postproceso_variabilidad.sbatch  -> ${_DRY_ID}"
        else
            STATS_JID="$(sbatch --parsable --export="ALL,VARDIR=${VARDIR}" \
                --dependency="${DEP}" tools/postproceso_variabilidad.sbatch)"
            echo
            echo "Estadistica final encadenada: job ${STATS_JID} (${DEP})"
        fi
    else
        echo "AVISO: no se envio ninguna replica, no hay a que encadenar la estadistica." >&2
    fi
fi

echo
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1: no se envio nada. Quite la variable para enviar de verdad."
else
    echo "Seguimiento:  squeue -u \"\$USER\""
    echo "Resultados en Fase_4/{GEMM,Convolution,Stencil}/${VARDIR}/"
    echo "Estadistica (al terminar todo) en stats_out_variabilidad/"
fi
