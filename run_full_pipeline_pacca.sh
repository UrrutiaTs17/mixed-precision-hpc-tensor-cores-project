#!/bin/bash
# run_full_pipeline_pacca.sh
#
# Orquestador ALTERNATIVO para un cluster con SLURM (PACCA). NO reemplaza a
# run_full_pipeline.sh: ese sigue siendo el camino para correr todo de un tiron
# en una maquina propia con GPU, y se deja intacto.
#
# POR QUE HACEN FALTA DOS
# -----------------------
# run_full_pipeline.sh invoca cada .sbatch con `bash`, en secuencia, dentro de
# UN solo proceso. Eso exige tener la GPU reservada de principio a fin en una
# sola sesion. Con las campanas ampliadas (cuatro tamanos x tres barridos de
# iteraciones x cuatro valores de K, en tres kernels) eso son facilmente
# decenas de horas seguidas: en un cluster compartido no hay forma de pedir esa
# reserva, y si el job se cae en la hora 30 se pierde todo.
#
# Este script, en cambio, ENVIA cada fase como un job independiente con
# `sbatch --parsable`, y las encadena con `--dependency=afterok:<job_id>`.
# SLURM se encarga del resto: cada job pide solo el tiempo que necesita, la
# cola los intercala con los de otros usuarios, y si uno falla los que dependen
# de el no arrancan (afterok, no afterany) en vez de correr sobre datos que no
# existen.
#
# GRAFO DE DEPENDENCIAS
# ---------------------
#                       +--> Fase1/Fase2 (opcionales, sin dependientes)
#   validacion          |
#   preliminar ---------+--> F3 GEMM   --> F4 GEMM   --+
#   (los tres           +--> F3 Conv   --> F4 Conv   --+--> post-proceso
#    kernels)           +--> F3 Stencil--> F4 Stencil--+    (sin GPU)
#
#   * Los tres kernels van EN PARALELO entre si: no comparten nada.
#   * Fase 4 de un kernel depende SOLO de su propia Fase 3.
#   * Fase 1 y Fase 2 cuelgan de la validacion preliminar pero NADIE depende de
#     ellas: no producen CSV_* que el post-proceso consuma (ver docs/MANUAL.md,
#     seccion Fase 1), asi que meterlas en la cadena de F3/F4 solo lograria que
#     un fallo suyo -- por ejemplo, CUTLASS sin clonar, que es opcional --
#     bloqueara una campana que no las necesita.
#   * El post-proceso depende de los SEIS jobs de F3/F4 a la vez
#     (afterok:J1:J2:...:J6) y NO pide GPU.
#
# La validacion preliminar (tools/validacion_preliminar.sbatch) va primero a
# proposito: verifica orden de operandos, humo de los tres kernels y los gates
# K=0/K=1 en minutos. Con afterok, si algo de eso falla NINGUNA campana llega a
# arrancar -- que es exactamente el punto de tenerla.
#
# USO
# ---
#   bash run_full_pipeline_pacca.sh                  # envia todo y sale
#   RUN_FASE1=0 RUN_FASE2=0 bash run_full_pipeline_pacca.sh
#   SKIP_VALIDACION=1 bash run_full_pipeline_pacca.sh   # sin la puerta previa
#   DRY_RUN=1 bash run_full_pipeline_pacca.sh        # imprime, no envia
#
# Parametrizar una fase: exporta las mismas variables que acepta su .sbatch
# ANTES de llamar a este script. Se propagan a los jobs via --export=ALL.
#   N_LIST="1024 2048" ANCHOR_LIST="0 1 5" bash run_full_pipeline_pacca.sh
#
# El script TERMINA en cuanto termina de enviar: no espera a que la campana
# corra. Imprime la tabla de job ids para seguirla con `squeue`/`sacct`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-0}"
SKIP_VALIDACION="${SKIP_VALIDACION:-0}"
RUN_FASE1="${RUN_FASE1:-1}"
RUN_FASE2="${RUN_FASE2:-1}"
RUN_FASE3="${RUN_FASE3:-1}"
RUN_FASE4="${RUN_FASE4:-1}"
RUN_POST="${RUN_POST:-1}"

# Particion del job de post-proceso. Se deja VACIA por defecto a proposito: no
# hay un nombre de particion sin GPU que sea portable entre clusters, y pasar
# uno inexistente hace que sbatch RECHACE el envio. Vacia = particion por
# defecto del cluster. En PACCA, si existe una particion de solo CPU, ponerla
# aqui libera la GPU durante todo el post-proceso:
#   POST_PARTITION=CPU bash run_full_pipeline_pacca.sh
POST_PARTITION="${POST_PARTITION:-}"

if [[ "${DRY_RUN}" != "1" ]] && ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: no se encontro sbatch. Este orquestador es especifico de un" >&2
    echo "cluster con SLURM. Fuera de un cluster use run_full_pipeline.sh, que" >&2
    echo "corre exactamente las mismas fases con \`bash\` en secuencia." >&2
    exit 1
fi

declare -a RESUMEN=()
_DRY_ID=1000
# Job id del ultimo envio. `enviar` lo deja AQUI en vez de imprimirlo para que
# el llamador lo capture con $(...): una sustitucion de comandos abre una
# SUBSHELL, y ahi los `RESUMEN+=(...)` y el contador del dry-run se perderian
# al volver. Es el tipo de bug que no se nota hasta que la tabla final sale
# vacia.
ULTIMO_JID=""

# enviar <etiqueta> <directorio> <script> [dependencia]
enviar() {
    local etiqueta="$1" dir="$2" script="$3" dep="${4:-}"
    local -a args=(--parsable --export=ALL)
    [[ -n "${dep}" ]] && args+=(--dependency="afterok:${dep}")
    # Sin comillas a proposito: EXTRA_SBATCH_ARGS puede traer varios flags.
    # shellcheck disable=SC2206
    [[ -n "${EXTRA_SBATCH_ARGS:-}" ]] && args+=(${EXTRA_SBATCH_ARGS})

    if [[ "${DRY_RUN}" == "1" ]]; then
        # Id sintetico y creciente, para que el grafo de dependencias del
        # dry-run se vea igual que el real.
        _DRY_ID=$(( _DRY_ID + 1 ))
        ULTIMO_JID="${_DRY_ID}"
        echo "[DRY_RUN] (cd ${dir} && sbatch ${args[*]} ${script})  -> ${ULTIMO_JID}"
    else
        ULTIMO_JID="$(cd "${dir}" && sbatch "${args[@]}" "${script}")"
        echo "Enviado ${etiqueta}: job ${ULTIMO_JID}${dep:+ (afterok:${dep})}"
    fi
    RESUMEN+=("$(printf '%-10s %-34s dep=%s' "${ULTIMO_JID}" "${etiqueta}" "${dep:-ninguna}")")
}

# --- Puerta previa: validacion preliminar a escala reducida ------------------
DEP_BASE=""
if [[ "${SKIP_VALIDACION}" == "1" ]]; then
    echo "SKIP_VALIDACION=1 -- las campanas se envian SIN la puerta previa." >&2
    echo "  (Tarea 8 de la auditoria: no hacerlo salvo que ya se haya corrido" >&2
    echo "   tools/validacion_preliminar.sbatch a mano y haya pasado.)" >&2
else
    enviar "validacion preliminar" "." "tools/validacion_preliminar.sbatch"
    DEP_BASE="${ULTIMO_JID}"
fi

# --- Fase 1 y Fase 2: opcionales, sin dependientes --------------------------
if [[ "${RUN_FASE1}" == "1" ]]; then
    enviar "F1 GEMM"        Fase_1/GEMM        run_gemm_fase1.sbatch    "${DEP_BASE}"
    enviar "F1 Convolucion" Fase_1/Convolution run_conv_fase1.sbatch    "${DEP_BASE}"
    enviar "F1 Stencil"     Fase_1/Stencil     run_stencil_fase1.sbatch "${DEP_BASE}"
fi
if [[ "${RUN_FASE2}" == "1" ]]; then
    enviar "F2 GEMM"        Fase_2/GEMM        run_gemm_tc.sbatch    "${DEP_BASE}"
    enviar "F2 Convolucion" Fase_2/Convolution run_conv_tc.sbatch    "${DEP_BASE}"
    enviar "F2 Stencil"     Fase_2/Stencil     run_stencil_tc.sbatch "${DEP_BASE}"
fi

# --- Fase 3 -> Fase 4, una cadena independiente por kernel ------------------
declare -a JOBS_CAMPANA=()

cadena_kernel() {
    local nombre="$1" dir_f3="$2" script_f3="$3" dir_f4="$4" script_f4="$5"
    local j3=""
    if [[ "${RUN_FASE3}" == "1" ]]; then
        enviar "F3 ${nombre}" "${dir_f3}" "${script_f3}" "${DEP_BASE}"
        j3="${ULTIMO_JID}"
        JOBS_CAMPANA+=("${j3}")
    fi
    if [[ "${RUN_FASE4}" == "1" ]]; then
        # Fase 4 depende de su PROPIA Fase 3 si se envio; si no (RUN_FASE3=0),
        # de la validacion preliminar. No de las Fases 3 de los otros kernels:
        # son independientes y encadenarlas solo alargaria el camino critico.
        enviar "F4 ${nombre}" "${dir_f4}" "${script_f4}" "${j3:-${DEP_BASE}}"
        JOBS_CAMPANA+=("${ULTIMO_JID}")
    fi
}

cadena_kernel "GEMM"        Fase_3/GEMM        run_gemm_chained.sbatch \
                            Fase_4/GEMM        run_gemm_chained.sbatch
cadena_kernel "Convolucion" Fase_3/Convolution run_conv_chained.sbatch \
                            Fase_4/Convolution run_conv_chained.sbatch
cadena_kernel "Stencil"     Fase_3/Stencil     run_stencil_tc.sbatch \
                            Fase_4/Stencil     run_stencil_tc.sbatch

# --- Post-proceso: depende de TODOS los jobs de campana ---------------------
if [[ "${RUN_POST}" == "1" ]]; then
    if [[ "${#JOBS_CAMPANA[@]}" -eq 0 ]]; then
        echo "AVISO: no se envio ningun job de Fase 3/4, asi que el post-proceso" >&2
        echo "  se envia sin dependencias, sobre lo que ya haya en results/." >&2
        DEP_POST="${DEP_BASE}"
    else
        # afterok con varios ids: "afterok:J1:J2:...". Si CUALQUIERA falla, el
        # post-proceso no arranca -- correcto: un ANOVA sobre una campana
        # incompleta es peor que no tener ANOVA, porque parece un resultado.
        DEP_POST="$(IFS=:; echo "${JOBS_CAMPANA[*]}")"
    fi
    # Se asigna y se limpia a mano en vez de con el prefijo `VAR=... enviar`:
    # para una FUNCION, bash no siempre restaura la variable al terminar la
    # llamada, y el flag se colaria en cualquier envio posterior.
    EXTRA_SBATCH_ARGS="${POST_PARTITION:+--partition=${POST_PARTITION}}"
    enviar "post-proceso (stats+Pareto)" "." "tools/postproceso.sbatch" "${DEP_POST}"
    unset EXTRA_SBATCH_ARGS
fi

echo
echo "################################################################"
echo "# Jobs enviados"
echo "################################################################"
printf '  %s\n' "${RESUMEN[@]}"
echo
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1: no se envio nada. Quite la variable para enviar de verdad."
else
    echo "Seguimiento:  squeue -u \"\$USER\"        (cola)"
    echo "              sacct -j <job_id> -X      (estado final de un job)"
    echo "Este script YA TERMINO: la campana sigue en la cola de SLURM."
fi
