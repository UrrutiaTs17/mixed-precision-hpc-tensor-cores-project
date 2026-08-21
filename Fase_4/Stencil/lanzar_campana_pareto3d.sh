#!/bin/bash
# ===========================================================================
# Orquestador de la campana Fase 4 -- frente de Pareto 3D (Stencil difusivo)
# ===========================================================================
#
# Tres subcomandos, cada uno lanza SU fase y nada mas:
#
#   exploratorio  1 job, 1 replica, ITERS=320. Su unico proposito es MEDIR el
#                 walltime real a 16384^2 con el operador difusivo, que no se
#                 conoce: el punto de comparacion mas cercano es el job 6325 de
#                 Fase 3 (31:52 con CHECKPOINT_EVERY=5), pero ese llevaba
#                 checkpoints y otro operador. Sin este dato, cualquier
#                 --time= de las fases siguientes es una adivinanza, y una
#                 adivinanza corta mata la campana a mitad.
#
#   piloto        3 replicas energeticas + 1 corrida numerica. Valida que la
#                 cadena completa produce rel_l2 finito y archivos legibles
#                 antes de comprometer el nodo 15 veces.
#
#   campana       15 replicas energeticas, como job array --array=1-15%1
#                 (%1 = una a la vez: dos replicas concurrentes en el mismo
#                 nodo se contaminan la medida de energia y de tiempo).
#
# Puertas: `piloto` exige un exploratorio COMPLETED, y `campana` exige un
# piloto validado. No se saltan con un flag; hay que borrar o corregir el
# estado en estado/ a mano, que es justo la friccion que se busca.
#
# Cada lanzamiento imprime los comandos de seguimiento con los JOB IDs REALES
# ya sustituidos.
#
# Uso:
#   ./lanzar_campana_pareto3d.sh exploratorio
#   ./lanzar_campana_pareto3d.sh piloto
#   ./lanzar_campana_pareto3d.sh campana
#   ./lanzar_campana_pareto3d.sh estado
#
# Variables reconocidas (todas con default): NX NY OP_MODE ALPHA CI_MODE CI_P
#   FP64_GPU CPU_FP64 CAMPAIGN_ID WALL_EXPLORATORIO WALL_ENERGY WALL_NUMERIC
#   CHECKPOINT_ITERS ARCHIVE_ITERS DRY_RUN

set -euo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AQUI}"
SBATCH_FILE="run_stencil_pareto3d.sbatch"
ESTADO_DIR="estado"
mkdir -p "${ESTADO_DIR}" logs results

# --- Diseno congelado -------------------------------------------------------
NX="${NX:-16384}"
NY="${NY:-16384}"
OP_MODE="${OP_MODE:-diffusive}"
ALPHA="${ALPHA:-0.1875}"
CI_MODE="${CI_MODE:-monomode}"
CI_P="${CI_P:-168}"
FP64_GPU="${FP64_GPU:-on}"
CPU_FP64="${CPU_FP64:-on}"
# Separador ":" y no ",": --export de SLURM parte por comas (ver el sbatch).
ITERS_CAMPANA="${ITERS_CAMPANA:-320:640}"
CHECKPOINT_ITERS="${CHECKPOINT_ITERS:-1:2:5:10:20:50:100:200:320:640}"
ARCHIVE_ITERS="${ARCHIVE_ITERS:-320:640}"
REPLICAS_PILOTO="${REPLICAS_PILOTO:-3}"
REPLICAS_CAMPANA="${REPLICAS_CAMPANA:-15}"
DRY_RUN="${DRY_RUN:-0}"

# Walltimes. El del exploratorio es GENEROSO a proposito (es lo que se esta
# midiendo); los otros dos quedan sin default y el script obliga a fijarlos con
# el dato que el exploratorio devuelva.
WALL_EXPLORATORIO="${WALL_EXPLORATORIO:-04:00:00}"

rojo()  { printf '\033[31m%s\033[0m\n' "$*"; }
verde() { printf '\033[32m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n' "$*"; }

morir() { rojo "ERROR: $*"; exit 1; }

# ---------------------------------------------------------------------------
# Seguimiento: comandos con el JOB ID real ya sustituido
# ---------------------------------------------------------------------------
imprimir_seguimiento() {
    local etiqueta="$1"; shift
    local ids=("$@")
    local lista; lista="$(IFS=,; echo "${ids[*]}")"
    echo
    bold "--- Seguimiento de ${etiqueta} (job IDs reales: ${lista}) ---"
    echo
    echo "  # cola y estado"
    echo "  squeue -j ${lista} -o \"%.10i %.12P %.20j %.8T %.10M %.10l %R\""
    echo
    echo "  # contabilidad al terminar (elapsed real -> calibra el walltime siguiente)"
    echo "  sacct -j ${lista} --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode"
    echo
    for id in "${ids[@]}"; do
        echo "  # log de ${id}"
        echo "  tail -f ${AQUI}/logs/f4_pareto3d_${id}.out"
    done
    echo
    echo "  # cancelar todo lo lanzado en esta fase"
    echo "  scancel ${lista}"
    echo
}

enviar() {   # imprime el comando y devuelve el job id por stdout
    local desc="$1"; shift
    echo "  -> ${desc}" >&2
    echo "     sbatch $*" >&2
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "DRYRUN$(date +%s%N | tail -c 5)"
        return 0
    fi
    local salida
    salida="$(sbatch "$@")" || morir "sbatch fallo para ${desc}"
    echo "${salida}" | grep -oE '[0-9]+$'
}

exportes_comunes() {
    echo "ALL,NX=${NX},NY=${NY},OP_MODE=${OP_MODE},ALPHA=${ALPHA},CI_MODE=${CI_MODE},CI_P=${CI_P},FP64_GPU=${FP64_GPU},CPU_FP64=${CPU_FP64},RUN_NCU=0"
}

# ---------------------------------------------------------------------------
# Puertas
# ---------------------------------------------------------------------------
estado_job() {
    sacct -j "$1" --format=State --noheader -X 2>/dev/null | head -1 | tr -d ' ' || echo DESCONOCIDO
}

exigir_exploratorio_ok() {
    local f="${ESTADO_DIR}/exploratorio.job"
    [[ -f "${f}" ]] || morir "no hay exploratorio lanzado. Corra primero: $0 exploratorio"
    local id; id="$(cat "${f}")"
    local st; st="$(estado_job "${id}")"
    [[ "${st}" == "COMPLETED" ]] || morir \
        "el exploratorio (job ${id}) esta en estado '${st}', no COMPLETED. El piloto necesita su
   walltime medido antes de comprometer el nodo. Revise:
     sacct -j ${id} --format=JobID,State,Elapsed,ExitCode"
    local elapsed; elapsed="$(sacct -j "${id}" --format=Elapsed --noheader -X 2>/dev/null | head -1 | tr -d ' ')"
    verde "Puerta OK: exploratorio ${id} COMPLETED, Elapsed=${elapsed}" >&2
    echo "${elapsed}"
}

exigir_piloto_ok() {
    local f="${ESTADO_DIR}/piloto.validado"
    [[ -f "${f}" ]] || morir \
        "el piloto no esta validado. Tras correr '$0 piloto' y revisar sus CSV, marque:
     echo '<motivo de la validacion>' > ${ESTADO_DIR}/piloto.validado"
    verde "Puerta OK: piloto validado ($(head -1 "${f}"))"
}

# ---------------------------------------------------------------------------
# Subcomandos
# ---------------------------------------------------------------------------
cmd_exploratorio() {
    bold "=== FASE: exploratorio (1 job, 1 replica, ITERS=320) ==="
    echo
    echo "Proposito: MEDIR el walltime real. No produce datos de campana."
    echo "  malla     : ${NX}x${NY}"
    echo "  operador  : ${OP_MODE} (alpha=${ALPHA})"
    echo "  CI        : ${CI_MODE} (p=${CI_P})"
    echo "  iters     : 320"
    echo "  run_kind  : energy (sin checkpoints ni archivado)"
    echo "  bloque    : A (SPATIAL_COMP=off, KAHAN_LIST='off on' -> none y kahan_local)"
    echo "  walltime  : ${WALL_EXPLORATORIO} (holgado a proposito: es lo que se mide)"
    echo
    local cid="${CAMPAIGN_ID:-f4_pareto3d_explor_$(date +%Y%m%d_%H%M%S)}"
    local id
    id="$(enviar "exploratorio bloque A" \
        --time="${WALL_EXPLORATORIO}" \
        --job-name=f4_p3d_explor \
        --export="$(exportes_comunes),RUN_KIND=energy,ITERS_LIST=320,SPATIAL_COMP=off,KAHAN_LIST=off:on,CAMPAIGN_ID=${cid}" \
        "${SBATCH_FILE}")"
    echo "${id}" > "${ESTADO_DIR}/exploratorio.job"
    echo "${cid}" > "${ESTADO_DIR}/exploratorio.campaign"
    verde "Lanzado: job ${id}  (campaign_id=${cid})"
    imprimir_seguimiento "el exploratorio" "${id}"
    echo "Cuando termine, lea el Elapsed y uselo para fijar WALL_ENERGY/WALL_NUMERIC:"
    echo "  sacct -j ${id} --format=JobID,State,Elapsed,MaxRSS,ExitCode"
    echo
    echo "Regla de dedo para el siguiente paso: el piloto corre ITERS_LIST='${ITERS_CAMPANA}',"
    echo "es decir ~3x el trabajo de este job (320 + 640 = 960 iteraciones frente a 320)."
    echo "Sume margen: WALL_ENERGY >= 3.5 x Elapsed."
}

cmd_piloto() {
    bold "=== FASE: piloto (${REPLICAS_PILOTO} replicas energeticas + 1 numerica) ==="
    local elapsed; elapsed="$(exigir_exploratorio_ok)"
    [[ -n "${WALL_ENERGY:-}" ]] || morir \
        "fije WALL_ENERGY con el dato del exploratorio (Elapsed=${elapsed}).
   Sugerencia: WALL_ENERGY >= 3.5 x ese Elapsed, porque el piloto corre 320 y 640
   iteraciones (960 en total) frente a las 320 del exploratorio.
     WALL_ENERGY=HH:MM:SS WALL_NUMERIC=HH:MM:SS $0 piloto"
    [[ -n "${WALL_NUMERIC:-}" ]] || morir \
        "fije tambien WALL_NUMERIC. La corrida numerica anade checkpoints y archivado:
   la referencia FP64 se vuelca a disco y cada ruta la relee, asi que pida al menos
   el doble de WALL_ENERGY."
    echo
    local cid="${CAMPAIGN_ID:-f4_pareto3d_piloto_$(date +%Y%m%d_%H%M%S)}"
    local ids=()
    for r in $(seq 1 "${REPLICAS_PILOTO}"); do
        local ra rb
        ra="$(enviar "piloto r${r} bloque A (none + kahan_local)" \
            --time="${WALL_ENERGY}" --job-name="f4_p3d_pil_r${r}A" \
            --export="$(exportes_comunes),RUN_KIND=energy,ITERS_LIST=${ITERS_CAMPANA},SPATIAL_COMP=off,KAHAN_LIST=off:on,CAMPAIGN_ID=${cid}" \
            "${SBATCH_FILE}")"
        rb="$(enviar "piloto r${r} bloque B (spatial)" \
            --time="${WALL_ENERGY}" --job-name="f4_p3d_pil_r${r}B" \
            --export="$(exportes_comunes),RUN_KIND=energy,ITERS_LIST=${ITERS_CAMPANA},SPATIAL_COMP=on,CAMPAIGN_ID=${cid}" \
            "${SBATCH_FILE}")"
        ids+=("${ra}" "${rb}")
    done
    # La corrida NUMERICA es una sola, y por bloque: el error es determinista,
    # asi que replicarla no aporta nada -- n=15 es para tiempo y energia.
    local na nb
    na="$(enviar "numerica bloque A" \
        --time="${WALL_NUMERIC}" --job-name=f4_p3d_num_A \
        --export="$(exportes_comunes),RUN_KIND=numeric,ITERS_LIST=${ITERS_CAMPANA},SPATIAL_COMP=off,KAHAN_LIST=off:on,CHECKPOINT_ITERS=${CHECKPOINT_ITERS},ARCHIVE_ITERS=${ARCHIVE_ITERS},CAMPAIGN_ID=${cid}" \
        "${SBATCH_FILE}")"
    nb="$(enviar "numerica bloque B" \
        --time="${WALL_NUMERIC}" --job-name=f4_p3d_num_B \
        --export="$(exportes_comunes),RUN_KIND=numeric,ITERS_LIST=${ITERS_CAMPANA},SPATIAL_COMP=on,CHECKPOINT_ITERS=${CHECKPOINT_ITERS},ARCHIVE_ITERS=${ARCHIVE_ITERS},CAMPAIGN_ID=${cid}" \
        "${SBATCH_FILE}")"
    ids+=("${na}" "${nb}")
    printf '%s\n' "${ids[@]}" > "${ESTADO_DIR}/piloto.jobs"
    echo "${cid}" > "${ESTADO_DIR}/piloto.campaign"
    verde "Lanzados ${#ids[@]} jobs (campaign_id=${cid})"
    imprimir_seguimiento "el piloto" "${ids[@]}"
    echo "Para habilitar la campana, revise los CSV y marque la validacion a mano:"
    echo "  echo 'rel_l2 finito en las 3 rutas; manifest legible' > ${ESTADO_DIR}/piloto.validado"
}

cmd_campana() {
    bold "=== FASE: campana (${REPLICAS_CAMPANA} replicas energeticas) ==="
    exigir_exploratorio_ok >/dev/null
    exigir_piloto_ok
    [[ -n "${WALL_ENERGY:-}" ]] || morir "fije WALL_ENERGY con el Elapsed medido en el piloto."
    echo
    local cid="${CAMPAIGN_ID:-f4_pareto3d_$(date +%Y%m%d_%H%M%S)}"
    # --array=1-N%1: una tarea a la vez. Dos replicas concurrentes en el mismo
    # nodo comparten GPU y potencia de pared, y esta campana mide exactamente
    # eso; el %1 no es cortesia con la cola, es una condicion de validez.
    local a b
    a="$(enviar "campana bloque A (array 1-${REPLICAS_CAMPANA}%1)" \
        --array="1-${REPLICAS_CAMPANA}%1" --time="${WALL_ENERGY}" --job-name=f4_p3d_A \
        --export="$(exportes_comunes),RUN_KIND=energy,ITERS_LIST=${ITERS_CAMPANA},SPATIAL_COMP=off,KAHAN_LIST=off:on,CAMPAIGN_ID=${cid}" \
        "${SBATCH_FILE}")"
    b="$(enviar "campana bloque B (array 1-${REPLICAS_CAMPANA}%1)" \
        --array="1-${REPLICAS_CAMPANA}%1" --time="${WALL_ENERGY}" --job-name=f4_p3d_B \
        --export="$(exportes_comunes),RUN_KIND=energy,ITERS_LIST=${ITERS_CAMPANA},SPATIAL_COMP=on,CAMPAIGN_ID=${cid}" \
        "${SBATCH_FILE}")"
    printf '%s\n%s\n' "${a}" "${b}" > "${ESTADO_DIR}/campana.jobs"
    echo "${cid}" > "${ESTADO_DIR}/campana.campaign"
    verde "Lanzados los arrays ${a} y ${b} (campaign_id=${cid})"
    imprimir_seguimiento "la campana" "${a}" "${b}"
}

cmd_estado() {
    bold "=== Estado de la campana ==="
    for fase in exploratorio piloto campana; do
        local f="${ESTADO_DIR}/${fase}.job"
        [[ -f "${ESTADO_DIR}/${fase}.jobs" ]] && f="${ESTADO_DIR}/${fase}.jobs"
        if [[ -f "${f}" ]]; then
            echo
            echo "-- ${fase} --"
            local ids; ids="$(tr '\n' ',' < "${f}" | sed 's/,$//')"
            sacct -j "${ids}" --format=JobID,JobName%22,State,Elapsed,MaxRSS,ExitCode 2>/dev/null \
                || echo "  (sacct no disponible; ids: ${ids})"
        else
            echo "-- ${fase}: sin lanzar --"
        fi
    done
    echo
    [[ -f "${ESTADO_DIR}/piloto.validado" ]] \
        && verde "piloto validado: $(head -1 "${ESTADO_DIR}/piloto.validado")" \
        || echo "piloto NO validado (la campana esta bloqueada)"
}

case "${1:-}" in
    exploratorio) cmd_exploratorio ;;
    piloto)       cmd_piloto ;;
    campana)      cmd_campana ;;
    estado)       cmd_estado ;;
    *)
        echo "Uso: $0 {exploratorio|piloto|campana|estado}"
        echo
        echo "  exploratorio  1 job, ITERS=320, para MEDIR el walltime real"
        echo "  piloto        ${REPLICAS_PILOTO} replicas energeticas + 1 numerica (exige exploratorio COMPLETED)"
        echo "  campana       ${REPLICAS_CAMPANA} replicas en job array (exige piloto validado)"
        echo "  estado        sacct de todo lo lanzado"
        exit 2
        ;;
esac
