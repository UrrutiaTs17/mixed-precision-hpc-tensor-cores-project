#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Re-lanzamiento SELECTIVO de una replica de la campana de variabilidad.
#
#   ./Fase_4/relanzar_replica.sh <dir_campana> <replica> <bloque A|B>
#
# Ejemplo: la tarea 7 del array del bloque A salio TIMEOUT.
#   ./Fase_4/relanzar_replica.sh Fase_4/f4_variabilidad_15r_20260818_0200 7 A
#
# Archiva lo que quedo de la corrida fallida en <replica>/<bloque>/fallidos/
# (nunca lo borra: una corrida cortada a medias no debe mezclarse con la
# buena, y el analisis la contaria como duplicado) y reenvia SOLO esa tarea
# con --array=<replica>, sin tocar el resto de la campana.
# ---------------------------------------------------------------------------
set -euo pipefail

CDIR="${1:?uso: relanzar_replica.sh <dir_campana> <replica> <bloque A|B>}"
REP="${2:?falta el numero de replica}"
BLQ="${3:?falta el bloque (A o B)}"

CDIR="$(cd "${CDIR}" && pwd)"
[[ -f "${CDIR}/campana.env" ]] || { echo "ERROR: ${CDIR} no es un dir de campana" >&2; exit 1; }
[[ "${BLQ}" == "A" || "${BLQ}" == "B" ]] || { echo "ERROR: bloque debe ser A o B" >&2; exit 1; }
[[ "${REP}" =~ ^[0-9]+$ ]] || { echo "ERROR: replica debe ser un numero" >&2; exit 1; }

# shellcheck source=/dev/null
source "${CDIR}/campana.env"
TD="${CDIR}/replicas/$(printf 'r%02d' "${REP}")/${BLQ}"
[[ -d "${TD}" ]] || { echo "ERROR: no existe ${TD}" >&2; exit 1; }

# Walltime: el mismo con el que se envio el array, leido del manifiesto.
WALL="$(awk -F, -v b="${BLQ}" -v r="${REP}" \
        'NR>1 && $3==b && $5==r {print $15; exit}' "${CDIR}/manifiesto_jobs.csv")"
[[ -n "${WALL}" ]] || { echo "ERROR: no encuentro el walltime del bloque ${BLQ} en el manifiesto" >&2; exit 1; }

SELLO="$(date +%Y%m%d_%H%M%S)"
if compgen -G "${TD}/results/*" >/dev/null || compgen -G "${TD}/telemetria_*" >/dev/null; then
    mkdir -p "${TD}/fallidos/${SELLO}"
    for patron in "${TD}"/results/* "${TD}"/telemetria_* "${TD}"/tarea_*.env; do
        [[ -e "${patron}" ]] && mv "${patron}" "${TD}/fallidos/${SELLO}/"
    done
    echo "Salidas anteriores archivadas en ${TD}/fallidos/${SELLO}/"
fi
mkdir -p "${TD}/results" "${TD}/logs"

cd "${CDIR}"
JID=$(sbatch --parsable --export=ALL \
        --array="${REP}" \
        --time="${WALL}" \
        "array_bloque${BLQ}.sbatch")
echo "Replica ${REP}, bloque ${BLQ}, relanzada como ${JID}_${REP} (walltime ${WALL})"
echo "  log: ${CDIR}/logs/f4var_${BLQ}_${JID}_${REP}.out"
echo "  dir: ${TD}"
echo
echo "NOTA: el nuevo JobID no esta en manifiesto_jobs.csv (que registra el envio"
echo "      original). La replica sigue siendo rastreable por su directorio y por"
echo "      la fila que la propia tarea anade a ${CDIR}/ejecuciones.csv."
squeue -u "$(whoami)" -j "${JID}" -o "%.18i %.20j %.9P %.8T %.10M %.10l %R" || true
