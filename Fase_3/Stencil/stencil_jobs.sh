#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Fase 3 - Campana de CIERRE (Stencil 2D): drift numerico, Kahan, compensacion
# espacial.
#
# Los jobs 4511-4514 (4 ago) predatan la ruta GPU_FP64 (commit 9220170,
# 11 ago) y la ruta CPU_FP64 cronometrada (commit 8d4e0e8, 16 ago): su
# esquema de CSV ya no es compatible con tools/extract_csv.py actual. Esta
# campana los relanza con el esquema vigente. Los jobs historicos se citan
# aqui solo como antecedente de POR QUE se relanza, nunca como fuente de
# datos validos para esta campana.
#
# Dos sub-campanas independientes, cada una invocable por separado:
#
#   --sub-a-exploratorio  1 job (Bloque A de run_stencil_tc.sbatch), unico
#                         objetivo: medir el walltime real con
#                         CHECKPOINT_EVERY=5 a 16384^2 (no medido antes).
#                         Manual, previa, NUNCA se encadena a --todo.
#   --sub-a-completa      2 jobs (Bloque A + Bloque B de run_stencil_tc.sbatch)
#   --sub-b-horizonte     6 jobs (3 tamanos x 2 configs de
#                         run_stencil_horizon.sbatch)
#   --todo                --sub-a-completa + --sub-b-horizonte (8 jobs)
#
# Cierre 2 (jobs 6325-6333 ya respondieron rel_l2 y n*; estas tres sub-campanas
# cubren los tres huecos que quedaban abiertos, con el MISMO fuente congelado):
#
#   --sub-c-energia       3 jobs (2 de energia con ventana valida + 1 control de
#                         independencia energia/fraccion de rutas finitas)
#   --sub-d-replicas      5 jobs (2 replicas de Bloque A + 3 de Bloque B) para
#                         poder reportar Cv de t_iter_ms
#   --sub-e-kahan-horizonte 1 job (Kahan local sobre el horizonte a 16384^2)
#   --validacion-final    --sub-c-energia + --sub-d-replicas
#                         + --sub-e-kahan-horizonte (9 jobs)
#
# Se ejecuta EN PACCA, desde cualquier punto del repositorio.
# ---------------------------------------------------------------------------
set -euo pipefail

msg() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

uso() {
    cat <<'USO_EOF'
Uso: stencil_jobs.sh <flag>

  --sub-a-exploratorio   1 job: Bloque A (SPATIAL_COMP=off, KAHAN_LIST="off on")
                         de run_stencil_tc.sbatch, RUN_NCU=0, --time 04:00:00.
                         Solo mide walltime real a 16384^2. No lanza nada mas.

  --sub-a-completa       2 jobs (Bloque A + Bloque B) de run_stencil_tc.sbatch,
                         RUN_NCU=0, --time=WALL_TC (env, default 04:00:00).

  --sub-b-horizonte      6 jobs (NX=NY en 4096/8192/16384 x {baseline,espacial})
                         de run_stencil_horizon.sbatch, --time por defecto del
                         propio script (02:00:00).

  --todo                 --sub-a-completa + --sub-b-horizonte (8 jobs).
                         NUNCA incluye --sub-a-exploratorio.

--- Cierre 2 (huecos que dejaron abiertos los jobs 6325-6333) ---

  --sub-c-energia        3 jobs de run_stencil_tc.sbatch con CHECKPOINT_EVERY=0
                         (un solo tramo => el umbral de window_reliable baja a
                         0.5 s de ventana total):
                           energia_off  ITERS_LIST=400  SPATIAL_COMP=off
                           energia_on   ITERS_LIST=400  SPATIAL_COMP=on
                           control_1000 ITERS_LIST=1000 SPATIAL_COMP=off
                         --time=WALL_ENERGIA (default 02:00:00) para los dos de
                         400 iters y WALL_CONTROL (default 04:00:00) para el de
                         1000.

  --sub-d-replicas       5 jobs de run_stencil_tc.sbatch con TODO en default
                         (ITERS_LIST="10 50 100 120", CHECKPOINT_EVERY=5):
                         2 replicas de Bloque A + 3 de Bloque B,
                         --time=WALL_TC (env, default 04:00:00).

  --sub-e-kahan-horizonte 1 job de run_stencil_horizon.sbatch a NX=NY=16384 con
                         SPATIAL_COMP=off y KAHAN_LIST en su default ("off on"):
                         la rama kahan=on del horizonte nunca se corrio.
                         --time por defecto del script (02:00:00, ya calibrado
                         para las dos pasadas de KAHAN).

  --validacion-final     --sub-c-energia + --sub-d-replicas
                         + --sub-e-kahan-horizonte (9 jobs).

Sin flag: imprime este uso y sale sin lanzar nada.
USO_EOF
}

MODO="${1:-}"
case "${MODO}" in
    --sub-a-exploratorio|--sub-a-completa|--sub-b-horizonte|--todo) ;;
    --sub-c-energia|--sub-d-replicas|--sub-e-kahan-horizonte|--validacion-final) ;;
    ""|-h|--help)
        uso
        exit 0
        ;;
    *)
        uso
        die "flag desconocida: ${MODO}"
        ;;
esac

WALL_TC="${WALL_TC:-04:00:00}"
# Sub-campana C: a 16384^2 el job 6325 midio 00:31:52 para 280 iters x 2 pasadas
# de KAHAN (~3.4 s por iteracion-pasada, incluyendo rutas CPU y checkpoints).
# 400 iters x 2 pasadas ~= 46 min y 1000 x 2 ~= 1 h 54 min: estos limites dejan
# margen >2x sobre ambas estimaciones.
WALL_ENERGIA="${WALL_ENERGIA:-02:00:00}"
WALL_CONTROL="${WALL_CONTROL:-04:00:00}"

# --- 1. Procedencia: rama, commit, arbol ------------------------------------
ROOT="$(git rev-parse --show-toplevel)" || die "no estas dentro del repositorio git"
cd "${ROOT}"
RAMA="$(git rev-parse --abbrev-ref HEAD)"
COMMIT="$(git rev-parse HEAD)"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    msg "Cambios locales sobre archivos VERSIONADOS (pueden alterar la campana):"
    git status --porcelain --untracked-files=no
    die "arbol sucio; no se lanza. Revise o descarte los cambios usted mismo (este script no toca su arbol)."
fi

# Los fuentes que la campana congela deben estar limpios ademas de versionados:
# un fichero sin seguimiento ahi dentro podria cambiar lo que se compila.
FUENTES=(Fase_3/Stencil Fase_2/common.cuh tools/common_ncu.sh)

# ...salvo los ARTEFACTOS que toda corrida previa deja en Fase_3/Stencil (el
# binario compilado, logs/, results/, reportes de ncu, campanas ya generadas
# por este mismo script y el propio lanzador antes de su primer commit): se
# listan, pero no abortan.
es_artefacto() {
    case "$1" in
        Fase_3/Stencil/stencil_tc|\
        Fase_3/Stencil/logs|Fase_3/Stencil/logs/|Fase_3/Stencil/logs/*|\
        Fase_3/Stencil/results|Fase_3/Stencil/results/|Fase_3/Stencil/results/*|\
        Fase_3/Stencil/stencil_jobs.sh|\
        Fase_3/Stencil/f3_cierre_*|Fase_3/Stencil/f3_cierre_*/*|\
        *.out|*.err|*.log|*.ncu-rep|*.nsys-rep|*.qdrep|*__pycache__*) return 0 ;;
    esac
    return 1
}

SUCIO_FUENTES=""
ARTEFACTOS=""
while IFS= read -r _linea; do
    [[ -z "${_linea}" ]] && continue
    _estado="${_linea:0:2}"
    _ruta="${_linea:3}"
    _ruta="${_ruta#\"}"; _ruta="${_ruta%\"}"
    if [[ "${_estado}" == "??" ]] && es_artefacto "${_ruta}"; then
        ARTEFACTOS+="${_linea}"$'\n'
    else
        SUCIO_FUENTES+="${_linea}"$'\n'
    fi
done < <(git status --porcelain -- "${FUENTES[@]}")

if [[ -n "${SUCIO_FUENTES}" ]]; then
    msg "Cambios (o ficheros sin seguimiento) en los fuentes de la campana:"
    printf '%s' "${SUCIO_FUENTES}"
    die "los fuentes que se congelan no estan limpios; no se lanza."
fi
if [[ -n "${ARTEFACTOS}" ]]; then
    msg "[i] Artefactos de corridas anteriores en los fuentes (no se congelan, no bloquean):"
    printf '%s' "${ARTEFACTOS}" | sed 's/^/      /'
fi

command -v sbatch >/dev/null || die "sbatch no disponible: ejecute este script en PACCA"

# --- 2. Directorio de campana NUEVO (jamas sobrescribe) ---------------------
CAMPANA="${CAMPANA:-f3_cierre_$(date +%Y%m%d_%H%M%S)}"
CDIR="${ROOT}/Fase_3/Stencil/${CAMPANA}"
[[ -e "${CDIR}" ]] && die "el directorio de campana ya existe: ${CDIR}"

SRC="${ROOT}/Fase_3/Stencil"
mkdir -p "${CDIR}/jobs/Fase_2" "${CDIR}/jobs/tools"

# Fuentes congelados en el nivel "jobs/": el .cu incluye
# "../../Fase_2/common.cuh" (relativo al PROPIO .cu, que vive en
# jobs/<grupo>/<nombre>/) y run_stencil_tc.sbatch busca
# "../../tools/common_ncu.sh" (relativo a su cwd, el mismo directorio). Ambas
# rutas resuelven exactamente en jobs/.
cp "${ROOT}/Fase_2/common.cuh"   "${CDIR}/jobs/Fase_2/"
cp "${ROOT}/tools/common_ncu.sh" "${CDIR}/jobs/tools/"

SHA_CU="$(sha256sum "${SRC}/stencil_tensor_activation.cu" | awk '{print $1}')"

MAPA="${CDIR}/manifiesto_jobs.csv"
echo "sub_campana,grupo,nombre,script,export_vars,walltime,job_id,dir_job" > "${MAPA}"

# Un directorio de trabajo por job: cada uno compila su propio binario y
# escribe su propio logs/ y results/. Ninguno comparte fichero con otro.
preparar_dir_job() {
    local jobdir="$1" script="$2"
    mkdir -p "${jobdir}/tools" "${jobdir}/logs" "${jobdir}/results"
    cp "${SRC}/stencil_tensor_activation.cu" "${jobdir}/"
    cp "${SRC}/${script}"                    "${jobdir}/"
    cp "${SRC}/tools/extract_csv.py"         "${jobdir}/tools/"
    cp "${SRC}/tools/power_sampling.h"       "${jobdir}/tools/"
    cp "${ROOT}/tools/common_ncu.sh"         "${jobdir}/"
    [[ "$(sha256sum "${jobdir}/stencil_tensor_activation.cu" | awk '{print $1}')" == "${SHA_CU}" ]] \
        || die "la copia del fuente en ${jobdir} no coincide con el original"
}

# --- 3. MANIFIESTO, escrito ANTES del primer sbatch -------------------------
TS_LANZAMIENTO="$(date -Iseconds)"
cat > "${CDIR}/MANIFIESTO.md" <<MAN_EOF
# Campana de cierre - Fase 3 - Stencil 2D

- **ID de campana**: \`${CAMPANA}\`
- **Timestamp de lanzamiento**: ${TS_LANZAMIENTO}
- **Rama**: \`${RAMA}\`
- **Commit**: \`${COMMIT}\`
- **Estado del arbol**: limpio en archivos versionados y en los fuentes congelados
- **sha256 del .cu congelado**: \`${SHA_CU}\`
- **Directorio**: \`Fase_3/Stencil/${CAMPANA}/\`
- **Modo invocado**: \`${MODO}\`

## Por que se relanza
Los jobs 4511-4514 (4 ago) predatan la ruta GPU_FP64 (commit 9220170, 11 ago)
y la ruta CPU_FP64 cronometrada (commit 8d4e0e8, 16 ago): su esquema de CSV
ya no es compatible con \`tools/extract_csv.py\` actual. Se citan aqui solo
como antecedente historico, **no** como fuente de datos validos para esta
campana.

## Sub-campana A - run_stencil_tc.sbatch (16384^2, ventana finita ITERS 10-120)
| Job | Bloque | SPATIAL_COMP | KAHAN_LIST | RUN_NCU | Proposito |
| --- | --- | --- | --- | --- | --- |
| a_exploratorio/unico | A | off | off on | 0 | Medir walltime real (CHECKPOINT_EVERY=5, 16384^2), no medido antes |
| a_completa/bloqueA | A | off | off on | 0 | none + kahan_local |
| a_completa/bloqueB | B | on (fuerza KAHAN_LIST=off) | off | 0 | espacial |

Parametros no tocados en sub-campana A (defaults del script):
ITERS_LIST="10 50 100 120", CHECKPOINT_EVERY=5, FP64_GPU=on, CPU_FP64=on.

## Sub-campana B - run_stencil_horizon.sbatch (horizonte de overflow, sin Kahan local)
| Job | NX=NY | SPATIAL_COMP | KAHAN_LIST |
| --- | --- | --- | --- |
| b_horizonte/nx4096_off  | 4096  | off | off |
| b_horizonte/nx4096_on   | 4096  | on  | (forzado off) |
| b_horizonte/nx8192_off  | 8192  | off | off |
| b_horizonte/nx8192_on   | 8192  | on  | (forzado off) |
| b_horizonte/nx16384_off | 16384 | off | off |
| b_horizonte/nx16384_on  | 16384 | on  | (forzado off) |

Kahan local se descarto en Fase 3 (indistinguible de no compensar, ver
comentarios de run_stencil_horizon.sbatch): no se reproduce esa rama aqui.
ITERS=1200, CHECKPOINT_EVERY=20 y --time=02:00:00 quedan en el default del
propio script (no se sobreescriben).

## Sub-campana C - energia con ventana valida (run_stencil_tc.sbatch, 16384^2)
| Job | ITERS_LIST | CHECKPOINT_EVERY | SPATIAL_COMP | Walltime | Proposito |
| --- | --- | --- | --- | --- | --- |
| c_energia/energia_off  | 400  | 0 | off | ${WALL_ENERGIA} | Energia sin compensar |
| c_energia/energia_on   | 400  | 0 | on  | ${WALL_ENERGIA} | Energia con compensacion espacial |
| c_energia/control_1000 | 1000 | 0 | off | ${WALL_CONTROL} | Control de independencia energia / rutas finitas |

\`CHECKPOINT_EVERY=0\` desactiva el particionado en tramos: con un solo tramo
\`window_reliable\` deja de exigir \`time_total_s >= 0.5 * gpu_segment_count\` y
el umbral baja a >=0.5 s de ventana total. A 16384^2 la ruta GPU mas rapida es
~2.78 ms/iter, asi que 400 iters dan ~1.11 s (margen ~2.2x sobre el umbral).

**Esperado, no es un bug:** sin checkpoints no hay snapshots de error, asi que
estos tres jobs NO producen \`drift_stencil\` ni \`store_stencil\` utiles. Son
jobs dedicados a energia.

El control a ITERS=1000 existe para *demostrar* (en vez de asumir) que el
consumo del kernel es memory-bound e independiente del contenido numerico: la
fraccion de rutas WMMA aun finitas cae de ~7% (a 400 iters) a ~2.8% (a 1000),
y si \`energy_gpu_j_per_iter\` no cambia significativamente frente a
\`energia_off\`, la independencia queda medida.

## Sub-campana D - replicas para coeficiente de variacion (run_stencil_tc.sbatch)
| Job | Bloque | SPATIAL_COMP | KAHAN_LIST | Walltime |
| --- | --- | --- | --- | --- |
| d_replicas/repA_r1 | A | off | off on | ${WALL_TC} |
| d_replicas/repA_r2 | A | off | off on | ${WALL_TC} |
| d_replicas/repB_r1 | B | on  | (forzado off) | ${WALL_TC} |
| d_replicas/repB_r2 | B | on  | (forzado off) | ${WALL_TC} |
| d_replicas/repB_r3 | B | on  | (forzado off) | ${WALL_TC} |

Todo lo no listado queda en el default del script (ITERS_LIST="10 50 100 120",
CHECKPOINT_EVERY=5, FP64_GPU=on, CPU_FP64=on), identico a los jobs 6325/6326
(Bloque A) y 6327 (Bloque B), para que las replicas sean comparables con ellos.
Bloque A pasa asi de 2 a 4 corridas y Bloque B de 1 a 4: >=3 replicas
independientes en ambos, minimo para reportar Cv de \`t_iter_ms\` (y de energia
donde la ventana lo permita).

## Sub-campana E - Kahan local sobre el horizonte (run_stencil_horizon.sbatch)
| Job | NX=NY | SPATIAL_COMP | KAHAN_LIST |
| --- | --- | --- | --- |
| e_kahan_horizonte/nx16384_kahan | 16384 | off | off on (default del script) |

Los jobs 6328/6330/6332 se lanzaron con \`KAHAN_LIST=off\` explicito, asi que la
rama kahan=on nunca se corrio en sub-campana B. Se sabe que Kahan local no
mueve el **error**; falta saber si mueve el **horizonte** n\*. Aqui NO se pasa
\`KAHAN_LIST\`: se deja el default "off on" para obtener las dos pasadas.
ITERS=1200, CHECKPOINT_EVERY=20 y --time=02:00:00 quedan en el default del
script (ese limite ya esta calibrado para las dos pasadas de KAHAN).

## Job IDs (anadidos tras el envio)

MAN_EOF

msg "=============================================================="
msg " Campana : ${CAMPANA}"
msg " Modo    : ${MODO}"
msg " Rama    : ${RAMA}"
msg " Commit  : ${COMMIT}"
msg " Destino : ${CDIR}"
msg " Manifiesto escrito ANTES del primer sbatch."
msg "=============================================================="
msg ""

# --- 4. Envio de jobs --------------------------------------------------------
JIDS=()

# kahan_list vacio ("") => NO se exporta KAHAN_LIST y el script usa su default.
# extra (7mo arg, opcional) se anade tal cual al final de --export.
lanzar_tc() {
    local sub="$1" grupo="$2" nombre="$3" spatial="$4" kahan_list="$5" wall="$6"
    local extra="${7:-}"
    local jobdir="${CDIR}/jobs/${grupo}/${nombre}"
    preparar_dir_job "${jobdir}" "run_stencil_tc.sbatch"

    local export_vars="ALL,RUN_NCU=0,SPATIAL_COMP=${spatial}"
    [[ -n "${kahan_list}" ]] && export_vars="${export_vars},KAHAN_LIST=${kahan_list}"
    [[ -n "${extra}"      ]] && export_vars="${export_vars},${extra}"
    local jid
    jid=$(cd "${jobdir}" && sbatch --parsable \
            --export="${export_vars}" \
            --time="${wall}" \
            run_stencil_tc.sbatch)
    JIDS+=("${jid}")

    printf '%s,%s,%s,%s,"%s",%s,%s,%s\n' \
        "${sub}" "${grupo}" "${nombre}" "run_stencil_tc.sbatch" \
        "${export_vars}" "${wall}" "${jid}" "jobs/${grupo}/${nombre}" >> "${MAPA}"

    {
        echo "| ${sub} | ${nombre} | \`${jid}\` | ${wall} |"
    } >> "${CDIR}/MANIFIESTO.md"

    msg "  [${sub}] ${nombre}: JobID ${jid}  (--time ${wall}, export ${export_vars})"
}

lanzar_horizon() {
    local sub="$1" grupo="$2" nombre="$3" nx="$4" spatial="$5" kahan_off_explicito="$6"
    local jobdir="${CDIR}/jobs/${grupo}/${nombre}"
    preparar_dir_job "${jobdir}" "run_stencil_horizon.sbatch"

    local export_vars="ALL,NX=${nx},NY=${nx},SPATIAL_COMP=${spatial}"
    if [[ "${kahan_off_explicito}" == "1" ]]; then
        export_vars="${export_vars},KAHAN_LIST=off"
    fi
    local jid
    jid=$(cd "${jobdir}" && sbatch --parsable \
            --export="${export_vars}" \
            run_stencil_horizon.sbatch)
    JIDS+=("${jid}")

    printf '%s,%s,%s,%s,"%s",%s,%s,%s\n' \
        "${sub}" "${grupo}" "${nombre}" "run_stencil_horizon.sbatch" \
        "${export_vars}" "02:00:00(default)" "${jid}" "jobs/${grupo}/${nombre}" >> "${MAPA}"

    {
        echo "| ${sub} | ${nombre} | \`${jid}\` | 02:00:00 (default del script) |"
    } >> "${CDIR}/MANIFIESTO.md"

    msg "  [${sub}] ${nombre}: JobID ${jid}  (export ${export_vars})"

    sleep 2
}

{
    echo "| Sub-campana | Job | JobID | Walltime |"
    echo "| --- | --- | --- | --- |"
} >> "${CDIR}/MANIFIESTO.md"

sub_a_exploratorio() {
    msg "Enviando sub-a-exploratorio (1 job, Bloque A, --time 04:00:00)..."
    lanzar_tc "A-exploratorio" "a_exploratorio" "unico" "off" "off on" "04:00:00"
}

sub_a_completa() {
    msg "Enviando sub-a-completa (2 jobs, Bloque A + Bloque B, --time ${WALL_TC})..."
    lanzar_tc "A-completa" "a_completa" "bloqueA" "off" "off on" "${WALL_TC}"
    lanzar_tc "A-completa" "a_completa" "bloqueB" "on"  "off"    "${WALL_TC}"
}

sub_b_horizonte() {
    msg "Enviando sub-b-horizonte (6 jobs, 3 tamanos x 2 configs)..."
    for NX_H in 4096 8192 16384; do
        lanzar_horizon "B-horizonte" "b_horizonte" "nx${NX_H}_off" "${NX_H}" "off" "1"
        lanzar_horizon "B-horizonte" "b_horizonte" "nx${NX_H}_on"  "${NX_H}" "on"  "0"
    done
}

# --- Cierre 2 ---------------------------------------------------------------
# CHECKPOINT_EVERY=0: un solo tramo de energia, umbral de window_reliable a
# 0.5 s totales en vez de 0.5 s x numero de tramos. Sin checkpoints no hay
# snapshots de error: drift_/store_stencil salen vacios a proposito.
sub_c_energia() {
    msg "Enviando sub-c-energia (3 jobs: 2 energia + 1 control)..."
    lanzar_tc "C-energia" "c_energia" "energia_off"  "off" "" "${WALL_ENERGIA}" \
        "CHECKPOINT_EVERY=0,ITERS_LIST=400"
    lanzar_tc "C-energia" "c_energia" "energia_on"   "on"  "" "${WALL_ENERGIA}" \
        "CHECKPOINT_EVERY=0,ITERS_LIST=400"
    lanzar_tc "C-energia" "c_energia" "control_1000" "off" "" "${WALL_CONTROL}" \
        "CHECKPOINT_EVERY=0,ITERS_LIST=1000"
}

# Replicas byte a byte de la configuracion de 6326 (Bloque A) y 6327 (Bloque B):
# solo cambia el directorio de trabajo, para poder estimar Cv de t_iter_ms.
sub_d_replicas() {
    msg "Enviando sub-d-replicas (5 jobs: 2 de Bloque A + 3 de Bloque B, --time ${WALL_TC})..."
    for R in 1 2; do
        lanzar_tc "D-replicas" "d_replicas" "repA_r${R}" "off" "off on" "${WALL_TC}"
    done
    for R in 1 2 3; do
        lanzar_tc "D-replicas" "d_replicas" "repB_r${R}" "on"  "off"    "${WALL_TC}"
    done
}

# SPATIAL_COMP=off SIN KAHAN_LIST explicito: es la unica via para que el script
# use su default "off on" y corra por fin la rama kahan=on del horizonte.
sub_e_kahan_horizonte() {
    msg "Enviando sub-e-kahan-horizonte (1 job, NX=NY=16384, KAHAN_LIST en default)..."
    lanzar_horizon "E-kahan-horizonte" "e_kahan_horizonte" "nx16384_kahan" \
        "16384" "off" "0"
}

case "${MODO}" in
    --sub-a-exploratorio)
        sub_a_exploratorio
        ;;
    --sub-a-completa)
        sub_a_completa
        ;;
    --sub-b-horizonte)
        sub_b_horizonte
        ;;
    --todo)
        sub_a_completa
        sub_b_horizonte
        ;;
    --sub-c-energia)
        sub_c_energia
        ;;
    --sub-d-replicas)
        sub_d_replicas
        ;;
    --sub-e-kahan-horizonte)
        sub_e_kahan_horizonte
        ;;
    --validacion-final)
        sub_c_energia
        sub_d_replicas
        sub_e_kahan_horizonte
        ;;
esac

# --- 5. Reporte ---------------------------------------------------------------
msg ""
msg "=============================================================="
msg " ID de campana     : ${CAMPANA}"
msg " Modo              : ${MODO}"
msg " Jobs enviados      : ${#JIDS[@]}"
msg " JobIDs             : ${JIDS[*]}"
msg " Manifiesto         : ${CDIR}/MANIFIESTO.md"
msg " Mapa de jobs        : ${MAPA}"
msg "=============================================================="
msg ""
msg "Estado inicial en cola:"
squeue -u "$(whoami)" -o "%.18i %.20j %.9P %.8T %.10M %.10l %.6D %R" || true
