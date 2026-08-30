#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Fase 4 - Campana PILOTO de variabilidad experimental (Stencil 2D, 16384^2)
#
# Objetivo: caracterizar la DISPERSION (std, CV%, IC) de tiempo, GFLOP/s,
# energia GPU (NVML) y energia CPU (RAPL) sobre 5 replicas independientes.
# NO es una campana numerica: en ITERS=320/640 el error sale NaN por diseno
# (la referencia FP64 supera first_nonfinite hacia ~1045). Eso es esperado.
#
# Diseno: 2 precisiones (FP16/BF16, una sola invocacion via --tc both)
#       x 3 tratamientos (none / kahan_local / spatial)
#       x 2 horizontes (320 / 640)
#       x 5 replicas = 60 observaciones WMMA.
#
# Se ejecuta EN PACCA, desde cualquier punto del repositorio.
# ---------------------------------------------------------------------------
set -euo pipefail

COMMIT_ESPERADO="8d4e0e82efaf264ec2ad74b3f8bb172d614f4847"
REPLICAS="${REPLICAS:-5}"
NX_C=16384
NY_C=16384
ITERS_C="320 640"
CKPT_C=0
NCU_C=0
# Walltime derivado del costo OBSERVADO en los jobs 5200/5201 (16384^2,
# ITERS_LIST="320 640", CHECKPOINT_EVERY=0, mismo commit, nodo paccaA100):
#   invocacion ITERS=320 -> ~11.2 min CPU + ~0.6 min GPU  ~= 12 min
#   invocacion ITERS=640 -> ~22.0 min CPU + ~1.2 min GPU  ~= 23 min
#   bloque A (4 invocaciones) ~= 70 min + compilacion ~= 73 min -> 2:30:00 (+105%)
#   bloque B (2 invocaciones) ~= 34 min + compilacion ~= 36 min -> 1:30:00 (+150%)
# Ambos margenes superan con holgura el 30% exigido.
WALL_A="${WALL_A:-02:30:00}"
WALL_B="${WALL_B:-01:30:00}"

msg() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# --- 1. Procedencia: rama, commit, arbol limpio ----------------------------
ROOT="$(git rev-parse --show-toplevel)" || die "no estas dentro del repositorio git"
cd "${ROOT}"
RAMA="$(git rev-parse --abbrev-ref HEAD)"
COMMIT="$(git rev-parse HEAD)"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    msg "Cambios locales sobre archivos versionados (pueden alterar la campana):"
    git status --porcelain --untracked-files=no
    die "arbol sucio; no se lanza la campana. Revise o descarte los cambios usted mismo."
fi
if [[ "${COMMIT}" != "${COMMIT_ESPERADO}" ]]; then
    msg "AVISO: HEAD=${COMMIT}"
    msg "       El costo/walltime se calibro sobre ${COMMIT_ESPERADO} (jobs 5200/5201)."
    msg "       Exporte ACEPTAR_COMMIT_DISTINTO=1 si aun asi desea lanzar."
    [[ "${ACEPTAR_COMMIT_DISTINTO:-0}" == "1" ]] || die "commit distinto al calibrado"
fi

command -v sbatch >/dev/null || die "sbatch no disponible: ejecute este script en PACCA"

# --- 2. Directorio de campana NUEVO ----------------------------------------
# Profundidad 2 desde la raiz (igual que Fase_4/Stencil): el .cu incluye
# "tools/power_sampling.h" y "../../Fase_2/common.cuh", y el .sbatch busca
# "../../tools/common_ncu.sh". Ambas rutas resuelven desde Fase_4/<ID>/.
CAMPANA="${CAMPANA:-f4_piloto_variabilidad_$(date +%Y%m%d_%H%M%S)}"
CDIR="${ROOT}/Fase_4/${CAMPANA}"
[[ -e "${CDIR}" ]] && die "el directorio de campana ya existe: ${CDIR}"
mkdir -p "${CDIR}/logs" "${CDIR}/results" "${CDIR}/tools"

SRC="${ROOT}/Fase_4/Stencil"
cp "${SRC}/stencil_tensor_activation.cu" "${CDIR}/"
cp "${SRC}/run_stencil_tc.sbatch"        "${CDIR}/"
cp "${SRC}/tools/extract_csv.py"         "${CDIR}/tools/"
cp "${SRC}/tools/power_sampling.h"       "${CDIR}/tools/"
SHA_CU="$(sha256sum "${CDIR}/stencil_tensor_activation.cu" | awk '{print $1}')"

MAPA="${CDIR}/manifiesto_jobs.csv"
echo "job_id,job_name,replica,bloque,tratamientos,precisiones,iters,nx,ny,kahan_list,spatial_comp,checkpoint_every,run_ncu,walltime,invocaciones,obs_wmma" > "${MAPA}"

# --- 3. Manifiesto ANTES del primer sbatch ---------------------------------
cat > "${CDIR}/MANIFIESTO.md" <<EOF
# Campana piloto de variabilidad - Fase 4 - Stencil 2D

- **ID de campana**: \`${CAMPANA}\`
- **Timestamp de lanzamiento**: $(date -Iseconds)
- **Rama**: \`${RAMA}\`
- **Commit**: \`${COMMIT}\`
- **Estado del arbol**: limpio (sin cambios en archivos versionados)
- **sha256 del .cu congelado en la campana**: \`${SHA_CU}\`
- **Directorio**: \`Fase_4/${CAMPANA}/\`

## Proposito
Caracterizar la **variabilidad experimental** (std muestral, CV%, IC) de
tiempo, GFLOP/s, energia GPU (NVML) y energia CPU (RAPL). Las corridas previas
son una sola observacion por combinacion, de modo que no permiten estimar
error experimental. Campana **energetica y de reproducibilidad**, no numerica.

## Parametros fijos
| Parametro | Valor |
| --- | --- |
| NX x NY | ${NX_C} x ${NY_C} |
| ITERS | ${ITERS_C} |
| CHECKPOINT_EVERY | ${CKPT_C} (desactivados) |
| RUN_NCU | ${NCU_C} (NCU altera tiempo y energia: fuera) |
| FP64_GPU / CPU_FP64 | on / on (defaults, identicos a jobs 5200/5201) |
| Replicas | ${REPLICAS} |
| Exclusividad | \`--exclusive\` (ya en el .sbatch) |
| Frecuencia GPU | sin fijar (nvidia-smi -lgc denegado); ruido tratado por repeticion |

## Factores
- **Precision** (2): FP16 y BF16, acumulacion FP32, WMMA. Ambas en **una sola
  invocacion** del binario mediante \`--tc both\`.
- **Compensacion** (3, mutuamente excluyentes):
  | Tratamiento | Flags | Etiqueta \`route\` en CSV |
  | --- | --- | --- |
  | none | \`--kahan off --spatial-comp off\` | \`WMMA_FP16\` / \`WMMA_BF16\` con \`kahan=off\` |
  | kahan_local | \`--kahan on --spatial-comp off\` | \`WMMA_FP16\` / \`WMMA_BF16\` con \`kahan=on\` |
  | spatial | \`--kahan off --spatial-comp on\` | \`WMMA_FP16_SP\` / \`WMMA_BF16_SP\` |
- **Horizonte** (2): ITERS = 320 y 640.

## Estructura de jobs
Dos bloques por replica; \`kahan=on\` y \`spatial=on\` **no pueden coexistir**:
- **Bloque A** (\`SPATIAL_COMP=off\`, \`KAHAN_LIST="off on"\`): tratamientos
  *none* y *kahan_local*. 4 invocaciones (2 kahan x 2 iters), 8 obs WMMA.
- **Bloque B** (\`SPATIAL_COMP=on\`): el .sbatch **fuerza** \`KAHAN_LIST=off\`.
  Tratamiento *spatial*. 2 invocaciones (2 iters), 4 obs WMMA.

Triple barrera contra \`kahan=on + spatial=on\`: (1) el lanzador valida cada
combinacion antes de enviarla, (2) el .sbatch fuerza \`KAHAN_LIST=off\` cuando
\`SPATIAL_COMP=on\`, (3) el binario aborta esa combinacion en \`parse_args\`.

## Conteo esperado
- Jobs SLURM: **$((REPLICAS * 2))** (${REPLICAS} replicas x 2 bloques)
- Invocaciones del binario: **$((REPLICAS * 6))** (${REPLICAS} x (4 + 2))
- Observaciones WMMA: **$((REPLICAS * 12))** = 2 precisiones x 3 tratamientos x 2 horizontes x ${REPLICAS} replicas

Cada invocacion emite ademas filas \`CPU_FP32\`, \`CPU_FP64\`, \`GPU_FP32\` y
\`GPU_FP64\`. Se conservan, pero **no** cuentan dentro de las $((REPLICAS * 12)).

## Trazabilidad
La replica se identifica por \`job_id\` (columna ya presente en todos los CSV,
que **no se modifica**); el mapa job_id -> replica esta en
\`manifiesto_jobs.csv\`. Dentro de un job, cada observacion queda determinada
por \`(route, kahan, iters)\`: \`route\` codifica precision y compensacion
espacial (sufijo \`_SP\`), \`kahan\` distingue *none* de *kahan_local*.

## Advertencia de interpretacion
En 320 y 640 iteraciones **rel_l2 / rel_linf / max_abs saldran NaN**: la
solucion supera \`first_nonfinite\` (~1045 para la referencia FP64). Es
esperado, no es fallo del job, y **no invalida la energia medida**. La campana
energetica y la numerica son independientes.

## Salidas
- Logs SLURM: \`Fase_4/${CAMPANA}/logs/mixed_precision_stencil_tc_f3_<JOBID>.{out,err}\`
- Log crudo por job: \`Fase_4/${CAMPANA}/results/run_<JOBID>.log\`
- CSV por job: \`Fase_4/${CAMPANA}/results/{summary,energy,drift,store,horizon}_stencil_<JOBID>.csv\`
EOF

msg "=============================================================="
msg " Campana : ${CAMPANA}"
msg " Rama    : ${RAMA}"
msg " Commit  : ${COMMIT}"
msg " Destino : ${CDIR}"
msg "=============================================================="

# --- 4. Envio ---------------------------------------------------------------
cd "${CDIR}"

enviar() {
    local replica="$1" bloque="$2" spatial="$3" kahan_list="$4" wall="$5"
    local tratamientos="$6" ninv="$7" nobs="$8"

    # Barrera dura: kahan=on junto a spatial=on jamas debe enviarse.
    if [[ "${spatial}" == "on" && "${kahan_list}" == *on* ]]; then
        die "combinacion prohibida kahan=on + spatial=on (replica ${replica}, bloque ${bloque})"
    fi

    local nombre="f4p_r${replica}_${bloque}_sp${spatial}"
    local jid
    jid=$(
        export NX="${NX_C}" NY="${NY_C}" ITERS_LIST="${ITERS_C}" \
               CHECKPOINT_EVERY="${CKPT_C}" RUN_NCU="${NCU_C}" \
               SPATIAL_COMP="${spatial}" KAHAN_LIST="${kahan_list}" \
               FP64_GPU=on CPU_FP64=on \
               OP_MODE="${OP_MODE:-stress}" ALPHA="${ALPHA:-0.1875}" \
               CI_MODE="${CI_MODE:-legacy}" CI_P="${CI_P:-168}" \
               CI_AMPLITUDE="${CI_AMPLITUDE:-1.0}"
        sbatch --parsable --export=ALL --job-name="${nombre}" --time="${wall}" \
               run_stencil_tc.sbatch
    )
    printf '%s,%s,%d,%s,%s,FP16+BF16,%s,%d,%d,%s,%s,%d,%d,%s,%d,%d\n' \
        "${jid}" "${nombre}" "${replica}" "${bloque}" "${tratamientos}" \
        "${ITERS_C// /+}" "${NX_C}" "${NY_C}" "${kahan_list// /+}" "${spatial}" \
        "${CKPT_C}" "${NCU_C}" "${wall}" "${ninv}" "${nobs}" >> "${MAPA}"
    printf '  replica %d | bloque %s | %-22s | %s | JobID %s\n' \
        "${replica}" "${bloque}" "${tratamientos}" "${wall}" "${jid}"
}

msg ""
msg "Enviando jobs..."
for r in $(seq 1 "${REPLICAS}"); do
    enviar "${r}" A off "off on" "${WALL_A}" "none+kahan_local" 4 8
    enviar "${r}" B on  "off"    "${WALL_B}" "spatial"          2 4
done

NJOBS=$(( REPLICAS * 2 ))
msg ""
msg "=============================================================="
msg " ID de campana             : ${CAMPANA}"
msg " Jobs SLURM enviados       : ${NJOBS}"
msg " Invocaciones del binario  : $(( REPLICAS * 6 ))"
msg " Observaciones WMMA esperadas: $(( REPLICAS * 12 ))"
msg " Mapa job->replica         : ${MAPA}"
msg " Manifiesto                : ${CDIR}/MANIFIESTO.md"
msg "=============================================================="
msg ""
column -s, -t "${MAPA}"
msg ""
msg "Estado inicial en cola:"
squeue -u "$(whoami)" -o "%.10i %.22j %.9P %.8T %.10M %.10l %.6D %R"
