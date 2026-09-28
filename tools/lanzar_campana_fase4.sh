#!/bin/bash
# tools/lanzar_campana_fase4.sh -- encola la campana corregida de Fase 4 para
# Stencil con el operador difusivo alpha=3/16 (matriz A8 de DECISIONS.md).
#
# SOLO ENCOLA (sbatch): no ejecuta nada pesado, se puede correr desde el login.
# Cada job corre con CAMPANA_STRICT=1 (abort si el esquema/alpha/comp_scheme no son
# los pedidos, CPU_FP64 una vez por celda, audit_coverage.py al final) y con
# ARCHIVE_DIR en el scratch local del nodo (una sola copia al final a OUT_BASE).
#
# Dos GRUPOS, porque Kahan local y spatial son alternativos (DECISIONS.md S6):
#   sp  : SPATIAL_COMP=on  -> comp_scheme=spatial (+ referencias), K en 0 1 8 32.
#   off : SPATIAL_COMP=off, KAHAN_LIST="off on" -> none y kahan_local, solo K=0
#         (el binario rechaza el ancla K>0 sin --spatial-comp on).
# CPU_FP64 (referencia de costo, no depende de K ni de la compensacion) se pide
# UNA vez por celda (nx,ny,iters) en todo el bloque: solo en el grupo `sp`.
#
# Por grupo, en orden (dependencia afterany: un fallo NO frena los siguientes, el
# gate lo reporta):
#   num_corta     : RUN_KIND=numeric, iters 10 50 100 120 (exploratorio: finitud/first_nonfinite)
#   num_S / num_L : RUN_KIND=numeric a la ventana energetica (4096 -> 4000; 8192,16384 -> 1500)
#                   para que exista error a los MISMOS iters que la energia (assert A2)
#   en_S / en_L   : RUN_KIND=energy en esas ventanas
#
# EXTENSION DE K (DECISIONS.md, errata 8): ademas del grupo Stencil de arriba, esta
# tanda puede encolar
#   stencil kext : SPATIAL_COMP=on, ANCHOR_LIST="2 4 16 64 128" (K=0,1,8,32 ya estan en `sp`),
#                  CPU_FP64=off (la referencia ya salio en `sp`), mismas 5 pasadas.
#   gemm / conv  : campana completa con ANCHOR_LIST="0 1 2 5 10 20 40" en UN solo barrido
#                  (numerica 20 40 80 + energia dedicada por cola; el pase de energia de
#                  GEMM/Conv ya emite el error final, no hacen falta pasadas a la ventana).
# Se elige con GROUPS_RUN (sp off kext) y KERNELS_RUN (stencil gemm conv). Con
# AFTER_JOB=<id> la cadena espera a ese job (afterany).
#
# Uso (desde la raiz del checkout de la campana en PACCA):
#   CI_MODE=monomode OUT_BASE=$HOME/campana_v2_out/fase4_a316 bash tools/lanzar_campana_fase4.sh
#   DRY_RUN=1 CI_MODE=monomode OUT_BASE=... bash tools/lanzar_campana_fase4.sh   # solo imprime
#   GROUPS_RUN="sp" restringe a un grupo.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CI_MODE:?defina CI_MODE=monomode|legacy (decision de diseno: ver DECISIONS.md)}"
: "${OUT_BASE:?defina OUT_BASE=directorio propio de esta campana (nunca results/ de 7145)}"
case "${OUT_BASE%/}" in results|*/results) echo "ERROR: OUT_BASE no puede terminar en results/" >&2; exit 2;; esac
DRY_RUN="${DRY_RUN:-0}"
GROUPS_RUN="${GROUPS_RUN:-sp off}"
KERNELS_RUN="${KERNELS_RUN:-stencil}"
AFTER_JOB="${AFTER_JOB:-}"
K_STENCIL_EXT="${K_STENCIL_EXT:-2 4 16 64 128}"
K_CHAINED_ALL="${K_CHAINED_ALL:-0 1 2 5 10 20 40}"
ALPHA_CAMPANA="0.1875"   # 3/16

# Ventanas energeticas por cola (DECISIONS.md S2): tamano -> iters.
TIER_S_NX="4096";        TIER_S_ITERS="4000"
TIER_L_NX="8192 16384";  TIER_L_ITERS="1500"
NUM_ITERS_CORTA="10 50 100 120"
CKPT_CORTA="5"
CKPT_VENTANA_S="500"     # 4000 iters -> 8 puntos de drift; el error final siempre se emite
CKPT_VENTANA_L="250"     # 1500 iters -> 6 puntos

KDIR="${REPO_ROOT}/Fase_4/Stencil"
LAST_JOB="${AFTER_JOB}"
mkdir -p "${OUT_BASE}"
JOBS_TSV="${OUT_BASE}/JOBS.tsv"
[[ "${DRY_RUN}" == "1" ]] || [[ -s "${JOBS_TSV}" ]] || printf 'fecha\tgrupo\tnombre\tjob_id\texport\n' > "${JOBS_TSV}"

encolar() {   # encolar <grupo> <nombre> <VAR=val ...>   (KSCRIPT/KDIR_RUN/OPFLAGS los fija el bucle)
    local grupo="$1" nombre="$2"; shift 2
    local exp="ALL,CAMPANA_STRICT=1,TC_FORMAT=both${OPFLAGS}"
    exp+=",ARCHIVE_DIR=/tmp/campana_v2_${grupo}_${nombre},FINAL_RESULTS_DIR=${OUT_BASE%/}/${grupo}_${nombre}"
    local kv; for kv in "$@"; do exp+=",${kv}"; done
    local dep=()
    if [[ -n "${LAST_JOB}" && "${LAST_JOB}" != "DRY" ]]; then dep=(--dependency="afterany:${LAST_JOB}"); fi
    local cmd=(sbatch --parsable --job-name="v2_${grupo}_${nombre}" ${dep[@]+"${dep[@]}"} --export="${exp}" "${KSCRIPT}")
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY_RUN] (cd ${KDIR_RUN} && ${cmd[*]})"
        LAST_JOB="DRY"
        return
    fi
    local jid; jid="$(cd "${KDIR_RUN}" && "${cmd[@]}")"
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${grupo}" "${nombre}" "${jid}" "${exp}" >> "${JOBS_TSV}"
    echo "encolado ${grupo}/${nombre}: job ${jid}"
    LAST_JOB="${jid}"
}

for kern in ${KERNELS_RUN}; do
  case "${kern}" in
  stencil)
    KDIR_RUN="${KDIR}"; KSCRIPT="run_stencil_tc.sbatch"
    OPFLAGS=",OP_MODE=diffusive,ALPHA=${ALPHA_CAMPANA},CI_MODE=${CI_MODE}"
    for g in ${GROUPS_RUN}; do
        case "${g}" in
            sp)   COMUN=(SPATIAL_COMP=on "ANCHOR_LIST=0 1 8 32" CPU_FP64=on) ;;
            off)  COMUN=(SPATIAL_COMP=off "KAHAN_LIST=off on" ANCHOR_LIST=0 CPU_FP64=off) ;;
            kext) COMUN=(SPATIAL_COMP=on "ANCHOR_LIST=${K_STENCIL_EXT}" CPU_FP64=off) ;;
            *)    echo "grupo desconocido: ${g}" >&2; exit 2 ;;
        esac
        encolar "${g}" num_corta "${COMUN[@]}" RUN_KIND=numeric "NX_LIST=4096 8192 16384" \
            "ITERS_LIST=${NUM_ITERS_CORTA}" "CHECKPOINT_EVERY=${CKPT_CORTA}"
        encolar "${g}" num_S "${COMUN[@]}" RUN_KIND=numeric "NX_LIST=${TIER_S_NX}" \
            "ITERS_LIST=${TIER_S_ITERS}" "CHECKPOINT_EVERY=${CKPT_VENTANA_S}"
        encolar "${g}" num_L "${COMUN[@]}" RUN_KIND=numeric "NX_LIST=${TIER_L_NX}" \
            "ITERS_LIST=${TIER_L_ITERS}" "CHECKPOINT_EVERY=${CKPT_VENTANA_L}"
        encolar "${g}" en_S "${COMUN[@]}" RUN_KIND=energy "NX_LIST=${TIER_S_NX}" "ITERS_LIST=${TIER_S_ITERS}"
        encolar "${g}" en_L "${COMUN[@]}" RUN_KIND=energy "NX_LIST=${TIER_L_NX}" "ITERS_LIST=${TIER_L_ITERS}"
    done ;;
  gemm)
    KDIR_RUN="${REPO_ROOT}/Fase_4/GEMM"; KSCRIPT="run_gemm_chained.sbatch"; OPFLAGS=""
    COMUN=("ANCHOR_LIST=${K_CHAINED_ALL}" "COMP_LIST=off on" RUN_NCU=0)
    encolar gemm num "${COMUN[@]}" RUN_KIND=numeric "N_LIST=1024 2048 4096 8192" "ITERS_LIST=20 40 80" CHECKPOINT_EVERY=5
    encolar gemm en_A "${COMUN[@]}" RUN_KIND=energy "N_LIST=1024 2048" "ITERS_LIST=24000"
    encolar gemm en_B "${COMUN[@]}" RUN_KIND=energy "N_LIST=4096 8192" "ITERS_LIST=500" ;;
  conv)
    KDIR_RUN="${REPO_ROOT}/Fase_4/Convolution"; KSCRIPT="run_conv_chained.sbatch"; OPFLAGS=""
    COMUN=("ANCHOR_LIST=${K_CHAINED_ALL}" "COMP_LIST=off on" RUN_NCU=0)
    encolar conv num "${COMUN[@]}" RUN_KIND=numeric "HW_LIST=64 128 256 512" "ITERS_LIST=20 40 80" CHECKPOINT_EVERY=5
    encolar conv en_A "${COMUN[@]}" RUN_KIND=energy "HW_LIST=64 128" "ITERS_LIST=37000"
    encolar conv en_B "${COMUN[@]}" RUN_KIND=energy "HW_LIST=256 512" "ITERS_LIST=2500" ;;
  *) echo "kernel desconocido: ${kern}" >&2; exit 2 ;;
  esac
done

echo
echo "Listo. Bitacora de jobs en ${JOBS_TSV}. Al terminar todo, gate de campana:"
echo "  python3 tools/audit_coverage.py --results-dir ${OUT_BASE} --recursive --mode campaign --out ${OUT_BASE}/audit_report.md"
