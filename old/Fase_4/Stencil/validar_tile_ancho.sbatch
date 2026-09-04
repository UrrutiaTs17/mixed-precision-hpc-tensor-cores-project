#!/bin/bash
#SBATCH --job-name=f4_tile_ancho
#SBATCH --output=logs/f4_tile_ancho_%j.out
#SBATCH --error=logs/f4_tile_ancho_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --time=03:00:00

# ===========================================================================
# ETAPA B: tile ancho 16x64 + swizzle XOR + ldmatrix.
#
# Las tres cosas van en un solo commit porque no son separables: sin swizzle,
# ensanchar EMPEORA el fragmento A de 2 a 8 vias de conflicto (calculado sobre
# el reparto real de las lanes). Para que el resultado siga siendo atribuible,
# lo que se separa no es el commit sino las VARIANTES DE COMPILACION:
#
#   base      revision anterior (tile 16x16, mma.sync)
#   ancho     16x64, sin swizzle, sin ldmatrix   <- debe ser el PEOR
#   swz       16x64 + swizzle                    <- el que se propone
#   swz_ldm   16x64 + swizzle + ldmatrix
#
# base->ancho aisla el ensanchamiento (menos sectores de banda, mas conflictos).
# ancho->swz aisla el swizzle. swz->swz_ldm aisla ldmatrix.
#
# CORRECTITUD: las tres variantes deben ser identicas BIT A BIT a base. El
# ensanchamiento no altera ninguna operacion de punto flotante -- los bloques de
# 16 columnas de cada warp caen en las mismas posiciones globales que los tiles
# de antes, y los bloques que no caben usan el respaldo por warp, que es el
# codigo anterior sin tocar. El swizzle es una permutacion de direcciones y
# ldmatrix es otra forma de leer los mismos bytes.
#
# Uso:
#   sbatch --export=ALL,BASE_COMMIT=61c031a validar_tile_ancho.sbatch
# ===========================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

REPO_ROOT="$(git rev-parse --show-toplevel)"
CUDA_ARCH="${CUDA_ARCH:-80}"
NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"
BASE_COMMIT="${BASE_COMMIT:-HEAD~1}"

JOB_ID="${SLURM_JOB_ID:-manual}"
WORK="tile_ancho_${JOB_ID}"
mkdir -p "${WORK}/salidas"

echo "Nodo        : $(hostname)"
echo "Fecha       : $(date -Iseconds)"
echo "Commit NUEVO: $(git -C "${REPO_ROOT}" rev-parse HEAD)"
echo "Commit BASE : $(git -C "${REPO_ROOT}" rev-parse "${BASE_COMMIT}")"
echo "Arquitectura: sm_${CUDA_ARCH}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

NVML_LINK_FLAG="${NVML_LINK_FLAG:--lnvidia-ml}"
if "${NVCC}" -E -x cu - 2>/dev/null <<< '#include <nvml.h>' >/dev/null &&
   (ldconfig -p 2>/dev/null | grep -Eq 'lib(nvidia-ml|nvml)\.so' ||
    [[ -f /usr/lib64/libnvidia-ml.so || -f /usr/lib64/libnvml.so ]]); then
    NVML_FLAGS=(-DUSE_NVML_TELEMETRY "${NVML_LINK_FLAG}" -lpthread)
else
    NVML_FLAGS=(-lpthread)
fi

# ===========================================================================
# ETAPA 1 — Compilacion de base y de las tres variantes
# ===========================================================================
echo "############################################################"
echo "# ETAPA 1: compilacion"
echo "############################################################"

BASE_TREE="${PWD}/${WORK}/arbol_base"
git -C "${REPO_ROOT}" worktree add --detach "${BASE_TREE}" "${BASE_COMMIT}"

CU_REL="Fase_3/Stencil/stencil_tensor_activation.cu"
SHA_CU_BASE="$(sha256sum "${BASE_TREE}/${CU_REL}" | awk '{print $1}')"
SHA_CU_NEW="$(sha256sum "${REPO_ROOT}/${CU_REL}"  | awk '{print $1}')"
if [[ "${SHA_CU_BASE}" == "${SHA_CU_NEW}" ]]; then
    echo "ERROR: ${CU_REL} es IDENTICO en las dos revisiones (sha256 ${SHA_CU_BASE})."
    echo "       El A/B seria una tautologia. Fija BASE_COMMIT correctamente."
    git -C "${REPO_ROOT}" worktree remove --force "${BASE_TREE}" || true
    exit 1
fi
echo "  .cu BASE : ${SHA_CU_BASE}"
echo "  .cu NUEVO: ${SHA_CU_NEW}"
echo

compilar() {   # $1 = arbol, $2 = binario, $3 = log ptxas, $4.. = flags extra
    local arbol="$1" salida="$2" logptx="$3"; shift 3
    ( cd "${arbol}/Fase_3/Stencil" && "${NVCC}" -std=c++17 stencil_tensor_activation.cu \
        -o "${salida}" \
        -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
        -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
        -Xptxas=-v "$@" \
        --allow-unsupported-compiler "${NVML_FLAGS[@]}" ) 2> "${logptx}"
}

VARIANTES=(base ancho swz swz_ldm)
declare -A BIN PTX
for v in "${VARIANTES[@]}"; do
    BIN[$v]="${PWD}/${WORK}/stencil_${v}"
    PTX[$v]="${PWD}/${WORK}/ptxas_${v}.txt"
done

echo "-- compilando --"
compilar "${BASE_TREE}" "${BIN[base]}"    "${PTX[base]}"
echo "   base      OK"
compilar "${REPO_ROOT}" "${BIN[ancho]}"   "${PTX[ancho]}"   -DSTENCIL_SWIZZLE=0 -DSTENCIL_LDMATRIX=0
echo "   ancho     OK"
compilar "${REPO_ROOT}" "${BIN[swz]}"     "${PTX[swz]}"     -DSTENCIL_SWIZZLE=1 -DSTENCIL_LDMATRIX=0
echo "   swz       OK"
compilar "${REPO_ROOT}" "${BIN[swz_ldm]}" "${PTX[swz_ldm]}" -DSTENCIL_SWIZZLE=1 -DSTENCIL_LDMATRIX=1
echo "   swz_ldm   OK"
echo

echo "-- registros y shared por instanciacion --"
resumen_ptxas() {   # $1 = log, $2 = etiqueta
    awk -v etq="$2" '
        /Compiling entry function/ && /stencil2d_wmma_kernel/ {
            tipo = (index($0, "__half") > 0) ? "FP16" : "BF16"
            i = index($0, "CompModeE"); m = substr($0, i + 9, 1)
            modo = (m == "0") ? "Off" : ((m == "1") ? "Local" : "Spatial")
            regs = ""; smem = ""; pend = 1; next
        }
        pend && /smem/ {
            for (j = 1; j <= NF; j++) if ($j ~ /smem/) smem = $(j-1)
        }
        pend && /Used [0-9]+ registers/ {
            for (j = 1; j <= NF; j++) if ($j == "Used") regs = $(j + 1)
            printf "     %-8s %-4s %-8s %3s reg  %6s B smem\n", etq, tipo, modo, regs, smem
            pend = 0
        }' "$1" | sort -k2,3
}
for v in "${VARIANTES[@]}"; do resumen_ptxas "${PTX[$v]}" "$v"; done
echo

spills=0
for v in ancho swz swz_ldm; do
    nz="$(grep -oE '[0-9]+ bytes spill (stores|loads)' "${PTX[$v]}" | grep -vE '^0 bytes' || true)"
    if [[ -n "${nz}" ]]; then
        echo "  ATENCION: ${v} reporta spills:"
        printf '%s\n' "${nz}" | sort | uniq -c | sed 's/^/     /'
        spills=$((spills + 1))
    fi
done
if [[ "${spills}" -eq 0 ]]; then echo "  OK: ninguna variante reporta spills."; fi
echo

# ===========================================================================
# ETAPA 2 — Igualdad bit a bit de cada variante contra base
# ===========================================================================
echo "############################################################"
echo "# ETAPA 2: igualdad bit a bit contra base"
echo "############################################################"
echo "Los dominios cubren los tres regimenes que el ensanchamiento distingue:"
echo "  nx-2 multiplo de 64  -> todos los bloques usan el camino ancho"
echo "  nx-2 NO multiplo     -> la ultima franja cae al respaldo por warp"
echo "  nx-2 < 64            -> TODO cae al respaldo (no hay bloque completo)"
echo

S="${PWD}/${WORK}/salidas"
fallos=0
casos=0

correr_caso() {   # $1 = etiqueta, $2.. = flags
    local etq="$1"; shift
    mkdir -p "${S}/${etq}"
    casos=$((casos + 1))
    local ok=1
    for v in "${VARIANTES[@]}"; do
        mkdir -p "${S}/${etq}/${v}"
        if ! "${BIN[$v]}" "$@" --archive-dir "${S}/${etq}/${v}" \
               > "${S}/${etq}/${v}.log" 2>&1; then
            echo "  ERROR ${etq}: la variante ${v} fallo al ejecutar"
            fallos=$((fallos + 1)); return
        fi
        ( cd "${S}/${etq}/${v}" && sha256sum *.bin | sort -k2 ) > "${S}/${etq}/${v}.sha"
    done
    local n; n=$(wc -l < "${S}/${etq}/base.sha")
    if [[ "${n}" -eq 0 ]]; then
        echo "  FALLO ${etq}: no se archivo ningun campo"
        fallos=$((fallos + 1)); return
    fi
    local difs=""
    for v in ancho swz swz_ldm; do
        if ! diff -q "${S}/${etq}/base.sha" "${S}/${etq}/${v}.sha" > /dev/null; then
            difs="${difs} ${v}"; ok=0
        fi
    done
    if [[ "${ok}" -eq 1 ]]; then
        echo "  OK    ${etq}  (${n} campos, 3 variantes identicas a base)"
    else
        echo "  FALLO ${etq}: DIFIEREN:${difs}"
        for v in ${difs}; do
            diff "${S}/${etq}/base.sha" "${S}/${etq}/${v}.sha" | head -6 | sed "s/^/        ${v}: /"
        done
        fallos=$((fallos + 1))
    fi
}

COMUN=(--iters 20 --tc both --fp64-gpu off --cpu-fp64 off
       --checkpoint-iters 10,20 --archive-iters 10,20)

# 66 y 130: nx-2 = 64 y 128, camino ancho puro (vec_ok falso, carga escalar).
# 512 y 520: nx-2 = 510 y 518, franja final al respaldo, vec_ok verdadero.
# 63 y 40  : demasiado pequenos para un bloque de 64, todo al respaldo.
for NX in 66 130 512 520 63 40; do
    correr_caso "nx${NX}_off"     --nx "${NX}" --ny "${NX}" "${COMUN[@]}"
    correr_caso "nx${NX}_local"   --nx "${NX}" --ny "${NX}" "${COMUN[@]}" --kahan on
    correr_caso "nx${NX}_spatial" --nx "${NX}" --ny "${NX}" "${COMUN[@]}" --spatial-comp on
done
# ny distinto de nx: obliga a tiles_y != tiles_x y destapa un mapeo mal derivado.
correr_caso "nx520_ny66_spatial" --nx 520 --ny 66 "${COMUN[@]}" --spatial-comp on
correr_caso "nx66_ny520_spatial" --nx 66 --ny 520 "${COMUN[@]}" --spatial-comp on
# Operador difusivo (la campana real) y CUDA Graphs.
correr_caso "nx520_dif_spatial" --nx 520 --ny 520 "${COMUN[@]}" \
    --spatial-comp on --op-mode diffusive --alpha 0.1875 --ci-mode monomode --ci-p 5
correr_caso "nx520_graph_off"     --nx 520 --ny 520 "${COMUN[@]}" \
    --execution-mode graph --graph-block 2
correr_caso "nx520_graph_spatial" --nx 520 --ny 520 "${COMUN[@]}" \
    --spatial-comp on --execution-mode graph --graph-block 2
echo

# ===========================================================================
# ETAPA 2b — compute-sanitizer
# ===========================================================================
echo "############################################################"
echo "# ETAPA 2b: compute-sanitizer (memcheck)"
echo "############################################################"
SANITIZER_BIN=""
_nvcc_dir="$(dirname "${NVCC}")"
for cand in "$(command -v compute-sanitizer || true)" \
            "${_nvcc_dir}/compute-sanitizer" \
            "${_nvcc_dir}/../../cuda/bin/compute-sanitizer" \
            "${_nvcc_dir}"/../../cuda/*/bin/compute-sanitizer; do
    if [[ -n "${cand}" && -x "${cand}" ]]; then SANITIZER_BIN="${cand}"; break; fi
done

sanit_fallos=0
if [[ -z "${SANITIZER_BIN}" ]]; then
    echo "  compute-sanitizer no encontrado; etapa omitida (no cuenta como fallo)."
else
    echo "  Binario: ${SANITIZER_BIN}"
    # racecheck ademas de memcheck: el tile ancho introduce shared COMPARTIDA
    # entre warps y un __syncthreads() nuevo. Una carrera ahi no la ve memcheck.
    for v in swz swz_ldm; do
        for MODO in "" "--kahan on" "--spatial-comp on"; do
            case "${MODO}" in
                "")                 etq="${v}_off" ;;
                "--kahan on")       etq="${v}_local" ;;
                *)                  etq="${v}_spatial" ;;
            esac
            for TOOL in memcheck racecheck; do
                log="${S}/sanit_${TOOL}_${etq}.log"
                if "${SANITIZER_BIN}" --tool "${TOOL}" --error-exitcode 1 \
                     "${BIN[$v]}" --nx 130 --ny 130 --iters 5 --tc both \
                     --fp64-gpu off --cpu-fp64 off ${MODO} > "${log}" 2>&1; then
                    echo "  OK    ${TOOL} ${etq}"
                else
                    echo "  FALLO ${TOOL} ${etq} (ver ${log})"
                    grep -iE "error|hazard|race|invalid|misaligned" "${log}" | head -8 | sed 's/^/        /'
                    sanit_fallos=$((sanit_fallos + 1))
                fi
            done
        done
    done
fi
echo

# ===========================================================================
# ETAPA 3 — Rendimiento
# ===========================================================================
echo "############################################################"
echo "# ETAPA 3: rendimiento"
echo "############################################################"
NX_MED="${NX_MED:-4096}"
ITERS_MED="${ITERS_MED:-50}"
REPLICAS="${REPLICAS:-4}"
CSV="${PWD}/${WORK}/mediciones.csv"

if [[ $((fallos + sanit_fallos)) -ne 0 ]]; then
    echo "  OMITIDA: la correctitud no paso (fallos=${fallos}, sanitizer=${sanit_fallos})."
    medido=0
else
    medido=1
    echo "Malla: ${NX_MED}^2 · ${ITERS_MED} iteraciones · ${REPLICAS} replicas"
    COMUN_MED=(--nx "${NX_MED}" --ny "${NX_MED}" --iters "${ITERS_MED}" --tc both
               --fp64-gpu off --cpu-fp64 off
               --op-mode diffusive --alpha 0.1875 --ci-mode monomode --ci-p 5)
    echo "variante,replica,modo,route,kahan,t_iter_ms,t_kernel_ms" > "${CSV}"

    correr_medida() {   # $1 = variante, $2 = replica, $3 = modo, $4.. = flags
        local v="$1" rep="$2" modo="$3"; shift 3
        local log="${PWD}/${WORK}/salidas/med_${v}_r${rep}_${modo}.log"
        "${BIN[$v]}" "${COMUN_MED[@]}" "$@" > "${log}" 2>&1
        awk -F, -v v="${v}" -v rep="${rep}" -v modo="${modo}" \
            '/^CSV_SUMMARY/ { printf "%s,%s,%s,%s,%s,%s,%s\n", v, rep, modo, $2, $6, $7, $12 }' \
            "${log}" >> "${CSV}"
    }

    echo "-- cebado (descartado) --"
    "${BIN[swz]}" "${COMUN_MED[@]}" > "${PWD}/${WORK}/salidas/med_cebado.log" 2>&1
    echo "   hecho"

    for rep in $(seq 1 "${REPLICAS}"); do
        # El orden ROTA entre replicas para que una deriva del nodo no caiga
        # siempre sobre la misma variante.
        ORDEN=()
        for k in 0 1 2 3; do
            ORDEN+=("${VARIANTES[$(( (k + rep) % 4 ))]}")
        done
        for v in "${ORDEN[@]}"; do
            correr_medida "${v}" "${rep}" off
            correr_medida "${v}" "${rep}" local   --kahan on
            correr_medida "${v}" "${rep}" spatial --spatial-comp on
        done
        echo "   replica ${rep}/${REPLICAS} completa (orden: ${ORDEN[*]})"
    done
    echo

    python3 - "${CSV}" <<'PY'
import csv, sys, statistics as st
filas = list(csv.DictReader(open(sys.argv[1])))
if not filas:
    print("  CSV vacio: revisa los .log de salidas/"); sys.exit(0)
orden = ["base", "ancho", "swz", "swz_ldm"]
datos = {}
for f in filas:
    datos.setdefault((f["route"], f["modo"]), {}).setdefault(f["variante"], []).append(float(f["t_iter_ms"]))
def stats(v):
    m = st.mean(v)
    return m, (st.stdev(v)/m*100 if len(v) > 1 and m else 0.0)
print(f"{'ruta':<16}{'modo':<9}" + "".join(f"{v:>12}" for v in orden)
      + f"{'swz/base':>10}{'swz/ancho':>11}{'ldm/swz':>9}{'CV%max':>8}")
print("-" * 116)
for k in sorted(datos):
    d = datos[k]
    if not all(v in d for v in orden): continue
    m = {}; cvs = []
    for v in orden:
        mm, cv = stats(d[v]); m[v] = mm; cvs.append(cv)
    print(f"{k[0]:<16}{k[1]:<9}" + "".join(f"{m[v]:12.5f}" for v in orden)
          + f"{m['base']/m['swz']:9.3f}x{m['ancho']/m['swz']:10.3f}x"
          + f"{m['swz']/m['swz_ldm']:8.3f}x{max(cvs):8.2f}")
print()
print("swz/base   > 1.000x = el cambio completo gana. Es la cifra que decide.")
print("swz/ancho  > 1.000x = el swizzle aporta. Deberia ser el salto grande:")
print("                      sin el, el fragmento A choca a 8 vias.")
print("ldm/swz    > 1.000x = ldmatrix aporta encima del swizzle.")
print("GPU_FP32 y CPU_FP32 son CONTROLES: sus filas deben salir planas.")
PY
fi
echo

# ===========================================================================
# ETAPA 4 — Veredicto
# ===========================================================================
echo "############################################################"
echo "# ETAPA 4: veredicto"
echo "############################################################"
echo "Casos ejecutados   : ${casos}"
echo "Casos con FALLO    : ${fallos}"
echo "sanitizer con FALLO: ${sanit_fallos}"
echo "Variantes con spill: ${spills}"
git -C "${REPO_ROOT}" worktree remove --force "${BASE_TREE}" || true
total=$((fallos + sanit_fallos + spills))
if [[ "${total}" -eq 0 ]]; then
    echo
    echo "VEREDICTO: el tile ancho reproduce la revision anterior BIT A BIT en"
    echo "las tres variantes, sin accesos invalidos, sin carreras y sin spills."
    if [[ "${medido}" -eq 1 ]]; then
        echo "La tabla de la ETAPA 3 dice si ademas gana."
    fi
    exit 0
else
    echo
    echo "VEREDICTO: NO autorizado."
    if [[ "${fallos}"       -ne 0 ]]; then echo "  - diferencias numericas: revisa ${WORK}/salidas/<caso>/*.sha"; fi
    if [[ "${sanit_fallos}" -ne 0 ]]; then echo "  - compute-sanitizer reporta accesos invalidos o carreras"; fi
    if [[ "${spills}"       -ne 0 ]]; then echo "  - ptxas reporta spills"; fi
    exit 1
fi
