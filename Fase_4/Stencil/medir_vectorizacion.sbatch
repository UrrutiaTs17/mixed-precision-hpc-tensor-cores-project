#!/bin/bash
#SBATCH --job-name=f4_medir_vec
#SBATCH --output=logs/f4_medir_vec_%j.out
#SBATCH --error=logs/f4_medir_vec_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --time=02:00:00

# ===========================================================================
# ETAPA 6: RENDIMIENTO de la alineacion y la vectorizacion.
#
# Tres revisiones, compiladas y medidas DENTRO DEL MISMO JOB, en el mismo nodo
# y en la misma sesion. No se compara contra ninguna corrida pasada: las
# condiciones del nodo (reloj, vecinos, estado termico) se cancelan porque las
# tres rutas las comparten.
#
#   R1  pre-alineacion   buffers sin padding, accesos escalares
#   R2  alineacion       padding + punteros desplazados, kernel SIN tocar
#   R3  vectorizado      accesos de 128 bits (el arbol de trabajo actual)
#
# R1->R2 aisla el efecto de mover el puntero base (el kernel es identico).
# R2->R3 aisla el efecto de la vectorizacion (la alineacion es identica).
# Las tres son BIT A BIT equivalentes: los jobs 6719 y 6722 ya lo establecieron,
# aqui solo se miden tiempos.
#
# GPU_FP32 es el CONTROL: su codigo no cambia en ninguna de las tres
# revisiones, asi que cualquier diferencia entre sus tiempos mide ruido del
# nodo, no el cambio. Si GPU_FP32 se mueve mas que las rutas WMMA, la
# comparacion no vale.
#
# Uso:
#   sbatch medir_vectorizacion.sbatch
#   sbatch --export=ALL,NX=16384,REPLICAS=3 medir_vectorizacion.sbatch
# ===========================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

REPO_ROOT="$(git rev-parse --show-toplevel)"
CUDA_ARCH="${CUDA_ARCH:-80}"
NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"

REV_R1="${REV_R1:-26e35ab}"      # pre-alineacion
REV_R2="${REV_R2:-63e651d}"      # alineacion sola
NX="${NX:-4096}"
ITERS="${ITERS:-50}"
REPLICAS="${REPLICAS:-5}"

JOB_ID="${SLURM_JOB_ID:-manual}"
WORK="medir_vec_${JOB_ID}"
mkdir -p "${WORK}/salidas"
CSV="${PWD}/${WORK}/mediciones.csv"

echo "Nodo        : $(hostname)"
echo "Fecha       : $(date -Iseconds)"
echo "Commit HEAD : $(git -C "${REPO_ROOT}" rev-parse HEAD)"
echo "R1 (pre-ali): $(git -C "${REPO_ROOT}" rev-parse "${REV_R1}")"
echo "R2 (alinea) : $(git -C "${REPO_ROOT}" rev-parse "${REV_R2}")"
echo "Malla       : ${NX}^2 · ${ITERS} iteraciones · ${REPLICAS} replicas"
echo "Arquitectura: sm_${CUDA_ARCH}"
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv,noheader || true
echo

NVML_LINK_FLAG="${NVML_LINK_FLAG:--lnvidia-ml}"
if "${NVCC}" -E -x cu - 2>/dev/null <<< '#include <nvml.h>' >/dev/null &&
   (ldconfig -p 2>/dev/null | grep -Eq 'lib(nvidia-ml|nvml)\.so' ||
    [[ -f /usr/lib64/libnvidia-ml.so || -f /usr/lib64/libnvml.so ]]); then
    NVML_FLAGS=(-DUSE_NVML_TELEMETRY "${NVML_LINK_FLAG}" -lpthread)
else
    NVML_FLAGS=(-lpthread)
fi

CU_REL="Fase_3/Stencil/stencil_tensor_activation.cu"

# ===========================================================================
# ETAPA 1 — Tres arboles, tres binarios
# ===========================================================================
echo "############################################################"
echo "# ETAPA 1: compilacion de las tres revisiones"
echo "############################################################"

TREE_R1="${PWD}/${WORK}/arbol_r1"
TREE_R2="${PWD}/${WORK}/arbol_r2"
git -C "${REPO_ROOT}" worktree add --detach "${TREE_R1}" "${REV_R1}" > /dev/null
git -C "${REPO_ROOT}" worktree add --detach "${TREE_R2}" "${REV_R2}" > /dev/null

limpiar_worktrees() {
    git -C "${REPO_ROOT}" worktree remove --force "${TREE_R1}" 2>/dev/null || true
    git -C "${REPO_ROOT}" worktree remove --force "${TREE_R2}" 2>/dev/null || true
}
trap limpiar_worktrees EXIT

# Guarda contra la comparacion vacia (ver validar_alineacion.sbatch): si dos de
# los tres fuentes son identicos, esa pareja no mide nada.
declare -A SHA_CU
SHA_CU[R1]="$(sha256sum "${TREE_R1}/${CU_REL}" | awk '{print $1}')"
SHA_CU[R2]="$(sha256sum "${TREE_R2}/${CU_REL}" | awk '{print $1}')"
SHA_CU[R3]="$(sha256sum "${REPO_ROOT}/${CU_REL}" | awk '{print $1}')"
for a in R1 R2; do
    for b in R2 R3; do
        if [[ "${a}" != "${b}" && "${SHA_CU[$a]}" == "${SHA_CU[$b]}" ]]; then
            echo "ERROR: el .cu de ${a} y ${b} es IDENTICO (${SHA_CU[$a]})."
            echo "       Esa pareja no medira nada. Revisa REV_R1 / REV_R2."
            exit 1
        fi
    done
done
echo "  .cu R1: ${SHA_CU[R1]}"
echo "  .cu R2: ${SHA_CU[R2]}"
echo "  .cu R3: ${SHA_CU[R3]}"
echo "  (los tres difieren: las dos comparaciones son reales)"
echo

declare -A BIN
compilar() {   # $1 = raiz, $2 = binario, $3 = log ptxas
    ( cd "$1/Fase_3/Stencil" && "${NVCC}" -std=c++17 stencil_tensor_activation.cu -o "$2" \
        -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
        -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
        -Xptxas=-v \
        --allow-unsupported-compiler "${NVML_FLAGS[@]}" ) 2> "$3"
}
for R in R1 R2 R3; do
    BIN[$R]="${PWD}/${WORK}/stencil_${R}"
done
compilar "${TREE_R1}"  "${BIN[R1]}" "${PWD}/${WORK}/ptxas_R1.txt"
compilar "${TREE_R2}"  "${BIN[R2]}" "${PWD}/${WORK}/ptxas_R2.txt"
compilar "${REPO_ROOT}" "${BIN[R3]}" "${PWD}/${WORK}/ptxas_R3.txt"

resumen_registros() {   # $1 = log, $2 = etiqueta
    awk -v etq="$2" '
        /Compiling entry function/ && /stencil2d_wmma_kernel/ {
            tipo = (index($0, "__half") > 0) ? "FP16" : "BF16"
            i = index($0, "CompModeE"); m = substr($0, i + 9, 1)
            modo = (m == "0") ? "Off" : ((m == "1") ? "Local" : "Spatial")
            pend = 1; next
        }
        pend && /Used [0-9]+ registers/ {
            for (j = 1; j <= NF; j++) if ($j == "Used") regs = $(j + 1)
            printf "     %-4s %-8s %-3s %3s\n", tipo, modo, etq, regs
            pend = 0
        }' "$1"
}
echo "-- Registros por instanciacion (el delta R2->R3 es el costo de vectorizar) --"
{ for R in R1 R2 R3; do resumen_registros "${PWD}/${WORK}/ptxas_${R}.txt" "${R}"; done; } | sort -k1,2
echo
for R in R1 R2 R3; do
    sp="$(grep -oE '[0-9]+ bytes spill (stores|loads)' "${PWD}/${WORK}/ptxas_${R}.txt" \
          | grep -vE '^0 bytes' || true)"
    if [[ -n "${sp}" ]]; then echo "  ATENCION ${R}: spills distintos de cero"; fi
done
echo

# ===========================================================================
# ETAPA 2 — Medicion
# ===========================================================================
echo "############################################################"
echo "# ETAPA 2: medicion"
echo "############################################################"

COMUN=(--nx "${NX}" --ny "${NX}" --iters "${ITERS}" --tc both
       --fp64-gpu off --cpu-fp64 off
       --op-mode diffusive --alpha 0.1875 --ci-mode monomode --ci-p 5)

echo "revision,replica,modo,route,kahan,t_iter_ms,t_kernel_ms" > "${CSV}"

correr() {   # $1 = revision, $2 = replica, $3 = etiqueta de modo, $4.. = flags extra
    local rev="$1" rep="$2" modo="$3"; shift 3
    local log="${PWD}/${WORK}/salidas/${rev}_r${rep}_${modo}.log"
    "${BIN[$rev]}" "${COMUN[@]}" "$@" > "${log}" 2>&1
    awk -F, -v rev="${rev}" -v rep="${rep}" -v modo="${modo}" \
        '/^CSV_SUMMARY/ { printf "%s,%s,%s,%s,%s,%s,%s\n", rev, rep, modo, $2, $6, $7, $12 }' \
        "${log}" >> "${CSV}"
}

# Corrida de cebado, DESCARTADA: la primera ejecucion sobre una GPU fria mide
# el arranque del reloj, no el kernel. No entra al CSV.
echo "-- cebado (descartado) --"
"${BIN[R3]}" "${COMUN[@]}" > "${PWD}/${WORK}/salidas/cebado.log" 2>&1
echo "   hecho"
echo

for rep in $(seq 1 "${REPLICAS}"); do
    # El orden de las revisiones ROTA entre replicas: si el nodo deriva
    # (temperatura, reloj), la deriva no cae siempre sobre la misma revision.
    case $(( rep % 3 )) in
        1) ORDEN=(R1 R2 R3) ;;
        2) ORDEN=(R2 R3 R1) ;;
        *) ORDEN=(R3 R1 R2) ;;
    esac
    for rev in "${ORDEN[@]}"; do
        correr "${rev}" "${rep}" off
        correr "${rev}" "${rep}" local   --kahan on
        correr "${rev}" "${rep}" spatial --spatial-comp on
    done
    echo "   replica ${rep}/${REPLICAS} completa (orden: ${ORDEN[*]})"
done
echo

# ===========================================================================
# ETAPA 3 — Resumen
# ===========================================================================
echo "############################################################"
echo "# ETAPA 3: resumen"
echo "############################################################"
python3 - "${CSV}" <<'PY'
import csv, sys, statistics as st

filas = list(csv.DictReader(open(sys.argv[1])))
if not filas:
    print("  CSV vacio: revisa los .log de salidas/"); sys.exit(1)

# clave: (route, kahan, modo) -> {revision: [t_iter_ms]}
datos = {}
for f in filas:
    k = (f["route"], f["kahan"], f["modo"])
    datos.setdefault(k, {}).setdefault(f["revision"], []).append(float(f["t_iter_ms"]))

def stats(v):
    m = st.mean(v)
    cv = (st.stdev(v) / m * 100) if len(v) > 1 and m else 0.0
    return m, cv

print(f"{'ruta':<16}{'modo':<9}{'R1 pre-ali':>12}{'R2 alinea':>12}{'R3 vector':>12}"
      f"{'R1->R2':>9}{'R2->R3':>9}{'R1->R3':>9}{'CV%max':>8}")
print("-" * 96)
for k in sorted(datos):
    route, kahan, modo = k
    d = datos[k]
    if not all(r in d for r in ("R1", "R2", "R3")):
        continue
    (m1, c1), (m2, c2), (m3, c3) = stats(d["R1"]), stats(d["R2"]), stats(d["R3"])
    print(f"{route:<16}{modo:<9}{m1:>12.5f}{m2:>12.5f}{m3:>12.5f}"
          f"{m1/m2:>8.3f}x{m2/m3:>8.3f}x{m1/m3:>8.3f}x{max(c1,c2,c3):>8.2f}")

print()
print("Lectura:")
print("  GPU_FP32 es el CONTROL: su codigo es identico en R1/R2/R3. Sus tres")
print("  columnas deben coincidir dentro del CV. Si no, el nodo derivo y las")
print("  demas filas no son atribuibles.")
print("  R1->R2 = efecto de la ALINEACION sola (kernel sin tocar).")
print("  R2->R3 = efecto de la VECTORIZACION sola (alineacion identica).")
print("  Un CV por encima del 2 % invalida diferencias menores que el propio CV.")
PY

echo
echo "CSV crudo: ${CSV}"
