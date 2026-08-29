#!/bin/bash
#SBATCH --job-name=f4_pad
#SBATCH --output=logs/f4_pad_%j.out
#SBATCH --error=logs/f4_pad_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --time=02:00:00

# ===========================================================================
# BARRIDO DE PADDING DE LOS TILES DE SHARED
#
# Motivacion (job 6730, ncu): el mayor stall del kernel WMMA es mio_throttle
# (14.93 contra 0.10 de GPU_FP32), no la espera de memoria global. La causa son
# los conflictos de banco en shared -- 1.1 M en lectura y 566 k en escritura,
# frente a 9.5 k y 1.9 k de GPU_FP32 -- porque los tiles no tienen padding y
# varias filas arrancan en el mismo banco.
#
# Cuantos bancos DISTINTOS usan las 16 filas como arranque (32 bancos de 4 B):
#
#   padTC  padF32   x_tile   out_tile        B/warp (off / spatial)
#       0       0        4          2            1664 / 2944   <- HOY
#       0       4        4          8            1920 / 3456
#       8       0        8          2            1920 / 3200
#       8       4        8          8            2176 / 3712
#      24       4        8          8            2688 / 4224
#       0      12        4          8            2432 / 4480
#
# El techo es 8, no 16: wmma exige ldm*sizeof(elemento) multiplo de 16 B, luego
# el paso de fila es siempre multiplo de 4 bancos y al menos dos filas coinciden.
#
# El padding es un intercambio, no un regalo: mas shared por bloque = menos
# bloques por SM = menos ocupancia. Por eso se BARRE en vez de fijar un valor.
#
# ETAPA 2 exige igualdad BIT A BIT contra el binario sin padding: el padding
# solo cambia donde vive cada dato en shared, no que dato es ni en que orden se
# opera. Una variante que no sea identica esta mal indexada y queda descartada
# antes de medirse.
#
# Uso:
#   sbatch barrer_padding.sbatch
#   sbatch --export=ALL,NX=16384,REPLICAS=2 barrer_padding.sbatch
# ===========================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

REPO_ROOT="$(git rev-parse --show-toplevel)"
CUDA_ARCH="${CUDA_ARCH:-80}"
NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"
NX="${NX:-4096}"
ITERS="${ITERS:-50}"
REPLICAS="${REPLICAS:-3}"

JOB_ID="${SLURM_JOB_ID:-manual}"
WORK="pad_${JOB_ID}"
mkdir -p "${WORK}/salidas" "${WORK}/ref"
CSV="${PWD}/${WORK}/mediciones.csv"

echo "Nodo        : $(hostname)"
echo "Fecha       : $(date -Iseconds)"
echo "Commit      : $(git -C "${REPO_ROOT}" rev-parse HEAD)"
echo "Malla       : ${NX}^2 · ${ITERS} iteraciones · ${REPLICAS} replicas"
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

# padTC:padF32 — la primera DEBE ser 0:0 (es la referencia de correctitud).
VARIANTES_STR="${VARIANTES_OVERRIDE:-0:0 0:4 8:0 8:4 24:4 0:12}"
read -r -a VARIANTES <<< "${VARIANTES_STR}"
if [[ " ${VARIANTES_STR} " != *" 0:0 "* ]]; then
    echo "ERROR: la lista de variantes debe incluir 0:0, que es la referencia"
    echo "       de correctitud y el denominador de las aceleraciones."
    exit 1
fi

# ===========================================================================
# ETAPA 1 — Un binario por variante
# ===========================================================================
echo "############################################################"
echo "# ETAPA 1: compilacion de las variantes"
echo "############################################################"
printf "  %-10s %-12s %-12s %-14s %s\n" variante "B/warp off" "B/warp spa" "B/bloque spa" "registros (FP16 off/local/spatial)"

declare -A BIN_DE
for v in "${VARIANTES[@]}"; do
    pt="${v%%:*}"; pf="${v##*:}"
    etq="pad${pt}_${pf}"
    bin="${PWD}/${WORK}/stencil_${etq}"
    ptx="${PWD}/${WORK}/ptxas_${etq}.txt"
    ( cd "${REPO_ROOT}/Fase_3/Stencil" && "${NVCC}" -std=c++17 stencil_tensor_activation.cu -o "${bin}" \
        -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
        -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
        -DSTENCIL_PAD_TC="${pt}" -DSTENCIL_PAD_F32="${pf}" \
        -Xptxas=-v \
        --allow-unsupported-compiler "${NVML_FLAGS[@]}" ) 2> "${ptx}"
    BIN_DE[$etq]="${bin}"

    off=$(( 16*(16+pt)*2 + 16*(16+pf)*4 + 128 ))
    spa=$(( off + 16*(16+pf)*4 + 256 ))
    regs=$(awk '/Compiling entry function/ && /stencil2d_wmma_kernel/ && /__half/ {
                    i=index($0,"CompModeE"); m=substr($0,i+9,1); pend=1; next }
                pend && /Used [0-9]+ registers/ {
                    for (j=1;j<=NF;j++) if ($j=="Used") r[m]=$(j+1); pend=0 }
                END { printf "%s/%s/%s", r["0"], r["1"], r["2"] }' "${ptx}")
    printf "  %-10s %-12s %-12s %-14s %s\n" "${etq}" "${off}" "${spa}" "$((4*spa))" "${regs}"

    sp="$(grep -oE '[0-9]+ bytes spill (stores|loads)' "${ptx}" | grep -vE '^0 bytes' || true)"
    if [[ -n "${sp}" ]]; then echo "     ATENCION ${etq}: spills distintos de cero"; fi
done
echo

# ===========================================================================
# ETAPA 2 — Igualdad bit a bit contra pad0_0
# ===========================================================================
echo "############################################################"
echo "# ETAPA 2: correctitud (todas deben ser identicas a pad0_0)"
echo "############################################################"
CHK=(--nx 512 --ny 512 --iters 20 --tc both --fp64-gpu off --cpu-fp64 off
     --checkpoint-iters 10,20 --archive-iters 10,20)
declare -A DESCARTADA
fallos_corr=0

for v in "${VARIANTES[@]}"; do
    pt="${v%%:*}"; pf="${v##*:}"; etq="pad${pt}_${pf}"
    malas=0
    for modo in off local spatial; do
        extra=()
        case "${modo}" in
            local)   extra=(--kahan on) ;;
            spatial) extra=(--spatial-comp on) ;;
        esac
        dir="${PWD}/${WORK}/ref/${etq}_${modo}"; mkdir -p "${dir}"
        "${BIN_DE[$etq]}" "${CHK[@]}" "${extra[@]}" --archive-dir "${dir}" \
            > "${PWD}/${WORK}/salidas/corr_${etq}_${modo}.log" 2>&1
        ( cd "${dir}" && sha256sum *.bin | sort -k2 ) > "${dir}.sha"
    done
    if [[ "${etq}" != "pad0_0" ]]; then
        for modo in off local spatial; do
            if ! diff -q "${PWD}/${WORK}/ref/pad0_0_${modo}.sha" \
                         "${PWD}/${WORK}/ref/${etq}_${modo}.sha" > /dev/null; then
                malas=$((malas + 1))
            fi
        done
        if [[ "${malas}" -eq 0 ]]; then
            echo "  OK    ${etq}"
        else
            echo "  FALLO ${etq}: ${malas}/3 modos difieren -> DESCARTADA, no se mide"
            DESCARTADA[$etq]=1
            fallos_corr=$((fallos_corr + 1))
        fi
    else
        n=$(wc -l < "${PWD}/${WORK}/ref/pad0_0_off.sha")
        echo "  ref   ${etq} (${n} campos por modo)"
    fi
done
echo

# ===========================================================================
# ETAPA 3 — Medicion
# ===========================================================================
echo "############################################################"
echo "# ETAPA 3: medicion"
echo "############################################################"
COMUN=(--nx "${NX}" --ny "${NX}" --iters "${ITERS}" --tc both
       --fp64-gpu off --cpu-fp64 off
       --op-mode diffusive --alpha 0.1875 --ci-mode monomode --ci-p 5)
echo "variante,replica,modo,route,kahan,t_iter_ms,t_kernel_ms" > "${CSV}"

VIVAS=()
for v in "${VARIANTES[@]}"; do
    etq="pad${v%%:*}_${v##*:}"
    # if explicito y no '[[ ]] && ...': este es el ULTIMO comando del cuerpo
    # del bucle, y ahi un test falso hace que el for devuelva 1 y set -e aborte.
    if [[ -z "${DESCARTADA[$etq]:-}" ]]; then VIVAS+=("${etq}"); fi
done

echo "-- cebado (descartado) --"
"${BIN_DE[pad0_0]}" "${COMUN[@]}" > "${PWD}/${WORK}/salidas/cebado.log" 2>&1
echo "   hecho"

for rep in $(seq 1 "${REPLICAS}"); do
    # Rota el orden de las variantes entre replicas: si el nodo deriva, la
    # deriva no cae siempre sobre la misma.
    n=${#VIVAS[@]}; desp=$(( (rep - 1) % n )); ORDEN=()
    for k in $(seq 0 $((n - 1))); do ORDEN+=("${VIVAS[$(( (k + desp) % n ))]}"); done
    for etq in "${ORDEN[@]}"; do
        for modo in off local spatial; do
            extra=()
            case "${modo}" in
                local)   extra=(--kahan on) ;;
                spatial) extra=(--spatial-comp on) ;;
            esac
            log="${PWD}/${WORK}/salidas/${etq}_r${rep}_${modo}.log"
            "${BIN_DE[$etq]}" "${COMUN[@]}" "${extra[@]}" > "${log}" 2>&1
            awk -F, -v e="${etq}" -v r="${rep}" -v m="${modo}" \
                '/^CSV_SUMMARY/ { printf "%s,%s,%s,%s,%s,%s,%s\n", e, r, m, $2, $6, $7, $12 }' \
                "${log}" >> "${CSV}"
        done
    done
    echo "   replica ${rep}/${REPLICAS} completa (orden: ${ORDEN[*]})"
done
echo

# ===========================================================================
# ETAPA 4 — Resumen
# ===========================================================================
echo "############################################################"
echo "# ETAPA 4: resumen (referencia = pad0_0, el estado actual)"
echo "############################################################"
python3 - "${CSV}" <<'PY'
import csv, sys, statistics as st

filas = list(csv.DictReader(open(sys.argv[1])))
if not filas:
    print("  CSV vacio: revisa salidas/"); sys.exit(1)

datos = {}
for f in filas:
    datos.setdefault((f["route"], f["modo"]), {}) \
         .setdefault(f["variante"], []).append(float(f["t_iter_ms"]))

variantes = sorted({f["variante"] for f in filas},
                   key=lambda v: (v != "pad0_0", v))

def stats(v):
    m = st.mean(v)
    return m, (st.stdev(v) / m * 100 if len(v) > 1 and m else 0.0)

anchoi = 30
print(" " * anchoi + "".join(f"{v:>16}" for v in variantes))
print("-" * (anchoi + 16 * len(variantes)))
for clave in sorted(datos):
    route, modo = clave
    d = datos[clave]
    base = stats(d["pad0_0"])[0] if "pad0_0" in d else None
    celdas = []
    for v in variantes:
        if v not in d:
            celdas.append(f"{'-':>16}"); continue
        m, cv = stats(d[v])
        celdas.append(f"{m:>9.5f}{('' if base is None else f' {base/m:>5.3f}x'):>7}")
    print(f"{route + ' ' + modo:<{anchoi}}" + "".join(celdas))

print()
print("Cada celda: ms/iter y aceleracion frente a pad0_0 (>1.000x = mas rapido).")
print("GPU_FP32 y CPU_FP32 son CONTROLES: su codigo no depende del padding, asi")
print("que sus filas deben salir planas. Si se mueven, el nodo derivo.")
PY

echo
echo "Correctitud: ${fallos_corr} variantes descartadas."
echo "CSV crudo: ${CSV}"
