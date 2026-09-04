#!/bin/bash
#SBATCH --job-name=f4_mma_sync
#SBATCH --output=logs/f4_mma_sync_%j.out
#SBATCH --error=logs/f4_mma_sync_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --time=02:00:00

# ===========================================================================
# ETAPA A de mma.sync: el kernel WMMA deja de usar nvcuda::wmma y emite
# mma.sync.aligned.m16n8k16 directamente, con el MISMO layout de memoria.
#
# Que se prueba y que NO.
#
#   SI: que el reparto lane->elemento que codifica el kernel coincide con el
#       que define el ISA de PTX. wmma m16n16k16 con acumulador FP32 se baja a
#       exactamente dos mma.m16n8k16 por mitad de n, asi que la aritmetica es
#       la misma instruccion de hardware sobre los mismos datos y la exigencia
#       es IDENTIDAD BYTE A BYTE, no "error dentro de tolerancia". Cualquier
#       diferencia es un indice de fragmento mal puesto.
#
#   NO: ninguna mejora de rendimiento. Este paso no cambia ni un acceso de
#       memoria. Su unico proposito es habilitar la ETAPA B (swizzle de la
#       shared), que necesita conocer el reparto del fragmento y que
#       store_matrix_sync impide porque solo acepta un ldm plano.
#
# Por eso la medicion de la ETAPA 3 tiene un criterio ASIMETRICO: no se espera
# ganancia, se comprueba que no haya perdida grande. wmma::load_matrix_sync
# puede estar emitiendo ldmatrix (LDSM) donde este codigo emite accesos
# escalares de 16 bits para el operando B; si eso cuesta caro, se vera aqui y
# la ETAPA B tendra que traerse ldmatrix consigo.
#
# BASE_COMMIT debe ser la revision ANTERIOR al commit de mma.sync.
#
# Uso:
#   sbatch --export=ALL,BASE_COMMIT=1387763 validar_mma_sync.sbatch
# ===========================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

REPO_ROOT="$(git rev-parse --show-toplevel)"
CUDA_ARCH="${CUDA_ARCH:-80}"
NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"
BASE_COMMIT="${BASE_COMMIT:-HEAD~1}"

JOB_ID="${SLURM_JOB_ID:-manual}"
WORK="mma_sync_${JOB_ID}"
mkdir -p "${WORK}/salidas"

echo "Nodo        : $(hostname)"
echo "Fecha       : $(date -Iseconds)"
echo "Commit NUEVO: $(git -C "${REPO_ROOT}" rev-parse HEAD)"
echo "Commit BASE : $(git -C "${REPO_ROOT}" rev-parse "${BASE_COMMIT}")"
echo "Arquitectura: sm_${CUDA_ARCH}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

# --- NVML (mismo bloque que los demas sbatch de la fase) --------------------
NVML_LINK_FLAG="${NVML_LINK_FLAG:--lnvidia-ml}"
if "${NVCC}" -E -x cu - 2>/dev/null <<< '#include <nvml.h>' >/dev/null &&
   (ldconfig -p 2>/dev/null | grep -Eq 'lib(nvidia-ml|nvml)\.so' ||
    [[ -f /usr/lib64/libnvidia-ml.so || -f /usr/lib64/libnvml.so ]]); then
    NVML_FLAGS=(-DUSE_NVML_TELEMETRY "${NVML_LINK_FLAG}" -lpthread)
else
    NVML_FLAGS=(-lpthread)
fi

# ===========================================================================
# ETAPA 1 — Dos binarios: BASE y NUEVO
# ===========================================================================
echo "############################################################"
echo "# ETAPA 1: compilacion de las dos revisiones"
echo "############################################################"

# El .cu incluye ../../Fase_2/common.cuh y tools/power_sampling.h por ruta
# RELATIVA, asi que la revision base necesita un arbol completo: worktree, no
# un 'git show' a un fichero suelto.
BASE_TREE="${PWD}/${WORK}/arbol_base"
git -C "${REPO_ROOT}" worktree add --detach "${BASE_TREE}" "${BASE_COMMIT}"

BIN_BASE="${PWD}/${WORK}/stencil_base"
BIN_NEW="${PWD}/${WORK}/stencil_nuevo"

compilar() {   # $1 = raiz del arbol, $2 = binario de salida, $3 = log de ptxas
    ( cd "$1/Fase_3/Stencil" && "${NVCC}" -std=c++17 stencil_tensor_activation.cu -o "$2" \
        -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
        -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
        -Xptxas=-v \
        --allow-unsupported-compiler "${NVML_FLAGS[@]}" ) 2> "$3"
}

# GUARDA CONTRA LA COMPARACION VACIA.
#
# Si BASE y NUEVO tienen el MISMO .cu, todos los campos salen identicos por
# construccion y el veredicto verde no significa nada. Paso el 2026-08-28: un
# commit que solo tocaba este sbatch dejo BASE_COMMIT=HEAD~1 apuntando al mismo
# fuente, y el job devolvio 18/18 OK sin haber comparado nada. Se corta aqui.
CU_REL="Fase_3/Stencil/stencil_tensor_activation.cu"
SHA_CU_BASE="$(sha256sum "${BASE_TREE}/${CU_REL}" | awk '{print $1}')"
SHA_CU_NEW="$(sha256sum "${REPO_ROOT}/${CU_REL}" | awk '{print $1}')"
if [[ "${SHA_CU_BASE}" == "${SHA_CU_NEW}" ]]; then
    echo
    echo "ERROR: ${CU_REL} es IDENTICO en las dos revisiones"
    echo "       (sha256 ${SHA_CU_BASE})."
    echo "       El A/B seria una tautologia: cualquier resultado saldria OK."
    echo "       Fija la revision base correcta, p.ej.:"
    echo "         sbatch --export=ALL,BASE_COMMIT=<hash> validar_mma_sync.sbatch"
    git -C "${REPO_ROOT}" worktree remove --force "${BASE_TREE}" || true
    exit 1
fi
echo "  .cu BASE : ${SHA_CU_BASE}"
echo "  .cu NUEVO: ${SHA_CU_NEW}"
echo "  (difieren: la comparacion es real)"
echo

PTX_BASE="${PWD}/${WORK}/ptxas_base.txt"
PTX_NEW="${PWD}/${WORK}/ptxas_nuevo.txt"

compilar "${BASE_TREE}" "${BIN_BASE}" "${PTX_BASE}"
echo "  BASE  compilado: $(sha256sum "${BIN_BASE}" | awk '{print $1}')"
compilar "${REPO_ROOT}" "${BIN_NEW}" "${PTX_NEW}"
echo "  NUEVO compilado: $(sha256sum "${BIN_NEW}" | awk '{print $1}')"
echo "  (los sha256 DEBEN diferir: son revisiones distintas)"
echo

# --- Presion de registros ---------------------------------------------------
# mma.sync obliga a mantener los fragmentos en registros de forma explicita:
# 4 regs de A por cada uno de los dos operandos, 2 de B y 8 de acumulador por
# mitad de n. wmma hacia lo mismo, pero el reparto lo decidia el compilador. Si
# la version explicita provoca spills, el cambio se paga en trafico a memoria
# local: es criterio de rechazo, no aviso.
# Registros por instanciacion, desmangleando lo justo (tipo y CompMode) para
# poder comparar BASE y NUEVO linea a linea.
resumen_registros() {   # $1 = log de ptxas, $2 = etiqueta
    awk -v etq="$2" '
        /Compiling entry function/ && /stencil2d_wmma_kernel/ {
            tipo = (index($0, "__half") > 0) ? "FP16" : "BF16"
            i = index($0, "CompModeE")
            m = substr($0, i + 9, 1)
            modo = (m == "0") ? "Off" : ((m == "1") ? "Local" : "Spatial")
            pend = 1; next
        }
        pend && /Used [0-9]+ registers/ {
            for (j = 1; j <= NF; j++) if ($j == "Used") regs = $(j + 1)
            printf "     %-5s %-4s %-8s %3s registros\n", etq, tipo, modo, regs
            pend = 0
        }' "$1" | sort -k2,3
}

echo "-- Registros por instanciacion del kernel WMMA --"
resumen_registros "${PTX_BASE}" "BASE"
resumen_registros "${PTX_NEW}"  "NUEVO"
echo

# Deteccion de spills.
#
# NO basta con buscar la palabra "spill": el fuente declara una clase
# ReferenceSpill (volcado a disco del campo de referencia) cuyo nombre aparece
# en los avisos del compilador, y las propias lineas de ptxas dicen "0 bytes
# spill stores" -- con la palabra presente y el contador a cero. La primera
# version de esta comprobacion daba positivo por las dos razones a la vez.
#
# Lo correcto es extraer el NUMERO de cada contador y exigir que todos sean 0.
#
# Y NO se usa 'grep -q' dentro de una tuberia: con 'set -o pipefail' activo,
# grep -q cierra la tuberia en cuanto encuentra la primera coincidencia, el
# grep de aguas arriba muere por SIGPIPE y el pipeline devuelve estado distinto
# de cero. Es decir: justo cuando SI hay spills, la condicion daria falso. Un
# falso negativo silencioso, peor que el falso positivo que esto reemplaza.
# Se captura la salida en una variable y se prueba si esta vacia.
spills_nuevo=0
spill_no_cero="$(grep -oE '[0-9]+ bytes spill (stores|loads)' "${PTX_NEW}" \
                 | grep -vE '^0 bytes' || true)"
if [[ -n "${spill_no_cero}" ]]; then
    echo "  ATENCION: la revision NUEVA reporta spills distintos de cero:"
    printf '%s\n' "${spill_no_cero}" | sort | uniq -c | sed 's/^/     /'
    spills_nuevo=1
else
    n_cero=$(grep -cE '[0-9]+ bytes spill stores' "${PTX_NEW}" || true)
    echo "  OK: cero spill stores y cero spill loads (${n_cero} funciones comprobadas)."
fi
echo

# ===========================================================================
# ETAPA 2 — Igualdad bit a bit de los campos archivados
# ===========================================================================
echo "############################################################"
echo "# ETAPA 2: igualdad bit a bit BASE vs NUEVO"
echo "############################################################"
echo "Mallas: 64 y 512 tienen nx multiplo de 8 -> el desplazamiento ALINEA."
echo "        63 y 511 no lo son -> el desplazamiento NO alinea, pero debe"
echo "        seguir siendo CORRECTO. Ese es justamente el caso que atrapa un"
echo "        error de aritmetica de punteros."
echo

S="${PWD}/${WORK}/salidas"
fallos=0
casos=0

correr_par() {   # $1 = etiqueta, $2.. = flags comunes a las dos revisiones
    local etiqueta="$1"; shift
    local dir_base="${S}/${etiqueta}/base"
    local dir_new="${S}/${etiqueta}/nuevo"
    mkdir -p "${dir_base}" "${dir_new}"
    casos=$((casos + 1))

    "${BIN_BASE}" "$@" --archive-dir "${dir_base}" \
        > "${S}/${etiqueta}/base.log" 2>&1 || { echo "  ERROR: BASE fallo en ${etiqueta}"; fallos=$((fallos+1)); return; }
    "${BIN_NEW}"  "$@" --archive-dir "${dir_new}"  \
        > "${S}/${etiqueta}/nuevo.log" 2>&1 || { echo "  ERROR: NUEVO fallo en ${etiqueta}"; fallos=$((fallos+1)); return; }

    # sha256 de cada campo archivado, ordenado por nombre de fichero.
    ( cd "${dir_base}" && sha256sum *.bin | sort -k2 ) > "${S}/${etiqueta}/base.sha"
    ( cd "${dir_new}"  && sha256sum *.bin | sort -k2 ) > "${S}/${etiqueta}/nuevo.sha"

    local n_base; n_base=$(wc -l < "${S}/${etiqueta}/base.sha")
    if [[ "${n_base}" -eq 0 ]]; then
        echo "  FALLO ${etiqueta}: no se archivo ningun campo (revisa --checkpoint-iters)"
        fallos=$((fallos + 1)); return
    fi

    if diff -q "${S}/${etiqueta}/base.sha" "${S}/${etiqueta}/nuevo.sha" > /dev/null; then
        echo "  OK    ${etiqueta}  (${n_base} campos identicos byte a byte)"
    else
        echo "  FALLO ${etiqueta}: los campos DIFIEREN"
        diff "${S}/${etiqueta}/base.sha" "${S}/${etiqueta}/nuevo.sha" | sed 's/^/        /'
        fallos=$((fallos + 1))
    fi
}

COMUN=(--iters 20 --tc both --fp64-gpu off --cpu-fp64 off
       --checkpoint-iters 10,20 --archive-iters 10,20)

for NX in 64 512 63 511; do
    correr_par "nx${NX}_off"     --nx "${NX}" --ny "${NX}" "${COMUN[@]}"
    correr_par "nx${NX}_local"   --nx "${NX}" --ny "${NX}" "${COMUN[@]}" --kahan on
    correr_par "nx${NX}_spatial" --nx "${NX}" --ny "${NX}" "${COMUN[@]}" --spatial-comp on
done

# Operador difusivo + CI monomodo: la configuracion real de la campana de Fase 4.
for NX in 64 512; do
    correr_par "nx${NX}_dif_spatial" --nx "${NX}" --ny "${NX}" "${COMUN[@]}" \
        --spatial-comp on --op-mode diffusive --alpha 0.1875 --ci-mode monomode --ci-p 5
done

# CUDA Graphs (requisito 17): los grafos hornean punteros en sus nodos, asi que
# esta es la comprobacion de que la captura recibio los desplazados.
for NX in 64 512; do
    correr_par "nx${NX}_graph_off"     --nx "${NX}" --ny "${NX}" "${COMUN[@]}" \
        --execution-mode graph --graph-block 2
    correr_par "nx${NX}_graph_spatial" --nx "${NX}" --ny "${NX}" "${COMUN[@]}" \
        --spatial-comp on --execution-mode graph --graph-block 2
done

echo
# ===========================================================================
# ETAPA 2b — compute-sanitizer sobre la ruta vectorizada
# ===========================================================================
# El riesgo NUEVO son los accesos de 32 y 64 bits que sustituyen a
# load/store_matrix_sync: tc_ld32 lee dos elementos de 16 bits como un uint32_t
# y el volcado del acumulador escribe float2. La aritmetica dice que ambas
# direcciones son multiplos de 4 y de 8 B respectivamente (kLdX, kLdF y kTile
# son pares); esto lo comprueba en el hardware.
echo "############################################################"
echo "# ETAPA 2b: compute-sanitizer (memcheck) sobre la ruta vectorizada"
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
    echo "  compute-sanitizer no encontrado; se omite esta etapa (no cuenta como fallo)."
else
    echo "  Binario: ${SANITIZER_BIN}"
    for MODO in "" "--kahan on" "--spatial-comp on"; do
        etiqueta="sanit_${MODO// /}"
        [[ -z "${MODO}" ]] && etiqueta="sanit_off"
        log="${S}/${etiqueta}.log"
        mkdir -p "${S}"
        if "${SANITIZER_BIN}" --tool memcheck --error-exitcode 1 \
             "${BIN_NEW}" --nx 64 --ny 64 --iters 5 --tc both \
             --fp64-gpu off --cpu-fp64 off ${MODO} > "${log}" 2>&1; then
            echo "  OK    memcheck ${etiqueta}"
        else
            echo "  FALLO memcheck ${etiqueta} (ver ${log})"
            grep -iE "error|invalid|misaligned" "${log}" | head -10 | sed 's/^/        /'
            sanit_fallos=$((sanit_fallos + 1))
        fi
    done
fi
echo

# ===========================================================================
# ETAPA 3 — Rendimiento BASE vs NUEVO
# ===========================================================================
# Solo se mide si la correctitud paso: cronometrar un kernel que da otro
# resultado no informa de nada.
#
# GPU_FP32 es el CONTROL. Su codigo es identico en las dos revisiones (mma.sync
# solo toca stencil2d_wmma_kernel), asi que la diferencia entre sus tiempos
# mide el ruido del nodo. Si GPU_FP32 se mueve tanto como las rutas WMMA, la
# comparacion no vale.
echo "############################################################"
echo "# ETAPA 3: rendimiento BASE vs NUEVO"
echo "############################################################"

NX_MED="${NX_MED:-4096}"
ITERS_MED="${ITERS_MED:-50}"
REPLICAS="${REPLICAS:-5}"
CSV="${PWD}/${WORK}/mediciones.csv"

if [[ $((fallos + sanit_fallos)) -ne 0 ]]; then
    echo "  OMITIDA: la correctitud no paso (fallos=${fallos}, memcheck=${sanit_fallos})."
    medido=0
else
    medido=1
    echo "Malla: ${NX_MED}^2 · ${ITERS_MED} iteraciones · ${REPLICAS} replicas"
    echo

    declare -A BIN=([BASE]="${BIN_BASE}" [NUEVO]="${BIN_NEW}")

    COMUN_MED=(--nx "${NX_MED}" --ny "${NX_MED}" --iters "${ITERS_MED}" --tc both
               --fp64-gpu off --cpu-fp64 off
               --op-mode diffusive --alpha 0.1875 --ci-mode monomode --ci-p 5)

    echo "revision,replica,modo,route,kahan,t_iter_ms,t_kernel_ms" > "${CSV}"

    correr_medida() {   # $1 = revision, $2 = replica, $3 = modo, $4.. = flags
        local rev="$1" rep="$2" modo="$3"; shift 3
        local log="${PWD}/${WORK}/salidas/med_${rev}_r${rep}_${modo}.log"
        "${BIN[$rev]}" "${COMUN_MED[@]}" "$@" > "${log}" 2>&1
        awk -F, -v rev="${rev}" -v rep="${rep}" -v modo="${modo}" \
            '/^CSV_SUMMARY/ { printf "%s,%s,%s,%s,%s,%s,%s\n", rev, rep, modo, $2, $6, $7, $12 }' \
            "${log}" >> "${CSV}"
    }

    # Cebado descartado: la primera corrida sobre una GPU fria mide el arranque
    # del reloj, no el kernel.
    echo "-- cebado (descartado) --"
    "${BIN_NEW}" "${COMUN_MED[@]}" > "${PWD}/${WORK}/salidas/med_cebado.log" 2>&1
    echo "   hecho"

    for rep in $(seq 1 "${REPLICAS}"); do
        # El orden ALTERNA entre replicas: si el nodo deriva, la deriva no cae
        # siempre sobre la misma revision.
        if (( rep % 2 == 1 )); then ORDEN=(BASE NUEVO); else ORDEN=(NUEVO BASE); fi
        for rev in "${ORDEN[@]}"; do
            correr_medida "${rev}" "${rep}" off
            correr_medida "${rev}" "${rep}" local   --kahan on
            correr_medida "${rev}" "${rep}" spatial --spatial-comp on
        done
        echo "   replica ${rep}/${REPLICAS} completa (orden: ${ORDEN[*]})"
    done
    echo

    python3 - "${CSV}" <<'PY'
import csv, sys, statistics as st

filas = list(csv.DictReader(open(sys.argv[1])))
if not filas:
    print("  CSV vacio: revisa los .log de salidas/"); sys.exit(0)

datos = {}
for f in filas:
    k = (f["route"], f["modo"])
    datos.setdefault(k, {}).setdefault(f["revision"], []).append(float(f["t_iter_ms"]))

def stats(v):
    m = st.mean(v)
    cv = (st.stdev(v) / m * 100) if len(v) > 1 and m else 0.0
    return m, cv

print(f"{'ruta':<16}{'modo':<9}{'BASE wmma':>12}{'NUEVO mma':>12}"
      f"{'NUEVO/BASE':>12}{'CV% BASE':>10}{'CV% NUEVO':>11}")
print("-" * 82)
for k in sorted(datos):
    route, modo = k
    d = datos[k]
    if "BASE" not in d or "NUEVO" not in d:
        continue
    mb, cb = stats(d["BASE"])
    mn, cn = stats(d["NUEVO"])
    rel = (mb / mn) if mn else 0.0
    print(f"{route:<16}{modo:<9}{mb:12.5f}{mn:12.5f}{rel:11.3f}x{cb:10.2f}{cn:11.2f}")
print()
print("NUEVO/BASE > 1.000x = mma.sync es mas rapido. No se espera ganancia:")
print("el criterio es que no haya PERDIDA grande. GPU_FP32 es el control y")
print("debe salir en 1.000x; si no, el nodo derivo y la tabla no vale.")
PY
fi
echo

# ===========================================================================
# ETAPA 4 — Veredicto
# ===========================================================================
echo "############################################################"
echo "# ETAPA 4: veredicto"
echo "############################################################"
echo "Casos ejecutados      : ${casos}"
echo "Casos con FALLO       : ${fallos}"
echo "memcheck con FALLO    : ${sanit_fallos}"
echo "Spills en la revision NUEVA: ${spills_nuevo}"
git -C "${REPO_ROOT}" worktree remove --force "${BASE_TREE}" || true
total=$((fallos + sanit_fallos + spills_nuevo))
if [[ "${total}" -eq 0 ]]; then
    echo
    echo "VEREDICTO: mma.sync reproduce wmma BIT A BIT, sin accesos invalidos y"
    echo "sin spills. El reparto de fragmentos es correcto y la ETAPA B"
    echo "(swizzle de la shared) queda habilitada."
    if [[ "${medido}" -eq 1 ]]; then
        echo "La tabla de la ETAPA 3 dice cuanto cuesta el paso intermedio."
    fi
    exit 0
else
    echo
    echo "VEREDICTO: NO autorizado."
    # if/then y no [[ ]] && echo: con set -e una condicion falsa abortaria el
    # script y se perderian las lineas siguientes.
    if [[ "${fallos}"       -ne 0 ]]; then echo "  - hay diferencias numericas: revisa ${WORK}/salidas/<caso>/{base,nuevo}.sha"; fi
    if [[ "${sanit_fallos}" -ne 0 ]]; then echo "  - compute-sanitizer reporta accesos invalidos"; fi
    if [[ "${spills_nuevo}" -ne 0 ]]; then echo "  - ptxas reporta spills: los fragmentos explicitos no caben en registros"; fi
    exit 1
fi
