#!/bin/bash
#SBATCH --job-name=f4_perfil_wmma
#SBATCH --output=logs/f4_perfil_wmma_%j.out
#SBATCH --error=logs/f4_perfil_wmma_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --exclusive
#SBATCH --time=01:30:00

# ===========================================================================
# PERFILADO: por que el kernel WMMA se queda al ~41 % del ancho de banda
# mientras GPU_FP32 llega al ~75 %.
#
# La campana de alineacion + vectorizacion (jobs 6719-6729) cerro solo entre
# 2.7 % y 7.5 % de ese hueco. La hipotesis microarquitectonica de la alineacion
# esta MEDIDA y es insuficiente: la causa esta en otro sitio y este job existe
# para encontrarla con contadores, no con razonamiento.
#
# Se perfila el kernel WMMA y, en la MISMA corrida, stencil2d_fp32_kernel como
# referencia: la pregunta no es "cuanto rinde WMMA" sino "en que se diferencia
# de la ruta que si satura".
#
# Metricas: se reutilizan los tres grupos ya definidos y probados en
# tools/common_ncu.sh (quick, coalescing, warp stalls). El cuarto grupo
# -- conflictos de banco y limitantes de ocupancia -- es NUEVO y por eso va en
# una invocacion aparte y tolerante a fallo: si algun nombre de metrica no
# existe en esta version de ncu, se pierde ese grupo y no el job entero.
#
# Uso:
#   sbatch perfilar_wmma.sbatch
#   sbatch --export=ALL,NX=16384 perfilar_wmma.sbatch
# ===========================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

REPO_ROOT="$(git rev-parse --show-toplevel)"
CUDA_ARCH="${CUDA_ARCH:-80}"
NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"
NX="${NX:-4096}"
ITERS="${ITERS:-10}"

JOB_ID="${SLURM_JOB_ID:-manual}"
WORK="perfil_wmma_${JOB_ID}"
mkdir -p "${WORK}/csv" "${WORK}/salidas"

echo "Nodo        : $(hostname)"
echo "Fecha       : $(date -Iseconds)"
echo "Commit      : $(git -C "${REPO_ROOT}" rev-parse HEAD)"
echo "Malla       : ${NX}^2 · ${ITERS} iteraciones"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

# --- ncu: misma convencion de ruta que run_stencil_tc.sbatch de Fase_3 -------
NCU="${NCU:-${HOME}/fcrojasv/nsight_compute/ncu}"
if [[ ! -x "${NCU}" ]]; then
    _d="$(dirname "${NVCC}")"
    for cand in "$(command -v ncu || true)" "${_d}/ncu" \
                "${_d}/../../cuda/bin/ncu" "${_d}"/../../cuda/*/bin/ncu \
                /opt/nvidia/nsight-compute/*/ncu; do
        if [[ -n "${cand}" && -x "${cand}" ]]; then NCU="${cand}"; break; fi
    done
fi
if [[ ! -x "${NCU}" ]]; then
    echo "ERROR: no se encontro ncu. Exporta NCU=/ruta/a/ncu y relanza."
    exit 1
fi
echo "ncu: ${NCU}  ($("${NCU}" --version 2>/dev/null | head -1 || echo version desconocida))"

# Definiciones compartidas: NCU_QUICK_METRICS, NCU_WARP_STALL_METRICS,
# NCU_COALESCING_METRICS y ncu_run(). Se declaran una sola vez en tools/.
for _c in "${REPO_ROOT}/tools/common_ncu.sh" "../../tools/common_ncu.sh"; do
    if [[ -f "${_c}" ]]; then . "${_c}"; COMMON_NCU="${_c}"; break; fi
done
if [[ -z "${COMMON_NCU:-}" ]]; then echo "ERROR: no se encontro tools/common_ncu.sh"; exit 1; fi
echo "metricas: ${COMMON_NCU}"
echo

# Grupo NUEVO. Responde dos preguntas que los otros tres no cubren:
#   - conflictos de banco en shared: la vectorizacion introdujo accesos de
#     128 bits a shared (LDS.128 del acumulador). Si generan conflictos, parte
#     del beneficio se esta pagando ahi.
#   - que limita la ocupancia: shared, registros, warps o bloques. El kernel
#     WMMA usa 32-40 registros y 6656/11776 B de shared por bloque; saber cual
#     es el techo decide si vale la pena tocar el reparto de shared.
NCU_EXTRA_METRICS="l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps,launch__occupancy_limit_blocks,launch__waves_per_multiprocessor,gpu__time_duration.sum"

NVML_LINK_FLAG="${NVML_LINK_FLAG:--lnvidia-ml}"
if "${NVCC}" -E -x cu - 2>/dev/null <<< '#include <nvml.h>' >/dev/null &&
   (ldconfig -p 2>/dev/null | grep -Eq 'lib(nvidia-ml|nvml)\.so' ||
    [[ -f /usr/lib64/libnvidia-ml.so || -f /usr/lib64/libnvml.so ]]); then
    NVML_FLAGS=(-DUSE_NVML_TELEMETRY "${NVML_LINK_FLAG}" -lpthread)
else
    NVML_FLAGS=(-lpthread)
fi

# ===========================================================================
# ETAPA 1 — Binario
# ===========================================================================
echo "############################################################"
echo "# ETAPA 1: compilacion"
echo "############################################################"
BIN="${PWD}/${WORK}/stencil_tc"
( cd "${REPO_ROOT}/Fase_3/Stencil" && "${NVCC}" -std=c++17 stencil_tensor_activation.cu -o "${BIN}" \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
    --allow-unsupported-compiler "${NVML_FLAGS[@]}" )
echo "  sha256: $(sha256sum "${BIN}" | awk '{print $1}')"
echo

# ===========================================================================
# ETAPA 2 — Perfilado
# ===========================================================================
echo "############################################################"
echo "# ETAPA 2: perfilado"
echo "############################################################"
# Se saltan las kWarmupIters=3 primeras invocaciones y se perfila UNA sola en
# regimen estacionario: perfilar un warm-up mediria el arranque, no el kernel.
SKIP="${NCU_LAUNCH_SKIP:-3}"
BASE_ARGS=(--nx "${NX}" --ny "${NX}" --iters "${ITERS}" --profile-only
           --op-mode diffusive --alpha 0.1875 --ci-mode monomode --ci-p 5)

perfilar() {   # $1 = etiqueta, $2 = regex de kernel, $3.. = flags del binario
    local etq="$1" regex="$2"; shift 2
    local pasos=0 ok=0
    for grupo in QUICK STALL EXTRA; do
        case "${grupo}" in
            QUICK) MET="${NCU_QUICK_METRICS},${NCU_COALESCING_METRICS}" ;;
            STALL) MET="${NCU_WARP_STALL_METRICS}" ;;
            EXTRA) MET="${NCU_EXTRA_METRICS}" ;;
        esac
        pasos=$((pasos + 1))
        local csv="${PWD}/${WORK}/csv/${etq}_${grupo}.csv"
        if ncu_run "${NCU}" --csv --page raw \
                --target-processes all \
                --kernel-name "regex:${regex}" \
                --launch-skip "${SKIP}" --launch-count 1 \
                --metrics "${MET}" \
                "${BIN}" "$@" > "${csv}" 2> "${PWD}/${WORK}/salidas/${etq}_${grupo}.err"; then
            ok=$((ok + 1))
        else
            echo "     AVISO: grupo ${grupo} fallo en ${etq} (ver salidas/${etq}_${grupo}.err)"
            rm -f "${csv}"
        fi
    done
    echo "  ${etq}: ${ok}/${pasos} grupos de metricas"
}

RX_WMMA=".*stencil2d_wmma_kernel.*"
RX_FP32=".*stencil2d_fp32_kernel.*"

# GPU_FP32 es la REFERENCIA: la ruta que si llega al 75 %. Se perfila desde la
# misma invocacion del binario, con la misma malla y el mismo operador.
perfilar "FP32_ref"     "${RX_FP32}" "${BASE_ARGS[@]}" --tc fp16
perfilar "WMMA_FP16_off"     "${RX_WMMA}" "${BASE_ARGS[@]}" --tc fp16
perfilar "WMMA_BF16_off"     "${RX_WMMA}" "${BASE_ARGS[@]}" --tc bf16
perfilar "WMMA_FP16_local"   "${RX_WMMA}" "${BASE_ARGS[@]}" --tc fp16 --kahan on
perfilar "WMMA_FP16_spatial" "${RX_WMMA}" "${BASE_ARGS[@]}" --tc fp16 --spatial-comp on
echo

# ===========================================================================
# ETAPA 3 — Tabla comparativa
# ===========================================================================
echo "############################################################"
echo "# ETAPA 3: comparativa (FP32 es la referencia que SI satura)"
echo "############################################################"
python3 - "${PWD}/${WORK}/csv" <<'PY'
import csv, glob, os, sys, re

carpeta = sys.argv[1]
datos = {}   # etiqueta -> {metrica: valor}
for ruta in sorted(glob.glob(os.path.join(carpeta, "*.csv"))):
    etq = os.path.basename(ruta).rsplit("_", 1)[0]
    with open(ruta, newline="") as fh:
        # ncu --csv antepone lineas de aviso antes de la cabecera real
        lineas = [l for l in fh if l.strip()]
    inicio = next((i for i, l in enumerate(lineas) if '"ID"' in l or l.startswith("ID,")), None)
    if inicio is None:
        continue
    for fila in csv.DictReader(lineas[inicio:]):
        nombre = (fila.get("Metric Name") or "").strip()
        bruto = (fila.get("Metric Value") or "").strip().replace(",", "")
        if not nombre:
            continue
        try:
            datos.setdefault(etq, {})[nombre] = float(bruto)
        except ValueError:
            datos.setdefault(etq, {})[nombre] = bruto

if not datos:
    print("  Sin datos: revisa los .err de salidas/")
    sys.exit(0)

FILAS = [
    ("DRAM % del pico",        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "{:.1f}"),
    ("SM % del pico",          "sm__throughput.avg.pct_of_peak_sustained_elapsed",       "{:.1f}"),
    ("ocupancia alcanzada %",  "sm__warps_active.avg.pct_of_peak_sustained_active",      "{:.1f}"),
    ("registros/hilo",         "launch__registers_per_thread",                           "{:.0f}"),
    ("DRAM leidos (MB)",       "dram__bytes_read.sum",                                   "{:.1f}"),
    ("DRAM escritos (MB)",     "dram__bytes_write.sum",                                  "{:.1f}"),
    ("sectores/request ld",    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio", "{:.2f}"),
    ("sectores/request st",    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio", "{:.2f}"),
    ("bytes/sector ld %",      "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct", "{:.1f}"),
    ("bytes/sector st %",      "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct", "{:.1f}"),
    ("acierto L1 %",           "l1tex__t_sector_hit_rate.pct",                           "{:.1f}"),
    ("stall long_scoreboard",  "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", "{:.2f}"),
    ("stall wait",             "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",           "{:.2f}"),
    ("stall barrier",          "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",        "{:.2f}"),
    ("stall mio_throttle",     "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",   "{:.2f}"),
    ("stall lg_throttle",      "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",    "{:.2f}"),
    ("issue activo %",         "smsp__issue_active.avg.pct_of_peak_sustained_active",    "{:.1f}"),
    ("conflictos banco ld",    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", "{:.0f}"),
    ("conflictos banco st",    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", "{:.0f}"),
    ("limite ocupancia shared","launch__occupancy_limit_shared_mem",                     "{:.0f}"),
    ("limite ocupancia regs",  "launch__occupancy_limit_registers",                      "{:.0f}"),
    ("limite ocupancia warps", "launch__occupancy_limit_warps",                          "{:.0f}"),
]
ORDEN = ["FP32_ref", "WMMA_FP16_off", "WMMA_BF16_off", "WMMA_FP16_local", "WMMA_FP16_spatial"]
cols = [c for c in ORDEN if c in datos] + [c for c in sorted(datos) if c not in ORDEN]

anchoi = max(len(f[0]) for f in FILAS) + 2
print(" " * anchoi + "".join(f"{c:>20}" for c in cols))
print("-" * (anchoi + 20 * len(cols)))
for etiqueta, metrica, fmt in FILAS:
    celdas = []
    for c in cols:
        v = datos[c].get(metrica)
        if v is None:
            celdas.append(f"{'-':>20}")
        elif isinstance(v, float):
            if "MB" in etiqueta:
                v = v / 1e6
            celdas.append(f"{fmt.format(v):>20}")
        else:
            celdas.append(f"{str(v):>20}")
    print(f"{etiqueta:<{anchoi}}" + "".join(celdas))

print()
print("Como leerla:")
print("  La columna FP32_ref es la ruta que llega al ~75 %. Cada fila donde")
print("  WMMA se separa mucho de ella es una hipotesis sobre el 41 %.")
print("  - DRAM % bajo + stall long_scoreboard alto -> latency-bound, no")
print("    bandwidth-bound: el kernel espera, no satura. La solucion seria mas")
print("    paralelismo en vuelo (ocupancia o accesos independientes), no menos")
print("    bytes.")
print("  - sectores/request > 4 en ld/st -> accesos aun descoalescidos.")
print("  - stall barrier alto -> los __syncwarp()/__syncthreads() del tile")
print("    dominan; el reparto 1 warp = 1 tile seria el sospechoso.")
print("  - conflictos de banco altos -> el precio de los accesos de 128 bits")
print("    a shared que introdujo la vectorizacion.")
print("  - 'limite ocupancia' mas bajo = el recurso que la esta capando.")
PY

echo
echo "CSV crudos: ${PWD}/${WORK}/csv/"
