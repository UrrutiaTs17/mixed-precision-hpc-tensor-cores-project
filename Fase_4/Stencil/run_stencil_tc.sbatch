#!/bin/bash
#SBATCH --job-name=mixed_precision_stencil_tc_f4
#SBATCH --output=logs/mixed_precision_stencil_tc_f4_%j.out
#SBATCH --error=logs/mixed_precision_stencil_tc_f4_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=48:00:00

# Fase 4: mismo binario y mismos parametros que Fase_3/Stencil/run_stencil_tc.sbatch
# (del que este script parte, ver Fase_4/Stencil/README.md), mas ANCHOR_LIST
# para barrer --anchor-every (ancla FP64). Con ANCHOR_LIST="0" (el default) el
# comportamiento es identico al de Fase 3 -- ver la seccion "ANCLA FP64" mas
# abajo para el resto de las variables nuevas.
#
#   sbatch run_stencil_tc.sbatch                              # ANCHOR_LIST=0 (sin ancla)
#   sbatch --export=ALL,ANCHOR_LIST="0 1" run_stencil_tc.sbatch   # gates K=0 / K=1
#   sbatch --export=ALL,ANCHOR_LIST="2 4 8 16" run_stencil_tc.sbatch  # barrido

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f stencil_tensor_activation.cu && -f Stencil/stencil_tensor_activation.cu ]]; then
    cd Stencil
fi

# Directorios de salida, parametrizados: ningun script de post-proceso ni
# ninguna ruta de log queda hardcodeada a "logs"/"results" salvo como default.
LOGS_DIR="${LOGS_DIR:-logs}"
RESULTS_DIR="${RESULTS_DIR:-results}"
mkdir -p "${LOGS_DIR}" "${RESULTS_DIR}"
JOB_ID="${SLURM_JOB_ID:-manual}"
RESULTS_LOG="${RESULTS_DIR}/run_${JOB_ID}.log"

# stencil_tc usa mma.h / cuda_fp16.h / cuda_bf16.h y, si el cluster lo expone,
# NVML para telemetria; OpenBLAS/cuDNN no aplican aqui.

# Arquitectura por defecto para A100.
# Puedes sobrescribirla con:
#   sbatch --export=ALL,CUDA_ARCH=86 run_stencil_tc.sbatch
# --- Deteccion portable del toolchain (tools/detect_toolchain.sh) --------
# Corre igual bajo SLURM en PACCA o con `bash` directo en cualquier otra
# maquina con el entorno conda de environment.yml activo -- ver el
# comentario de cabecera de ese script.
for _toolchain in "../../tools/detect_toolchain.sh" "./detect_toolchain.sh"; do
    if [[ -f "${_toolchain}" ]]; then
        # shellcheck source=/dev/null
        source "${_toolchain}"
        break
    fi
done
CUDA_ARCH="$(detect_cuda_arch)"  # A100=80, RTX 3050=86, V100=70, Ada=89, Hopper=90

# Rutas del toolchain: especificas del cluster historico (PACCA/HPC SDK).
# Sobrescribir con --export si el cluster de destino tiene otro layout.
NVCC="$(detect_nvcc)"
NCU="$(detect_ncu || true)"
if [[ -z "${NCU}" ]]; then
    echo "AVISO: Nsight Compute (ncu) no encontrado -- se deshabilita el perfilado (RUN_NCU forzado a 0)." >&2
    RUN_NCU=0
fi

# NVML es opcional: cuando nvml.h y la biblioteca del driver estan visibles,
# se habilita la telemetria y se enlaza con libnvidia-ml. Si el cluster no
# expone alguno de los dos, el mismo fuente compila con el stub sin
# telemetria (ver common/power_sampling.h).
NVML_FLAGS=()
# El driver NVIDIA suele instalarse como -lnvidia-ml; algunos modulos de
# cluster ofrecen -lnvml como alias.
NVML_LINK_FLAG="${NVML_LINK_FLAG:--lnvidia-ml}"
if [[ "${DISABLE_NVML_TELEMETRY:-0}" != "1" ]] &&
   "${NVCC}" -E -x cu - 2>/dev/null <<< '#include <nvml.h>' >/dev/null &&
   (ldconfig -p 2>/dev/null | grep -Eq 'lib(nvidia-ml|nvml)\.so' ||
    [[ -f /usr/lib64/libnvidia-ml.so || -f /usr/lib64/libnvml.so ||
       -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so ||
       -f /usr/lib/x86_64-linux-gnu/libnvml.so ]]); then
    NVML_FLAGS=(-DUSE_NVML_TELEMETRY "${NVML_LINK_FLAG}" -lpthread)
    echo "NVML habilitado para telemetria GPU."
else
    NVML_FLAGS=(-lpthread)
    echo "NVML no disponible para el enlace; se usara el stub de telemetria GPU."
fi

SMOKE_TEST="${SMOKE_TEST:-0}"
if [[ "${SMOKE_TEST}" == "1" ]]; then
    RUN_NCU="${RUN_NCU:-0}"
else
    RUN_NCU="${RUN_NCU:-1}"
fi
NCU_SET="${NCU_SET:-full}"
# Se perfilan tres kernels en invocaciones ncu SEPARADAS (un regex cada una):
# el WMMA (template sobre T = __half / __nv_bfloat16 y sobre CompMode, asi que
# el regex cubre todas las instanciaciones; cual se lanza lo deciden los flags
# del binario en cada invocacion, no el nombre), el GPU FP32 clasico y las
# conversiones float<->16 bits que corren dentro de la ruta WMMA.
NCU_KERNEL_REGEX_WMMA="${NCU_KERNEL_REGEX_WMMA:-.*stencil2d_wmma_kernel.*}"
NCU_KERNEL_REGEX_FP32="${NCU_KERNEL_REGEX_FP32:-.*stencil2d_fp32_kernel.*}"
NCU_KERNEL_REGEX_CONVERT="${NCU_KERNEL_REGEX_CONVERT:-.*convert_float_to_(half|bfloat16)_kernel.*}"
# La compensacion espacial NO es un kernel aparte: es la instanciacion
# stencil2d_wmma_kernel<T, CompMode::Spatial> del MISMO template (ver
# launch_wmma en stencil_tensor_activation.cu, que elige Off/Local/Spatial en
# tiempo de compilacion). Por eso el regex es el mismo por defecto: lo que
# distingue una politica de otra no es el nombre del kernel sino los flags con
# que se invoca el binario, y cada llamada a ncu lanza una sola instanciacion.
NCU_KERNEL_REGEX_WMMA_SP="${NCU_KERNEL_REGEX_WMMA_SP:-${NCU_KERNEL_REGEX_WMMA}}"
# Debe coincidir con kWarmupIters en stencil_tensor_activation.cu (este mismo
# directorio, Fase_3/Stencil); se deriva del propio fuente en vez de repetir
# el literal para que ambos no puedan desincronizarse si kWarmupIters cambia.
KWARMUP_ITERS_CU="$(grep -m1 'kWarmupIters *=' stencil_tensor_activation.cu | grep -oE '[0-9]+' | head -n1)"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-${KWARMUP_ITERS_CU}}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-1}"

# --- Tamano de malla y barrido de iteraciones -------------------------------
# NX/NY: SIN limite superior impuesto por este script -- el default (16384) es
# el tamano historicamente calibrado (presupuesto de memoria, walltime), pero
# nada impide correr a 512/1024/2048. AUDITORIA (ver Fase_3/Stencil/README.md,
# "Hallazgos de la auditoria"): el histerico del proyecto nunca ejercito una
# campana real en el rango 512^2-2048^2 pese a que el flag siempre lo permitio;
# antes de generar datos en ese rango conviene revalidar al menos el horizonte
# de overflow medido alli.
NX="${NX:-16384}"
NY="${NY:-16384}"
# Formato(s) Tensor Core de las corridas de MEDICION (--tc del binario).
TC_FORMAT="${TC_FORMAT:-both}"
# Barrido de ITERACIONES: 10..120 caen dentro del horizonte de FP32/BF16/FP16
# a los NX/NY por defecto (produce L2/Linf finitas). El horizonte de
# divergencia en si (donde SI importa iters grande) se mide aparte, en
# run_stencil_horizon.sbatch.
ITERS_LIST="${ITERS_LIST:-10 50 100 120}"
# Necesario para que storage_rel (error relativo de guardar en FP16/BF16) sea
# evaluable en las corridas donde la ruta diverge antes de ITERS (p.ej. FP16
# con n*~28 en corridas con ITERS>28): sin --checkpoint-every, stencil_tc solo
# puede capturar un estado FP32 finito en la ULTIMA iteracion medida, y si esa
# iteracion ya diverguio no hay forma de recuperar un estado finito posterior
# al hecho (ver storage_rel_eval_iter en benchmark_gpu_tensor_core_stencil,
# stencil_tensor_activation.cu). El default (5) fue calibrado para NX/NY=16384;
# a mallas mucho mas chicas o con otro --op-mode/--ci-mode puede convenir otro
# valor (donde diverja FP16 con esa condicion inicial).
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"

# --- Tipo de corrida: energetica vs numerica --------------------------------
# El gate de energy_window_reliable es el de common/power_sampling.h:
#
#     window_reliable  <=>  time_total_s >= 0.5 s * gpu_segment_count
#
# y cada checkpoint abre un tramo, porque el contador NVML se cuantiza POR
# TRAMO y no sobre la suma (ver "REGIMEN DE VALIDEZ" en
# common/power_sampling.h). Con CHECKPOINT_EVERY=5 e ITERS=120 son 24 tramos:
# 12 s de ventana exigidos contra un tiempo de ruta GPU tipicamente muy por
# debajo de eso a mallas grandes. Por eso una corrida con checkpoints activos
# puede reportar reliable=0 en casi todas sus filas pese a que NVML SI este
# midiendo: no falla el contador, falla el protocolo.
#
#   energy  -> sin checkpoints ni perfilado. Se REPLICA: es la unica via para
#              reportar variabilidad, y cualquier instrumentacion contamina
#              por igual t_iter_ms y la ventana de energia.
#   numeric -> con checkpoints (comportamiento historico, default). Una sola
#              corrida basta: el error es determinista.
#
# Las dos variables se fuerzan DESPUES de leer el entorno, a proposito: un
# --export=ALL arrastra el CHECKPOINT_EVERY o el RUN_NCU que quedara exportado
# de otra sesion, y una sola corrida instrumentada dentro de una campana
# energetica arruina la replica sin dejar rastro evidente en el CSV.
RUN_KIND="${RUN_KIND:-numeric}"
case "${RUN_KIND}" in
    energy)
        if [[ "${CHECKPOINT_EVERY}" != "0" ]]; then
            echo "AVISO: RUN_KIND=energy ignora CHECKPOINT_EVERY=${CHECKPOINT_EVERY} y lo fija en 0."
        fi
        CHECKPOINT_EVERY=0
        if [[ "${RUN_NCU}" != "0" ]]; then
            echo "AVISO: RUN_KIND=energy ignora RUN_NCU=${RUN_NCU}: perfilar invalida la energia."
        fi
        RUN_NCU=0
        ;;
    numeric)
        ;;
    *)
        echo "ERROR: RUN_KIND debe ser 'energy' o 'numeric' (recibido: ${RUN_KIND})." >&2
        exit 2
        ;;
esac

# Restriccion de tiempo: NCU solo se corre para el valor de ITERS de la lista
# que coincida con ITERS_NCU, no para todos -- perfilar en cada ITERS
# multiplicaria el tiempo de cola. Reportes por corrida perfilada: 9 = FP32
# clasico (1) + conversion fp16/bf16 (2) + WMMA fp16/bf16 x las 3 politicas de
# NCU_COMP_LIST (6).
#
# ATENCION: ITERS_NCU e ITERS_LIST son independientes y hay que alinearlas a
# mano. Si ningun valor de ITERS_LIST es igual a ITERS_NCU no se genera ningun
# reporte; el script lo avisa (al arrancar y en cada corrida omitida) en vez de
# saltarse el bloque en silencio.
ITERS_NCU="${ITERS_NCU:-100}"
# Diseno factorial {fp16,bf16} x {kahan off,on} x ITERS_LIST: cada valor de
# ITERS_LIST corre completo con --kahan off y con --kahan on. Kahan solo
# cambia las rutas WMMA (ver stencil_tensor_activation.cu); GPU FP32 clasico y
# las referencias CPU FP32/FP64 dan identico resultado en ambas pasadas, pero
# se recalculan de todas formas (una invocacion == una corrida completa).
# OJO: con el default actual (SPATIAL_COMP=on, ver abajo) esta lista se fuerza
# a "off" y el barrido factorial de Kahan NO corre en las mediciones. Para
# recuperarlo hay que pedir explicitamente SPATIAL_COMP=off.
KAHAN_LIST="${KAHAN_LIST:-off on}"

# --- Politica de compensacion (none / kahan_local / spatial) ----------------
# El binario expone esto como DOS flags booleanos independientes, no como un
# unico selector: --kahan off|on (politica "kahan_local" en el README) y
# --spatial-comp off|on (politica "spatial"), mutuamente excluyentes; "none"
# es --kahan off --spatial-comp off, el default del binario. Este script
# respeta esa misma forma en vez de inventar un tercer nombre de variable.
#
# El default de ESTE SCRIPT se invirtio de "off" a "on" porque la espacial es
# la unica de las tres politicas que efectivamente mitiga el error medido:
# Kahan LOCAL puede ser indistinguible de no compensar o incluso empeorar el
# error segun el operador activo (ver Fase_3/Stencil/README.md, "Compensacion
# del redondeo de almacenamiento"), a un costo de ~1.97x en t_iter_ms; la
# espacial baja rel_l2 varios ordenes de magnitud a ~1.56x.
#
# El binario RECHAZA --kahan on junto con --spatial-comp on: con SPATIAL_COMP=on
# se fuerza KAHAN_LIST=off aqui, porque con `set -e` una combinacion invalida
# mataria el job a mitad del barrido. SPATIAL_COMP=off es la via para obtener
# el barrido factorial de Kahan local (kahan off y on):
#   sbatch run_stencil_tc.sbatch                     -> WMMA_*_SP
#   sbatch --export=ALL,SPATIAL_COMP=off run_stencil_tc.sbatch
#                                     -> WMMA_FP16/BF16 con kahan off y on
#
# Nota: esto solo fija la politica de las corridas de MEDICION. El perfilado
# NCU no depende de esta variable para el kernel WMMA: perfila las tres
# politicas en la misma corrida (ver NCU_COMP_LIST).
SPATIAL_COMP="${SPATIAL_COMP:-on}"
if [[ "${SPATIAL_COMP}" == "on" ]]; then
    KAHAN_LIST="off"
fi

# --- ANCLA FP64 (Fase 4) -----------------------------------------------------
# ANCHOR_LIST: valores de K (--anchor-every) a barrer, separados por espacio.
# Por defecto "0" -- deshabilitado, comportamiento identico a Fase 3. Ver
# Fase_4/Stencil/README.md y la seccion 01/02 del documento de plan para el
# mecanismo, la escala de K recomendada por formato, y los tres gates que hay
# que correr ANTES de confiar en cualquier K>0 de una campana real:
#   sbatch --export=ALL,ANCHOR_LIST="0 1" run_stencil_tc.sbatch   # gates K=0/K=1
#   sbatch --export=ALL,ANCHOR_LIST="2 4 8 16" run_stencil_tc.sbatch  # barrido
#
# El binario RECHAZA --anchor-every > 0 sin --spatial-comp on (ver parse_args
# en el .cu): con `set -e` una combinacion invalida mataria el job a mitad del
# barrido, asi que se valida aqui primero, con un mensaje claro, en vez de
# dejar que el binario aborte a mitad de una campana larga.
ANCHOR_LIST="${ANCHOR_LIST:-0}"
if [[ "${SPATIAL_COMP}" != "on" ]]; then
    for _anchor_val in ${ANCHOR_LIST}; do
        if [[ "${_anchor_val}" != "0" ]]; then
            echo "ERROR: ANCHOR_LIST='${ANCHOR_LIST}' incluye K=${_anchor_val} > 0, pero" \
                 "SPATIAL_COMP='${SPATIAL_COMP}' (debe ser 'on'): el ancla FP64 requiere" \
                 "--spatial-comp on. Agregue --export=ALL,SPATIAL_COMP=on o quite los" \
                 "valores distintos de 0 de ANCHOR_LIST." >&2
            exit 1
        fi
    done
fi

# --- Operador y condicion inicial -------------------------------------------
# Los DEFAULTS de abajo reproducen el comportamiento historico (stress +
# legacy, los defaults del binario): una corrida que no exporte nada da el
# mismo resultado que antes de exponer estos flags. El operador difusivo se
# pide explicitamente:
#   sbatch --export=ALL,OP_MODE=diffusive,CI_MODE=monomode run_stencil_tc.sbatch
#
# ALPHA solo se pasa con --op-mode diffusive y CI_P/CI_AMPLITUDE solo con
# --ci-mode monomode: el binario RECHAZA --alpha junto a stress (es un error,
# no un no-op silencioso), asi que con `set -e` pasarlo siempre mataria el job.
OP_MODE="${OP_MODE:-stress}"
ALPHA="${ALPHA:-0.1875}"
CI_MODE="${CI_MODE:-legacy}"
CI_P="${CI_P:-168}"
CI_AMPLITUDE="${CI_AMPLITUDE:-1.0}"

OP_ARGS=(--op-mode "${OP_MODE}" --ci-mode "${CI_MODE}")
[[ "${OP_MODE}" == "diffusive" ]] && OP_ARGS+=(--alpha "${ALPHA}")
[[ "${CI_MODE}" == "monomode" ]] && OP_ARGS+=(--ci-p "${CI_P}" --ci-amplitude "${CI_AMPLITUDE}")

# Ruta GPU_FP64 (referencia de maxima precision EN GPU): ON por defecto, de modo
# que TODA corrida del barrido la produzca para los MISMOS NX/NY/ITERS que las
# rutas WMMA. Esa coincidencia es el requisito para poder unir sus filas por
# (nx, ny, iters) y calcular el speedup de las rutas WMMA contra un
# denominador GPU, sin cruzar dispositivos.
# FP64_GPU=off la omite (por ejemplo, para reducir memoria a mallas grandes):
#   sbatch --export=ALL,FP64_GPU=off run_stencil_tc.sbatch
FP64_GPU="${FP64_GPU:-on}"

# Ruta CPU_FP64: cronometra el patron de oro IEEE 754 (double serial en CPU),
# referencia de COSTO del proyecto. El ERROR contra FP64 de CPU se mide siempre
# (es el ground truth de todas las rutas); esta ruta agrega su TIEMPO y su
# ENERGIA. Cuesta una segunda pasada FP64 sobre la malla, asi que a ITERS
# grandes es la parte mas cara de la corrida.
# CPU_FP64=off la omite (barridos donde solo interesan las rutas GPU):
#   sbatch --export=ALL,CPU_FP64=off run_stencil_tc.sbatch
CPU_FP64="${CPU_FP64:-on}"

# Definiciones compartidas de perfilado (NCU_QUICK_METRICS). Se declara una
# sola vez en tools/common_ncu.sh (fuera de cualquier Fase_N) y se reutiliza
# entre los .sbatch de todas las fases; este script corre desde
# Fase_3/Stencil, de ahi "../../tools/common_ncu.sh" como ruta principal.
for _common_ncu in "../../tools/common_ncu.sh" "./common_ncu.sh"; do
    if [[ -f "${_common_ncu}" ]]; then
        # shellcheck source=/dev/null
        source "${_common_ncu}"
        break
    fi
done

# Separacion validacion/caracterizacion en NCU:
#   quick -> --metrics "${NCU_QUICK_METRICS}" (rapido, valida Tensor Cores)
#   full  -> --set "${NCU_SET}"               (caracterizacion completa)
NCU_MODE="${NCU_MODE:-quick}"

# Metricas extra del modo quick para discriminar por que la compensacion
# espacial (5 lecturas de comp[] por celda) resulta MAS rapida que la Kahan
# local (1 lectura): stalls por warp -> hipotesis de latencia serializada;
# sectores/request -> hipotesis de coalescencia. Ambos grupos se definen y
# documentan en tools/common_ncu.sh. Vaciar la variable
# (NCU_COMP_ANALYSIS_METRICS="") vuelve al conjunto quick historico, util si
# la version de ncu del cluster rechaza algun nombre de metrica.
NCU_COMP_ANALYSIS_METRICS="${NCU_COMP_ANALYSIS_METRICS:-${NCU_WARP_STALL_METRICS:-},${NCU_COALESCING_METRICS:-}}"

# Politicas de compensacion a perfilar en el kernel WMMA, independientes de
# SPATIAL_COMP/KAHAN_LIST (que fijan las corridas de MEDICION). Se perfilan las
# tres en la misma corrida, con la misma nx/iters, porque comparar sus perfiles
# lado a lado es justamente lo que explica la anomalia de rendimiento; ncu
# invoca el binario aparte con --profile-only, asi que puede pedir una politica
# distinta de la que corre el barrido sin alterarlo.
#   off     -> --kahan off --spatial-comp off  (base, reporte ..._kahan0)
#   local   -> --kahan on  --spatial-comp off  (Kahan local, ..._kahan1)
#   spatial -> --kahan off --spatial-comp on   (espacial, ..._sp)
# Nunca se combinan kahan on + spatial on: el binario rechaza esa combinacion.
NCU_COMP_LIST="${NCU_COMP_LIST:-off local spatial}"
TIMEFORMAT=">>> Tiempo real de la llamada ncu (NCU_MODE=${NCU_MODE}): %R s (user %U, sys %S)"

echo "Nodo: $(hostname)"
echo "Fecha: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-no definido}"
echo "Arquitectura CUDA: sm_${CUDA_ARCH}"
echo "Operador         : OP_MODE=${OP_MODE} CI_MODE=${CI_MODE} (args: ${OP_ARGS[*]})"
echo "NVCC=${NVCC}"
echo "NCU=${NCU}"
echo "SMOKE_TEST=${SMOKE_TEST}"
echo "RUN_NCU=${RUN_NCU}"
# RUN_KIND y CHECKPOINT_EVERY se imprimen juntos: en RUN_KIND=energy el segundo
# ya viene forzado a 0, y es el par que decide si la ventana de energia sera
# fiable. Dejarlo en el .out permite auditarlo sin releer el CSV.
echo "RUN_KIND=${RUN_KIND} (CHECKPOINT_EVERY=${CHECKPOINT_EVERY}, ITERS_LIST=\"${ITERS_LIST}\")"

nvidia-smi || true

echo "Compilando stencil_tc (Fase_3/Stencil, incluye common/cuda_checks.cuh," \
     "common/metrics.cuh y common/power_sampling.h por ruta relativa)..."
"${NVCC}" -std=c++17 stencil_tensor_activation.cu -o stencil_tc \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
    --allow-unsupported-compiler "${NVML_FLAGS[@]}"

# Procedencia del binario/corrida: permite auditar de que commit/flags/hash
# salio cada CSV, sin depender de que el usuario recuerde como se compilo.
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo desconocido)"
if git diff --quiet 2>/dev/null; then
    GIT_STATUS="limpio"
else
    GIT_STATUS="sucio"
fi
BUILD_DATE="$(date -Iseconds)"
NVCC_FLAGS="-std=c++17 -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} -Xcompiler \"-O3 -funroll-loops -ffp-contract=off\" --allow-unsupported-compiler ${NVML_FLAGS[*]}"
BIN_SHA256="$(sha256sum stencil_tc | awk '{print $1}')"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "GIT_STATUS=${GIT_STATUS}"
echo "BUILD_DATE=${BUILD_DATE}"
echo "NVCC_FLAGS=${NVCC_FLAGS}"
echo "BIN_SHA256=${BIN_SHA256}"
echo "RUN_PARAMS=NX=${NX} NY=${NY} TC_FORMAT=${TC_FORMAT} ITERS_LIST=${ITERS_LIST} CHECKPOINT_EVERY=${CHECKPOINT_EVERY} KAHAN_LIST=${KAHAN_LIST} SPATIAL_COMP=${SPATIAL_COMP} FP64_GPU=${FP64_GPU} CPU_FP64=${CPU_FP64} ANCHOR_LIST=${ANCHOR_LIST}"

# Validacion SASS pre-perfilado: si el binario no contiene instrucciones HMMA,
# las instanciaciones WMMA (FP16 y BF16 emiten ambas HMMA en Ampere) no
# generaron codigo Tensor Core y perfilar seria inutil. CUOBJDUMP se busca por
# PATH, junto a NVCC, o en el arbol cuda/ del HPC SDK (layout tipico
# .../<version>/cuda/bin/cuobjdump).
_nvcc_dir="$(dirname "${NVCC}")"
if [[ -z "${CUOBJDUMP:-}" ]]; then
    if command -v cuobjdump >/dev/null 2>&1; then
        CUOBJDUMP="$(command -v cuobjdump)"
    else
        for _cand in \
            "${_nvcc_dir}/cuobjdump" \
            "${_nvcc_dir}/../../cuda/bin/cuobjdump" \
            "${_nvcc_dir}"/../../cuda/*/bin/cuobjdump; do
            if [[ -x "${_cand}" ]]; then
                CUOBJDUMP="${_cand}"
                break
            fi
        done
    fi
fi

if [[ -x "${CUOBJDUMP:-}" ]] || command -v "${CUOBJDUMP:-}" >/dev/null 2>&1; then
    HMMA_COUNT=$("${CUOBJDUMP}" --dump-sass stencil_tc | grep -c HMMA || true)
    echo "Instrucciones HMMA detectadas en binario: ${HMMA_COUNT} (cuobjdump: ${CUOBJDUMP})"
    if [[ "${RUN_NCU}" == "1" && "${HMMA_COUNT}" -eq 0 ]]; then
        echo "ERROR: binario sin instrucciones HMMA; revisar la instanciacion" \
             "del kernel stencil2d_wmma_kernel antes de perfilar." >&2
        exit 1
    fi
else
    echo "AVISO: no se encontro cuobjdump; se omite la validacion HMMA." \
         "Exporta CUOBJDUMP=/ruta/a/cuobjdump para habilitarla." >&2
fi

echo "Ejecutando benchmarks de stencil 2D con Tensor Cores (Fase 3: encadenamiento genuino)..."

if [[ "${SMOKE_TEST}" == "1" ]]; then
    declare -a RUNS=(
        "512 512 3 ${TC_FORMAT}"
    )
else
    # NX/NY son configurables via variables de entorno; ver el bloque
    # "Tamano de malla y barrido de iteraciones" mas arriba (incluye el
    # hallazgo de auditoria sobre el rango 512^2-2048^2 nunca ejercitado).
    declare -a RUNS=()
    for ITERS_VAL in ${ITERS_LIST}; do
        RUNS+=("${NX} ${NY} ${ITERS_VAL} ${TC_FORMAT}")
    done
fi

# Perfila UN solo kernel (regex + tc_type que lo activa) con --profile-only en
# el binario: evita recalcular la referencia CPU FP32 y la FP64 encadenada
# antes de llegar al kernel bajo perfil. Usa NX/NY/ITERS/NCU_CAPTURE_ARGS del
# ambito que la invoca (bucle principal, mas abajo). kahan (5to arg,
# "off"/"on") se pasa tal cual a --kahan; default "off" para los kernels que
# Kahan no toca (FP32 clasico, conversion), que solo se perfilan una vez.
# spatial (6to arg) default "${SPATIAL_COMP}": los kernels que la compensacion
# no toca (FP32 clasico, conversion) siguen la politica del barrido sin tener
# que declararla; el kernel WMMA la recibe explicita, una por politica de
# NCU_COMP_LIST.
ncu_profile_kernel() {
    local label="$1" regex="$2" tc_type="$3" report_tag="$4" kahan="${5:-off}"
    local spatial="${6:-${SPATIAL_COMP}}"
    local report="${LOGS_DIR}/ncu_${report_tag}_nx${NX}_ny${NY}_job${SLURM_JOB_ID:-manual}"
    echo
    echo "Perfilando ${label} con Nsight Compute (NCU_MODE=${NCU_MODE}," \
         "kahan=${kahan}, spatial-comp=${spatial})..."
    echo "Reporte: ${report}.ncu-rep"
    # `time` (y stderr de ncu) se redirigen al .out para correlacionar
    # duracion vs #SBATCH --time en corridas con --set full.
    { time ncu_run "${NCU}" "${NCU_CAPTURE_ARGS[@]}" \
        --target-processes all \
        --kernel-name "regex:${regex}" \
        --launch-skip "${NCU_LAUNCH_SKIP}" \
        --launch-count "${NCU_LAUNCH_COUNT}" \
        --export "${report}" \
        --force-overwrite \
        ./stencil_tc --nx "${NX}" --ny "${NY}" --iters "${ITERS}" --tc "${tc_type}" \
            --kahan "${kahan}" --spatial-comp "${spatial}" \
            "${OP_ARGS[@]}" --profile-only; } 2>&1
}

# Primer valor de KAHAN_LIST: la pasada en la que se concentra todo el
# perfilado NCU (ver el bloque RUN_NCU mas abajo).
read -r KAHAN_FIRST _ <<< "${KAHAN_LIST}"

# Aviso temprano: si NINGUN valor de ITERS_LIST coincide con ITERS_NCU, la
# corrida entera terminara sin un solo .ncu-rep. Detectarlo aqui evita
# descubrirlo despues de horas de cola.
if [[ "${RUN_NCU}" == "1" ]]; then
    NCU_ITERS_MATCH=0
    for _iters_val in ${ITERS_LIST}; do
        [[ "${_iters_val}" == "${ITERS_NCU}" ]] && NCU_ITERS_MATCH=1
    done
    if [[ "${NCU_ITERS_MATCH}" == "0" ]]; then
        echo "AVISO: RUN_NCU=1 pero ITERS_NCU=${ITERS_NCU} no esta en" \
             "ITERS_LIST='${ITERS_LIST}': esta corrida NO generara ningun reporte NCU." \
             "Alinee ITERS_NCU con algun valor de ITERS_LIST para perfilar." >&2
    fi
fi

# Una sola invocacion de tee envuelve TODO el bucle (todas las combinaciones
# de RUNS x KAHAN_LIST) en vez de reabrirse por cada pasada de KAHAN: ninguna
# pasada individual puede perderse sin que se pierdan todas.
{
for run in "${RUNS[@]}"; do
    read -r NX NY ITERS TC <<< "${run}"
    for KAHAN in ${KAHAN_LIST}; do
      # ANCLA FP64 (Fase 4): bucle anidado sobre ANCHOR_LIST. Con el default
      # ("0"), este bucle tiene una sola pasada y el comportamiento es
      # identico a Fase_3/Stencil/run_stencil_tc.sbatch.
      read -r ANCHOR_FIRST _ <<< "${ANCHOR_LIST}"
      for ANCHOR in ${ANCHOR_LIST}; do
        echo
        echo "============================================================"
        echo "Corrida: NX=${NX}, NY=${NY}, ITERS=${ITERS}, TC=${TC}, KAHAN=${KAHAN}, ANCHOR_EVERY=${ANCHOR}"
        echo "============================================================"
        # --fp64-gpu se pasa SOLO aqui (corridas de medicion). Las invocaciones
        # de ncu usan --profile-only, que no ejecuta la ruta GPU_FP64.
        ./stencil_tc --nx "${NX}" --ny "${NY}" --iters "${ITERS}" --tc "${TC}" \
            --checkpoint-every "${CHECKPOINT_EVERY}" --kahan "${KAHAN}" \
            --spatial-comp "${SPATIAL_COMP}" --fp64-gpu "${FP64_GPU}" \
            --cpu-fp64 "${CPU_FP64}" --anchor-every "${ANCHOR}" "${OP_ARGS[@]}"

        # El perfilado NCU (bloque de abajo) no depende de ANCHOR: mide el
        # kernel WMMA de pasos normales, que el ancla no modifica (ver
        # Fase_4/Stencil/README.md -- el ancla sustituye el paso, no cambia
        # stencil2d_wmma_kernel). Se restringe a la primera pasada de ANCHOR_LIST
        # por el mismo motivo que ya se restringe a KAHAN_FIRST: no perfilar
        # el mismo kernel N veces sin ganar informacion nueva.
        if [[ "${ANCHOR}" != "${ANCHOR_FIRST}" ]]; then
            continue
        fi

        if [[ "${RUN_NCU}" == "1" && "${ITERS}" == "${ITERS_NCU}" ]]; then
            # Con --tc both el binario lanza primero todos los kernels FP16 y luego
            # los BF16; como --launch-skip cuenta lanzamientos que matchean el regex
            # de forma global, un solo pase de ncu con skip=3 solo capturaria la
            # instanciacion FP16. Se perfila cada tipo por separado (--tc fp16 /
            # --tc bf16): un reporte por precision con skip/count deterministas.
            if [[ "${TC}" == "both" ]]; then
                NCU_TC_TYPES=(fp16 bf16)
            else
                NCU_TC_TYPES=("${TC}")
            fi
            if [[ "${NCU_MODE}" == "full" ]]; then
                NCU_CAPTURE_ARGS=(--set "${NCU_SET}")
            else
                NCU_CAPTURE_ARGS=(--metrics "${NCU_QUICK_METRICS}${NCU_COMP_ANALYSIS_METRICS:+,${NCU_COMP_ANALYSIS_METRICS}}")
            fi

            # Todo el perfilado se hace en UNA sola pasada del bucle de KAHAN
            # (la primera): ncu invoca el binario aparte, con --profile-only y
            # con la politica de compensacion que el mismo elige (ver
            # NCU_COMP_LIST), asi que no gana nada repitiendose por cada valor
            # de KAHAN_LIST.
            if [[ "${KAHAN}" == "${KAHAN_FIRST}" ]]; then
                ncu_profile_kernel "GPU FP32 clasico" "${NCU_KERNEL_REGEX_FP32}" \
                    "${NCU_TC_TYPES[0]}" "fp32_stencil"

                for TC_TYPE in "${NCU_TC_TYPES[@]}"; do
                    ncu_profile_kernel "conversion ${TC_TYPE}" "${NCU_KERNEL_REGEX_CONVERT}" \
                        "${TC_TYPE}" "convert_${TC_TYPE}"

                    for COMP_POLICY in ${NCU_COMP_LIST}; do
                        case "${COMP_POLICY}" in
                            off)
                                NCU_KAHAN=off; NCU_SPATIAL=off
                                COMP_TAG="kahan0"; COMP_LABEL="sin compensacion"
                                COMP_REGEX="${NCU_KERNEL_REGEX_WMMA}"
                                ;;
                            local)
                                NCU_KAHAN=on;  NCU_SPATIAL=off
                                COMP_TAG="kahan1"; COMP_LABEL="Kahan local"
                                COMP_REGEX="${NCU_KERNEL_REGEX_WMMA}"
                                ;;
                            spatial)
                                NCU_KAHAN=off; NCU_SPATIAL=on
                                COMP_TAG="sp";  COMP_LABEL="compensacion espacial"
                                COMP_REGEX="${NCU_KERNEL_REGEX_WMMA_SP}"
                                ;;
                            *)
                                echo "AVISO: politica desconocida en NCU_COMP_LIST:" \
                                     "'${COMP_POLICY}' (validas: off local spatial)." \
                                     "Se omite." >&2
                                continue
                                ;;
                        esac
                        ncu_profile_kernel "WMMA ${TC_TYPE} (${COMP_LABEL})" \
                            "${COMP_REGEX}" "${TC_TYPE}" \
                            "wmma_stencil_${TC_TYPE}_${COMP_TAG}" \
                            "${NCU_KAHAN}" "${NCU_SPATIAL}"
                    done
                done
            fi
        elif [[ "${RUN_NCU}" == "1" ]]; then
            echo
            echo "AVISO: NCU omitido en esta corrida: ITERS=${ITERS} no coincide con" \
                 "ITERS_NCU=${ITERS_NCU}. El perfilado solo se dispara en el valor de" \
                 "ITERS_LIST igual a ITERS_NCU (ITERS_LIST='${ITERS_LIST}'); si ninguno" \
                 "coincide, esta corrida no generara ningun .ncu-rep." >&2
        fi
      done  # for ANCHOR in ${ANCHOR_LIST}
    done
done
} 2>&1 | tee -a "${RESULTS_LOG}"

# tools/extract_csv.py (post-proceso de CSV) migra por separado, en
# Fase_3/tools/ (ver README.md de la raiz del repo -- no es parte de este
# entregable). Se prueban varias rutas candidatas, como ya se hace arriba con
# common_ncu.sh, y si no se encuentra ninguna se avisa y se omite la
# extraccion en vez de fallar todo el job: el log crudo (${RESULTS_LOG}) queda
# de todas formas disponible para post-procesar manualmente despues.
EXTRACT_CSV_SCRIPT="${EXTRACT_CSV_SCRIPT:-}"
if [[ -z "${EXTRACT_CSV_SCRIPT}" ]]; then
    for _cand in "../tools/extract_csv.py" "tools/extract_csv.py" "../../tools/extract_csv.py"; do
        if [[ -f "${_cand}" ]]; then
            EXTRACT_CSV_SCRIPT="${_cand}"
            break
        fi
    done
fi

if [[ -n "${EXTRACT_CSV_SCRIPT}" ]]; then
    echo
    echo "Extrayendo CSVs desde ${RESULTS_LOG} (via ${EXTRACT_CSV_SCRIPT})..."
    python3 "${EXTRACT_CSV_SCRIPT}" \
        --input  "${RESULTS_LOG}" \
        --outdir "${RESULTS_DIR}" \
        --job-id "${JOB_ID}" \
        --kernel stencil

    echo "CSVs generados:"
    for csv_path in \
        "${RESULTS_DIR}/drift_stencil_${JOB_ID}.csv" \
        "${RESULTS_DIR}/summary_stencil_${JOB_ID}.csv" \
        "${RESULTS_DIR}/horizon_stencil_${JOB_ID}.csv" \
        "${RESULTS_DIR}/store_stencil_${JOB_ID}.csv" \
        "${RESULTS_DIR}/energy_stencil_${JOB_ID}.csv"; do
        rows=$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' "${csv_path}" 2>/dev/null || echo 0)
        echo "  ${csv_path}: ${rows} filas"
    done
else
    echo
    echo "AVISO: extract_csv.py no encontrado (se probo ../tools/, tools/ y" \
         "../../tools/ relativos a esta carpeta). Se omite la extraccion de" \
         "CSVs; el log crudo con los tokens CSV_* queda en ${RESULTS_LOG}." \
         "Defina EXTRACT_CSV_SCRIPT=/ruta/a/extract_csv.py para forzar una" \
         "ubicacion." >&2
fi
