#!/bin/bash
#SBATCH --job-name=mixed_precision_stencil_tc_f3
#SBATCH --output=logs/mixed_precision_stencil_tc_f3_%j.out
#SBATCH --error=logs/mixed_precision_stencil_tc_f3_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=48:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f stencil_tensor_activation.cu && -f Stencil/stencil_tensor_activation.cu ]]; then
    cd Stencil
fi
mkdir -p logs results
JOB_ID="${SLURM_JOB_ID:-manual}"
RESULTS_LOG="results/run_${JOB_ID}.log"

# stencil_tc usa mma.h / cuda_fp16.h / cuda_bf16.h y, si PACCA lo expone,
# NVML para telemetria; OpenBLAS/cuDNN no aplican aqui.

# Arquitectura por defecto para A100.
# Puedes sobrescribirla con:
#   sbatch --export=ALL,CUDA_ARCH=86 run_stencil_tc.sbatch
CUDA_ARCH="${CUDA_ARCH:-80}"  # A100=80, RTX 3050=86, V100=70

NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"
NCU="${NCU:-${HOME}/fcrojasv/nsight_compute/ncu}"

# NVML es opcional: cuando nvml.h y la biblioteca del driver estan visibles,
# se habilita la telemetria y se enlaza con libnvidia-ml. Si PACCA no expone
# alguno de los dos, el mismo fuente compila con el stub sin telemetria.
NVML_FLAGS=()
# El driver NVIDIA suele instalarse como -lnvidia-ml; PACCA puede sobreescribir
# este valor con -lnvml si su modulo ofrece ese alias.
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
# Se declara como variable propia para poder apuntarla a un regex sobre el
# nombre demangled (que si lleva el argumento de plantilla) si alguna vez hace
# falta: ncu usa --kernel-name-base=function por defecto, que recorta los
# argumentos de plantilla, asi que ese caso exigiria ademas
# --kernel-name-base demangled.
NCU_KERNEL_REGEX_WMMA_SP="${NCU_KERNEL_REGEX_WMMA_SP:-${NCU_KERNEL_REGEX_WMMA}}"
# Debe coincidir con kWarmupIters en stencil_tensor_activation.cu (Fase_3/Stencil);
# se deriva del propio fuente en vez de repetir el literal para que ambos no
# puedan desincronizarse si kWarmupIters cambia.
KWARMUP_ITERS_CU="$(grep -m1 'kWarmupIters *=' stencil_tensor_activation.cu | grep -oE '[0-9]+' | head -n1)"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-${KWARMUP_ITERS_CU}}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-1}"
# Barrido de METRICAS (16384^2): 10..120 caen dentro del horizonte de
# FP32/BF16/FP16 (produce L2/Linf finitas). Se elimino 200: a este tamano
# divergen las tres rutas GPU (n* FP32~142, BF16~138, FP16~28) y toda la
# corrida sale "REFERENCIA NO FINITA" sin ningun dato comparable -- era la
# corrida mas cara del job sin aportar nada. El horizonte de divergencia en
# si (donde SI importa iters grande) se mide aparte, a 4096^2, en
# run_stencil_horizon.sbatch.
ITERS_LIST="${ITERS_LIST:-10 50 100 120}"
# Necesario para que storage_rel (error relativo de guardar en FP16/BF16) sea
# evaluable en las corridas donde la ruta diverge antes de ITERS (p.ej. FP16
# con n*~28 en corridas con ITERS>28): sin --checkpoint-every, stencil_tc solo
# puede capturar un estado FP32 finito en la ULTIMA iteracion medida, y si esa
# iteracion ya diverguio no hay forma de recuperar un estado finito posterior
# al hecho (ver comentario storage_rel_eval_iter en
# benchmark_gpu_tensor_core_stencil, stencil_tensor_activation.cu). 5 deja
# checkpoints en 5,10,15,20,25,... -- el ultimo antes de n*~28 (FP16, el
# formato que diverge mas temprano) cae en 25, bien dentro del rango finito.
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
# Restriccion de tiempo: NCU solo se corre para el valor de ITERS de la lista
# que coincida con ITERS_NCU, no para todos -- perfilar en cada ITERS
# multiplicaria el tiempo de cola. Reportes por corrida perfilada: 9 = FP32
# clasico (1) + conversion fp16/bf16 (2) + WMMA fp16/bf16 x las 3 politicas de
# NCU_COMP_LIST (6).
#
# ATENCION: ITERS_NCU e ITERS_LIST son independientes y hay que alinearlas a
# mano. Si ningun valor de ITERS_LIST es igual a ITERS_NCU no se genera ningun
# reporte; el script lo avisa (al arrancar y en cada corrida omitida) en vez de
# saltarse el bloque en silencio como hacia antes.
ITERS_NCU="${ITERS_NCU:-100}"
# Diseno factorial {fp16,bf16} x {kahan off,on} x ITERS_LIST (bloque 5): cada
# valor de ITERS_LIST corre completo con --kahan off y con --kahan on. Kahan
# solo cambia las rutas WMMA (ver stencil_tensor_activation.cu); GPU FP32
# clasico y las referencias CPU FP32/FP64 dan identico resultado en ambas
# pasadas, pero se recalculan de todas formas (una invocacion == una corrida
# completa) -- el costo extra es ese recalculo redundante, no trabajo nuevo.
# OJO: con el default actual (SPATIAL_COMP=on, ver abajo) esta lista se fuerza
# a "off" y el barrido factorial de Kahan NO corre en las mediciones. Para
# recuperarlo hay que pedir explicitamente SPATIAL_COMP=off.
KAHAN_LIST="${KAHAN_LIST:-off on}"

# Compensacion ESPACIAL (error feedback de vecinos): politica POR DEFECTO,
# alternativa a --kahan off|on (ver CompMode en stencil_tensor_activation.cu).
# El default se invirtio de "off" a "on" porque es la unica de las tres
# politicas que efectivamente mitiga el error medido: Kahan LOCAL es
# indistinguible de no compensar (rel_l2 igual hasta la 4a cifra con --kahan
# off, horizonte sin mover: FP16=28, BF16=138) a un costo de ~1.97x en
# t_iter_ms, mientras la espacial baja rel_l2 ~5-6 ordenes de magnitud
# (BF16 a nx=16384: 6.01e-01 -> 3.16e-07) y lleva el horizonte de BF16 de 138
# a 142, el mismo de FP32 clasico, por ~1.56x. Verificado a nx=4096/8192/16384
# (jobs 4501/4502 y 4511-4514).
#
# El binario RECHAZA --kahan on junto con --spatial-comp on (son dos politicas
# alternativas de la misma compensacion, no dos capas): con SPATIAL_COMP=on se
# fuerza KAHAN_LIST=off aqui, porque con `set -e` una combinacion invalida
# mataria el job a mitad del barrido. Por eso SPATIAL_COMP=off es la UNICA via
# para obtener el barrido factorial de Kahan local (kahan off y on), que se
# conserva intacto:
#   sbatch run_stencil_tc.sbatch                     -> WMMA_*_SP
#   sbatch --export=ALL,SPATIAL_COMP=off run_stencil_tc.sbatch
#                                     -> WMMA_FP16/BF16 con kahan off y on
#
# Las rutas WMMA de una corrida espacial se reportan como WMMA_FP16_SP /
# WMMA_BF16_SP, asi que sus filas conviven sin ambiguedad con las de
# KAHAN_LIST="off on" en un mismo CSV extraido.
#
# Nota: esto solo fija la politica de las corridas de MEDICION. El perfilado
# NCU no depende de esta variable para el kernel WMMA: perfila las tres
# politicas en la misma corrida (ver NCU_COMP_LIST).
SPATIAL_COMP="${SPATIAL_COMP:-on}"
if [[ "${SPATIAL_COMP}" == "on" ]]; then
    KAHAN_LIST="off"
fi

NX="${NX:-16384}"
NY="${NY:-16384}"

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
# la version de ncu de PACCA rechaza algun nombre de metrica.
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
echo "NVCC=${NVCC}"
echo "NCU=${NCU}"
echo "SMOKE_TEST=${SMOKE_TEST}"
echo "RUN_NCU=${RUN_NCU}"

nvidia-smi || true

echo "Compilando stencil_tc (Fase_3/Stencil, incluye Fase_2/common.cuh por ruta relativa)..."
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
echo "RUN_PARAMS=NX=${NX} NY=${NY} ITERS_LIST=${ITERS_LIST} CHECKPOINT_EVERY=${CHECKPOINT_EVERY} KAHAN_LIST=${KAHAN_LIST} SPATIAL_COMP=${SPATIAL_COMP}"

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
        "512 512 3 both"
    )
else
    # Dimensiones de referencia del proyecto (mismas del hint Nsight en el .cu),
    # barridas sobre ITERS_LIST dentro de la ventana finita (10..120): todas
    # las metricas (L2/Linf/gflops/etc.) quedan comparables entre corridas.
    # NX/NY son configurables via variables de entorno (mismo mecanismo que
    # ITERS_LIST); el valor por defecto (16384) se conserva por compatibilidad
    # con las corridas de diagnostico ya hechas a ese tamano.
    declare -a RUNS=()
    for ITERS_VAL in ${ITERS_LIST}; do
        RUNS+=("${NX} ${NY} ${ITERS_VAL} both")
    done
fi

# Perfila UN solo kernel (regex + tc_type que lo activa) con --profile-only en
# el binario: evita recalcular la referencia CPU FP32 (~742s) y la FP64
# encadenada (~900s) antes de llegar al kernel bajo perfil (~1723s de pared
# por llamada a ncu antes de este cambio). Usa NX/NY/ITERS/NCU_CAPTURE_ARGS
# del ambito que la invoca (bucle principal, mas abajo). kahan (5to arg,
# "off"/"on") se pasa tal cual a --kahan; default "off" para los kernels que
# Kahan no toca (FP32 clasico, conversion), que solo se perfilan una vez.
# spatial (6to arg) default "${SPATIAL_COMP}": los kernels que la compensacion
# no toca (FP32 clasico, conversion) siguen la politica del barrido sin tener
# que declararla; el kernel WMMA la recibe explicita, una por politica de
# NCU_COMP_LIST.
ncu_profile_kernel() {
    local label="$1" regex="$2" tc_type="$3" report_tag="$4" kahan="${5:-off}"
    local spatial="${6:-${SPATIAL_COMP}}"
    local report="logs/ncu_${report_tag}_nx${NX}_ny${NY}_job${SLURM_JOB_ID:-manual}"
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
            --kahan "${kahan}" --spatial-comp "${spatial}" --profile-only; } 2>&1
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
# de RUNS x KAHAN_LIST) en vez de reabrirse por cada pasada de KAHAN: mismo
# patron ya usado para el .out de SLURM (una unica redireccion persistente
# durante toda la corrida, que nunca perdio datos). Reabrir un proceso tee
# por pasada dejaba a cada pasada dependiendo de que ESA invocacion puntual
# sobreviviera intacta; con una sola invocacion para todo el bucle, ninguna
# pasada individual puede perderse sin que se pierdan todas.
{
for run in "${RUNS[@]}"; do
    read -r NX NY ITERS TC <<< "${run}"
    for KAHAN in ${KAHAN_LIST}; do
        echo
        echo "============================================================"
        echo "Corrida: NX=${NX}, NY=${NY}, ITERS=${ITERS}, TC=${TC}, KAHAN=${KAHAN}"
        echo "============================================================"
        ./stencil_tc --nx "${NX}" --ny "${NY}" --iters "${ITERS}" --tc "${TC}" \
            --checkpoint-every "${CHECKPOINT_EVERY}" --kahan "${KAHAN}" \
            --spatial-comp "${SPATIAL_COMP}"

        if [[ "${RUN_NCU}" == "1" && "${ITERS}" == "${ITERS_NCU}" ]]; then
            # Con --tc both el binario lanza primero todos los kernels FP16 y luego
            # los BF16; como --launch-skip cuenta lanzamientos que matchean el regex
            # de forma global, un solo pase de ncu con skip=3 solo capturaria la
            # instanciacion FP16. Se perfila cada tipo por separado (--tc fp16 /
            # --tc bf16): un reporte por precision con skip/count deterministas.
            # --profile-only exige --tc fp16|bf16 (nunca both), asi que este split
            # ya es obligatorio, no solo una optimizacion de --launch-skip.
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
            # de KAHAN_LIST. Se ancla al primer elemento y no al literal "off"
            # para que KAHAN_LIST="on" (o cualquier lista de un solo valor)
            # tampoco se quede sin perfilar.
            if [[ "${KAHAN}" == "${KAHAN_FIRST}" ]]; then
                # El valor de --tc que se le pasa al FP32 clasico es
                # irrelevante para ese kernel; se usa el primero de
                # NCU_TC_TYPES.
                ncu_profile_kernel "GPU FP32 clasico" "${NCU_KERNEL_REGEX_FP32}" \
                    "${NCU_TC_TYPES[0]}" "fp32_stencil"

                for TC_TYPE in "${NCU_TC_TYPES[@]}"; do
                    # Las conversiones float<->16 bits no dependen de la
                    # politica de compensacion (no las toca, ver
                    # stencil_tensor_activation.cu): un reporte por precision.
                    ncu_profile_kernel "conversion ${TC_TYPE}" "${NCU_KERNEL_REGEX_CONVERT}" \
                        "${TC_TYPE}" "convert_${TC_TYPE}"

                    # El kernel WMMA SI depende de la politica: se perfila una
                    # vez por politica de NCU_COMP_LIST, con reportes
                    # distinguibles por sufijo (kahan0 / kahan1 / sp). Los tres
                    # salen de la MISMA corrida, nx e iters -- que es lo que
                    # permite compararlos lado a lado.
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
            # Sin este aviso el bloque NCU se saltaba en silencio: ITERS_NCU e
            # ITERS_LIST son variables independientes, y toda corrida cuyo
            # ITERS no coincidiera con ITERS_NCU terminaba sin un solo
            # .ncu-rep sin que nada lo dijera en el log.
            echo
            echo "AVISO: NCU omitido en esta corrida: ITERS=${ITERS} no coincide con" \
                 "ITERS_NCU=${ITERS_NCU}. El perfilado solo se dispara en el valor de" \
                 "ITERS_LIST igual a ITERS_NCU (ITERS_LIST='${ITERS_LIST}'); si ninguno" \
                 "coincide, esta corrida no generara ningun .ncu-rep." >&2
        fi
    done
done
} 2>&1 | tee -a "${RESULTS_LOG}"

echo
echo "Extrayendo CSVs desde ${RESULTS_LOG}..."
python3 tools/extract_csv.py \
    --input  "${RESULTS_LOG}" \
    --outdir "results" \
    --job-id "${JOB_ID}" \
    --kernel stencil

echo "CSVs generados:"
for csv_path in \
    "results/drift_stencil_${JOB_ID}.csv" \
    "results/summary_stencil_${JOB_ID}.csv" \
    "results/horizon_stencil_${JOB_ID}.csv" \
    "results/store_stencil_${JOB_ID}.csv" \
    "results/energy_stencil_${JOB_ID}.csv"; do
    rows=$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' "${csv_path}")
    echo "  ${csv_path}: ${rows} filas"
done
