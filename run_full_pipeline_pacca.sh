#!/bin/bash
# run_full_pipeline_pacca.sh
#
# Orquestador ALTERNATIVO para un cluster con SLURM (PACCA). NO reemplaza a
# run_full_pipeline.sh: ese sigue siendo el camino para correr todo de un tiron
# en una maquina propia con GPU, y se deja intacto.
#
# POR QUE HACEN FALTA DOS
# -----------------------
# run_full_pipeline.sh invoca cada .sbatch con `bash`, en secuencia, dentro de
# UN solo proceso. Eso exige tener la GPU reservada de principio a fin en una
# sola sesion. Con las campanas ampliadas (cuatro tamanos x tres barridos de
# iteraciones x cuatro valores de K, en tres kernels) eso son facilmente
# decenas de horas seguidas: en un cluster compartido no hay forma de pedir esa
# reserva, y si el job se cae en la hora 30 se pierde todo.
#
# Este script, en cambio, ENVIA cada fase como un job independiente con
# `sbatch --parsable`, y las encadena con `--dependency=afterany:<job_id>`.
# Cada job dependiente usa tools/oom_guard.sh via BASH_ENV: continua si el
# predecesor fue OOM y se bloquea si termino por otro error.
# SLURM se encarga del resto: cada job pide solo el tiempo que necesita, la
# cola los intercala con los de otros usuarios, y si uno falla los que dependen
# de el no arrancan si fue un error normal; un OOM queda registrado y permite
# continuar con los datos parciales disponibles.
#
# GRAFO DE DEPENDENCIAS
# ---------------------
#                       +--> Fase1/Fase2 (opcionales, sin dependientes)
#   validacion          |
#   preliminar ---------+--> F3 GEMM   (+energia) --> F4 GEMM   (+energia) --+
#   (los tres           +--> F3 Conv   (+energia) --> F4 Conv   (+energia) --+--> post-proceso
#    kernels)           +--> F3 Stencil(+energia) --> F4 Stencil(+energia)--+    (sin GPU)
#
#                        campana de variabilidad (aparte, RUN_VARIABILIDAD=1)
#                        tools/lanzar_campana_variabilidad.sh --> su propio
#                        post-proceso (tools/postproceso_variabilidad.sbatch)
#
#   * Los tres kernels van EN PARALELO entre si: no comparten nada.
#   * Fase 4 de un kernel depende SOLO de su propia Fase 3.
#   * Cada F3/F4 manda DOS jobs -- RUN_ENERGY_PASS=1 es el default: la pasada
#     normal (numerica, defaults del .sbatch) y una pasada SOLO de energia
#     (RUN_KIND=energy, ITERS_LIST grande) -- ver la nota de esa variable mas
#     abajo. Ambas dependen de lo mismo que dependeria una sola.
#   * Fase 1 y Fase 2 cuelgan de la validacion preliminar pero NADIE depende de
#     ellas: no producen CSV_* que el post-proceso consuma (ver docs/MANUAL.md,
#     seccion Fase 1), asi que meterlas en la cadena de F3/F4 solo lograria que
#     un fallo suyo -- por ejemplo, CUTLASS sin clonar, que es opcional --
#     bloqueara una campana que no las necesita.
#   * El post-proceso depende de TODOS los jobs de F3/F4 (normales + energia)
#     a la vez (afterany:J1:J2:...); oom_guard.sh deja pasar OOM y bloquea
#     otros errores. No pide GPU.
#   * La campana de variabilidad es un grafo APARTE con su propio
#     post-proceso: no comparte resultados con el post-proceso de arriba
#     (nunca mezclar produccion con variabilidad en el mismo run_statistics.py
#     -- son puntos de operacion distintos, ver tools/postproceso*.sbatch).
#
# La validacion preliminar (tools/validacion_preliminar.sbatch) va primero a
# proposito: verifica orden de operandos, humo de los tres kernels y los gates
# K=0/K=1 en minutos. Si la validacion falla por OOM, oom_guard.sh permite que
# la campana continue; si falla por otra causa, las dependencias se bloquean.
#
# USO
# ---
#   bash run_full_pipeline_pacca.sh                  # envia todo y sale
#   RUN_FASE1=0 RUN_FASE2=0 bash run_full_pipeline_pacca.sh
#   RUN_ENERGY_PASS=0 bash run_full_pipeline_pacca.sh   # sin la pasada de energia
#   RUN_VARIABILIDAD=0 bash run_full_pipeline_pacca.sh  # sin campana de replicas
#   SKIP_VALIDACION=1 bash run_full_pipeline_pacca.sh   # sin la puerta previa
#   DRY_RUN=1 bash run_full_pipeline_pacca.sh        # imprime, no envia
#
# Parametrizar una fase: exporta las mismas variables que acepta su .sbatch
# ANTES de llamar a este script. Se propagan a los jobs via --export=ALL.
#   N_LIST="1024 2048" ANCHOR_LIST="0 1 5" bash run_full_pipeline_pacca.sh
#
# El script TERMINA en cuanto termina de enviar: no espera a que la campana
# corra. Imprime la tabla de job ids para seguirla con `squeue`/`sacct`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-0}"
SKIP_VALIDACION="${SKIP_VALIDACION:-0}"
RUN_FASE1="${RUN_FASE1:-1}"
RUN_FASE2="${RUN_FASE2:-1}"
RUN_FASE3="${RUN_FASE3:-1}"
RUN_FASE4="${RUN_FASE4:-1}"
RUN_POST="${RUN_POST:-1}"
OOM_FAILURE_LOG="${OOM_FAILURE_LOG:-${REPO_ROOT}/oom_failures_pacca_$(date +%Y%m%d_%H%M%S).log}"

# --- Pase de energia (RUN_KIND=energy), Fase 3/4 de los 3 kernels ------------
#
# QUE PROBLEMA CIERRA: sin esto, una corrida "de una sola pasada" con los
# defaults de cada .sbatch (ITERS_LIST chico: 20/40/80 en GEMM/Conv, el
# default de Stencil) SIEMPRE da energy_window_reliable=0 en casi todas las
# filas -- confirmado en produccion (jobs 6925/6926/6927/6928/6866/6890):
# 96/96, 96/96, 276/276, 276/276 filas GPU respectivamente. La causa, en dos
# capas:
#   1. CHECKPOINT_EVERY>0 (el default) fragmenta la ventana NVML en tramos:
#      con muchos tramos cortos el umbral minimo de tiempo total termina muy
#      por encima de lo que dura la corrida (kEnergyWindowReliableSeconds *
#      gpu_segments, ver common/power_sampling.h). RUN_KIND=energy fuerza
#      CHECKPOINT_EVERY=0 y RUN_NCU=0 (perfilar invalida la energia) -- ver el
#      bloque RUN_KIND en cada run_*.sbatch.
#   2. Con CHECKPOINT_EVERY=0, TODAS las rutas de GEMM/Conv (GPU_FP64,
#      _none Y _comp, en cualquier K) cierran 2 tramos, no 1 -- verificado
#      con datos reales, no solo lectura de codigo: la columna gpu_segments
#      de summary_{gemm,conv}_*.csv en la campana de variabilidad
#      (jobs 7116-7129, 7114-7127) sale '2' en el 100% de las 480+480 filas,
#      sin excepcion de ruta ni de K. La razon es que run_fp64_reference() Y
#      el bucle de cada ruta comparten la MISMA es_checkpoint(), que dispara
#      igual en la ULTIMA iteracion sin importar CHECKPOINT_EVERY -- umbral
#      real 1000 ms para las 5 rutas, no 500. (Stencil es distinto: ahi solo
#      las rutas WMMA_*_SP cierran 2 tramos: GPU_FP32/GPU_FP64 solo cierran
#      via checkpoint_due(), sin el disparo incondicional -- 1 tramo, 500 ms.
#      Confirmado igual con datos: ver B1 de la auditoria de Fase 4.)
#
# La correccion NO es cambiar el ITERS_LIST default de la pasada normal (esa
# pasada existe para caracterizar tiempo/error a bajo costo, con muchos
# checkpoints -- eso es justo lo que RUN_KIND=energy sacrifica). La solucion
# validada esta campana es una SEGUNDA pasada, solo para energia, con un
# ITERS_LIST mucho mas grande (para que el tiempo total cruce el umbral en
# la ruta MAS RAPIDA del barrido -- esa es la que manda, porque todas las
# rutas de una invocacion comparten el mismo --iters) y sin checkpoints
# intermedios. Job id y CSV quedan aparte (job_id nuevo), asi que conviven en
# results/ con los de la pasada numerica sin pisarlos -- el post-proceso los
# toma a ambos por igual (mas replicas para el mismo tamano/formato, no un
# reemplazo).
#
# CUIDADO CON LA RAMPA DE RELOJ DE GPU AL DIMENSIONAR ESTO: un job largo con
# muchas invocaciones cortas seguidas (el barrido normal de produccion) NO
# corre a reloj estable -- se midio hasta 40% de diferencia en t_iter_ms
# entre la primera y la ultima invocacion del MISMO job (mismo tamano/ruta),
# monotonamente decreciente. Dimensionar con la MEDIANA de esas invocaciones
# (como salio la primera version de estos defaults: GEMM 12000, Conv 18000)
# subestima el caso real, porque una pasada de energia dedicada es una
# invocacion LARGA y sostenida que alcanza el reloj rapido de estado
# estable -- el mismo que representa el MINIMO observado, no la mediana. Es
# el mismo error, en la misma direccion, que dejo corta la tanda D de
# Stencil (~30-40% de iters faltantes). Los defaults de abajo usan el MINIMO
# t_iter_ms medido por ruta (no la mediana) en produccion real
# (jobs 6927/6928/6867/6893), sobre la ruta mas rapida del barrido de
# tamanos (N=1024 en GEMM, HW=64 en Conv, malla 4096 en Stencil), con margen
# adicional del ~25-30% sobre ese minimo:
#   GEMM  N=1024, FP16_none: min 0.0557 ms/it -> 17966 iters minimos
#         -> default 24000 (+33.6% sobre el minimo).
#   Conv  HW=64,  FP16_none: min 0.0358 ms/it -> 27962 iters minimos
#         -> default 37000 (+32.3% sobre el minimo).
#   Stencil malla 4096, WMMA_FP16_SP: min 0.3058 ms/it (umbral 1000 ms,
#         2 tramos) -> 3270 iters minimos -- MAYOR que el de GPU_FP32/FP64 en
#         esa misma malla (2685, umbral 500 ms/1 tramo): con reloj estable la
#         ruta que manda en Stencil a 4096 es la WMMA, no la referencia.
#         -> default 4000 (+22.3% sobre el minimo).
#   A tamanos MAYORES esos mismos ITERS sobran (el umbral se cruza antes) --
#   no hace falta ajustar por tamano, solo cuesta un poco mas de GPU, nunca
#   menos confiabilidad; esta decision (un solo ITERS_LIST compartido en vez
#   de uno por tamano) tambien esta documentada en el README, seccion "Por
#   donde empezar". Si el presupuesto de cola aprieta, lanzar la pasada de
#   energia por tamano por separado con NX_LIST/N_LIST/HW_LIST de un solo
#   valor y su propio ITERS_LIST ajustado -- como el lanzamiento de Stencil
#   de esta campana (jobs 7219/7220/7221, uno por malla), aunque ESE
#   lanzamiento en particular se dimensiono con la MEDIANA, no el minimo:
#   8192 y 16384 pueden haber quedado con menos margen del que parece (923 y
#   312 iters minimos con el minimo real, contra 900 y 250 lanzados) -- a
#   revisar cuando esos jobs terminen, no se tocan mientras siguen en cola.
# Default 1: este script ES la campana completa (asi se describe en la
# cabecera de arriba) -- no existe aqui un modo "solo exactitud" o "smoke"
# que proteger, y la campana no queda realmente terminada sin el eje de
# energia/Pareto. `bash run_full_pipeline_pacca.sh`, sin nada mas, ya manda
# todo: F3/F4 normal + energia de los 3 kernels, y la campana de
# variabilidad completa (GEMM/Conv en energia, Stencil en sus dos variantes
# -- ver la nota de RUN_VARIABILIDAD mas abajo). Para desactivarlo (por
# ejemplo, iterando rapido sobre un solo kernel con RUN_FASE1=0...):
#   RUN_ENERGY_PASS=0 bash run_full_pipeline_pacca.sh
RUN_ENERGY_PASS="${RUN_ENERGY_PASS:-1}"
ENERGY_ITERS_GEMM="${ENERGY_ITERS_GEMM:-24000}"
ENERGY_ITERS_CONV="${ENERGY_ITERS_CONV:-37000}"
ENERGY_ITERS_STENCIL="${ENERGY_ITERS_STENCIL:-4000}"

# --- Campana de variabilidad (replicas para el ANOVA/Tukey, Etapa 7) --------
#
# QUE PROBLEMA CIERRA: run_statistics.py necesita >=2 observaciones
# INDEPENDIENTES (job_id distinto) por celda del diseno para estimar
# varianza -- sin eso, Dataset.warn_low_replicas avisa pero el ANOVA ya sale
# mal (sumas de cuadrados NaN o negativas, ver postproceso de produccion:
# 144 celdas con 1 sola replica, exactamente las que tienen K>0). La campana
# de arriba (F3/F4 normal + pase de energia) corre cada celda UNA vez -- para
# replicas hace falta tools/lanzar_campana_variabilidad.sh, que fija un
# tamano chico y barato por kernel y repite esa MISMA configuracion REPLICAS
# veces como jobs de SLURM independientes. Es un orquestador aparte (envia
# sus propios jobs via sbatch y encadena su propio post-proceso,
# tools/postproceso_variabilidad.sbatch) -- se invoca aqui con `bash`, no se
# reimplementa.
RUN_VARIABILIDAD="${RUN_VARIABILIDAD:-1}"
VARIABILIDAD_REPLICAS="${VARIABILIDAD_REPLICAS:-8}"

# Particion del job de post-proceso. Se deja VACIA por defecto a proposito: no
# hay un nombre de particion sin GPU que sea portable entre clusters, y pasar
# uno inexistente hace que sbatch RECHACE el envio. Vacia = particion por
# defecto del cluster. En PACCA, si existe una particion de solo CPU, ponerla
# aqui libera la GPU durante todo el post-proceso:
#   POST_PARTITION=CPU bash run_full_pipeline_pacca.sh
POST_PARTITION="${POST_PARTITION:-}"

if [[ "${DRY_RUN}" != "1" ]] && ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: no se encontro sbatch. Este orquestador es especifico de un" >&2
    echo "cluster con SLURM. Fuera de un cluster use run_full_pipeline.sh, que" >&2
    echo "corre exactamente las mismas fases con \`bash\` en secuencia." >&2
    exit 1
fi

declare -a RESUMEN=()
_DRY_ID=1000
# Job id del ultimo envio. `enviar` lo deja AQUI en vez de imprimirlo para que
# el llamador lo capture con $(...): una sustitucion de comandos abre una
# SUBSHELL, y ahi los `RESUMEN+=(...)` y el contador del dry-run se perderian
# al volver. Es el tipo de bug que no se nota hasta que la tabla final sale
# vacia.
ULTIMO_JID=""

# enviar <etiqueta> <directorio> <script> [dependencia]
enviar() {
    local etiqueta="$1" dir="$2" script="$3" dep="${4:-}"
    local export_values submit_dir submit_script phase_label
    # EXPORT_EXTRA se pega DENTRO del mismo --export=ALL,... (no como un
    # segundo --export): sbatch no documenta que "gana el ultimo" si se pasa
    # --export dos veces, asi que en vez de confiar en eso se arma un solo
    # flag. Es lo que usa el pase de energia para llevar RUN_KIND=energy e
    # ITERS_LIST al job sin tocar la firma de enviar() para todo lo demas.
    export_values="ALL${EXPORT_EXTRA:+,${EXPORT_EXTRA}}"
    submit_dir="${dir}"
    submit_script="${script}"
    if [[ -n "${dep}" ]]; then
        phase_label="${etiqueta// /_}"
        export_values+=",OOM_DEP_JOBS=${dep},OOM_FAILURE_LOG=${OOM_FAILURE_LOG},OOM_PHASE_LABEL=${phase_label},BASH_ENV=${REPO_ROOT}/tools/oom_guard.sh"
    fi
    local -a args=(--parsable "--export=${export_values}")
    [[ -n "${dep}" ]] && args+=(--dependency="afterany:${dep}")
    # Sin comillas a proposito: EXTRA_SBATCH_ARGS puede traer varios flags NO
    # relacionados con --export (p.ej. --partition=X del post-proceso).
    # shellcheck disable=SC2206
    [[ -n "${EXTRA_SBATCH_ARGS:-}" ]] && args+=(${EXTRA_SBATCH_ARGS})

    if [[ "${DRY_RUN}" == "1" ]]; then
        # Id sintetico y creciente, para que el grafo de dependencias del
        # dry-run se vea igual que el real.
        _DRY_ID=$(( _DRY_ID + 1 ))
        ULTIMO_JID="${_DRY_ID}"
        echo "[DRY_RUN] (cd ${submit_dir} && sbatch ${args[*]} ${submit_script})  -> ${ULTIMO_JID}"
    else
        ULTIMO_JID="$(cd "${submit_dir}" && sbatch "${args[@]}" "${submit_script}")"
        echo "Enviado ${etiqueta}: job ${ULTIMO_JID}${dep:+ (afterany:${dep})}"
    fi
    RESUMEN+=("$(printf '%-10s %-34s dep=%s' "${ULTIMO_JID}" "${etiqueta}" "${dep:-ninguna}")")
}

# --- Puerta previa: validacion preliminar a escala reducida ------------------
DEP_BASE=""
if [[ "${SKIP_VALIDACION}" == "1" ]]; then
    echo "SKIP_VALIDACION=1 -- las campanas se envian SIN la puerta previa." >&2
    echo "  (Tarea 8 de la auditoria: no hacerlo salvo que ya se haya corrido" >&2
    echo "   tools/validacion_preliminar.sbatch a mano y haya pasado.)" >&2
else
    enviar "validacion preliminar" "." "tools/validacion_preliminar.sbatch"
    DEP_BASE="${ULTIMO_JID}"
fi

# --- Fase 1 y Fase 2: opcionales, sin dependientes --------------------------
if [[ "${RUN_FASE1}" == "1" ]]; then
    enviar "F1 GEMM"        Fase_1/GEMM        run_gemm_fase1.sbatch    "${DEP_BASE}"
    enviar "F1 Convolucion" Fase_1/Convolution run_conv_fase1.sbatch    "${DEP_BASE}"
    enviar "F1 Stencil"     Fase_1/Stencil     run_stencil_fase1.sbatch "${DEP_BASE}"
fi
if [[ "${RUN_FASE2}" == "1" ]]; then
    enviar "F2 GEMM"        Fase_2/GEMM        run_gemm_tc.sbatch    "${DEP_BASE}"
    enviar "F2 Convolucion" Fase_2/Convolution run_conv_tc.sbatch    "${DEP_BASE}"
    enviar "F2 Stencil"     Fase_2/Stencil     run_stencil_tc.sbatch "${DEP_BASE}"
fi

# --- Fase 3 -> Fase 4, una cadena independiente por kernel ------------------
declare -a JOBS_CAMPANA=()

cadena_kernel() {
    local nombre="$1" dir_f3="$2" script_f3="$3" dir_f4="$4" script_f4="$5" energy_iters="$6"
    local j3=""
    if [[ "${RUN_FASE3}" == "1" ]]; then
        enviar "F3 ${nombre}" "${dir_f3}" "${script_f3}" "${DEP_BASE}"
        j3="${ULTIMO_JID}"
        JOBS_CAMPANA+=("${j3}")
        if [[ "${RUN_ENERGY_PASS}" == "1" && "${SMOKE_TEST:-0}" != "1" ]]; then
            # Segunda pasada, SOLO energia: mismo tamano/formato/K que la de
            # arriba (no se tocan N_LIST/HW_LIST/NX_LIST/ANCHOR_LIST, cada
            # .sbatch usa su propio default), pero con RUN_KIND=energy e
            # ITERS_LIST grande -- ver la nota de cabecera de este script.
            EXPORT_EXTRA="RUN_KIND=energy,ITERS_LIST=${energy_iters}"
            enviar "F3 ${nombre} (energia)" "${dir_f3}" "${script_f3}" "${DEP_BASE}"
            unset EXPORT_EXTRA
            JOBS_CAMPANA+=("${ULTIMO_JID}")
        fi
    fi
    if [[ "${RUN_FASE4}" == "1" ]]; then
        # Fase 4 depende de su PROPIA Fase 3 si se envio; si no (RUN_FASE3=0),
        # de la validacion preliminar. No de las Fases 3 de los otros kernels:
        # son independientes y encadenarlas solo alargaria el camino critico.
        enviar "F4 ${nombre}" "${dir_f4}" "${script_f4}" "${j3:-${DEP_BASE}}"
        JOBS_CAMPANA+=("${ULTIMO_JID}")
        if [[ "${RUN_ENERGY_PASS}" == "1" && "${SMOKE_TEST:-0}" != "1" ]]; then
            EXPORT_EXTRA="RUN_KIND=energy,ITERS_LIST=${energy_iters}"
            enviar "F4 ${nombre} (energia)" "${dir_f4}" "${script_f4}" "${j3:-${DEP_BASE}}"
            unset EXPORT_EXTRA
            JOBS_CAMPANA+=("${ULTIMO_JID}")
        fi
    fi
}

cadena_kernel "GEMM"        Fase_3/GEMM        run_gemm_chained.sbatch \
                            Fase_4/GEMM        run_gemm_chained.sbatch "${ENERGY_ITERS_GEMM}"
cadena_kernel "Convolucion" Fase_3/Convolution run_conv_chained.sbatch \
                            Fase_4/Convolution run_conv_chained.sbatch "${ENERGY_ITERS_CONV}"
cadena_kernel "Stencil"     Fase_3/Stencil     run_stencil_tc.sbatch \
                            Fase_4/Stencil     run_stencil_tc.sbatch "${ENERGY_ITERS_STENCIL}"

# --- Campana de variabilidad: replicas para el ANOVA/Tukey ------------------
# Orquestador aparte (envia sus propios jobs de SLURM y su propio
# post-proceso encadenado) -- se corre con `bash`, no se integra en la cadena
# enviar()/afterany de arriba porque maneja su propio grafo de dependencias
# internamente. No depende de la validacion preliminar ni de F3/F4: usa un
# tamano chico y propio (NX=NY=1024 en Stencil, N=1024 en GEMM, HW=64 en
# Conv), pensado para ser barato y repetirse muchas veces, no para reusar los
# jobs grandes de arriba.
#
# OJO -- esto NO estaba cubierto hasta esta correccion: tools/lanzar_campana_
# variabilidad.sh trae sus propios defaults (REPL_{GEMM,CONV,STENCIL}_RUN_KIND
# = numeric, ITERS_LIST chico) que, sin overrides, REPRODUCEN el problema
# original -- energy_window_reliable=0 en toda la campana de replicas, exacto
# lo que costo diagnosticar y corregir esta sesion (jobs 7114-7129 en GEMM/
# Conv, tanda D en Stencil). Una llamada bash tools/lanzar_campana_
# variabilidad.sh "en seco" NO hereda la correccion -- hay que pasarle los
# mismos overrides que se usaron para validar los datos que ya estan en
# campana_final_20260912/, o un relanzamiento del pipeline completo vuelve a
# dejar sin energia confiable justo la campana que el post-proceso (ANOVA,
# Etapa 7) necesita con mas replicas.
#
# GEMM/Convolucion: la UNICA variante de variabilidad que se valido y se usa
# en las figuras (F1/F5/F7/F8) es la de energia -- no existe una campana
# "solo numerica" de GEMM/Conv separada que preservar, asi que se fuerza
# RUN_KIND=energy con el ITERS_LIST validado, SIEMPRE que corre variabilidad
# (no depende de RUN_ENERGY_PASS: son datos distintos, con un solo proposito).
#
# Stencil es distinto: la campana SI tiene dos variantes con proposito
# distinto que conviven en campana_final_20260912/ -- tandas A/B/C (defaults,
# RUN_KIND=numeric, checkpoints densos: de ahi sale el drift fino de F3/F4/F6)
# y tanda D (RUN_KIND=energy, SPATIAL_COMP=off, ITERS_LIST grande: la UNICA
# fuente de energia Tensor Core, aun incompleta -- ver jobs 7219-7221 en
# cola). Por eso Stencil se manda en DOS llamadas: la primera con sus propios
# defaults (reproduce A/B/C), la segunda SOLO si RUN_ENERGY_PASS=1 (reproduce
# D) -- RUN_GEMM=0/RUN_CONV=0 ahi para no repetir GEMM/Conv, que ya se
# mandaron en la primera llamada.
if [[ "${RUN_VARIABILIDAD}" == "1" && "${SMOKE_TEST:-0}" != "1" ]]; then
    echo
    echo "################################################################"
    echo "# Campana de variabilidad (replicas, ${VARIABILIDAD_REPLICAS}x)"
    echo "################################################################"
    export REPL_GEMM_RUN_KIND="${REPL_GEMM_RUN_KIND:-energy}"
    export REPL_GEMM_ITERS_LIST="${REPL_GEMM_ITERS_LIST:-19000 38000}"
    export REPL_CONV_RUN_KIND="${REPL_CONV_RUN_KIND:-energy}"
    export REPL_CONV_ITERS_LIST="${REPL_CONV_ITERS_LIST:-30000 60000}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        DRY_RUN=1 REPLICAS="${VARIABILIDAD_REPLICAS}" bash tools/lanzar_campana_variabilidad.sh
    else
        REPLICAS="${VARIABILIDAD_REPLICAS}" bash tools/lanzar_campana_variabilidad.sh
    fi

    if [[ "${RUN_ENERGY_PASS}" == "1" ]]; then
        echo
        echo "################################################################"
        echo "# Campana de variabilidad -- Stencil, SOLO energia (tipo tanda D)"
        echo "################################################################"
        if [[ "${DRY_RUN}" == "1" ]]; then
            DRY_RUN=1 RUN_GEMM=0 RUN_CONV=0 REPLICAS="${VARIABILIDAD_REPLICAS}" \
                REPL_STENCIL_RUN_KIND=energy REPL_STENCIL_SPATIAL_COMP=off \
                REPL_STENCIL_ITERS_LIST="26000 52000" \
                bash tools/lanzar_campana_variabilidad.sh
        else
            RUN_GEMM=0 RUN_CONV=0 REPLICAS="${VARIABILIDAD_REPLICAS}" \
                REPL_STENCIL_RUN_KIND=energy REPL_STENCIL_SPATIAL_COMP=off \
                REPL_STENCIL_ITERS_LIST="26000 52000" \
                bash tools/lanzar_campana_variabilidad.sh
        fi
    fi
    unset REPL_GEMM_RUN_KIND REPL_GEMM_ITERS_LIST REPL_CONV_RUN_KIND REPL_CONV_ITERS_LIST
fi

# --- Post-proceso: depende de TODOS los jobs de campana ---------------------
if [[ "${RUN_POST}" == "1" ]]; then
    if [[ "${#JOBS_CAMPANA[@]}" -eq 0 ]]; then
        echo "AVISO: no se envio ningun job de Fase 3/4, asi que el post-proceso" >&2
        echo "  se envia sin dependencias, sobre lo que ya haya en results/." >&2
        DEP_POST="${DEP_BASE}"
    else
        # afterany con varios ids: "afterany:J1:J2:...". oom_guard.sh clasifica
        # cada job y deja pasar solo los fallos de memoria; otros errores
        # mantienen bloqueado el post-proceso.
        DEP_POST="$(IFS=:; echo "${JOBS_CAMPANA[*]}")"
    fi
    # Se asigna y se limpia a mano en vez de con el prefijo `VAR=... enviar`:
    # para una FUNCION, bash no siempre restaura la variable al terminar la
    # llamada, y el flag se colaria en cualquier envio posterior.
    EXTRA_SBATCH_ARGS="${POST_PARTITION:+--partition=${POST_PARTITION}}"
    enviar "post-proceso (stats+Pareto)" "." "tools/postproceso.sbatch" "${DEP_POST}"
    unset EXTRA_SBATCH_ARGS
fi

echo
echo "################################################################"
echo "# Jobs enviados"
echo "################################################################"
printf '  %s\n' "${RESUMEN[@]}"
echo
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1: no se envio nada. Quite la variable para enviar de verdad."
else
    echo "Seguimiento:  squeue -u \"\$USER\"        (cola)"
    echo "              sacct -j <job_id> -X      (estado final de un job)"
    echo "Este script YA TERMINO: la campana sigue en la cola de SLURM."
fi
