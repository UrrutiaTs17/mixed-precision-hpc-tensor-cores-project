#!/bin/bash
#SBATCH --job-name=validacion_preliminar
#SBATCH --output=logs/validacion_preliminar_%j.out
#SBATCH --error=logs/validacion_preliminar_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# VALIDACION PRELIMINAR A ESCALA REDUCIDA -- los tres kernels.
#
# POR QUE EXISTE
# --------------
# El requisito de hardware del proyecto es Ampere+ (sm_80+): common/wmma_gemm.cuh
# usa cp.async, que no existe antes de Ampere (ver REQUIREMENTS.md). Donde no
# haya una GPU asi, PACCA es la unica ruta de ejecucion -- pero eso no significa
# que la primera corrida en PACCA deba ser la campana completa. Este job es el
# paso barato y corto que va ANTES: confirma que todo compila, corre y produce
# los CSV esperados, con el tamano mas chico de cada lista, antes de
# comprometer decenas de horas de cola de una GPU compartida en algo que
# podria fallar por un error trivial de compilacion.
#
# QUE VERIFICA, EN ORDEN DE LO MAS BARATO A LO MAS CARO
# -----------------------------------------------------
#   PASO 1  Orden de operandos de cublasDgemm en GEMM y Convolucion
#           (Fase_3/tools/verificar_orden_operandos_*.py). Es lo mas importante
#           y lo mas barato: N=32 y hw=64, segundos. Si esto falla, NINGUNA
#           columna rel_l2/rel_linf de ninguna campana de esos dos kernels es
#           valida, asi que se corre primero.
#   PASO 2  Prueba de humo de los tres kernels (SMOKE_TEST=1 en el .sbatch de
#           Fase 4 de cada uno): compila, corre el tamano mas chico con pocas
#           iteraciones y extrae los CSV. Ademas comprueba que la columna
#           anchor_every EXISTE y que en GEMM/Convolucion VARIA POR RUTA
#           (_none=0, _comp=K) dentro de la misma corrida.
#   PASO 3  Gate K=0 y K=1 de los tres kernels (gate3_ancla.sbatch de cada
#           carpeta de Fase 4).
#   PASO 4  Gate de MEDICION en GEMM y Convolucion: que t_iter_ms y
#           energy_gpu_j distingan una ruta de otra y que la referencia FP64
#           se reporte aparte (Fase_4/tools/gate4_medicion.py). Reutiliza los
#           logs del paso 3, asi que no cuesta ninguna corrida extra. Stencil
#           no se evalua: su medicion siempre fue por ruta.
#
# Solo cuando este job termina con codigo 0 tiene sentido enviar las campanas
# completas (run_full_pipeline_pacca.sh ya lo encadena asi: las campanas
# dependen de este job con afterok).
#
#   sbatch tools/validacion_preliminar.sbatch
#   bash   tools/validacion_preliminar.sbatch          # sin SLURM
#   SALTAR_PASO3=1 bash tools/validacion_preliminar.sbatch
#
# Codigo de salida: 0 si TODO paso; el numero de pasos fallidos si no.
# NO usa `set -e` a proposito: se corren todos los pasos aunque uno falle, para
# que un solo job diga todo lo que hay que arreglar en vez de una cosa por
# envio.

set -uo pipefail

# SLURM copia este script a un directorio de spool del nodo y lo ejecuta
# desde ahi -- BASH_SOURCE[0] apunta a esa copia, no al archivo real del
# repo. SLURM_SUBMIT_DIR si apunta al directorio real de envio (root del
# repo, ya que run_full_pipeline_pacca.sh hace `sbatch tools/....sbatch`
# desde ahi), asi que es la fuente correcta cuando existe. Solo se cae al
# truco de BASH_SOURCE para el modo sin SLURM (`bash archivo.sbatch`
# directo, ver REQUIREMENTS.md), donde no hay spool y BASH_SOURCE si es
# el archivo real.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

FALLOS=0
declare -a RESUMEN=()

paso() {
    local etiqueta="$1"; shift
    echo
    echo "################################################################"
    echo "# ${etiqueta}"
    echo "################################################################"
    local rc=0
    "$@" || rc=$?
    if [[ "${rc}" -eq 0 ]]; then
        RESUMEN+=("OK    ${etiqueta}")
    else
        RESUMEN+=("FALLA ${etiqueta}  (codigo ${rc})")
        FALLOS=$((FALLOS + 1))
    fi
    return 0   # nunca aborta el job: se siguen corriendo los demas pasos
}

echo "Nodo: $(hostname)"
echo "Fecha: $(date)"
echo "Repositorio: ${REPO_ROOT}"
nvidia-smi || true

# --- PASO 1: orden de operandos ---------------------------------------------
paso "PASO 1a - orden de operandos, GEMM (gpu_fp64_step)" \
    python3 Fase_3/tools/verificar_orden_operandos_gemm.py --n 32
paso "PASO 1b - orden de operandos + im2col + padding, Convolucion" \
    python3 Fase_3/tools/verificar_orden_operandos_conv.py --hw 64

# --- PASO 2: humo de los tres kernels ---------------------------------------
# Se usan los .sbatch de FASE 4 (no los de Fase 3) porque son los unicos que
# ejercitan tambien el camino del ancla y la columna anchor_every. SMOKE_TEST=1
# los recorta al punto mas chico de cada lista.
# RESULTS_DIR propio: los .sbatch hacen `tee -a` sobre results/run_<job>.log, y
# fuera de SLURM el job id es la constante "manual". Sin un directorio aparte,
# una segunda corrida de esta validacion APENDARIA sus lineas CSV_* a las de la
# primera y el CSV extraido mezclaria pasadas -- que es justo lo que la
# comprobacion de anchor_every de mas abajo no debe tener que adivinar.
HUMO_DIR="${HUMO_DIR:-results/humo_preliminar}"
humo() {
    local dir="$1" script="$2"
    rm -rf "${REPO_ROOT}/${dir}/${HUMO_DIR}"
    ( cd "${REPO_ROOT}/${dir}" && SMOKE_TEST=1 RUN_NCU=0 RESULTS_DIR="${HUMO_DIR}" \
        bash "${script}" )
}
paso "PASO 2a - humo GEMM (Fase 4, SMOKE_TEST=1)"        humo Fase_4/GEMM        run_gemm_chained.sbatch
paso "PASO 2b - humo Convolucion (Fase 4, SMOKE_TEST=1)" humo Fase_4/Convolution run_conv_chained.sbatch
paso "PASO 2c - humo Stencil (Fase 4, SMOKE_TEST=1)"     humo Fase_4/Stencil     run_stencil_tc.sbatch

# Comprobacion de la columna anchor_every sobre los CSV que acaba de extraer el
# paso anterior. En GEMM/Convolucion tiene que VARIAR POR RUTA dentro de la
# misma corrida; en Stencil es constante por invocacion (ver la explicacion en
# Fase_4/tools/README.md). Se comprueba lo primero, que es lo que la auditoria
# pidio confirmar explicitamente.
comprobar_columna() {
    local kernel="$1" dir="$2"
    python3 - "${REPO_ROOT}/${dir}/${HUMO_DIR}" "${kernel}" <<'PYEOF'
import csv, glob, os, sys
resdir, kernel = sys.argv[1], sys.argv[2]
patron = os.path.join(resdir, "summary_%s_*.csv" % kernel)
archivos = sorted(glob.glob(patron), key=os.path.getmtime)
if not archivos:
    print("FALLA: no se genero ningun %s" % patron)
    sys.exit(1)
ruta = archivos[-1]
filas = list(csv.DictReader(open(ruta, newline="", encoding="utf-8")))
if not filas:
    print("FALLA: %s no tiene filas" % ruta)
    sys.exit(1)
if "anchor_every" not in filas[0]:
    print("FALLA: %s no trae la columna anchor_every" % ruta)
    sys.exit(1)
por_comp = {}
for f in filas:
    por_comp.setdefault(f["comp"], set()).add(f["anchor_every"])
print("Archivo: %s (%d filas)" % (ruta, len(filas)))
for comp, valores in sorted(por_comp.items()):
    print("  comp=%-3s -> anchor_every en {%s}" % (comp, ",".join(sorted(valores))))
if por_comp.get("off", set()) != {"0"}:
    print("FALLA: la ruta _none (comp=off) deberia reportar anchor_every=0 SIEMPRE, "
          "y reporta {%s}." % ",".join(sorted(por_comp.get("off", set()))))
    sys.exit(1)
if not (por_comp.get("on", set()) - {"0"}):
    print("FALLA: ninguna fila de la ruta _comp reporta anchor_every>0. La corrida "
          "de humo deberia incluir K=1 (ANCHOR_LIST='0 1' con SMOKE_TEST=1), asi que "
          "o el barrido no se ejecuto o la columna no esta variando por ruta.")
    sys.exit(1)
print("OK: anchor_every varia por ruta dentro de la misma corrida "
      "(_none=0, _comp incluye K>0).")
PYEOF
}
paso "PASO 2d - columna anchor_every por ruta (GEMM)"        comprobar_columna gemm Fase_4/GEMM
paso "PASO 2e - columna anchor_every por ruta (Convolucion)" comprobar_columna conv Fase_4/Convolution

# --- PASO 3: gates del ancla ------------------------------------------------
if [[ "${SALTAR_PASO3:-0}" == "1" ]]; then
    echo
    echo "SALTAR_PASO3=1 -- se omiten los gates K=0/K=1."
else
    gate() {
        local dir="$1"
        ( cd "${REPO_ROOT}/${dir}" && bash gate3_ancla.sbatch )
    }
    paso "PASO 3a - gate K=0/K=1 GEMM"        gate Fase_4/GEMM
    paso "PASO 3b - gate K=0/K=1 Convolucion" gate Fase_4/Convolution
    paso "PASO 3c - gate K=0/K=1 Stencil"     gate Fase_4/Stencil

    # --- PASO 4: la medicion distingue una ruta de otra (GATE 4) ------------
    # Reutiliza los logs que gate3_ancla.sbatch acaba de generar (f4_k0.log y
    # f4_k1.log en results/gate3_<job>/): son exactamente el par K=0 / K=1 de
    # la misma configuracion que este gate necesita, asi que no hace falta
    # volver a correr nada.
    #
    # Stencil NO se evalua: su medicion siempre fue por ruta, el bug era de
    # GEMM y Convolucion.
    gate4() {
        local dir="$1" kernel="$2"
        local gdir
        gdir="$(ls -d "${REPO_ROOT}/${dir}/results/gate3_"* 2>/dev/null | tail -1)"
        if [[ -z "${gdir}" || ! -f "${gdir}/f4_k0.log" || ! -f "${gdir}/f4_k1.log" ]]; then
            echo "NO EVALUABLE: no se encontraron los logs de gate3 en" \
                 "${dir}/results/gate3_*. El PASO 3 tiene que haber corrido antes." >&2
            return 2
        fi
        python3 "${REPO_ROOT}/Fase_4/tools/gate4_medicion.py" --kernel "${kernel}" \
            --etiqueta "validacion preliminar" \
            --k0 "${gdir}/f4_k0.log" --k1 "${gdir}/f4_k1.log"
    }
    paso "PASO 4a - medicion por ruta GEMM"        gate4 Fase_4/GEMM        gemm
    paso "PASO 4b - medicion por ruta Convolucion" gate4 Fase_4/Convolution conv
fi

# --- Resumen ----------------------------------------------------------------
echo
echo "################################################################"
echo "# RESUMEN DE LA VALIDACION PRELIMINAR"
echo "################################################################"
for linea in "${RESUMEN[@]}"; do
    echo "  ${linea}"
done
echo
if [[ "${FALLOS}" -eq 0 ]]; then
    echo "TODO PASO. Recien ahora tiene sentido enviar las campanas completas"
    echo "(run_full_pipeline_pacca.sh, o cada .sbatch de Fase 3/4 por separado)."
    exit 0
fi
echo "${FALLOS} paso(s) fallaron. NO enviar las campanas completas hasta"
echo "resolverlos: una campana lanzada sobre un binario que no pasa estas"
echo "puertas produce numeros que despues no se pueden reportar."
exit "${FALLOS}"
