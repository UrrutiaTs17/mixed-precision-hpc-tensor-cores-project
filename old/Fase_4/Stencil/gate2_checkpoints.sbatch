#!/bin/bash
#SBATCH --job-name=f4_gate2_checkpoints
#SBATCH --output=logs/f4_gate2_%j.out
#SBATCH --error=logs/f4_gate2_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#
# GATE 2 de la campana de Pareto 3D (Fase 4): checkpoints por lista y archivado.
#
# Verifica, a 512^2 con el operador difusivo y la CI monomodo:
#   - exactamente un CSV_CKPT por ruta y por iteracion de --checkpoint-iters,
#   - un CSV_NORM por iteracion de checkpoint,
#   - un campo archivado por ruta en --archive-iters, con manifest.json valido
#     y sha256 verificable,
#   - first_nonfinite = -1 en todas las rutas,
#   - pico de RSS que NO crece ni al subir --iters ni al duplicar el numero de
#     checkpoints (la referencia FP64 va a disco, ver ReferenceSpill).
#
# ---------------------------------------------------------------------------
# Sobre --ci-p en este gate
# ---------------------------------------------------------------------------
# El enunciado del GATE 2 fija la malla en 512^2 y no toca --ci-p, con lo que
# queda el default 168. Ese p es legal a 512^2 (el modo m=2p=336 cabe en los 510
# armonicos de la malla, el binario lo acepta) pero corresponde a k=2.07 rad,
# casi Nyquist: el factor de amortiguamiento es |g| = 0.106 por iteracion y en
# 20 iteraciones el campo cae a ~3e-20 de su norma inicial. FP16, cuyo minimo
# normal es 6.1e-5, se hunde a CERO alrededor de la iteracion 5.
#
# El criterio "first_nonfinite = -1" seguiria pasando -- cero es finito -- de
# modo que el gate daria verde sin haber probado nada para FP16: su rel_l2 seria
# exactamente 1 (el error es la referencia entera). Por eso se corren DOS
# variantes:
#
#   A) LITERAL      --ci-p 168 : el comando exacto del enunciado. Se valida con
#                   --esperar-colapso, y sirve para dejar constancia medida del
#                   colapso en vez de afirmarlo.
#   B) REPRESENTATIVA --ci-p 5 : k = 0.0615 rad, practicamente el mismo numero de
#                   onda que la campana real usa a 16384^2 (k = 0.0644 con
#                   p=168). |g|^20 = 0.972, campo vivo. Es la variante cuyo
#                   veredicto cuenta.
#
# Uso:
#   sbatch gate2_checkpoints.sbatch
# Variables: NX, NY (def. 512), CI_P_REPRESENTATIVO (def. 5), CUDA_ARCH, NVCC.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

REPO_ROOT="$(git rev-parse --show-toplevel)"
NX="${NX:-512}"
NY="${NY:-512}"
CI_P_REPRESENTATIVO="${CI_P_REPRESENTATIVO:-5}"
CUDA_ARCH="${CUDA_ARCH:-80}"
NVCC="${NVCC:-/opt/ohpc/pub/devtools/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvcc}"

JOB_ID="${SLURM_JOB_ID:-manual}"
WORK="gate2_${JOB_ID}"
mkdir -p "${WORK}"

CKPT_LISTA="1,2,5,10,20"
CKPT_LISTA_LARGA="1,2,5,10,20,30,40,60,80,100"
ARCH_LISTA="20"

echo "Nodo      : $(hostname)"
echo "Fecha     : $(date -Iseconds)"
echo "HEAD      : $(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
echo "Malla     : ${NX}x${NY}"
nvidia-smi || true
echo

NVML_FLAGS=()
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

BIN="${PWD}/${WORK}/stencil_tc"
echo "== Compilando HEAD =="
( cd "${REPO_ROOT}/Fase_3/Stencil" && "${NVCC}" -std=c++17 stencil_tensor_activation.cu -o "${BIN}" \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    -Xcompiler "-O3 -funroll-loops -ffp-contract=off" \
    --allow-unsupported-compiler "${NVML_FLAGS[@]}" )
echo "sha256 binario: $(sha256sum "${BIN}" | awk '{print $1}')"
echo

fallos=0
# Extrae "Maximum resident set size" de /usr/bin/time -v. Es el pico de RSS del
# proceso: la magnitud que el gate exige que NO escale.
rss_kb() { grep -E "Maximum resident set size" "$1" | grep -oE "[0-9]+" | tail -1; }

# $1 nombre  $2 ci_p  $3 iters  $4 lista_ckpt  $5 lista_archivo|-  $6 flags extra
corrida() {
    local nombre="$1" ci_p="$2" iters="$3" lista="$4" arch="$5"; shift 5
    local dir="${WORK}/${nombre}"
    mkdir -p "${dir}/archive"
    local args=(--nx "${NX}" --ny "${NY}" --iters "${iters}" --tc both
                --op-mode diffusive --ci-mode monomode --ci-p "${ci_p}"
                --checkpoint-iters "${lista}")
    if [[ "${arch}" != "-" ]]; then
        args+=(--archive-iters "${arch}" --archive-dir "${dir}/archive")
    else
        args+=(--archive-dir "${dir}/archive")
    fi
    echo "### ${nombre}: ci_p=${ci_p} iters=${iters} ckpt=[${lista}] archivo=[${arch}]"
    /usr/bin/time -v "${BIN}" "${args[@]}" > "${dir}/salida.log" 2> "${dir}/time.log" || {
        echo "   X el binario devolvio rc!=0"; tail -20 "${dir}/time.log"; fallos=$((fallos+1)); }
    echo "   RSS pico: $(rss_kb "${dir}/time.log") kB"
}

corrida literal        168                        20  "${CKPT_LISTA}"       "${ARCH_LISTA}"
corrida representativa "${CI_P_REPRESENTATIVO}"   20  "${CKPT_LISTA}"       "${ARCH_LISTA}"
corrida rss_iters100   "${CI_P_REPRESENTATIVO}"  100  "${CKPT_LISTA}"       -
corrida rss_ckpt10     "${CI_P_REPRESENTATIVO}"  100  "${CKPT_LISTA_LARGA}" -
echo

echo "############ Validacion de la variante REPRESENTATIVA (la que decide) ############"
set +e
python3 tools/validar_gate2.py --log "${WORK}/representativa/salida.log" \
    --iters-ckpt "${CKPT_LISTA}" --iters-archivo "${ARCH_LISTA}" \
    --archive-dir "${WORK}/representativa/archive"
[[ $? -ne 0 ]] && fallos=$((fallos+1))
set -e
echo

echo "############ Variante LITERAL del enunciado (ci_p=168, colapso esperado) ############"
set +e
python3 tools/validar_gate2.py --log "${WORK}/literal/salida.log" \
    --iters-ckpt "${CKPT_LISTA}" --iters-archivo "${ARCH_LISTA}" \
    --archive-dir "${WORK}/literal/archive" --esperar-colapso
[[ $? -ne 0 ]] && fallos=$((fallos+1))
set -e
echo
echo "-- rel_l2 final por ruta en la variante literal (1.0 = ruta hundida a cero) --"
grep -E "^CSV_CKPT,[^,]+,20," "${WORK}/literal/salida.log" || true
echo

echo "############ Pico de RSS: no debe escalar ############"
r_base=$(rss_kb "${WORK}/representativa/time.log")
r_it=$(rss_kb "${WORK}/rss_iters100/time.log")
r_ck=$(rss_kb "${WORK}/rss_ckpt10/time.log")
printf "  iters=20  ckpt=5    : %8d kB  (referencia)\n" "${r_base}"
printf "  iters=100 ckpt=5    : %8d kB  (%+d kB)\n" "${r_it}" "$((r_it - r_base))"
printf "  iters=100 ckpt=10   : %8d kB  (%+d kB)\n" "${r_ck}" "$((r_ck - r_base))"
# Un campo FP64 a 512^2 son 2 MiB (~2048 kB). Si el spill no funcionara, cinco
# checkpoints mas costarian ~10 MiB; el margen de 25% absorbe el ruido de
# asignacion del allocator sin dejar pasar eso.
umbral=$(( r_base / 4 ))
for par in "iters:${r_it}" "ckpt:${r_ck}"; do
    delta=$(( ${par#*:} - r_base ))
    if [[ ${delta} -gt ${umbral} ]]; then
        echo "  X ${par%%:*}: el RSS crecio ${delta} kB (umbral ${umbral} kB): el spill no esta conteniendo la referencia"
        fallos=$((fallos+1))
    else
        echo "  OK ${par%%:*}: crecimiento ${delta} kB dentro del umbral ${umbral} kB"
    fi
done

echo
echo "############################################################"
if [[ ${fallos} -eq 0 ]]; then
    echo "# GATE 2: PASA."
else
    echo "# GATE 2: FALLA en ${fallos} comprobacion(es). NO avanzar."
fi
echo "# Salidas crudas en: ${WORK}/"
echo "############################################################"
exit "${fallos}"
