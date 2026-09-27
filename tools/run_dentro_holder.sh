#!/bin/bash
# Corre Fase 3 -> Fase 4, UN KERNEL A LA VEZ, como job-steps de `srun`
# DENTRO de una asignacion SLURM ya existente (el holder del equipo, JOBID)
# en vez de encolar jobs nuevos con `sbatch`.
#
# Por que es seguro para el holder: un job-step que falla NO tumba la
# asignacion que lo contiene -- demostrado con el propio historial de este
# holder (`sacct -j $JOBID`, 35+ steps en estado FAILED mientras el job
# principal siguio RUNNING sin interrupcion). Este driver nunca usa `set -e`
# a proposito: un kernel que falla se registra y se sigue con el siguiente,
# nunca aborta el script completo ni toca la asignacion.
#
# Orden: TODA la Fase 3 (GEMM, Convolucion, Stencil; cada uno normal +
# energia) primero, despues TODA la Fase 4 en el mismo orden. Nunca dos
# `srun` en paralelo -- siempre se espera a que termine el anterior.
#
# Uso (lanzar SIEMPRE detached, esto corre horas y no debe morir si se
# cierra el ssh):
#   cd ~/mixed-precision-hpc-tensor-cores-project
#   JOBID=7145 nohup bash tools/run_dentro_holder.sh \
#       > "logs_holder_run/driver_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
#   disown
#
# Para detener la SECUENCIA (no el holder) antes de que termine:
#   touch logs_holder_run/HOLDER_RUN_STOP
# El holder sigue vivo; solo deja de lanzarse el siguiente kernel.

set -uo pipefail

JOBID="${JOBID:?export JOBID=<id del holder>, ej. JOBID=7145}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNDIR="${REPO_ROOT}/logs_holder_run"
STATUS_LOG="${RUNDIR}/estado_$(date +%Y%m%d_%H%M%S).log"
STOP_FILE="${RUNDIR}/HOLDER_RUN_STOP"
mkdir -p "${RUNDIR}"
rm -f "${STOP_FILE}"

ENERGY_ITERS_GEMM="${ENERGY_ITERS_GEMM:-24000}"
ENERGY_ITERS_CONV="${ENERGY_ITERS_CONV:-37000}"
ENERGY_ITERS_STENCIL="${ENERGY_ITERS_STENCIL:-4000}"

estado() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${STATUS_LOG}"; }

holder_vivo() {
    sacct -j "${JOBID}" --format=State -n -X 2>/dev/null | grep -q RUNNING
}

# Ejecuta un kernel/pasada como job-step dentro de $JOBID. NO usa `set -e`:
# el llamador decide que hacer con el codigo de salida. extra_env es una
# cadena tipo "RUN_KIND=energy ITERS_LIST=24000" (vacia si no aplica).
correr_paso() {
    local etiqueta="$1" dir="$2" script="$3" extra_env="${4:-}"
    local log="${RUNDIR}/${etiqueta// /_}.log"

    if [[ -f "${STOP_FILE}" ]]; then
        estado "HOLDER_RUN_STOP detectado -- se detiene la SECUENCIA (el holder ${JOBID} sigue vivo)."
        return 2
    fi
    if ! holder_vivo; then
        estado "ALERTA: el holder ${JOBID} ya no esta RUNNING segun sacct -- se detiene la secuencia por seguridad, sin tocar nada."
        return 2
    fi

    estado "INICIA: ${etiqueta} (dir=${dir} script=${script} env=${extra_env:-ninguno})"
    if srun --jobid="${JOBID}" --job-name="mp_${etiqueta// /_}" \
         bash -c "cd '${REPO_ROOT}/${dir}' && ${extra_env} bash '${script}'" \
         > "${log}" 2>&1; then
        estado "OK: ${etiqueta} -> ${log}"
        return 0
    else
        local rc=$?
        estado "FALLO (rc=${rc}): ${etiqueta} -> ${log} -- el holder ${JOBID} sigue vivo, se continua con el siguiente paso."
        return 1
    fi
}

estado "=== Inicio de secuencia dentro del holder ${JOBID} (PID driver=$$) ==="

correr_paso "Validacion preliminar" "." "tools/validacion_preliminar.sbatch"
rc_validacion=$?
if [[ "${rc_validacion}" -eq 2 ]]; then
    exit 0
elif [[ "${rc_validacion}" -ne 0 ]]; then
    estado "La validacion preliminar fallo -- es la puerta de entrada obligatoria, se detiene TODA la secuencia aqui. El holder ${JOBID} sigue vivo, solo se para este driver."
    exit 1
fi

# Corre un paso y sale del driver (sin tocar el holder) si devolvio 2 (stop
# file o holder caido). NO usar `cmd || cond && exit` para esto: por
# precedencia de operadores bash evalua ((cmd || cond) && exit), que con
# cmd=0 (exito) hace corto-circuito y termina el driver en el primer paso
# bueno -- justo el bug que se queria evitar.
paso_o_salir() {
    correr_paso "$@"
    local rc=$?
    if [[ "${rc}" -eq 2 ]]; then
        exit 0
    fi
}

# --- Fase 3: un kernel a la vez ---------------------------------------------
paso_o_salir "F3 GEMM"                Fase_3/GEMM        run_gemm_chained.sbatch
paso_o_salir "F3 GEMM energia"        Fase_3/GEMM        run_gemm_chained.sbatch "RUN_KIND=energy ITERS_LIST=${ENERGY_ITERS_GEMM}"
paso_o_salir "F3 Convolucion"         Fase_3/Convolution run_conv_chained.sbatch
paso_o_salir "F3 Convolucion energia" Fase_3/Convolution run_conv_chained.sbatch "RUN_KIND=energy ITERS_LIST=${ENERGY_ITERS_CONV}"
paso_o_salir "F3 Stencil"             Fase_3/Stencil     run_stencil_tc.sbatch
paso_o_salir "F3 Stencil energia"     Fase_3/Stencil     run_stencil_tc.sbatch "RUN_KIND=energy ITERS_LIST=${ENERGY_ITERS_STENCIL}"

# --- Fase 4: un kernel a la vez, DESPUES de que termine toda la Fase 3 ------
paso_o_salir "F4 GEMM"                Fase_4/GEMM        run_gemm_chained.sbatch
paso_o_salir "F4 GEMM energia"        Fase_4/GEMM        run_gemm_chained.sbatch "RUN_KIND=energy ITERS_LIST=${ENERGY_ITERS_GEMM}"
paso_o_salir "F4 Convolucion"         Fase_4/Convolution run_conv_chained.sbatch
paso_o_salir "F4 Convolucion energia" Fase_4/Convolution run_conv_chained.sbatch "RUN_KIND=energy ITERS_LIST=${ENERGY_ITERS_CONV}"
paso_o_salir "F4 Stencil"             Fase_4/Stencil     run_stencil_tc.sbatch
paso_o_salir "F4 Stencil energia"     Fase_4/Stencil     run_stencil_tc.sbatch "RUN_KIND=energy ITERS_LIST=${ENERGY_ITERS_STENCIL}"

estado "=== Secuencia completa. Detalle por paso en ${RUNDIR}/ ==="
