#!/bin/bash
# Permite continuar una dependencia SLURM solo cuando el predecesor termino por OOM.

if [[ -z "${OOM_DEP_JOBS:-}" ]]; then
    return 0 2>/dev/null || exit 0
fi

if ! command -v sacct >/dev/null 2>&1; then
    echo "ERROR: no se encontro sacct para clasificar OOM_DEP_JOBS=${OOM_DEP_JOBS}." >&2
    return 1 2>/dev/null || exit 1
fi

is_oom_state() {
    local state="$1" exit_code="$2"
    [[ "${state}" == OUT_OF_MEMORY* || "${exit_code}" == *:9 || "${exit_code}" == *:137 ]]
}

for dependency_job in ${OOM_DEP_JOBS//:/ }; do
    dependency_record="$(sacct -X -n -P -j "${dependency_job}" -o State,ExitCode 2>/dev/null | sed -n '/[^|]/p' | head -n 1)"
    if [[ -z "${dependency_record}" ]]; then
        echo "ERROR: no se pudo consultar el estado de SLURM job ${dependency_job}." >&2
        return 1 2>/dev/null || exit 1
    fi

    dependency_state="${dependency_record%%|*}"
    dependency_exit="${dependency_record#*|}"
    if [[ "${dependency_state}" == COMPLETED* && "${dependency_exit}" == 0:0* ]]; then
        continue
    fi

    if is_oom_state "${dependency_state}" "${dependency_exit}"; then
        printf '%s\t%s\tOOM upstream job=%s state=%s exit=%s\n' \
            "$(date --iso-8601=seconds)" "${OOM_PHASE_LABEL:-unknown_phase}" \
            "${dependency_job}" "${dependency_state}" "${dependency_exit}" \
            >> "${OOM_FAILURE_LOG:-oom_failures_pacca.log}"
        echo "OOM upstream detectado en job ${dependency_job}; se continua con ${OOM_PHASE_LABEL:-la siguiente etapa}." >&2
        continue
    fi

    echo "ERROR: job upstream ${dependency_job} termino en ${dependency_state} (${dependency_exit}); no es OOM." >&2
    return 1 2>/dev/null || exit 1
done

unset OOM_DEP_JOBS OOM_DEPENDENCY_LABEL
return 0 2>/dev/null || true
