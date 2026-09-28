#!/usr/bin/env python3
import argparse
import csv
import os
import re
import sys


# Post-proceso de CSV_* de stencil_tensor_activation.cu. Este archivo es
# IDENTICO en Fase_3/tools/ y Fase_4/tools/ (igual que extract_csv_chained.py):
# los dos binarios de Stencil emiten el mismo esquema de columnas, y mantener
# dos variantes divergentes solo invitaba a que una se quedara atras.
#
# anchor_every (ancla FP64) es una COLUMNA REAL de CSV_DRIFT, CSV_SUMMARY y
# CSV_ENERGY: el binario la escribe como ULTIMO campo de cada una de esas tres
# lineas. En Fase 4 sale de g_anchor_every_csv (ver
# Fase_4/Stencil/stencil_tensor_activation.cu); en Fase 3, que no tiene ancla,
# es un 0 literal -- misma columna, mismo esquema, contenido correcto en ambas.
#
# En Stencil es contexto de INVOCACION COMPLETA, no de ruta: todas las filas de
# una corrida comparten el mismo valor, incluidas las rutas de referencia
# (gpu_fp64, cpu_fp64) que nunca ejecutan el ancla. Es correcto -- el ancla es
# un parametro de la corrida. En GEMM/Convolucion, en cambio, varia POR FILA
# dentro de la misma corrida ("_none"=0, "_comp"=K): ver el comentario de
# extract_csv_chained.py. Son dos semanticas distintas bajo el mismo nombre de
# columna, y ninguna herramienta aguas abajo debe asumir la otra.
#
# RESPALDO PARA LOGS VIEJOS: los logs anteriores a la columna traen un campo
# menos. Para esos se sigue reconstruyendo desde la linea de configuracion
# "Ancla FP64 (anchor-every) : K" que el binario imprime una vez por
# invocacion (ANCHOR_RE, mas abajo).
#
# La columna va al FINAL de cada header (no donde "logicamente" iria junto a
# kahan): insertarla en medio correria los indices posicionales que
# SUMMARY_VALUE_FIELDS/SUMMARY_FIELD_COUNT ya usan para mapear parts[] de la
# linea CSV_SUMMARY.
DRIFT_HEADER = [
    "job_id", "kernel", "nx", "ny", "iters", "kahan", "route",
    "iter", "ref_l2", "abs_l2", "rel_l2", "max_abs",
    "anchor_every",
    # Columnas de la campana corregida (DECISIONS.md S5). Van al final y se
    # DERIVAN aqui de lo que el binario ya emitio (no cambian la linea CSV_*):
    "device", "comp_scheme", "error_evaluable", "motivo_exclusion",
]

# Esquema POSICIONAL de la linea CSV_SUMMARY (incluye speedup_cpu porque el
# binario todavia lo emite y ocupa su sitio en parts[]). NO se publica: el
# speedup contra CPU no se usa en ningun analisis (ver DECISIONS.md S5), asi que
# el CSV de salida (SUMMARY_HEADER) ya no lo lleva.
SUMMARY_PARSE_HEADER = [
    "job_id", "kernel", "nx", "ny", "iters", "kahan", "route",
    "t_iter_ms", "t_total_ms", "gflops", "speedup_cpu", "speedup_fp32",
    "t_kernel_ms", "t_convert_ms", "t_checkpoint_ms", "rel_l2",
    "rel_linf", "max_abs", "rel_l2_prop", "rel_linf_prop",
    "first_nonfinite", "store_rel_norm", "store_rel_max_guarded",
    "store_excluded_count", "store_eval_iter", "energy_gpu_j", "energy_cpu_j",
    "energy_total_j", "edp_j_s", "joules_per_gflop", "onset_checkpoint",
    "anchor_every",
]

SUMMARY_HEADER = [c for c in SUMMARY_PARSE_HEADER if c != "speedup_cpu"] + [
    # Leidas de la propia linea CSV_SUMMARY (indices 29, 30, 31, 35):
    "op_mode", "alpha", "ci_mode", "reference_role",
    # Derivadas (DECISIONS.md S5):
    "device", "gpu_valid", "comp_scheme", "error_evaluable", "motivo_exclusion",
]
SUMMARY_ROW_KEYS = list(dict.fromkeys(SUMMARY_PARSE_HEADER + SUMMARY_HEADER))

# -2, no -1: excluye tanto onset_checkpoint (se llena aparte via
# handle_onset) como el nuevo anchor_every (se llena aparte via identity(),
# desde el contexto -- ninguno de los dos viene posicionalmente en parts[]).
SUMMARY_VALUE_FIELDS = SUMMARY_PARSE_HEADER[7:-2]

HORIZON_HEADER = [
    "job_id", "kernel", "nx", "ny", "iters", "kahan", "format",
    "h_predicho", "h_medido", "lambda", "r2", "n_puntos_fit",
    "semilla_A", "nyquist_u0", "piso_siembra", "fit_status",
]

STORE_HEADER = [
    "job_id", "kernel", "nx", "ny", "iters", "kahan", "route",
    "store_rel_norm", "store_rel_max_guarded", "store_excluded_count",
    "store_eval_iter", "ulp_formato",
]

ENERGY_HEADER = [
    "job_id", "kernel", "nx", "ny", "iters", "kahan", "route",
    "energy_gpu_j", "energy_cpu_j", "energy_total_j", "edp_j_s",
    "joules_per_gflop", "time_total_s", "flops_total_billions",
    "energy_gpu_j_per_iter", "energy_window_reliable",
    "anchor_every",
    "device", "gpu_valid", "comp_scheme", "n_cpu_fp64_invocaciones",
]

RUN_RE = re.compile(
    r"^(?:Corrida|Barrido de horizonte): NX=(\d+), NY=(\d+), "
    r"ITERS=(\d+), TC=[^,]+, KAHAN=(off|on)"
)
DIM_RE = re.compile(r"^Dimensiones \(nx, ny\)\s*:\s*(\d+),\s*(\d+)")
ITERS_RE = re.compile(r"^Iteraciones\s*:\s*(\d+)")
KAHAN_RE = re.compile(r"^Kahan \(residuo almacen\.\)\s*:\s*(off|on)")
# "Ancla FP64 (anchor-every)  : 5 (activa)" / "... : 0 (deshabilitada)" --
# impresa una vez por invocacion por el binario de Fase 4. Solo se usa como
# RESPALDO para logs anteriores a que anchor_every fuera columna. Logs de
# Fase 3 (que nunca imprimen esta linea) no hacen match y se quedan en "0",
# que es la lectura correcta: sin ancla, no hubo ancla.
ANCHOR_RE = re.compile(r"^Ancla FP64 \(anchor-every\)\s*:\s*(\d+)")

# Indice (0-based, contando el token CSV_*) de la columna anchor_every en cada
# linea que la lleva. Son los ULTIMOS campos de sus respectivas lineas:
#   CSV_DRIFT  : token,route,iter,ref_l2,abs_l2,rel_l2,max_abs,anchor_every
#   CSV_ENERGY : ...,energy_gpu_j_per_iter,energy_window_reliable,anchor_every
#   CSV_SUMMARY: ...,reference_role,execution_mode,anchor_every
DRIFT_ANCHOR_IDX = 7
ENERGY_ANCHOR_IDX = 15
SUMMARY_ANCHOR_IDX = 37
# Numero de tokens (contando CSV_*) de una linea completa de cada tipo, y sitio
# de las columnas de Fase 4 que se publican tal cual desde CSV_SUMMARY.
DRIFT_TOKENS_FULL = 8
ENERGY_TOKENS_FULL = 16
SUMMARY_TOKENS_FULL = 38
SUMMARY_OP_MODE_IDX = 29
SUMMARY_ALPHA_IDX = 30
SUMMARY_CI_MODE_IDX = 31
SUMMARY_REFERENCE_ROLE_IDX = 35


def anchor_de_fila(parts, idx, context):
    """anchor_every de la fila, con respaldo a la linea de configuracion.

    Se llama SIEMPRE con `parts` sin rellenar: pad() taparia con "NaN" la
    ausencia de la columna en un log viejo y el respaldo nunca se activaria.
    """
    if len(parts) > idx:
        valor = clean(parts[idx])
        if valor != "NaN":
            return valor
    return context.get("anchor_every", "0")


# "-NAN"/"+NAN": iostream imprime asi los NaN con signo (p. ej. "-nan").
NONFINITE_TOKENS = {"NONFINITE", "NA", "NAN", "-NAN", "+NAN", "NO_EVALUABLE",
                    "INF", "+INF", "-INF"}


def clean(value):
    value = value.strip()
    upper = value.upper()
    if value == "" or upper in NONFINITE_TOKENS:
        return "NaN"
    return value


def pad(fields, expected):
    if len(fields) >= expected:
        return fields[:expected]
    return fields + ["NaN"] * (expected - len(fields))


def parse_csv_line(line):
    if not line.startswith("CSV_"):
        return None
    try:
        return next(csv.reader([line]))
    except csv.Error:
        return None


def identity(context, job_id, kernel, route_or_format_key, route_or_format):
    return {
        "job_id": job_id,
        "kernel": kernel,
        "nx": context.get("nx", "NaN"),
        "ny": context.get("ny", "NaN"),
        "iters": context.get("iters", "NaN"),
        "kahan": context.get("kahan", "NaN"),
        "anchor_every": context.get("anchor_every", "0"),
        route_or_format_key: route_or_format,
    }


def reset_run(context, summary_by_route):
    summary_by_route.clear()
    context["nx"] = "NaN"
    context["ny"] = "NaN"
    context["iters"] = "NaN"
    context["kahan"] = "NaN"
    # NO se resetea aqui a "NaN": a diferencia de nx/ny/iters/kahan (que SI
    # vienen en la linea "Corrida:"), anchor_every llega en una linea de
    # configuracion SEPARADA, impresa antes de la primera "Corrida:" de cada
    # invocacion -- resetearlo aqui lo borraria justo antes de que
    # update_context_from_run_header lo vuelva a necesitar. Su default vive
    # en identity() ("0"), no aqui.


def ensure_summary_row(summary_rows, summary_by_route, context, job_id, kernel, route):
    if route in summary_by_route:
        return summary_by_route[route]
    row = {key: "NaN" for key in SUMMARY_ROW_KEYS}
    row.update(identity(context, job_id, kernel, "route", route))
    summary_rows.append(row)
    summary_by_route[route] = row
    return row


def update_context_from_run_header(line, context, summary_by_route):
    match = RUN_RE.match(line)
    if match:
        reset_run(context, summary_by_route)
        context["nx"], context["ny"], context["iters"], context["kahan"] = match.groups()
        return

    match = DIM_RE.match(line)
    if match:
        summary_by_route.clear()
        context["nx"], context["ny"] = match.groups()
        return

    match = ITERS_RE.match(line)
    if match:
        context["iters"] = match.group(1)
        return

    match = KAHAN_RE.match(line)
    if match:
        context["kahan"] = match.group(1)
        return

    match = ANCHOR_RE.match(line)
    if match:
        context["anchor_every"] = match.group(1)


def motivo_drift(parts):
    """(error_evaluable, motivo_exclusion) de una fila CSV_DRIFT, mirando los
    tokens CRUDOS: emit_csv_drift_row imprime NONFINITE en los 4 campos si la
    REFERENCIA no es finita, y solo en los 3 de error si lo no finito es la
    SOLUCION -- clean() los colapsaria a NaN y se perderia esa distincion."""
    raw_ref = parts[3].strip().upper() if len(parts) > 3 else ""
    raw_rel = parts[5].strip().upper() if len(parts) > 5 else ""
    if raw_ref in NONFINITE_TOKENS:
        return "0", "reference_non_finite"
    if raw_rel in NONFINITE_TOKENS:
        return "0", "solution_non_finite"
    return "1", "ok"


def handle_drift(parts, rows, summary_rows, summary_by_route, context, job_id, kernel):
    if len(parts) < 7:
        return
    route = clean(parts[1])
    ensure_summary_row(summary_rows, summary_by_route, context, job_id, kernel, route)
    row = identity(context, job_id, kernel, "route", route)
    evaluable, motivo = motivo_drift(parts)
    row.update({
        "iter": clean(parts[2]),
        "ref_l2": clean(parts[3]),
        "abs_l2": clean(parts[4]),
        "rel_l2": clean(parts[5]),
        "max_abs": clean(parts[6]),
        # La columna real de la fila gana sobre el contexto de identity().
        "anchor_every": anchor_de_fila(parts, DRIFT_ANCHOR_IDX, context),
        "error_evaluable": evaluable,
        "motivo_exclusion": motivo,
    })
    rows.append(row)


# Campos de una linea CSV_SUMMARY contando el token inicial. El binario llego a
# emitir dos columnas derivadas, speedup_fp64_gpu y speedup_fp64_cpu, en las
# posiciones 11 y 12 (justo tras speedup_fp32); se retiraron porque el speedup
# contra una referencia FP64 se calcula en el analisis a partir de los tiempos
# crudos, que es lo que el CSV publica. Los logs generados mientras existieron
# traen 30 o 31 campos en vez de 29: truncarlos por la derecha con pad() dejaria
# esas columnas ocupando el sitio de t_kernel_ms y correria todo el resto del
# esquema, asi que se descartan en su posicion, de la mas reciente a la mas
# antigua para que los indices no se muevan bajo los pies.
SUMMARY_FIELD_COUNT = 29
# (numero de campos del log heredado, indice de la columna sobrante). Las reglas
# encadenan: una linea de 31 pierde la 12 con la primera, queda en 30 y pierde la
# 11 con la segunda.
SUMMARY_LEGACY_DROPS = [(31, 12), (30, 11)]


def handle_summary(parts, rows, summary_by_route, context, job_id, kernel):
    # ANTES de cualquier recorte: pad() y los SUMMARY_LEGACY_DROPS mueven o
    # tapan la ultima columna, que es justo anchor_every. Los logs heredados de
    # 30/31 campos no la traen y caen solos al respaldo por contexto.
    anchor = anchor_de_fila(parts, SUMMARY_ANCHOR_IDX, context)
    # Columnas de Fase 4 que SI viajan en la linea (ver el orden en
    # emit_csv_summary_row del .cu). Solo existen en logs con la linea completa
    # (SUMMARY_TOKENS_FULL); los logs anteriores las dejan en NaN.
    extras = {}
    if len(parts) >= SUMMARY_TOKENS_FULL:
        extras = {"op_mode": clean(parts[SUMMARY_OP_MODE_IDX]),
                  "alpha": clean(parts[SUMMARY_ALPHA_IDX]),
                  "ci_mode": clean(parts[SUMMARY_CI_MODE_IDX]),
                  "reference_role": clean(parts[SUMMARY_REFERENCE_ROLE_IDX])}
    for legacy_count, drop_at in SUMMARY_LEGACY_DROPS:
        if len(parts) == legacy_count:
            parts = parts[:drop_at] + parts[drop_at + 1:]
    parts = pad(parts, SUMMARY_FIELD_COUNT)
    route = clean(parts[1])
    context["nx"] = clean(parts[2])
    context["ny"] = clean(parts[3])
    context["iters"] = clean(parts[4])
    context["kahan"] = clean(parts[5])
    row = ensure_summary_row(rows, summary_by_route, context, job_id, kernel, route)
    row.update(identity(context, job_id, kernel, "route", route))
    row["anchor_every"] = anchor
    row.update(extras)
    for field, value in zip(SUMMARY_VALUE_FIELDS, parts[6:SUMMARY_FIELD_COUNT]):
        row[field] = clean(value)


def handle_onset(parts, summary_rows, summary_by_route, context, job_id, kernel):
    if len(parts) < 3:
        return
    route = clean(parts[1])
    row = ensure_summary_row(summary_rows, summary_by_route, context, job_id, kernel, route)
    row["onset_checkpoint"] = clean(parts[2])


def handle_horizon(parts, rows, job_id, kernel):
    parts = pad(parts, 15)
    row = {
        "job_id": job_id,
        "kernel": kernel,
        "format": clean(parts[1]),
        "nx": clean(parts[2]),
        "ny": clean(parts[3]),
        "iters": clean(parts[4]),
        "kahan": clean(parts[5]),
        "h_predicho": clean(parts[6]),
        "h_medido": clean(parts[7]),
        "lambda": clean(parts[8]),
        "r2": clean(parts[9]),
        "n_puntos_fit": clean(parts[10]),
        "semilla_A": clean(parts[11]),
        "nyquist_u0": clean(parts[12]),
        "piso_siembra": clean(parts[13]),
        "fit_status": clean(parts[14]),
    }
    rows.append(row)


def handle_store(parts, rows, job_id, kernel):
    parts = pad(parts, 11)
    row = {
        "job_id": job_id,
        "kernel": kernel,
        "route": clean(parts[1]),
        "nx": clean(parts[2]),
        "ny": clean(parts[3]),
        "iters": clean(parts[4]),
        "kahan": clean(parts[5]),
        "store_rel_norm": clean(parts[6]),
        "store_rel_max_guarded": clean(parts[7]),
        "store_excluded_count": clean(parts[8]),
        "store_eval_iter": clean(parts[9]),
        "ulp_formato": clean(parts[10]),
    }
    rows.append(row)


def new_energy_filter_stats():
    return {"filas_gpu": 0, "aceptadas": 0, "ventana_corta": 0, "gpu_invalido": 0}


# El contador de energia de NVML se cuantiza en saltos del orden de ~20-25 ms
# de GPU cargada (ver REGIMEN DE VALIDEZ en tools/power_sampling.h), asi que
# una ventana corta produce un numero que no es comparable con el de otra ruta.
# Esas filas se EXCLUYEN del promedio anulando energy_gpu_j_per_iter -- que es
# la columna con la que se comparan formatos entre si, y cualquier media
# posterior salta los NaN sola -- en vez de borrar la fila: energy_gpu_j y
# energy_window_reliable se conservan crudos para poder auditar el descarte.
# El motivo se contabiliza y se reporta al terminar: el filtrado nunca es
# silencioso.
def apply_energy_window_filter(row, stats):
    if row["energy_window_reliable"] == "NaN":
        # Ruta CPU -- que fija gpu_valid sin leer NVML y no tiene ventana de
        # GPU que juzgar -- o log anterior a estas columnas. En ambos casos
        # energy_gpu_j_per_iter ya viene NaN desde el binario.
        return
    stats["filas_gpu"] += 1
    if row["energy_gpu_j"] == "NaN":
        motivo = "gpu_invalido"
    elif row["energy_window_reliable"] != "1":
        motivo = "ventana_corta"
    else:
        stats["aceptadas"] += 1
        return
    row["energy_gpu_j_per_iter"] = "NaN"
    stats[motivo] += 1


def handle_energy(parts, rows, summary_rows, summary_by_route, context, job_id, kernel, stats):
    anchor = anchor_de_fila(parts, ENERGY_ANCHOR_IDX, context)  # antes del pad()
    parts = pad(parts, 15)
    route = clean(parts[1])
    row = identity(context, job_id, kernel, "route", route)
    row["anchor_every"] = anchor
    row["nx"] = clean(parts[2])
    row["ny"] = clean(parts[3])
    row["iters"] = clean(parts[4])
    row["kahan"] = clean(parts[5])
    context["nx"] = row["nx"]
    context["ny"] = row["ny"]
    context["iters"] = row["iters"]
    context["kahan"] = row["kahan"]
    row.update({
        "energy_gpu_j": clean(parts[6]),
        "energy_cpu_j": clean(parts[7]),
        "energy_total_j": clean(parts[8]),
        "edp_j_s": clean(parts[9]),
        "joules_per_gflop": clean(parts[10]),
        "time_total_s": clean(parts[11]),
        "flops_total_billions": clean(parts[12]),
        "energy_gpu_j_per_iter": clean(parts[13]),
        "energy_window_reliable": clean(parts[14]),
    })
    apply_energy_window_filter(row, stats)
    rows.append(row)

    summary = ensure_summary_row(summary_rows, summary_by_route, context, job_id, kernel, route)
    summary.update({
        "energy_gpu_j": row["energy_gpu_j"],
        "energy_cpu_j": row["energy_cpu_j"],
        "energy_total_j": row["energy_total_j"],
        "edp_j_s": row["edp_j_s"],
        "joules_per_gflop": row["joules_per_gflop"],
    })


def device_de_ruta(route):
    """cpu | gpu, por el prefijo de la ruta (CPU_FP32, CPU_FP64 -> cpu)."""
    return "cpu" if route.upper().startswith("CPU_") else "gpu"


def comp_scheme_de(route, kahan):
    """Esquema de compensacion EFECTIVO de la fila, segun lo que el binario
    emitio: sufijo _SP en la ruta -> spatial; ruta WMMA con kahan=on ->
    kahan_local; el resto (referencias GPU_/CPU_ y WMMA sin compensar) -> none.
    El binario rechaza kahan+spatial a la vez, asi que los tres son excluyentes."""
    r = route.upper()
    if r.endswith("_SP"):
        return "spatial"
    if r.startswith("WMMA") and kahan == "on":
        return "kahan_local"
    return "none"


def error_primario(row):
    """Metrica de error con la que se juzga la fila: rel_l2_prop en las rutas
    WMMA (error del estado propagado en 16 bits), rel_l2 en las demas."""
    if row["route"].upper().startswith("WMMA"):
        return row.get("rel_l2_prop", "NaN")
    return row.get("rel_l2", "NaN")


def derivar_columnas(drift_rows, summary_rows, energy_rows):
    """Rellena las columnas derivadas de DECISIONS.md S5 y devuelve
    (n_cpu_fp64_por_celda, comp_schemes_wmma)."""
    for row in drift_rows:
        row["device"] = device_de_ruta(row["route"])
        row["comp_scheme"] = comp_scheme_de(row["route"], row["kahan"])

    comp_wmma = set()
    for row in summary_rows:
        route = row["route"]
        row["device"] = device_de_ruta(route)
        row["comp_scheme"] = comp_scheme_de(route, row["kahan"])
        if route.upper().startswith("WMMA"):
            comp_wmma.add(row["comp_scheme"])
        # gpu_valid: la ruta CPU nunca tiene captura de GPU valida; en las de GPU,
        # el binario imprime NaN en energy_gpu_j cuando la captura NVML fallo.
        row["gpu_valid"] = "1" if (row["device"] == "gpu" and row["energy_gpu_j"] != "NaN") else "0"
        if error_primario(row) != "NaN":
            row["error_evaluable"], row["motivo_exclusion"] = "1", "ok"
        else:
            row["error_evaluable"] = "0"
            try:
                nf = int(float(row["first_nonfinite"]))
            except ValueError:
                nf = -1
            row["motivo_exclusion"] = "solution_non_finite" if nf >= 0 else "not_evaluable"

    n_cpu = {}
    for row in energy_rows:
        if row["route"] == "CPU_FP64":
            celda = (row["nx"], row["ny"], row["iters"])
            n_cpu[celda] = n_cpu.get(celda, 0) + 1
    for row in energy_rows:
        row["device"] = device_de_ruta(row["route"])
        row["comp_scheme"] = comp_scheme_de(row["route"], row["kahan"])
        # gpu_valid: el binario deja energy_window_reliable en NaN cuando no hubo
        # captura de GPU valida (y en las rutas CPU). NO se toca energy_gpu_j: un
        # 0.0 en una ruta CPU queda crudo, pero gpu_valid=0 lo marca.
        row["gpu_valid"] = "1" if (row["device"] == "gpu" and row["energy_window_reliable"] != "NaN") else "0"
        row["n_cpu_fp64_invocaciones"] = str(n_cpu.get((row["nx"], row["ny"], row["iters"]), 0))
    return n_cpu, comp_wmma


def read_log(path, job_id, kernel):
    context = {"nx": "NaN", "ny": "NaN", "iters": "NaN", "kahan": "NaN", "anchor_every": "0"}
    summary_by_route = {}
    drift_rows = []
    summary_rows = []
    horizon_rows = []
    store_rows = []
    energy_rows = []
    energy_filter_stats = new_energy_filter_stats()
    tokens_por_linea = {"CSV_DRIFT": {}, "CSV_SUMMARY": {}, "CSV_ENERGY": {}}

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            update_context_from_run_header(line, context, summary_by_route)
            parts = parse_csv_line(line)
            if not parts:
                continue
            token = parts[0]
            if token in tokens_por_linea:
                cuenta = tokens_por_linea[token]
                cuenta[len(parts)] = cuenta.get(len(parts), 0) + 1
            if token == "CSV_DRIFT":
                handle_drift(parts, drift_rows, summary_rows, summary_by_route, context, job_id, kernel)
            elif token == "CSV_SUMMARY":
                handle_summary(parts, summary_rows, summary_by_route, context, job_id, kernel)
            elif token == "CSV_ONSET":
                handle_onset(parts, summary_rows, summary_by_route, context, job_id, kernel)
            elif token == "CSV_HORIZON":
                handle_horizon(parts, horizon_rows, job_id, kernel)
            elif token == "CSV_STORE":
                handle_store(parts, store_rows, job_id, kernel)
            elif token == "CSV_ENERGY":
                handle_energy(parts, energy_rows, summary_rows, summary_by_route,
                              context, job_id, kernel, energy_filter_stats)

    n_cpu, comp_wmma = derivar_columnas(drift_rows, summary_rows, energy_rows)
    return (drift_rows, summary_rows, horizon_rows, store_rows, energy_rows,
            energy_filter_stats, tokens_por_linea, n_cpu, comp_wmma)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def report_energy_filter(stats):
    descartadas = stats["ventana_corta"] + stats["gpu_invalido"]
    print("[extract_csv] energia GPU: %d filas de ruta GPU, %d aceptadas para "
          "promediar, %d descartadas (ventana_corta=%d, gpu_invalido=%d)."
          % (stats["filas_gpu"], stats["aceptadas"], descartadas,
             stats["ventana_corta"], stats["gpu_invalido"]))
    if descartadas > 0:
        print("[extract_csv] descartar = energy_gpu_j_per_iter -> NaN. "
              "energy_gpu_j y energy_window_reliable quedan crudos en el CSV "
              "para auditar el descarte.")


def verificar_campana(args, summary_rows, tokens_por_linea, n_cpu, comp_wmma):
    """Chequeos duros de la campana corregida (DECISIONS.md S1, S5-S7). Cada uno
    es OPT-IN por flag, para que los logs historicos (7145 y anteriores) se
    sigan pudiendo extraer sin abortar. Devuelve la lista de fallos."""
    fallos = []

    esperado = {"CSV_DRIFT": DRIFT_TOKENS_FULL, "CSV_ENERGY": ENERGY_TOKENS_FULL,
                "CSV_SUMMARY": SUMMARY_TOKENS_FULL}
    for token, completo in esperado.items():
        for n_tokens, n_lineas in sorted(tokens_por_linea[token].items()):
            print("[extract_csv] esquema %s: %d lineas con %d campos%s"
                  % (token, n_lineas, n_tokens, "" if n_tokens == completo else " (!= %d)" % completo))
            if args.strict_schema and n_tokens != completo:
                fallos.append("esquema %s: %d lineas con %d campos, se esperaban %d"
                              % (token, n_lineas, n_tokens, completo))

    if args.expect_comp_scheme and comp_wmma - {args.expect_comp_scheme}:
        fallos.append("comp_scheme efectivo en las rutas WMMA = %s, se pidio %s"
                      % (sorted(comp_wmma), args.expect_comp_scheme))

    if args.expect_op_mode:
        vistos = {r["op_mode"] for r in summary_rows if r["op_mode"] != "NaN"}
        if vistos != {args.expect_op_mode}:
            fallos.append("op_mode efectivo = %s, se pidio %s" % (sorted(vistos), args.expect_op_mode))

    if args.expect_alpha is not None:
        vistos = set()
        for r in summary_rows:
            if r["alpha"] not in ("NaN", "NA"):
                vistos.add(float(r["alpha"]))
        if not vistos or any(abs(a - args.expect_alpha) > 1e-9 for a in vistos):
            fallos.append("alpha efectivo = %s, se pidio %s (o ninguna fila lo trae)"
                          % (sorted(vistos), args.expect_alpha))

    if n_cpu:
        peor = max(n_cpu.values())
        print("[extract_csv] CPU_FP64: %d invocaciones en %d celdas (nx,ny,iters); "
              "maximo por celda = %d (razon max = %d)." % (sum(n_cpu.values()), len(n_cpu), peor, peor))
        if args.max_cpu_fp64_per_cell and peor > args.max_cpu_fp64_per_cell:
            fallos.append("CPU_FP64 corre %d veces en una misma celda (nx,ny,iters); el maximo "
                          "permitido es %d (la referencia se esta relanzando por ruta o por K)"
                          % (peor, args.max_cpu_fp64_per_cell))
    return fallos


def main():
    parser = argparse.ArgumentParser(description="Extract CSV_* tokens from stencil logs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--kernel", required=True)
    # Chequeos de la campana corregida (todos opcionales; ver verificar_campana).
    parser.add_argument("--expect-comp-scheme", choices=["none", "kahan_local", "spatial"],
                        help="abortar si las rutas WMMA no emitieron este esquema de compensacion")
    parser.add_argument("--expect-op-mode", choices=["stress", "diffusive"])
    parser.add_argument("--expect-alpha", type=float,
                        help="abortar si el alpha emitido por el binario difiere (p. ej. 0.1875)")
    parser.add_argument("--max-cpu-fp64-per-cell", type=int, default=0,
                        help="abortar si CPU_FP64 corre mas veces que esto en una celda (nx,ny,iters)")
    parser.add_argument("--strict-schema", action="store_true",
                        help="abortar si alguna linea CSV_* no trae el numero completo de campos")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    (drift_rows, summary_rows, horizon_rows, store_rows, energy_rows,
     energy_filter_stats, tokens_por_linea, n_cpu, comp_wmma) = read_log(
        args.input, args.job_id, args.kernel)

    outputs = [
        (os.path.join(args.outdir, "drift_%s_%s.csv" % (args.kernel, args.job_id)), DRIFT_HEADER, drift_rows),
        (os.path.join(args.outdir, "summary_%s_%s.csv" % (args.kernel, args.job_id)), SUMMARY_HEADER, summary_rows),
        (os.path.join(args.outdir, "horizon_%s_%s.csv" % (args.kernel, args.job_id)), HORIZON_HEADER, horizon_rows),
        (os.path.join(args.outdir, "store_%s_%s.csv" % (args.kernel, args.job_id)), STORE_HEADER, store_rows),
        (os.path.join(args.outdir, "energy_%s_%s.csv" % (args.kernel, args.job_id)), ENERGY_HEADER, energy_rows),
    ]
    for path, header, rows in outputs:
        write_csv(path, header, rows)

    report_energy_filter(energy_filter_stats)

    # Los CSV se escriben ANTES de abortar: si un chequeo falla, los datos siguen
    # disponibles para auditarlos, y el fallo queda como exit code != 0.
    fallos = verificar_campana(args, summary_rows, tokens_por_linea, n_cpu, comp_wmma)
    for fallo in fallos:
        print("[extract_csv] FALLO: " + fallo, file=sys.stderr)
    if fallos:
        sys.exit(3)


if __name__ == "__main__":
    main()
