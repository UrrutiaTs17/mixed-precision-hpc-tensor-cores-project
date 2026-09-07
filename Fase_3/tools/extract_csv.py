#!/usr/bin/env python3
import argparse
import csv
import os
import re


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
]

SUMMARY_HEADER = [
    "job_id", "kernel", "nx", "ny", "iters", "kahan", "route",
    "t_iter_ms", "t_total_ms", "gflops", "speedup_cpu", "speedup_fp32",
    "t_kernel_ms", "t_convert_ms", "t_checkpoint_ms", "rel_l2",
    "rel_linf", "max_abs", "rel_l2_prop", "rel_linf_prop",
    "first_nonfinite", "store_rel_norm", "store_rel_max_guarded",
    "store_excluded_count", "store_eval_iter", "energy_gpu_j", "energy_cpu_j",
    "energy_total_j", "edp_j_s", "joules_per_gflop", "onset_checkpoint",
    "anchor_every",
]

# -2, no -1: excluye tanto onset_checkpoint (se llena aparte via
# handle_onset) como el nuevo anchor_every (se llena aparte via identity(),
# desde el contexto -- ninguno de los dos viene posicionalmente en parts[]).
SUMMARY_VALUE_FIELDS = SUMMARY_HEADER[7:-2]

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


def clean(value):
    value = value.strip()
    upper = value.upper()
    if value == "" or upper in {"NONFINITE", "NA", "NAN", "NO_EVALUABLE", "INF", "+INF", "-INF"}:
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
    row = {key: "NaN" for key in SUMMARY_HEADER}
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


def handle_drift(parts, rows, summary_rows, summary_by_route, context, job_id, kernel):
    if len(parts) < 7:
        return
    route = clean(parts[1])
    ensure_summary_row(summary_rows, summary_by_route, context, job_id, kernel, route)
    row = identity(context, job_id, kernel, "route", route)
    row.update({
        "iter": clean(parts[2]),
        "ref_l2": clean(parts[3]),
        "abs_l2": clean(parts[4]),
        "rel_l2": clean(parts[5]),
        "max_abs": clean(parts[6]),
        # La columna real de la fila gana sobre el contexto de identity().
        "anchor_every": anchor_de_fila(parts, DRIFT_ANCHOR_IDX, context),
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


def read_log(path, job_id, kernel):
    context = {"nx": "NaN", "ny": "NaN", "iters": "NaN", "kahan": "NaN", "anchor_every": "0"}
    summary_by_route = {}
    drift_rows = []
    summary_rows = []
    horizon_rows = []
    store_rows = []
    energy_rows = []
    energy_filter_stats = new_energy_filter_stats()

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            update_context_from_run_header(line, context, summary_by_route)
            parts = parse_csv_line(line)
            if not parts:
                continue
            token = parts[0]
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

    return (drift_rows, summary_rows, horizon_rows, store_rows, energy_rows,
            energy_filter_stats)


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


def main():
    parser = argparse.ArgumentParser(description="Extract CSV_* tokens from stencil logs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--kernel", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    (drift_rows, summary_rows, horizon_rows, store_rows, energy_rows,
     energy_filter_stats) = read_log(args.input, args.job_id, args.kernel)

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


if __name__ == "__main__":
    main()
