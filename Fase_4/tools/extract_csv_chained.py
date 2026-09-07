#!/usr/bin/env python3
"""Post-proceso de CSV_DRIFT/CSV_SUMMARY para gemm_chained.cu y conv_chained.cu.

Esquema DISTINTO al de extract_csv.py (ese es especifico de Stencil -- ver su
propio comentario de cabecera y Fase_3/GEMM/README.md / Fase_3/Convolution/
README.md sobre por que no comparten script: columnas distintas, "n" vs "hw",
sin CSV_ONSET/HORIZON/STORE/ENERGY). Cada fila de CSV_DRIFT/CSV_SUMMARY que
emiten estos dos binarios ya trae "n"/"hw" e "iters" -- no hace falta
reconstruir contexto desde una linea de cabecera aparte (a diferencia de
Stencil, donde varias columnas de contexto NO viajan en cada fila CSV_*).

anchor_every: COLUMNA REAL, ya no reconstruida
----------------------------------------------
Desde que gemm_chained.cu/conv_chained.cu escriben anchor_every como ULTIMA
columna de CSV_DRIFT y CSV_SUMMARY, este script la lee directo de la fila.
Es una columna POR FILA, no de la corrida: la ruta "_none" reporta 0 y la
"_comp" reporta el K real, porque ambas corren en la MISMA invocacion del
binario (el ancla solo aplica a la ruta con compensacion; parse_args exige
--comp on para K>0). Los binarios de Fase 3, que no tienen ancla, emiten 0 en
las dos rutas -- mismo esquema de columnas, distinto contenido.

RESPALDO PARA LOGS VIEJOS: los logs generados ANTES de que la columna
existiera traen una columna menos. Para esos, el script sigue reconstruyendo
anchor_every desde la linea de configuracion que main() imprime al arrancar
cada corrida:

    N=1024 iters=20 comp=off checkpoint_every=0 anchor_every=5 (activa)
    HW=64 C=K=64 iters=20 comp=off checkpoint_every=0 anchor_every=5 (activa)

Esa ruta de respaldo se mantiene a proposito: borrarla convertiria en
inanalizable cualquier log anterior al cambio, y no cuesta nada conservarla.
"""
import argparse
import csv
import os
import re


DRIFT_HEADER = [
    "job_id", "kernel", "size", "format", "comp", "anchor_every",
    "route", "iter", "rel_l2", "rel_linf", "solution_finite",
]

SUMMARY_HEADER = [
    "job_id", "kernel", "size", "format", "comp", "anchor_every", "route",
    "iters", "t_iter_ms", "t_total_ms", "gflops", "energy_gpu_j",
    "window_reliable", "gpu_segments",
]

# GEMM: "N=1024 iters=20 comp=off checkpoint_every=0 anchor_every=5 (activa)"
GEMM_HEADER_RE = re.compile(
    r"^N=(\d+) iters=(\d+) comp=(off|on) checkpoint_every=(\d+) anchor_every=(\d+)"
)
# Convolucion: "HW=64 C=K=64 iters=20 comp=off checkpoint_every=0 anchor_every=5 (activa)"
CONV_HEADER_RE = re.compile(
    r"^HW=(\d+) C=K=\d+ iters=(\d+) comp=(off|on) checkpoint_every=(\d+) anchor_every=(\d+)"
)


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


def route_format(route):
    # route = "<FORMATO>_none" o "<FORMATO>_comp" (ver gemm_chained.cu /
    # conv_chained.cu, format_label + "_none"/"_comp"). rsplit por si algun
    # formato futuro trajera un "_" propio en el nombre.
    fmt, _, suffix = route.rpartition("_")
    return (fmt or route), suffix


def update_context_from_header(line, context):
    match = GEMM_HEADER_RE.match(line)
    if match:
        context["size"], context["iters"], context["comp"], \
            context["checkpoint_every"], context["anchor_every"] = match.groups()
        return
    match = CONV_HEADER_RE.match(line)
    if match:
        context["size"], context["iters"], context["comp"], \
            context["checkpoint_every"], context["anchor_every"] = match.groups()


# Indice (0-based, contando el token CSV_*) de la columna anchor_every en cada
# linea, y numero total de campos que trae una linea que SI la lleva.
DRIFT_ANCHOR_IDX = 7
DRIFT_FIELDS = 8
SUMMARY_ANCHOR_IDX = 10
SUMMARY_FIELDS = 11


def anchor_every_de_fila(parts, idx, context, suffix):
    """anchor_every de la fila, con respaldo para logs anteriores a la columna.

    Camino normal: la fila trae la columna y se lee tal cual -- incluido el 0
    de la ruta "_none", que el binario ya escribe explicitamente.

    Respaldo (logs viejos, sin la columna): se reconstruye desde la linea de
    configuracion, con la misma regla que usaba este script antes -- el ancla
    solo corre sobre la ruta "comp", asi que "none" es 0 sin importar con que
    --anchor-every se haya lanzado la corrida.
    """
    if len(parts) > idx:
        valor = clean(parts[idx])
        if valor != "NaN":
            return valor
    return context["anchor_every"] if suffix == "comp" else "0"


def handle_drift(parts, rows, context, job_id, kernel):
    crudas = parts  # sin rellenar: pad() taparia la ausencia de anchor_every
    parts = pad(parts, DRIFT_FIELDS)
    route = clean(parts[1])
    fmt, suffix = route_format(route)
    anchor = anchor_every_de_fila(crudas, DRIFT_ANCHOR_IDX, context, suffix)
    rows.append({
        "job_id": job_id,
        "kernel": kernel,
        "size": clean(parts[2]),
        "format": fmt,
        "comp": "on" if suffix == "comp" else "off",
        "anchor_every": anchor,
        "route": route,
        "iter": clean(parts[3]),
        "rel_l2": clean(parts[4]),
        "rel_linf": clean(parts[5]),
        "solution_finite": clean(parts[6]),
    })


def handle_summary(parts, rows, context, job_id, kernel):
    crudas = parts  # ver la nota en handle_drift
    parts = pad(parts, SUMMARY_FIELDS)
    route = clean(parts[1])
    fmt, suffix = route_format(route)
    anchor = anchor_every_de_fila(crudas, SUMMARY_ANCHOR_IDX, context, suffix)
    rows.append({
        "job_id": job_id,
        "kernel": kernel,
        "size": clean(parts[2]),
        "format": fmt,
        "comp": "on" if suffix == "comp" else "off",
        "anchor_every": anchor,
        "route": route,
        "iters": clean(parts[3]),
        "t_iter_ms": clean(parts[4]),
        "t_total_ms": clean(parts[5]),
        "gflops": clean(parts[6]),
        "energy_gpu_j": clean(parts[7]),
        "window_reliable": clean(parts[8]),
        "gpu_segments": clean(parts[9]),
    })


def read_log(path, job_id, kernel):
    context = {"size": "NaN", "iters": "NaN", "comp": "NaN",
               "checkpoint_every": "NaN", "anchor_every": "0"}
    drift_rows = []
    summary_rows = []

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            update_context_from_header(line, context)
            parts = parse_csv_line(line)
            if not parts:
                continue
            token = parts[0]
            if token == "CSV_DRIFT":
                handle_drift(parts, drift_rows, context, job_id, kernel)
            elif token == "CSV_SUMMARY":
                handle_summary(parts, summary_rows, context, job_id, kernel)

    return drift_rows, summary_rows


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Extrae CSV_DRIFT/CSV_SUMMARY de logs de gemm_chained/conv_chained.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--kernel", required=True, choices=["gemm", "conv"])
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    drift_rows, summary_rows = read_log(args.input, args.job_id, args.kernel)

    outputs = [
        (os.path.join(args.outdir, "drift_%s_%s.csv" % (args.kernel, args.job_id)), DRIFT_HEADER, drift_rows),
        (os.path.join(args.outdir, "summary_%s_%s.csv" % (args.kernel, args.job_id)), SUMMARY_HEADER, summary_rows),
    ]
    for path, header, rows in outputs:
        write_csv(path, header, rows)
        print("[extract_csv_chained] %s: %d filas" % (path, len(rows)))


if __name__ == "__main__":
    main()
