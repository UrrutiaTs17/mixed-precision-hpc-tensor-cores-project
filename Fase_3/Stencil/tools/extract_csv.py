#!/usr/bin/env python3
import argparse
import csv
import os
import re


DRIFT_HEADER = [
    "job_id", "kernel", "nx", "ny", "iters", "kahan", "route",
    "iter", "ref_l2", "abs_l2", "rel_l2", "max_abs",
]

SUMMARY_HEADER = [
    "job_id", "kernel", "nx", "ny", "iters", "kahan", "route",
    "t_iter_ms", "t_total_ms", "gflops", "speedup_cpu", "speedup_fp32",
    "t_kernel_ms", "t_convert_ms", "t_checkpoint_ms", "rel_l2",
    "rel_linf", "max_abs", "rel_l2_prop", "rel_linf_prop",
    "first_nonfinite", "store_rel_norm", "store_rel_max_guarded",
    "store_excluded_count", "store_eval_iter", "energy_gpu_j", "energy_cpu_j",
    "energy_total_j", "edp_j_s", "joules_per_gflop", "onset_checkpoint",
]

SUMMARY_VALUE_FIELDS = SUMMARY_HEADER[7:-1]

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
]

RUN_RE = re.compile(
    r"^(?:Corrida|Barrido de horizonte): NX=(\d+), NY=(\d+), "
    r"ITERS=(\d+), TC=[^,]+, KAHAN=(off|on)"
)
DIM_RE = re.compile(r"^Dimensiones \(nx, ny\)\s*:\s*(\d+),\s*(\d+)")
ITERS_RE = re.compile(r"^Iteraciones\s*:\s*(\d+)")
KAHAN_RE = re.compile(r"^Kahan \(residuo almacen\.\)\s*:\s*(off|on)")


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
        route_or_format_key: route_or_format,
    }


def reset_run(context, summary_by_route):
    summary_by_route.clear()
    context["nx"] = "NaN"
    context["ny"] = "NaN"
    context["iters"] = "NaN"
    context["kahan"] = "NaN"


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
    })
    rows.append(row)


def handle_summary(parts, rows, summary_by_route, context, job_id, kernel):
    parts = pad(parts, 29)
    route = clean(parts[1])
    context["nx"] = clean(parts[2])
    context["ny"] = clean(parts[3])
    context["iters"] = clean(parts[4])
    context["kahan"] = clean(parts[5])
    row = ensure_summary_row(rows, summary_by_route, context, job_id, kernel, route)
    row.update(identity(context, job_id, kernel, "route", route))
    for field, value in zip(SUMMARY_VALUE_FIELDS, parts[6:29]):
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


def handle_energy(parts, rows, summary_rows, summary_by_route, context, job_id, kernel):
    parts = pad(parts, 13)
    route = clean(parts[1])
    row = identity(context, job_id, kernel, "route", route)
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
    })
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
    context = {"nx": "NaN", "ny": "NaN", "iters": "NaN", "kahan": "NaN"}
    summary_by_route = {}
    drift_rows = []
    summary_rows = []
    horizon_rows = []
    store_rows = []
    energy_rows = []

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
                              context, job_id, kernel)

    return drift_rows, summary_rows, horizon_rows, store_rows, energy_rows


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Extract CSV_* tokens from stencil logs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--kernel", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    drift_rows, summary_rows, horizon_rows, store_rows, energy_rows = read_log(
        args.input, args.job_id, args.kernel
    )

    outputs = [
        (os.path.join(args.outdir, "drift_%s_%s.csv" % (args.kernel, args.job_id)), DRIFT_HEADER, drift_rows),
        (os.path.join(args.outdir, "summary_%s_%s.csv" % (args.kernel, args.job_id)), SUMMARY_HEADER, summary_rows),
        (os.path.join(args.outdir, "horizon_%s_%s.csv" % (args.kernel, args.job_id)), HORIZON_HEADER, horizon_rows),
        (os.path.join(args.outdir, "store_%s_%s.csv" % (args.kernel, args.job_id)), STORE_HEADER, store_rows),
        (os.path.join(args.outdir, "energy_%s_%s.csv" % (args.kernel, args.job_id)), ENERGY_HEADER, energy_rows),
    ]
    for path, header, rows in outputs:
        write_csv(path, header, rows)


if __name__ == "__main__":
    main()
