#!/usr/bin/env python3
"""Resume los CSV de Nsight Compute en una tabla comparativa.

Se ejecuta APARTE del sbatch, sobre los CSV ya generados, para que un fallo de
parseo no obligue a repetir el perfilado (que es lo que paso en el job 6730).

Acepta los DOS formatos de ncu --csv, porque dependen de --page y de la version:

  largo  (--page details): columnas "Metric Name" / "Metric Value", una fila
                           por metrica.
  ancho  (--page raw):     una COLUMNA por metrica, una fila por lanzamiento,
                           a veces con una segunda fila de unidades.

Uso:
    python3 resumir_ncu.py <directorio_con_csv>
    python3 resumir_ncu.py <directorio_con_csv> --debug    # vuelca cabeceras
"""
import csv
import glob
import os
import sys

FILAS = [
    ("DRAM % del pico",        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "{:.1f}", None),
    ("SM % del pico",          "sm__throughput.avg.pct_of_peak_sustained_elapsed",       "{:.1f}", None),
    ("ocupancia alcanzada %",  "sm__warps_active.avg.pct_of_peak_sustained_active",      "{:.1f}", None),
    ("issue activo %",         "smsp__issue_active.avg.pct_of_peak_sustained_active",    "{:.1f}", None),
    ("registros/hilo",         "launch__registers_per_thread",                           "{:.0f}", None),
    ("DRAM leidos (MB)",       "dram__bytes_read.sum",                                   "{:.1f}", 1e6),
    ("DRAM escritos (MB)",     "dram__bytes_write.sum",                                  "{:.1f}", 1e6),
    ("sectores/request ld",    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio", "{:.2f}", None),
    ("sectores/request st",    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio", "{:.2f}", None),
    ("bytes/sector ld %",      "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct", "{:.1f}", None),
    ("bytes/sector st %",      "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct", "{:.1f}", None),
    ("acierto L1 %",           "l1tex__t_sector_hit_rate.pct",                           "{:.1f}", None),
    ("--- stalls por issue activo ---", None, None, None),
    ("long_scoreboard",        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", "{:.2f}", None),
    ("short_scoreboard",       "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio", "{:.2f}", None),
    ("wait",                   "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",            "{:.2f}", None),
    ("barrier",                "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",         "{:.2f}", None),
    ("mio_throttle",           "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",    "{:.2f}", None),
    ("lg_throttle",            "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",     "{:.2f}", None),
    ("math_pipe_throttle",     "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio", "{:.2f}", None),
    ("not_selected",           "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",    "{:.2f}", None),
    ("--- shared y ocupancia ---", None, None, None),
    ("conflictos banco ld",    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", "{:.0f}", None),
    ("conflictos banco st",    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", "{:.0f}", None),
    ("limite ocup. shared",    "launch__occupancy_limit_shared_mem",                     "{:.0f}", None),
    ("limite ocup. registros", "launch__occupancy_limit_registers",                      "{:.0f}", None),
    ("limite ocup. warps",     "launch__occupancy_limit_warps",                          "{:.0f}", None),
    ("limite ocup. bloques",   "launch__occupancy_limit_blocks",                         "{:.0f}", None),
    ("--- Tensor Cores ---", None, None, None),
    ("HMMA emitidas",          "sm__inst_executed_pipe_tensor_op_hmma.sum",              "{:.0f}", None),
    ("tensor % del pico",      "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", "{:.2f}", None),
]

ORDEN = ["FP32_ref", "WMMA_FP16_off", "WMMA_BF16_off", "WMMA_FP16_local", "WMMA_FP16_spatial"]


def a_numero(bruto):
    if bruto is None:
        return None
    t = str(bruto).strip().replace('"', "").replace(",", "")
    if not t or t in ("n/a", "N/A", "-"):
        return None
    try:
        return float(t)
    except ValueError:
        return t


def leer_csv(ruta, debug=False):
    """Devuelve {metrica: valor}. Tolera ambos formatos y lineas de aviso."""
    with open(ruta, newline="", errors="replace") as fh:
        lineas = [l for l in fh if l.strip()]
    if not lineas:
        return {}, "vacio"

    # La cabecera real es la primera linea que menciona una columna conocida de
    # ncu. Antes puede haber avisos ("==WARNING==", texto suelto).
    inicio = None
    for i, l in enumerate(lineas):
        if "Kernel Name" in l or "Metric Name" in l or l.lstrip().startswith('"ID"'):
            inicio = i
            break
    if inicio is None:
        return {}, "sin cabecera reconocible"

    filas = list(csv.DictReader(lineas[inicio:]))
    if not filas:
        return {}, "cabecera sin filas"
    campos = [c for c in (filas[0].keys()) if c]
    if debug:
        print(f"    [debug] {os.path.basename(ruta)}: {len(campos)} columnas, "
              f"{len(filas)} filas; primeras: {campos[:6]}")

    # ---- formato largo -----------------------------------------------------
    if any(c.strip() == "Metric Name" for c in campos):
        salida = {}
        for f in filas:
            nombre = (f.get("Metric Name") or "").strip()
            if nombre:
                salida[nombre] = a_numero(f.get("Metric Value"))
        return salida, "largo"

    # ---- formato ancho -----------------------------------------------------
    # Las columnas de metrica son las que llevan "__" en el nombre (convencion
    # de ncu) o empiezan por "launch__". Puede haber una fila de unidades justo
    # debajo de la cabecera: se descarta quedandose con la primera fila que
    # tenga algun valor numerico.
    cols_metrica = [c for c in campos if "__" in c]
    if not cols_metrica:
        return {}, "sin columnas de metrica"
    for f in filas:
        vals = {c.strip(): a_numero(f.get(c)) for c in cols_metrica}
        if any(isinstance(v, float) for v in vals.values()):
            return vals, "ancho"
    return {}, "ancho sin valores numericos"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    carpeta = sys.argv[1]
    debug = "--debug" in sys.argv

    datos, diag = {}, []
    for ruta in sorted(glob.glob(os.path.join(carpeta, "*.csv"))):
        etq = os.path.basename(ruta).rsplit("_", 1)[0]
        vals, forma = leer_csv(ruta, debug)
        diag.append((os.path.basename(ruta), forma, len(vals)))
        if vals:
            datos.setdefault(etq, {}).update(vals)

    print("Ficheros leidos:")
    for nombre, forma, n in diag:
        print(f"  {nombre:<34} formato={forma:<24} metricas={n}")
    print()

    if not datos:
        print("No se extrajo ninguna metrica. Relanza con --debug para ver las")
        print("cabeceras reales, o pega la primera linea de uno de los CSV.")
        return 1

    cols = [c for c in ORDEN if c in datos] + [c for c in sorted(datos) if c not in ORDEN]
    anchoi = max(len(f[0]) for f in FILAS) + 2
    print(" " * anchoi + "".join(f"{c:>20}" for c in cols))
    print("-" * (anchoi + 20 * len(cols)))
    for etiqueta, metrica, fmt, divisor in FILAS:
        if metrica is None:                       # separador de seccion
            print(etiqueta)
            continue
        celdas = []
        for c in cols:
            v = datos[c].get(metrica)
            if v is None:
                celdas.append(f"{'-':>20}")
            elif isinstance(v, float):
                celdas.append(f"{fmt.format(v / divisor if divisor else v):>20}")
            else:
                celdas.append(f"{str(v)[:19]:>20}")
        print(f"{etiqueta:<{anchoi}}" + "".join(celdas))
    return 0


if __name__ == "__main__":
    sys.exit(main())
