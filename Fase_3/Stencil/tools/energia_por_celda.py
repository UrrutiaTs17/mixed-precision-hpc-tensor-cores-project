#!/usr/bin/env python3
"""Energia normalizada por actualizacion de celda + Cv entre replicas.

Porta al post-proceso de Fase 3 la metrica energy_per_cell_update_j que Fase 4
emite desde el binario (ver energy_per_cell_update_j en
Fase_3/Stencil/stencil_tensor_activation.cu, rama de Fase 4). Aqui NO se toca el
.cu: los tres factores ya estan en el CSV de energia, asi que la columna es
derivable y el fuente congelado conserva su sha256.

    energy_per_cell_update_j = energy_total_j / ((nx-2) * (ny-2) * iters)

El (nx-2)*(ny-2) es interior_cells() del .cu: el borde no se actualiza. La
normalizacion es lo que hace comparables corridas con distinta malla o distinto
numero de iteraciones -- la energia absoluta no lo es.

Solo se agregan filas con energy_window_reliable=1. Las demas se cuentan y se
reportan aparte en vez de descartarse en silencio: en los jobs 6325-6333 eran
el 100% de las filas, y ese recuento es justamente el diagnostico.

Uso:
    energia_por_celda.py energy_stencil_*.csv
    energia_por_celda.py --csv salida.csv  jobs/f_energia/*/results/energy_*.csv
"""

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict

# Orden de ENERGY_HEADER en tools/extract_csv.py. Se leen por nombre (DictReader),
# pero se declara aqui para fallar con un mensaje claro si el esquema cambia.
COLUMNAS_REQUERIDAS = (
    "job_id", "nx", "ny", "iters", "kahan", "route",
    "energy_gpu_j", "energy_total_j", "energy_window_reliable",
)


def a_float(texto):
    """NaN, vacio y basura -> None. El CSV usa el literal 'NaN'."""
    if texto is None:
        return None
    texto = texto.strip()
    if not texto or texto == "NaN":
        return None
    try:
        valor = float(texto)
    except ValueError:
        return None
    return valor if math.isfinite(valor) else None


def celdas_interiores(nx, ny):
    return float(nx - 2) * float(ny - 2)


def leer_filas(rutas):
    """Devuelve (filas_fiables, n_no_fiables, n_totales)."""
    filas, no_fiables, totales = [], 0, 0
    for ruta in rutas:
        # newline="" + strip por los CSV con CRLF que dejaron algunas corridas.
        with open(ruta, newline="", encoding="utf-8") as fh:
            lector = csv.DictReader(fh)
            faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in (lector.fieldnames or [])]
            if faltantes:
                sys.exit(f"ERROR: {ruta} no trae las columnas {faltantes}. "
                         "Esquema de extract_csv.py incompatible.")
            for cruda in lector:
                totales += 1
                fila = {k: (v.strip() if isinstance(v, str) else v) for k, v in cruda.items()}
                if fila["energy_window_reliable"] != "1":
                    no_fiables += 1
                    continue
                nx, ny = int(fila["nx"]), int(fila["ny"])
                iters = int(fila["iters"])
                total_j = a_float(fila["energy_total_j"])
                gpu_j = a_float(fila["energy_gpu_j"])
                if iters <= 0 or gpu_j is None:
                    no_fiables += 1
                    continue
                celdas = celdas_interiores(nx, ny) * iters
                filas.append({
                    "archivo": ruta,
                    "job_id": fila["job_id"],
                    "nx": nx, "ny": ny, "iters": iters,
                    "kahan": fila["kahan"], "route": fila["route"],
                    "energy_gpu_j": gpu_j,
                    "energy_total_j": total_j,
                    "energy_gpu_j_per_iter": gpu_j / iters,
                    # total_j puede venir NaN aunque gpu_j sea valido (lectura
                    # RAPL perdida): la columna sale vacia en vez de inventarse.
                    "energy_per_cell_update_j": (total_j / celdas) if total_j is not None else None,
                })
    return filas, no_fiables, totales


def cv_porcentual(valores):
    """Coeficiente de variacion en %. Requiere n>=2 (stdev muestral)."""
    if len(valores) < 2:
        return None
    media = statistics.fmean(valores)
    if media == 0.0:
        return None
    return statistics.stdev(valores) / abs(media) * 100.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entradas", nargs="+", metavar="ENTRADA",
                    help="ficheros energy_stencil_*.csv")
    ap.add_argument("--csv", metavar="SALIDA",
                    help="escribe las filas fiables con la columna derivada")
    args = ap.parse_args()

    filas, no_fiables, totales = leer_filas(args.entradas)

    print(f"Filas leidas          : {totales}")
    print(f"  con ventana fiable  : {len(filas)}")
    print(f"  descartadas         : {no_fiables}"
          "   (energy_window_reliable != 1, o iters/energia invalidos)")
    if not filas:
        print("\nNinguna fila con ventana fiable: nada que agregar.")
        print("Relanzar con RUN_KIND=energy (CHECKPOINT_EVERY=0) e ITERS suficiente")
        print("para superar el umbral de 0.5 s de ventana GPU.")
        return 0

    if args.csv:
        campos = ["job_id", "nx", "ny", "iters", "kahan", "route",
                  "energy_gpu_j", "energy_total_j",
                  "energy_gpu_j_per_iter", "energy_per_cell_update_j"]
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            escritor = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
            escritor.writeheader()
            for fila in filas:
                escritor.writerow(fila)
        print(f"\nCSV derivado escrito  : {args.csv}")

    # Agregacion por (ruta, malla, iters, kahan): son las corridas que se
    # supone identicas, asi que su dispersion ES la variabilidad del sistema.
    grupos = defaultdict(list)
    for fila in filas:
        grupos[(fila["route"], fila["nx"], fila["iters"], fila["kahan"])].append(fila)

    print(f"\n{'ruta':<14} {'nx':>6} {'iters':>6} {'kahan':>6} {'n':>3} "
          f"{'J/iter medio':>14} {'Cv %':>8} {'J/celda-upd':>14}")
    print("-" * 78)
    for clave in sorted(grupos):
        ruta, nx, iters, kahan = clave
        grupo = grupos[clave]
        por_iter = [f["energy_gpu_j_per_iter"] for f in grupo]
        por_celda = [f["energy_per_cell_update_j"] for f in grupo
                     if f["energy_per_cell_update_j"] is not None]
        cv = cv_porcentual(por_iter)
        txt_cv = f"{cv:8.2f}" if cv is not None else "     n/a"
        txt_celda = f"{statistics.fmean(por_celda):14.6e}" if por_celda else "           n/a"
        print(f"{ruta:<14} {nx:>6} {iters:>6} {kahan:>6} {len(grupo):>3} "
              f"{statistics.fmean(por_iter):14.6e} {txt_cv} {txt_celda}")

    if all(len(g) < 2 for g in grupos.values()):
        print("\nAviso: ningun grupo tiene 2+ corridas, no hay Cv que calcular.")
        print("Cv exige replicas: lanzar con --sub-f-energia (REPLICAS>=3).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
