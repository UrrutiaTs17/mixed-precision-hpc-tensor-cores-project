#!/usr/bin/env python3
"""Comparador de regresion del GATE 1 (Fase 4).

Contrasta la salida del binario ANTERIOR a la parametrizacion del operador
contra la del binario NUEVO corrido con los defaults (--op-mode stress,
--ci-mode legacy), y decide si la refactorizacion preservo el comportamiento.

Por que no basta un `diff` literal
----------------------------------
El stdout del binario mezcla dos clases de contenido:

  * DETERMINISTA: rel_l2, rel_linf, max_abs, rel_l2_prop, store_*, n_star y
    todas las filas CSV_DRIFT / CSV_HORIZON / CSV_ONSET. Para una misma malla,
    una misma CI y un mismo operador, esto sale bit a bit igual en cada
    corrida. Es lo que el GATE 1 tiene que blindar: si cambia, la
    parametrizacion altero la aritmetica (tipicamente por una contraccion FMA
    distinta, ver el comentario de stencil2d_fp32_kernel).

  * NO DETERMINISTA: t_iter_ms, t_total_ms, gflops, speedups, t_kernel_ms,
    t_convert_ms, t_checkpoint_ms y las cinco columnas de energia. Dos
    ejecuciones del MISMO binario ya difieren aqui por ruido de medicion
    (reloj, estado termico, contador NVML). Exigirles igualdad byte a byte
    haria fallar el gate siempre, por una razon que no tiene nada que ver con
    el cambio.

Asi que el veredicto se decide solo sobre la clase determinista, y la clase de
medicion se reporta aparte como desviacion relativa, para que quede a la vista
que es ruido y no un cambio de regimen.

Uso:
    comparar_gate1.py --base BASE.log --nuevo NUEVO.log [--etiqueta TXT]
                      [--tolerancia-medicion 0.15]
Salida: informe por stdout; codigo 0 si el gate pasa, 1 si no.
"""
import argparse
import csv
import math
import sys

# Indice (0-based, contando el token CSV_SUMMARY) -> nombre, para las 29
# columnas que ya existian antes de Fase 4. Las 7 nuevas (29..35) se listan
# aparte: no tienen contraparte en el log base y por definicion no pueden
# compararse.
SUMMARY_LEGACY = [
    "token", "route", "nx", "ny", "iters", "kahan",
    "t_iter_ms", "t_total_ms", "gflops", "speedup_cpu", "speedup_fp32",
    "t_kernel_ms", "t_convert_ms", "t_checkpoint_ms",
    "rel_l2", "rel_linf", "max_abs", "rel_l2_prop", "rel_linf_prop",
    "first_nonfinite", "store_rel_norm", "store_rel_max_guarded",
    "store_excluded_count", "store_eval_iter",
    "energy_gpu_j", "energy_cpu_j", "energy_total_j", "edp_j_s",
    "joules_per_gflop",
]
SUMMARY_NUEVAS = [
    "op_mode", "alpha", "ci_mode", "ci_p",
    "cell_updates_per_s", "energy_per_cell_update_j", "reference_role",
    "execution_mode",
]

# Columnas de medicion dentro de SUMMARY_LEGACY: se comparan con tolerancia
# relativa y NO deciden el veredicto.
MEDICION = {
    "t_iter_ms", "t_total_ms", "gflops", "speedup_cpu", "speedup_fp32",
    "t_kernel_ms", "t_convert_ms", "t_checkpoint_ms",
    "energy_gpu_j", "energy_cpu_j", "energy_total_j", "edp_j_s",
    "joules_per_gflop",
}
# Columnas de identidad: no son un resultado, son la clave de la fila.
IDENTIDAD = {"token", "route", "nx", "ny", "iters", "kahan"}

# Esquemas de las demas lineas CSV_*. Todas sus columnas son deterministas
# salvo las de CSV_ENERGY, que es integramente telemetria.
ESQUEMAS = {
    "CSV_DRIFT":   (["token", "route", "iter", "ref_l2", "abs_l2", "rel_l2", "max_abs"],
                    ("route", "iter")),
    "CSV_STORE":   (["token", "route", "nx", "ny", "iters", "kahan",
                     "store_rel_norm", "store_rel_max_guarded",
                     "store_excluded_count", "store_eval_iter", "ulp_formato"],
                    ("route",)),
    "CSV_HORIZON": (["token", "format", "nx", "ny", "iters", "kahan",
                     "h_predicho", "h_medido", "lambda", "r2", "n_puntos_fit",
                     "semilla_A", "nyquist_u0", "piso_siembra", "fit_status"],
                    ("format",)),
    "CSV_ONSET":   (["token", "route", "onset"], ("route",)),
}


def leer_lineas_csv(path):
    """Agrupa las lineas CSV_* del log por tipo. CSV_REGION se descarta: son
    timestamps de pared, no un resultado."""
    por_tipo = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for linea in fh:
            linea = linea.rstrip("\n")
            if not linea.startswith("CSV_"):
                continue
            token = linea.split(",", 1)[0]
            if token == "CSV_REGION":
                continue
            try:
                campos = next(csv.reader([linea]))
            except csv.Error:
                continue
            por_tipo.setdefault(token, []).append(campos)
    return por_tipo


def indexar(filas, nombres, clave_cols):
    """Indexa filas por su clave de identidad, tolerando duplicados."""
    idx = {}
    for campos in filas:
        d = dict(zip(nombres, campos))
        clave = tuple(d.get(c, "") for c in clave_cols)
        idx.setdefault(clave, []).append(d)
    return idx


def desviacion_relativa(a, b):
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(fa) and math.isfinite(fb)):
        return None
    if fa == 0.0 and fb == 0.0:
        return 0.0
    denom = max(abs(fa), abs(fb))
    return abs(fa - fb) / denom if denom else 0.0


def comparar_bloque(nombre, base_filas, nuevo_filas, nombres, clave_cols,
                    medicion_cols, tolerancia, informe):
    base_idx = indexar(base_filas, nombres, clave_cols)
    nuevo_idx = indexar(nuevo_filas, nombres, clave_cols)

    fallos, avisos, medidas = [], [], []

    solo_base = sorted(set(base_idx) - set(nuevo_idx))
    solo_nuevo = sorted(set(nuevo_idx) - set(base_idx))
    for clave in solo_base:
        fallos.append("%s %s: la fila existe en BASE y falta en NUEVO" % (nombre, clave))
    for clave in solo_nuevo:
        fallos.append("%s %s: la fila existe en NUEVO y falta en BASE" % (nombre, clave))

    for clave in sorted(set(base_idx) & set(nuevo_idx)):
        bs, ns = base_idx[clave], nuevo_idx[clave]
        if len(bs) != len(ns):
            fallos.append("%s %s: %d filas en BASE vs %d en NUEVO"
                          % (nombre, clave, len(bs), len(ns)))
            continue
        for b, n in zip(bs, ns):
            for col in nombres:
                if col in IDENTIDAD:
                    continue
                vb, vn = b.get(col, "<falta>"), n.get(col, "<falta>")
                if col in medicion_cols:
                    d = desviacion_relativa(vb, vn)
                    if d is None:
                        if vb != vn:
                            avisos.append("%s %s %s: %s -> %s (no numerico)"
                                          % (nombre, clave, col, vb, vn))
                    else:
                        medidas.append((d, nombre, clave, col, vb, vn))
                        if d > tolerancia:
                            avisos.append("%s %s %s: %s -> %s (%.1f %% de desviacion)"
                                          % (nombre, clave, col, vb, vn, 100.0 * d))
                elif vb != vn:
                    fallos.append("%s %s %s: BASE=%s  NUEVO=%s"
                                  % (nombre, clave, col, vb, vn))
    informe["fallos"].extend(fallos)
    informe["avisos"].extend(avisos)
    informe["medidas"].extend(medidas)
    informe["comparadas"][nombre] = len(set(base_idx) & set(nuevo_idx))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="log del binario anterior")
    ap.add_argument("--nuevo", required=True, help="log del binario nuevo con defaults")
    ap.add_argument("--etiqueta", default="", help="nombre de la configuracion comparada")
    ap.add_argument("--tolerancia-medicion", type=float, default=0.15,
                    help="desviacion relativa admitida en columnas de tiempo/energia")
    args = ap.parse_args()

    base = leer_lineas_csv(args.base)
    nuevo = leer_lineas_csv(args.nuevo)

    informe = {"fallos": [], "avisos": [], "medidas": [], "comparadas": {}}

    titulo = "GATE 1 :: %s" % (args.etiqueta or "(sin etiqueta)")
    print("=" * 72)
    print(titulo)
    print("  BASE  : %s" % args.base)
    print("  NUEVO : %s" % args.nuevo)
    print("=" * 72)

    if "CSV_SUMMARY" in base or "CSV_SUMMARY" in nuevo:
        comparar_bloque("CSV_SUMMARY",
                        base.get("CSV_SUMMARY", []), nuevo.get("CSV_SUMMARY", []),
                        SUMMARY_LEGACY, ("route",), MEDICION,
                        args.tolerancia_medicion, informe)
    for token, (nombres, clave) in ESQUEMAS.items():
        if token in base or token in nuevo:
            medicion = set() if token != "CSV_ENERGY" else set(nombres)
            comparar_bloque(token, base.get(token, []), nuevo.get(token, []),
                            nombres, clave, medicion,
                            args.tolerancia_medicion, informe)

    print("\n-- Filas comparadas por bloque --")
    for nombre, n in sorted(informe["comparadas"].items()):
        print("   %-14s %d" % (nombre, n))

    print("\n-- Columnas NUEVAS presentes solo en el binario nuevo (no comparables) --")
    filas_nuevas = nuevo.get("CSV_SUMMARY", [])
    if filas_nuevas:
        anchura = len(filas_nuevas[0])
        print("   CSV_SUMMARY trae %d campos (esperado 36: 29 previos + 7 nuevos)" % anchura)
        for fila in filas_nuevas:
            extra = fila[len(SUMMARY_LEGACY):]
            if extra:
                print("   %-14s %s" % (fila[1], dict(zip(SUMMARY_NUEVAS, extra))))
    else:
        print("   (ninguna: el log nuevo no trae filas CSV_SUMMARY)")

    if informe["medidas"]:
        peores = sorted(informe["medidas"], reverse=True)[:10]
        print("\n-- Mayores desviaciones en columnas de MEDICION (tiempo/energia) --")
        print("   Tolerancia: %.1f %%. Estas columnas NO deciden el veredicto." %
              (100.0 * args.tolerancia_medicion))
        for d, nombre, clave, col, vb, vn in peores:
            print("   %6.2f %%  %s %s %s: %s -> %s" % (100.0 * d, nombre, clave, col, vb, vn))

    if informe["avisos"]:
        print("\n-- AVISOS (medicion fuera de tolerancia; revisar, no bloquea) --")
        for a in informe["avisos"]:
            print("   ! " + a)

    print("\n-- Veredicto sobre las columnas DETERMINISTAS --")
    if informe["fallos"]:
        print("   FALLA: %d diferencia(s) en contenido que deberia ser identico.\n"
              % len(informe["fallos"]))
        for f in informe["fallos"]:
            print("   X " + f)
        print("\n   El GATE 1 NO pasa: la parametrizacion del operador cambio la")
        print("   aritmetica del caso stress/legacy. Sospechoso principal: la")
        print("   contraccion FMA de los kernels (ver stencil2d_fp32_kernel).")
        return 1

    total = sum(informe["comparadas"].values())
    print("   PASA: %d fila(s) comparadas, CERO diferencias deterministas." % total)
    print("   Las unicas diferencias son las 7 columnas nuevas y el ruido de medicion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
