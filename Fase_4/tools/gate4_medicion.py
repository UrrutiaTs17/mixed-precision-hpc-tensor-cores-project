#!/usr/bin/env python3
"""GATE 4 -- la medicion de tiempo y energia distingue una ruta de otra.

QUE BUG VIGILA
--------------
Hasta la correccion de la "Tarea 9", gemm_chained.cu y conv_chained.cu media
las tres trayectorias de cada iteracion (referencia FP64 + WMMA sin
compensacion + WMMA con compensacion) con UN SOLO cronometro que las envolvia
a las tres, y publicaba ese mismo numero en las dos filas CSV_SUMMARY:

    CSV_SUMMARY,FP16_none,64,4,20.8294,83.3177,14.4982,NA,0,3,0
    CSV_SUMMARY,FP16_comp,64,4,20.8294,83.3177,14.4982,NA,0,3,2
                              ^^^^^^^ identico -- ese era el bug

La energia era peor: los dos PowerBuffer se abrian y cerraban en los MISMOS
instantes sobre nvmlDeviceGetTotalEnergyConsumption, que es un contador de
TODO EL DISPOSITIVO, asi que las dos columnas energy_gpu_j eran literalmente
el mismo numero.

Consecuencia: dos de los tres ejes del Frente de Pareto 3D (Etapa 9 del plan)
eran inutilizables para esos dos kernels -- no podian separar comp=off de
comp=on, y ademas ambos incluian el costo de la referencia FP64, que en A100
domina. El eje de error siempre estuvo bien.

Stencil nunca tuvo este problema: cronometra y mide energia por ruta.

QUE VERIFICA, Y POR QUE ESAS COMPROBACIONES
-------------------------------------------
El gate no comprueba "el tiempo es correcto" (no hay contra que contrastarlo
sin un segundo instrumento). Comprueba la propiedad que el bug rompia y que
el Pareto necesita: **que cada numero responda a la carga de SU propia ruta y
solo a ella**. Cuatro comprobaciones, todas falsables:

  A. SEPARACION ENTRE RUTAS (tiempo). En una corrida con --comp on, la ruta
     _comp hace DOS productos WMMA por iteracion (el termino principal y la
     correccion) donde _none hace uno. Su t_iter_ms tiene que ser
     estrictamente mayor. Antes del fix eran bit a bit identicos, que es la
     firma exacta del bug.

  B. SEPARACION ENTRE RUTAS (energia). energy_gpu_j de _none y de _comp no
     pueden ser el mismo numero. Si NVML no estuvo disponible (columna NaN)
     la comprobacion se omite con aviso: no se puede fallar por telemetria
     ausente.

  C. LA RUTA RESPONDE A SU PROPIA CARGA. Con --anchor-every 1 la ruta _comp
     ejecuta ademas un paso FP64 completo por iteracion; con --anchor-every 0
     ninguno. Su t_iter_ms tiene que subir. Es la comprobacion mas fuerte de
     las cuatro: no depende de comparar dos rutas entre si, sino la MISMA
     ruta contra si misma bajo dos cargas distintas.

  D. AISLAMIENTO ENTRE RUTAS. La ruta _none NUNCA ancla, sin importar con que
     --anchor-every se lanzo la corrida. Su t_iter_ms tiene que ser
     insensible a K. Si sube con K, es que sigue contaminada por el trabajo
     de la otra ruta -- exactamente el bug, en su otra direccion.

  E. LA REFERENCIA FP64 SE REPORTA APARTE. Tras el fix, la trayectoria de
     referencia deja de estar dentro del tiempo de las rutas de baja
     precision y pasa a publicarse como su propia fila (ruta GPU_FP64), que
     ademas es el punto de Pareto "todo en FP64" que a estos dos kernels les
     faltaba (Stencil ya lo tenia). El gate exige que esa fila EXISTA -- sin
     ella no hay forma de comprobar que el costo se excluyo en vez de
     desaparecer sin dejar rastro.

Stencil NO se evalua aqui: su medicion ya era por ruta. --kernel solo acepta
gemm y conv a proposito.

USO
---
    python3 gate4_medicion.py --kernel gemm --k0 k0.log --k1 k1.log

Los dos logs tienen que ser de UNA sola invocacion cada uno, con los MISMOS
--n/--hw, --iters y --tc, y ambos con --comp on; solo debe cambiar
--anchor-every (0 en uno, 1 en el otro). El gate lo verifica.

Codigos de salida:
    0  la medicion distingue las rutas.
    1  FALLA alguna comprobacion.
    2  no evaluable (falta un log, o los dos logs no son comparables).
       NO es un "pasa".
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

# Esquema de CSV_SUMMARY de gemm_chained/conv_chained, contando el token.
SUMMARY = ["token", "route", "size", "iters", "t_iter_ms", "t_total_ms",
           "gflops", "energy_gpu_j", "window_reliable", "gpu_segments",
           "anchor_every"]

RUTA_REFERENCIA = "GPU_FP64"


def num(valor):
    if valor is None:
        return None
    v = str(valor).strip()
    if v == "" or v.upper() in {"NONFINITE", "NA", "NAN", "INF", "+INF", "-INF"}:
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return f if math.isfinite(f) else None


def leer_summary(path):
    """{route: fila} de las lineas CSV_SUMMARY del log."""
    filas = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for linea in fh:
            if not linea.startswith("CSV_SUMMARY,"):
                continue
            try:
                campos = next(csv.reader([linea.rstrip("\n")]))
            except csv.Error:
                continue
            d = dict(zip(SUMMARY, campos))
            filas[d.get("route", "")] = d
    return filas


def formatos_presentes(filas):
    """Formatos con par _none/_comp completo (FP16, BF16, ...)."""
    fmts = []
    for route in filas:
        if route.endswith("_none"):
            fmt = route[: -len("_none")]
            if fmt + "_comp" in filas:
                fmts.append(fmt)
    return sorted(fmts)


def rel(a, b):
    """Diferencia relativa |a-b| / max(|a|,|b|)."""
    den = max(abs(a), abs(b))
    return abs(a - b) / den if den else 0.0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", required=True, choices=["gemm", "conv"],
                    help="Stencil no aplica: su medicion ya era por ruta.")
    ap.add_argument("--k0", required=True, help="log de una corrida --comp on --anchor-every 0")
    ap.add_argument("--k1", required=True, help="log de una corrida --comp on --anchor-every 1")
    ap.add_argument("--etiqueta", default="")
    ap.add_argument("--margen-separacion", type=float, default=0.10,
                    help="cuanto mas lenta, como minimo, tiene que ser la ruta con "
                         "mas trabajo (default 0.10 = 10%%). La diferencia REAL "
                         "esperada es mucho mayor (la ruta _comp hace el doble de "
                         "productos WMMA); el umbral es bajo a proposito, porque lo "
                         "que tiene que atrapar es la IGUALDAD exacta del bug, no "
                         "medir la diferencia con precision.")
    ap.add_argument("--tolerancia-ruido", type=float, default=0.25,
                    help="variacion relativa admitida en una ruta que NO deberia "
                         "cambiar entre las dos corridas (default 0.25). Es ruido de "
                         "medicion entre dos invocaciones distintas del binario, no "
                         "una cantidad fisica.")
    args = ap.parse_args()

    print("=" * 72)
    print("GATE 4 :: MEDICION POR RUTA :: kernel=%s :: %s"
          % (args.kernel, args.etiqueta or "(sin etiqueta)"))
    print("=" * 72)

    for p in (args.k0, args.k1):
        if not os.path.isfile(p):
            print("NO EVALUABLE: no existe el log %s" % p, file=sys.stderr)
            return 2

    k0, k1 = leer_summary(args.k0), leer_summary(args.k1)
    if not k0 or not k1:
        print("NO EVALUABLE: alguno de los logs no trae filas CSV_SUMMARY "
              "(k0=%d rutas, k1=%d rutas)." % (len(k0), len(k1)), file=sys.stderr)
        return 2

    fmts = formatos_presentes(k0)
    if not fmts:
        print("NO EVALUABLE: el log K=0 no trae ningun par _none/_comp completo. "
              "Correr con --comp on: sin la ruta compensada no hay dos rutas que "
              "separar.", file=sys.stderr)
        return 2
    if formatos_presentes(k1) != fmts:
        print("NO EVALUABLE: los dos logs no traen los mismos formatos "
              "(K=0: %s, K=1: %s)." % (fmts, formatos_presentes(k1)), file=sys.stderr)
        return 2

    # Los dos logs tienen que ser la misma configuracion salvo el ancla.
    for fmt in fmts:
        for suf in ("_none", "_comp"):
            a, b = k0[fmt + suf], k1[fmt + suf]
            if (a["size"], a["iters"]) != (b["size"], b["iters"]):
                print("NO EVALUABLE: %s%s tiene size/iters distintos entre los dos "
                      "logs (%s/%s vs %s/%s). El gate compara la MISMA carga bajo "
                      "dos valores de K." % (fmt, suf, a["size"], a["iters"],
                                             b["size"], b["iters"]), file=sys.stderr)
                return 2
    if any(num(k0[f + "_comp"]["anchor_every"]) != 0 for f in fmts):
        print("NO EVALUABLE: el log --k0 no fue generado con --anchor-every 0.",
              file=sys.stderr)
        return 2
    if any(num(k1[f + "_comp"]["anchor_every"]) != 1 for f in fmts):
        print("NO EVALUABLE: el log --k1 no fue generado con --anchor-every 1.",
              file=sys.stderr)
        return 2

    fallos, avisos = [], []
    m = 1.0 + args.margen_separacion

    for fmt in fmts:
        n0, c0 = k0[fmt + "_none"], k0[fmt + "_comp"]
        n1, c1 = k1[fmt + "_none"], k1[fmt + "_comp"]
        tn0, tc0 = num(n0["t_iter_ms"]), num(c0["t_iter_ms"])
        tn1, tc1 = num(n1["t_iter_ms"]), num(c1["t_iter_ms"])

        print("\n-- %s --" % fmt)
        print("   t_iter_ms  K=0: _none=%s  _comp=%s" % (n0["t_iter_ms"], c0["t_iter_ms"]))
        print("   t_iter_ms  K=1: _none=%s  _comp=%s" % (n1["t_iter_ms"], c1["t_iter_ms"]))
        print("   energy_gpu_j K=0: _none=%s  _comp=%s"
              % (n0["energy_gpu_j"], c0["energy_gpu_j"]))

        if None in (tn0, tc0, tn1, tc1):
            fallos.append("%s: algun t_iter_ms no es un numero finito." % fmt)
            continue

        # --- A. separacion entre rutas (tiempo) ---
        if tc0 == tn0:
            fallos.append(
                "%s A: t_iter_ms de _none y _comp son EXACTAMENTE iguales (%g). Es la "
                "firma del bug: un solo cronometro envolviendo las dos rutas."
                % (fmt, tn0))
        elif tc0 < tn0 * m:
            fallos.append(
                "%s A: t_iter_ms(_comp)=%g no supera a t_iter_ms(_none)=%g por al menos "
                "%.0f%%. La ruta compensada hace el DOBLE de productos WMMA por "
                "iteracion; si no se nota, el cronometro no es suyo."
                % (fmt, tc0, tn0, 100.0 * args.margen_separacion))
        else:
            print("   A OK: _comp es %.2fx _none en K=0." % (tc0 / tn0))

        # --- B. separacion entre rutas (energia) ---
        en0, ec0 = num(n0["energy_gpu_j"]), num(c0["energy_gpu_j"])
        if en0 is None or ec0 is None:
            avisos.append("%s B: energy_gpu_j no disponible (NVML ausente o ventana "
                          "invalida); la comprobacion de energia se omite." % fmt)
        elif en0 == ec0:
            fallos.append(
                "%s B: energy_gpu_j de _none y _comp son EXACTAMENTE iguales (%g J). "
                "Las dos ventanas de PowerBuffer siguen cubriendo el mismo intervalo "
                "sobre un contador NVML de todo el dispositivo." % (fmt, en0))
        else:
            print("   B OK: energias distintas (%g vs %g J)." % (en0, ec0))

        # --- C. la ruta responde a su propia carga ---
        if tc1 < tc0 * m:
            fallos.append(
                "%s C: t_iter_ms(_comp) no sube al pasar de K=0 (%g) a K=1 (%g). Con "
                "--anchor-every 1 esa ruta ejecuta un paso FP64 completo por iteracion; "
                "si su tiempo no lo refleja, no esta midiendo su propio trabajo."
                % (fmt, tc0, tc1))
        else:
            print("   C OK: _comp pasa de %g a %g ms/iter al activar el ancla (%.2fx)."
                  % (tc0, tc1, tc1 / tc0))

        # --- D. aislamiento entre rutas ---
        d = rel(tn0, tn1)
        if d > args.tolerancia_ruido:
            fallos.append(
                "%s D: t_iter_ms(_none) cambia un %.1f%% entre K=0 (%g) y K=1 (%g), por "
                "encima del %.0f%% de ruido admitido. La ruta _none nunca ancla: si su "
                "tiempo se mueve con K, sigue contaminado por el trabajo de _comp."
                % (fmt, 100.0 * d, tn0, tn1, 100.0 * args.tolerancia_ruido))
        else:
            print("   D OK: _none varia %.1f%% entre K=0 y K=1 (ruido)." % (100.0 * d))

    # --- E. la referencia FP64 se reporta aparte ---
    print("\n-- referencia FP64 --")
    if RUTA_REFERENCIA not in k0:
        fallos.append(
            "E: no hay fila CSV_SUMMARY de la ruta %s. Tras el fix la trayectoria de "
            "referencia deja de estar dentro del tiempo de las rutas de baja precision "
            "y debe publicarse como su propia fila -- que es ademas el punto de Pareto "
            "'todo en FP64' que a este kernel le faltaba. Sin esa fila no hay forma de "
            "comprobar que el costo se excluyo en vez de desaparecer." % RUTA_REFERENCIA)
    else:
        ref = k0[RUTA_REFERENCIA]
        print("   %s: t_iter_ms=%s  energy_gpu_j=%s"
              % (RUTA_REFERENCIA, ref["t_iter_ms"], ref["energy_gpu_j"]))
        tref = num(ref["t_iter_ms"])
        if tref is None:
            fallos.append("E: la fila %s existe pero su t_iter_ms no es finito (%s)."
                          % (RUTA_REFERENCIA, ref["t_iter_ms"]))
        else:
            print("   E OK: la referencia FP64 se reporta como ruta propia.")
            for fmt in fmts:
                tn0 = num(k0[fmt + "_none"]["t_iter_ms"])
                if tn0 is not None and tn0 >= tref:
                    avisos.append(
                        "E: %s_none (%g ms/iter) no es mas rapida que la referencia "
                        "FP64 (%g ms/iter). No es un fallo de medicion -- es un "
                        "resultado, y a tamanos de campana seria sorprendente: "
                        "revisar antes de reportarlo." % (fmt, tn0, tref))

    if avisos:
        print("\n-- AVISOS (no bloquean) --")
        for a in avisos:
            print("   ! " + a)

    print("\n-- Veredicto --")
    if fallos:
        print("   FALLA: %d comprobacion(es).\n" % len(fallos))
        for f in fallos:
            print("   X " + f)
        print("\n   Mientras esto falle, los ejes de TIEMPO y ENERGIA del Frente de")
        print("   Pareto 3D de este kernel no son reportables. El eje de ERROR si.")
        return 1

    print("   PASA: la medicion distingue las rutas y la referencia FP64 va aparte.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
