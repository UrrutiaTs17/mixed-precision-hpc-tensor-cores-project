#!/usr/bin/env python3
"""Validador del GATE 2 (checkpoints por lista y archivado de campos).

Comprueba sobre el log de una corrida:
  1. Exactamente len(checkpoint_iters) filas CSV_CKPT por ruta, en las
     iteraciones pedidas y sin repeticiones.
  2. Una fila CSV_NORM por iteracion de checkpoint.
  3. Un fichero archivado por ruta en cada iteracion de --archive-iters,
     con manifest.json valido, sha256 correcto y forma legible por numpy.
  4. first_nonfinite = -1 en todas las rutas.
  5. Que ninguna metrica quedara en NONFINITE.

Uso:
  validar_gate2.py --log RUTA --iters-ckpt 1,2,5,10,20 --iters-archivo 20
                   [--archive-dir DIR] [--esperar-colapso]
Codigo 0 si el gate pasa, 1 si no.
"""
import argparse
import csv
import hashlib
import json
import os
import sys

# Rutas que emiten checkpoints. CPU_FP32 no tiene CheckpointContext (ver el
# comentario de write_csv_row en el .cu), asi que no aparece en CSV_CKPT.
# CPU_FP64 no es una "ruta con checkpoint": es la referencia, y lo que emite es
# CSV_NORM. Se archiva, eso si.
RUTAS_CKPT_ESPERADAS = {"GPU_FP32", "GPU_FP64"}
PREFIJOS_WMMA = ("WMMA_FP16", "WMMA_BF16")


def leer(log):
    filas = {"CSV_CKPT": [], "CSV_NORM": [], "CSV_ARCHIVE": [], "CSV_SUMMARY": [],
             "CSV_CI_SHA256": []}
    with open(log, encoding="utf-8", errors="replace") as fh:
        for linea in fh:
            linea = linea.rstrip("\n")
            tok = linea.split(",", 1)[0]
            if tok in filas:
                try:
                    filas[tok].append(next(csv.reader([linea])))
                except csv.Error:
                    pass
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--iters-ckpt", required=True)
    ap.add_argument("--iters-archivo", required=True)
    ap.add_argument("--archive-dir", default=None)
    ap.add_argument("--esperar-colapso", action="store_true",
                    help="admite rel_l2 == 1 (campo hundido a cero): usar solo en la "
                         "variante degenerada del gate, nunca en la representativa")
    a = ap.parse_args()

    esperados = sorted({int(x) for x in a.iters_ckpt.split(",") if x.strip()})
    archivo_esperado = sorted({int(x) for x in a.iters_archivo.split(",") if x.strip()})
    f = leer(a.log)
    fallos, avisos = [], []

    print("=" * 70)
    print("GATE 2 :: %s" % a.log)
    print("  checkpoints esperados : %s" % esperados)
    print("  archivado esperado    : %s" % archivo_esperado)
    print("=" * 70)

    # ---- 1. CSV_CKPT: una fila por (ruta, iter), exactamente las pedidas ----
    por_ruta = {}
    for p in f["CSV_CKPT"]:
        if len(p) < 6:
            fallos.append("CSV_CKPT con %d campos (esperado 6): %s" % (len(p), ",".join(p)))
            continue
        por_ruta.setdefault(p[1], []).append(p)

    if not por_ruta:
        fallos.append("no se emitio ninguna fila CSV_CKPT")

    print("\n-- CSV_CKPT por ruta --")
    for ruta in sorted(por_ruta):
        filas = por_ruta[ruta]
        its = [int(x[2]) for x in filas]
        marca = "OK "
        if sorted(its) != esperados:
            fallos.append("%s: iteraciones %s, esperadas %s" % (ruta, sorted(its), esperados))
            marca = "X  "
        if len(its) != len(set(its)):
            fallos.append("%s: iteraciones repetidas en CSV_CKPT" % ruta)
            marca = "X  "
        nf = [x for x in filas if "NONFINITE" in x]
        if nf:
            fallos.append("%s: %d checkpoint(s) NONFINITE" % (ruta, len(nf)))
            marca = "X  "
        ultimo = [x for x in filas if int(x[2]) == esperados[-1]]
        rel = ultimo[0][3] if ultimo and "NONFINITE" not in ultimo[0] else "n/d"
        print("   %s%-16s %d filas  iters=%s  rel_l2(final)=%s"
              % (marca, ruta, len(filas), sorted(its), rel))
        # Un rel_l2 de exactamente 1 significa que la ruta se fue a cero y el
        # error es la referencia entera: finito, pero sin contenido.
        if not a.esperar_colapso and ultimo and "NONFINITE" not in ultimo[0]:
            try:
                if float(ultimo[0][3]) >= 0.999:
                    fallos.append("%s: rel_l2 = %s en iter %d -- la ruta colapso a cero, el "
                                  "checkpoint es finito pero vacio de informacion"
                                  % (ruta, ultimo[0][3], esperados[-1]))
            except ValueError:
                pass

    faltan = (RUTAS_CKPT_ESPERADAS - set(por_ruta))
    if faltan:
        fallos.append("faltan rutas en CSV_CKPT: %s" % sorted(faltan))
    if not any(r.startswith(PREFIJOS_WMMA) for r in por_ruta):
        fallos.append("ninguna ruta WMMA emitio CSV_CKPT")

    # ---- 2. CSV_NORM ----
    norms = sorted(int(p[2]) for p in f["CSV_NORM"] if len(p) >= 5)
    print("\n-- CSV_NORM (normas de la referencia) --")
    print("   iters=%s" % norms)
    if norms != esperados:
        fallos.append("CSV_NORM en %s, esperado %s" % (norms, esperados))

    # ---- 3. Archivado ----
    print("\n-- Archivado --")
    adir = a.archive_dir
    if adir is None:
        print("   (sin --archive-dir: no se revisa el manifest)")
    else:
        man_path = os.path.join(adir, "manifest.json")
        if not os.path.exists(man_path):
            fallos.append("no existe %s" % man_path)
        else:
            try:
                man = json.load(open(man_path, encoding="utf-8"))
            except json.JSONDecodeError as e:
                fallos.append("manifest.json no es JSON valido: %s" % e)
                man = None
            if man is not None:
                campos = man.get("fields", [])
                print("   manifest.json valido, %d campos" % len(campos))
                rutas_arch = {}
                for c in campos:
                    rutas_arch.setdefault(c["route"], []).append(int(c["iter"]))
                    ruta_bin = os.path.join(adir, c["file"])
                    if not os.path.exists(ruta_bin):
                        fallos.append("falta el .bin %s" % ruta_bin)
                        continue
                    datos = open(ruta_bin, "rb").read()
                    if len(datos) != c["bytes"]:
                        fallos.append("%s: %d bytes en disco, %d en el manifest"
                                      % (c["file"], len(datos), c["bytes"]))
                    h = hashlib.sha256(datos).hexdigest()
                    ok_h = (h == c["sha256"])
                    if not ok_h:
                        fallos.append("%s: sha256 no coincide" % c["file"])
                    ancho = {"float32": 4, "float64": 8}[c["dtype"]]
                    n_esp = c["shape"][0] * c["shape"][1] * ancho
                    if n_esp != c["bytes"]:
                        fallos.append("%s: shape %s x %s no cuadra con %d bytes"
                                      % (c["file"], c["shape"], c["dtype"], c["bytes"]))
                    print("   %-16s iter=%-4d %-8s %s  %d B  sha256=%s"
                          % (c["route"], c["iter"], c["dtype"], c["shape"], c["bytes"],
                             "OK" if ok_h else "MAL"))
                for ruta, its in sorted(rutas_arch.items()):
                    if sorted(set(its)) != archivo_esperado:
                        fallos.append("%s archivada en %s, esperado %s"
                                      % (ruta, sorted(set(its)), archivo_esperado))
                # Toda ruta que emitio checkpoints debe haberse archivado, mas
                # CPU_FP64 (la referencia).
                esperadas_arch = set(por_ruta) | {"CPU_FP64"}
                faltan_arch = esperadas_arch - set(rutas_arch)
                if faltan_arch:
                    fallos.append("rutas sin archivar: %s" % sorted(faltan_arch))
                if not man.get("ci_sha256"):
                    fallos.append("manifest.json sin ci_sha256")

    # ---- 4. first_nonfinite y CSV_CI_SHA256 ----
    print("\n-- first_nonfinite por ruta (CSV_SUMMARY, campo 19) --")
    for p in f["CSV_SUMMARY"]:
        if len(p) < 20:
            continue
        fn = p[19]
        marca = "OK " if fn.strip() == "-1" else "X  "
        if fn.strip() != "-1":
            fallos.append("%s: first_nonfinite = %s (esperado -1)" % (p[1], fn))
        print("   %s%-16s first_nonfinite=%s" % (marca, p[1], fn))

    if f["CSV_CI_SHA256"]:
        print("\n-- CSV_CI_SHA256 --\n   %s" % f["CSV_CI_SHA256"][0][1])
    else:
        fallos.append("no se emitio CSV_CI_SHA256")

    print("\n-- Veredicto --")
    if fallos:
        print("   FALLA: %d problema(s)." % len(fallos))
        for x in fallos:
            print("   X " + x)
        return 1
    for x in avisos:
        print("   ! " + x)
    print("   PASA.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
