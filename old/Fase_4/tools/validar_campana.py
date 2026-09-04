#!/usr/bin/env python3
"""Validacion de integridad de una campana de variabilidad de Fase 4 (Stencil 2D).

Comprueba: las 15 replicas completas, FP16 y BF16 presentes, los tres
tratamientos presentes, ambos horizontes presentes, ningun CSV ausente o vacio,
sin duplicados, tiempos > 0, energia GPU y CPU no negativas, telemetria GPU
marcada como valida (energy_window_reliable=1), ExitCode=0 en todas las tareas
de los arrays y AUSENCIA TOTAL de corridas kahan=on + spatial=on.

NOTA: rel_l2/rel_linf/max_abs en NaN es ESPERADO a ITERS=320/640 (la solucion
supera first_nonfinite). No se valida el error: esta campana es energetica, y
la campana energetica y la numerica son independientes.

Admite las dos disposiciones de campana:
  - job arrays:  <campana>/replicas/r<NN>/<bloque>/results/*.csv
  - plana:       <campana>/results/*.csv   (replica via manifiesto_jobs.csv)

Uso:
    python3 validar_campana.py --campana Fase_4/f4_variabilidad_15r_<...>
"""
import argparse
import csv
import glob
import os
import re
import subprocess
import sys

CLAVE_OBS = ("precision", "compensacion", "iters", "replica")
RUTA_REPLICA_RE = re.compile(r"replicas[/\\]r(\d+)[/\\]([AB])[/\\]results[/\\]")


class Informe:
    def __init__(self):
        self.fallos, self.avisos, self.ok = [], [], []

    def check(self, cond, etiqueta, detalle=""):
        (self.ok if cond else self.fallos).append((etiqueta, detalle))

    def aviso(self, etiqueta, detalle=""):
        self.avisos.append((etiqueta, detalle))

    def imprimir(self):
        for e, d in self.ok:
            print(f"  [OK]    {e}" + (f" -- {d}" if d else ""))
        for e, d in self.avisos:
            print(f"  [AVISO] {e}" + (f" -- {d}" if d else ""))
        for e, d in self.fallos:
            print(f"  [FALLO] {e}" + (f" -- {d}" if d else ""))
        print()
        if self.fallos:
            print(f"RESULTADO: {len(self.fallos)} FALLO(S), "
                  f"{len(self.avisos)} aviso(s), {len(self.ok)} comprobacion(es) correctas")
            return 1
        print(f"RESULTADO: CAMPANA VALIDA -- {len(self.ok)} comprobaciones correctas"
              + (f", {len(self.avisos)} aviso(s)" if self.avisos else ""))
        return 0


def num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def clasificar(route, kahan):
    if "FP16" in route:
        precision = "FP16"
    elif "BF16" in route:
        precision = "BF16"
    else:
        return None
    if route.endswith("_SP"):
        return precision, "spatial"
    return precision, ("kahan_local" if kahan == "on" else "none")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campana", required=True)
    p.add_argument("--replicas", type=int, default=15)
    args = p.parse_args()

    camp = args.campana.rstrip("/")
    reps_esperadas = set(range(1, args.replicas + 1))
    inf = Informe()
    print("=" * 78)
    print(f"VALIDACION -- {os.path.basename(camp)}  ({args.replicas} replicas esperadas)")
    print("=" * 78)

    # --- manifiesto del diseno ---------------------------------------------
    ruta_mapa = os.path.join(camp, "manifiesto_jobs.csv")
    if not os.path.exists(ruta_mapa):
        sys.exit(f"falta el manifiesto {ruta_mapa}")
    with open(ruta_mapa, newline="") as fh:
        tareas = list(csv.DictReader(fh))
    esperadas = args.replicas * 2
    inf.check(len(tareas) == esperadas, "numero de tareas en el manifiesto",
              f"{len(tareas)} (esperado {esperadas})")

    arrays = sorted({t["array_job_id"] for t in tareas if t.get("array_job_id")})
    inf.check(len(arrays) == 2, "dos job arrays registrados", f"{arrays}")

    # --- 1. prohibicion kahan=on + spatial=on, ya en el plan ----------------
    malas = [(t.get("array_job_id"), t.get("array_task_id")) for t in tareas
             if t.get("spatial_comp") == "on" and "on" in t.get("kahan_list", "").split("+")]
    inf.check(not malas, "ninguna tarea planificada con kahan=on + spatial=on",
              f"infractoras: {malas}" if malas else "manifiesto limpio")

    # --- 2. CSV presentes y no vacios ---------------------------------------
    faltantes, vacios, dirs_sin_csv = [], [], []
    for t in tareas:
        d = t.get("dir_replica")
        if not d:
            continue
        resdir = os.path.join(camp, d, "results")
        encontrados = {tipo: sorted(glob.glob(os.path.join(resdir, f"{tipo}_stencil_*.csv")))
                       for tipo in ("summary", "energy")}
        if not encontrados["energy"] or not encontrados["summary"]:
            dirs_sin_csv.append(d)
            continue
        for tipo, rutas in encontrados.items():
            for ruta in rutas:
                with open(ruta, newline="") as fh:
                    if len(list(csv.DictReader(fh))) == 0:
                        vacios.append(os.path.relpath(ruta, camp))
    inf.check(not dirs_sin_csv, "toda tarea produjo sus CSV summary y energy",
              f"sin CSV: {dirs_sin_csv}" if dirs_sin_csv else f"{len(tareas)} tareas")
    inf.check(not vacios, "ningun CSV vacio",
              f"vacios: {vacios}" if vacios else "todos con filas")
    inf.check(not faltantes, "sin ficheros esperados ausentes", "")

    # --- 3. carga de observaciones WMMA -------------------------------------
    replica_de_job = {}
    ruta_ejec = os.path.join(camp, "ejecuciones.csv")
    ejec = []
    if os.path.exists(ruta_ejec):
        with open(ruta_ejec, newline="") as fh:
            ejec = list(csv.DictReader(fh))
        for e in ejec:
            replica_de_job[e["job_id"]] = int(e["replica"])
    for t in tareas:
        if t.get("job_id") and t.get("replica"):
            replica_de_job[t["job_id"]] = int(t["replica"])

    patrones = (sorted(glob.glob(os.path.join(camp, "replicas", "*", "*", "results",
                                              "energy_stencil_*.csv"))) +
                sorted(glob.glob(os.path.join(camp, "results", "energy_stencil_*.csv"))))
    obs, brutas, sin_replica = [], 0, 0
    for ruta in patrones:
        m = RUTA_REPLICA_RE.search(ruta)
        rep_ruta = int(m.group(1)) if m else None
        with open(ruta, newline="") as fh:
            for fila in csv.DictReader(fh):
                brutas += 1
                cls = clasificar(fila["route"], fila["kahan"])
                if cls is None:
                    continue  # CPU_*/GPU_*: se conservan, no se cuentan
                rep = rep_ruta if rep_ruta is not None else replica_de_job.get(fila["job_id"])
                if rep is None:
                    sin_replica += 1
                obs.append({
                    "precision": cls[0], "compensacion": cls[1],
                    "iters": int(fila["iters"]), "job_id": fila["job_id"],
                    "replica": rep, "route": fila["route"], "kahan": fila["kahan"],
                    "nx": fila["nx"], "ny": fila["ny"],
                    "t": num(fila["time_total_s"]),
                    "egpu": num(fila["energy_gpu_j"]),
                    "ecpu": num(fila["energy_cpu_j"]),
                    "fiable": fila.get("energy_window_reliable", ""),
                    "csv": os.path.relpath(ruta, camp),
                })
    n_esp = 2 * 3 * 2 * args.replicas
    inf.check(len(obs) == n_esp, "numero de observaciones WMMA",
              f"{len(obs)} (esperado {n_esp}); filas CSV_ENERGY totales: {brutas}")
    inf.check(sin_replica == 0, "toda observacion es rastreable a su replica",
              f"{sin_replica} sin replica" if sin_replica else f"{len(obs)} rastreables")

    # --- 4. cobertura del diseno --------------------------------------------
    inf.check({o["precision"] for o in obs} == {"FP16", "BF16"},
              "ambas precisiones presentes (FP16 y BF16)",
              str(sorted({o["precision"] for o in obs})))
    trat = {o["compensacion"] for o in obs}
    inf.check(trat == {"none", "kahan_local", "spatial"},
              "los tres tratamientos presentes", str(sorted(trat)))
    hz = {o["iters"] for o in obs}
    inf.check(hz == {320, 640}, "ambos horizontes presentes", str(sorted(hz)))
    mallas = {(o["nx"], o["ny"]) for o in obs}
    inf.check(mallas == {("16384", "16384")}, "malla unica 16384x16384", str(mallas))

    # --- 5. replicas completas ----------------------------------------------
    reps = {o["replica"] for o in obs if o["replica"] is not None}
    faltan_reps = sorted(reps_esperadas - reps)
    inf.check(not faltan_reps, f"las {args.replicas} replicas presentes",
              f"faltan: {faltan_reps}" if faltan_reps else f"replicas {min(reps)}..{max(reps)}")

    celdas = {}
    for o in obs:
        celdas.setdefault((o["precision"], o["compensacion"], o["iters"]), set()).add(o["replica"])
    incompletas = {str(k): sorted(reps_esperadas - v) for k, v in celdas.items()
                   if len(v & reps_esperadas) != args.replicas}
    inf.check(len(celdas) == 12, "las 12 celdas del diseno estan pobladas", f"{len(celdas)}/12")
    inf.check(not incompletas, f"todas las celdas con {args.replicas} replicas",
              f"incompletas: {incompletas}" if incompletas
              else f"12 celdas x {args.replicas} replicas = {12 * args.replicas}")

    # --- 6. duplicados -------------------------------------------------------
    vistos, dups = set(), []
    for o in obs:
        k = tuple(o[c] for c in CLAVE_OBS)
        (dups.append(k) if k in vistos else vistos.add(k))
    inf.check(not dups, "sin observaciones duplicadas",
              f"duplicados: {dups}" if dups else f"{len(vistos)} claves unicas")

    # --- 7. dominios numericos ----------------------------------------------
    t_malos = [(o["job_id"], o["route"], o["iters"], o["t"]) for o in obs
               if o["t"] is None or o["t"] <= 0]
    inf.check(not t_malos, "todos los tiempos > 0",
              f"invalidos: {t_malos}" if t_malos else f"{len(obs)} tiempos positivos")

    e_malos = [(o["job_id"], o["route"], o["iters"], o["egpu"], o["ecpu"]) for o in obs
               if (o["egpu"] is None or o["egpu"] < 0) or (o["ecpu"] is None or o["ecpu"] < 0)]
    inf.check(not e_malos, "energia GPU y CPU no negativas y definidas",
              f"invalidas: {e_malos}" if e_malos else f"{len(obs)} observaciones")

    # --- 8. telemetria GPU valida -------------------------------------------
    no_fiables = [(o["job_id"], o["route"], o["iters"]) for o in obs if o["fiable"] != "1"]
    inf.check(not no_fiables, "telemetria GPU valida (energy_window_reliable=1)",
              f"energy_window_reliable != 1 en {no_fiables}" if no_fiables
              else f"{len(obs)}/{len(obs)} con energy_window_reliable=1")

    # --- 9. ausencia real de kahan=on + spatial=on en los datos -------------
    infractoras = [(o["job_id"], o["route"], o["kahan"], o["csv"]) for o in obs
                   if o["route"].endswith("_SP") and o["kahan"] == "on"]
    inf.check(not infractoras, "ninguna corrida ejecutada con kahan=on + spatial=on",
              f"infractoras: {infractoras}" if infractoras
              else "0 filas con route *_SP y kahan=on")

    # --- 10. registro de ejecucion (rc y nodos) -----------------------------
    if ejec:
        rc_malos = [(e["job_id"], e["replica"], e["bloque"], e["rc"]) for e in ejec
                    if e.get("rc") not in ("0",)]
        inf.check(not rc_malos, "rc=0 en todas las tareas registradas",
                  f"anomalas: {rc_malos}" if rc_malos else f"{len(ejec)} tareas")
        inf.check(len(ejec) == esperadas, "ejecuciones.csv con todas las tareas",
                  f"{len(ejec)}/{esperadas}")
        nodos = sorted({e["nodo"] for e in ejec if e.get("nodo")})
        if len(nodos) > 1:
            inf.aviso("las replicas no corrieron todas en el mismo nodo",
                      f"{nodos} -- el CV mezcla variabilidad temporal y entre nodos")
        elif nodos:
            inf.ok.append(("todas las tareas en un unico nodo", nodos[0]))
    else:
        inf.aviso("sin ejecuciones.csv", "no se puede comprobar rc ni nodo por tarea")

    # --- 11. estado SLURM de las tareas del array ---------------------------
    if not arrays:
        inf.aviso("sacct no consultado", "manifiesto sin array_job_id")
    else:
        try:
            salida = subprocess.run(
                ["sacct", "-j", ",".join(arrays), "-X", "-n", "-P",
                 "--format=JobID,JobName,State,ExitCode,Elapsed"],
                capture_output=True, text=True, timeout=120, check=True).stdout
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            inf.aviso("estado SLURM no verificable", f"sacct no disponible ({exc})")
        else:
            estados, anomalos = [], []
            for linea in salida.strip().splitlines():
                campos = linea.split("|")
                if len(campos) < 5:
                    continue
                jid, nombre, estado, code, elapsed = campos[:5]
                if "_" not in jid:      # la fila resumen del array, sin tarea
                    continue
                estados.append((jid, estado, code, elapsed))
                if code.split(":")[0] != "0" or estado != "COMPLETED":
                    anomalos.append((jid, nombre, estado, code, elapsed))
            inf.check(len(estados) == esperadas, "sacct devuelve las tareas de los arrays",
                      f"{len(estados)}/{esperadas}")
            inf.check(not anomalos, "todas las tareas COMPLETED con ExitCode 0",
                      f"anomalas: {anomalos}" if anomalos else f"{len(estados)} tareas")
            timeouts = [a for a in anomalos if a[2] == "TIMEOUT"]
            if timeouts:
                inf.aviso("TAREAS CORTADAS POR TIMEOUT",
                          f"{timeouts} -- esas replicas contaminan el CV: relanzarlas")

    print()
    sys.exit(inf.imprimir())


if __name__ == "__main__":
    main()
