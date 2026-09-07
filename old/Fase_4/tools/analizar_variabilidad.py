#!/usr/bin/env python3
"""Analisis de VARIABILIDAD de una campana de Fase 4 (Stencil 2D).

Agrupa las observaciones WMMA por (precision, compensacion, iteraciones) y
calcula, sobre tiempo, GFLOP/s, energia GPU (NVML), energia CPU (RAPL) y
metricas derivadas: n, media, desviacion estandar muestral (ddof=1), mediana,
minimo, maximo y CV% = 100 * std / media. Las observaciones individuales
quedan siempre visibles.

Tres analisis:
  1. Estadisticos por celda del diseno, con las observaciones crudas.
  2. ANIDADO: los mismos estadisticos usando solo las primeras 5 replicas,
     luego las primeras 10, luego las 15 -- para ver si el CV se estabiliza al
     aumentar n y justificar el tamano muestral.
  3. DERIVA TEMPORAL: media y CV por orden de replica (mitad temprana vs
     mitad tardia, y tendencia replica a replica), con la temperatura y el
     clock observados si hay telemetria de contexto.

Admite las dos disposiciones de campana:
  - job arrays:  <campana>/replicas/r<NN>/<bloque>/results/*.csv
  - plana:       <campana>/results/*.csv   (replica via manifiesto_jobs.csv)

Solo biblioteca estandar: no requiere pandas ni numpy en el cluster.

Uso:
    python3 analizar_variabilidad.py --campana Fase_4/f4_variabilidad_15r_<...>
    python3 analizar_variabilidad.py --campana <...> --anidado 5,10,15 \
        --csv-salida <...>/estadisticos.csv
"""
import argparse
import csv
import glob
import os
import re
import statistics as st
import sys

# Metricas analizadas: (nombre visible, fichero de origen, columna, unidad)
METRICAS = [
    ("tiempo_total_s",      "energy",  "time_total_s",           "s"),
    ("t_iter_ms",           "summary", "t_iter_ms",              "ms"),
    ("rendimiento_gflops",  "summary", "gflops",                 "GFLOP/s"),
    ("energia_gpu_j",       "energy",  "energy_gpu_j",           "J"),
    ("energia_cpu_j",       "energy",  "energy_cpu_j",           "J"),
    ("energia_total_j",     "energy",  "energy_total_j",         "J"),
    ("energia_gpu_j_iter",  "energy",  "energy_gpu_j_per_iter",  "J/iter"),
    ("edp_j_s",             "energy",  "edp_j_s",                "J*s"),
    ("joules_por_gflop",    "energy",  "joules_per_gflop",       "J/GFLOP"),
    ("potencia_gpu_w",      "derivada", None,                    "W"),
    ("potencia_total_w",    "derivada", None,                    "W"),
]

# Metricas sobre las que se hace el analisis anidado y el de deriva: las tres
# variables de decision de la campana mas el rendimiento.
FOCO = ["tiempo_total_s", "rendimiento_gflops", "energia_gpu_j", "energia_cpu_j"]

CLAVE = ("job_id", "nx", "ny", "iters", "kahan", "route")
RUTA_REPLICA_RE = re.compile(r"replicas[/\\]r(\d+)[/\\]([AB])[/\\]results[/\\]")


def num(valor):
    try:
        x = float(valor)
    except (TypeError, ValueError):
        return None
    return x if x == x else None  # descarta NaN


def clasificar(route, kahan):
    """(precision, compensacion) a partir de la etiqueta de ruta y kahan."""
    if "FP16" in route:
        precision = "FP16"
    elif "BF16" in route:
        precision = "BF16"
    else:
        return None
    if route.endswith("_SP"):
        compensacion = "spatial"
    elif kahan == "on":
        compensacion = "kahan_local"
    else:
        compensacion = "none"
    return precision, compensacion


def patrones(campana, tipo):
    """CSV de un tipo, en las dos disposiciones posibles de campana."""
    return (sorted(glob.glob(os.path.join(
                campana, "replicas", "*", "*", "results", f"{tipo}_stencil_*.csv"))) +
            sorted(glob.glob(os.path.join(
                campana, "results", f"{tipo}_stencil_*.csv"))))


def leer(campana, tipo, replica_de_job):
    """{clave -> fila}, con la replica resuelta por ruta o por manifiesto."""
    filas = {}
    for ruta in patrones(campana, tipo):
        m = RUTA_REPLICA_RE.search(ruta)
        rep_ruta = int(m.group(1)) if m else None
        bloque = m.group(2) if m else None
        with open(ruta, newline="") as fh:
            for fila in csv.DictReader(fh):
                fila["_replica"] = rep_ruta if rep_ruta is not None else \
                    replica_de_job.get(fila.get("job_id", ""))
                fila["_bloque"] = bloque
                fila["_csv"] = ruta
                filas[tuple(fila[k] for k in CLAVE)] = fila
    return filas


def leer_ejecuciones(campana):
    """job_id -> {replica, bloque, nodo, inicio, rc} (lo escribe cada tarea)."""
    ruta = os.path.join(campana, "ejecuciones.csv")
    ejec = {}
    if os.path.exists(ruta):
        with open(ruta, newline="") as fh:
            for f in csv.DictReader(fh):
                ejec[f["job_id"]] = f
    return ejec


def leer_manifiesto(campana):
    """job_id -> replica, para campanas de la disposicion plana (sin arrays)."""
    ruta = os.path.join(campana, "manifiesto_jobs.csv")
    mapa = {}
    if os.path.exists(ruta):
        with open(ruta, newline="") as fh:
            for f in csv.DictReader(fh):
                jid = f.get("job_id")
                if jid and f.get("replica"):
                    mapa[jid] = int(f["replica"])
    return mapa


def leer_telemetria(campana):
    """replica -> {temp_media, temp_max, clock_medio} desde nvidia-smi."""
    por_replica = {}
    patron = os.path.join(campana, "replicas", "*", "*", "telemetria_gpu_*.csv")
    for ruta in sorted(glob.glob(patron)):
        m = re.search(r"replicas[/\\]r(\d+)[/\\]", ruta)
        if not m:
            continue
        rep = int(m.group(1))
        temps, clocks = [], []
        with open(ruta, newline="") as fh:
            for linea in fh:
                campos = [c.strip() for c in linea.split(",")]
                if len(campos) < 4 or campos[0].startswith("timestamp"):
                    continue
                c = re.match(r"([\d.]+)", campos[1])
                t = re.match(r"([\d.]+)", campos[3])
                if c:
                    clocks.append(float(c.group(1)))
                if t:
                    temps.append(float(t.group(1)))
        if temps or clocks:
            d = por_replica.setdefault(rep, {"temps": [], "clocks": []})
            d["temps"].extend(temps)
            d["clocks"].extend(clocks)
    return {rep: {"temp_media": st.fmean(d["temps"]) if d["temps"] else None,
                  "temp_max": max(d["temps"]) if d["temps"] else None,
                  "clock_medio": st.fmean(d["clocks"]) if d["clocks"] else None,
                  "n_muestras": max(len(d["temps"]), len(d["clocks"]))}
            for rep, d in por_replica.items()}


def cargar(campana):
    replica_de_job = leer_manifiesto(campana)
    ejec = leer_ejecuciones(campana)
    for jid, f in ejec.items():
        if f.get("replica"):
            replica_de_job.setdefault(jid, int(f["replica"]))

    energy = leer(campana, "energy", replica_de_job)
    summary = leer(campana, "summary", replica_de_job)
    if not energy:
        sys.exit(f"No hay CSV de energia bajo {campana} "
                 "(los jobs aun no han terminado o fallaron)")

    obs = []
    for clave, e in energy.items():
        cls = clasificar(e["route"], e["kahan"])
        if cls is None:
            continue  # CPU_FP32/CPU_FP64/GPU_FP32/GPU_FP64: se conservan, no se cuentan
        precision, compensacion = cls
        s = summary.get(clave, {})
        info = ejec.get(e["job_id"], {})
        registro = {
            "job_id": e["job_id"],
            "replica": e["_replica"],
            "bloque": e["_bloque"],
            "nodo": info.get("nodo", ""),
            "inicio": info.get("inicio", ""),
            "precision": precision,
            "compensacion": compensacion,
            "iters": int(e["iters"]),
            "route": e["route"],
            "kahan": e["kahan"],
            "fiable": e.get("energy_window_reliable", "NaN"),
        }
        for nombre, origen, columna, _ in METRICAS:
            if origen == "energy":
                registro[nombre] = num(e.get(columna))
            elif origen == "summary":
                registro[nombre] = num(s.get(columna))
        t = registro["tiempo_total_s"]
        egpu, etot = registro["energia_gpu_j"], registro["energia_total_j"]
        registro["potencia_gpu_w"] = egpu / t if (t and egpu is not None and t > 0) else None
        registro["potencia_total_w"] = etot / t if (t and etot is not None and t > 0) else None
        obs.append(registro)
    return obs


def resumen(valores):
    v = [x for x in valores if x is not None]
    n = len(v)
    if n == 0:
        return None
    media = st.fmean(v)
    desv = st.stdev(v) if n > 1 else 0.0
    cv = 100.0 * desv / media if media else float("nan")
    return dict(n=n, media=media, std=desv, mediana=st.median(v),
                minimo=min(v), maximo=max(v), cv=cv)


def agrupar(obs):
    grupos = {}
    for r in obs:
        grupos.setdefault((r["precision"], r["compensacion"], r["iters"]), []).append(r)
    return grupos


def orden_grupos(grupos):
    return sorted(grupos, key=lambda k: (k[2], k[0], k[1]))


# --------------------------------------------------------------------------
# 1. Estadisticos por celda, con las observaciones individuales
# --------------------------------------------------------------------------
def bloque_estadisticos(grupos, salida):
    for clave in orden_grupos(grupos):
        precision, compensacion, iters = clave
        filas = sorted(grupos[clave],
                       key=lambda r: (r["replica"] if r["replica"] is not None else 1e9,
                                      r["job_id"]))
        print(f"\n### {precision} | {compensacion} | ITERS={iters}   (n={len(filas)})")
        print("  observaciones individuales:")
        print(f"    {'replica':>7} {'job_id':>10} {'nodo':>12} {'fiable':>6} "
              f"{'tiempo_s':>10} {'GFLOP/s':>10} {'E_gpu_J':>10} {'E_cpu_J':>10} {'P_gpu_W':>9}")
        for r in filas:
            def f(x, d=4):
                return "NaN" if x is None else f"{x:.{d}f}"
            rep = "?" if r["replica"] is None else r["replica"]
            print(f"    {rep:>7} {r['job_id']:>10} {(r['nodo'] or '-'):>12} {r['fiable']:>6} "
                  f"{f(r['tiempo_total_s']):>10} {f(r['rendimiento_gflops'],2):>10} "
                  f"{f(r['energia_gpu_j'],2):>10} {f(r['energia_cpu_j'],2):>10} "
                  f"{f(r['potencia_gpu_w'],1):>9}")

        print("  estadisticos:")
        print(f"    {'metrica':<20} {'unidad':>8} {'n':>3} {'media':>13} {'std(ddof=1)':>13} "
              f"{'mediana':>13} {'min':>13} {'max':>13} {'CV%':>8}")
        for nombre, _, _, unidad in METRICAS:
            s = resumen([r.get(nombre) for r in filas])
            if s is None:
                print(f"    {nombre:<20} {unidad:>8} {0:>3}  (sin datos validos)")
                continue
            print(f"    {nombre:<20} {unidad:>8} {s['n']:>3} {s['media']:>13.5g} "
                  f"{s['std']:>13.5g} {s['mediana']:>13.5g} {s['minimo']:>13.5g} "
                  f"{s['maximo']:>13.5g} {s['cv']:>8.2f}")
            salida.append(dict(nivel_n="todas", precision=precision,
                               compensacion=compensacion, iters=iters,
                               metrica=nombre, unidad=unidad, **s))


# --------------------------------------------------------------------------
# 2. Analisis ANIDADO: primeras k replicas, k = 5, 10, 15
# --------------------------------------------------------------------------
def bloque_anidado(obs, niveles, salida):
    print("\n" + "=" * 110)
    print("ANALISIS ANIDADO -- estadisticos con las primeras k replicas "
          f"(k = {', '.join(str(k) for k in niveles)})")
    print("Si el CV% deja de moverse al pasar de un k al siguiente, el tamano "
          "muestral ya es suficiente.")
    print("=" * 110)

    for metrica in FOCO:
        print(f"\n--- {metrica} ---")
        cab = f"{'precision':<9} {'compensacion':<13} {'iters':>6}"
        for k in niveles:
            cab += f" | {'n':>3} {'media(k=' + str(k) + ')':>14} {'CV%':>7}"
        print(cab)
        for clave in orden_grupos(agrupar(obs)):
            precision, compensacion, iters = clave
            linea = f"{precision:<9} {compensacion:<13} {iters:>6}"
            for k in niveles:
                sub = [r for r in obs
                       if (r["precision"], r["compensacion"], r["iters"]) == clave
                       and r["replica"] is not None and r["replica"] <= k]
                s = resumen([r.get(metrica) for r in sub])
                if s is None:
                    linea += f" | {0:>3} {'n/d':>14} {'n/d':>7}"
                else:
                    linea += f" | {s['n']:>3} {s['media']:>14.5g} {s['cv']:>7.2f}"
                    salida.append(dict(nivel_n=k, precision=precision,
                                       compensacion=compensacion, iters=iters,
                                       metrica=metrica, unidad="", **s))
            print(linea)

    # Estabilizacion: cuanto cambia el CV al pasar de un nivel al siguiente.
    print("\n--- estabilizacion del CV% (variacion absoluta en puntos de CV) ---")
    print(f"{'metrica':<22} " + " ".join(
        f"{'k=' + str(a) + '->' + str(b):>14}" for a, b in zip(niveles, niveles[1:])))
    for metrica in FOCO:
        celdas = []
        for a, b in zip(niveles, niveles[1:]):
            deltas = []
            for clave, filas in agrupar(obs).items():
                sa = resumen([r.get(metrica) for r in filas
                              if r["replica"] is not None and r["replica"] <= a])
                sb = resumen([r.get(metrica) for r in filas
                              if r["replica"] is not None and r["replica"] <= b])
                if sa and sb and sa["cv"] == sa["cv"] and sb["cv"] == sb["cv"]:
                    deltas.append(abs(sb["cv"] - sa["cv"]))
            celdas.append("n/d" if not deltas else f"max {max(deltas):.2f}")
        print(f"{metrica:<22} " + " ".join(f"{c:>14}" for c in celdas))


# --------------------------------------------------------------------------
# 3. DERIVA TEMPORAL: las replicas tardias, comparadas con las tempranas
# --------------------------------------------------------------------------
def bloque_deriva(obs, telemetria, salida_csv):
    print("\n" + "=" * 110)
    print("DERIVA TEMPORAL -- las replicas tardias frente a las tempranas")
    print("Cada observacion se normaliza por la media de SU celda del diseno, "
          "asi las 12 celdas son comparables entre si.")
    print("=" * 110)

    grupos = agrupar(obs)
    medias = {}
    for clave, filas in grupos.items():
        for m in FOCO:
            s = resumen([r.get(m) for r in filas])
            if s and s["media"]:
                medias[(clave, m)] = s["media"]

    por_replica = {}
    for r in obs:
        if r["replica"] is None:
            continue
        clave = (r["precision"], r["compensacion"], r["iters"])
        d = por_replica.setdefault(r["replica"], {m: [] for m in FOCO})
        for m in FOCO:
            base = medias.get((clave, m))
            v = r.get(m)
            if base and v is not None:
                d[m].append(v / base)

    print(f"\n{'replica':>7} {'n_obs':>6} " +
          " ".join(f"{m + ' (norm)':>22}" for m in FOCO) +
          f" {'temp_media_C':>13} {'temp_max_C':>11} {'clock_MHz':>10}")
    filas_csv = []
    for rep in sorted(por_replica):
        d = por_replica[rep]
        celdas = []
        fila = {"replica": rep, "n_obs": len(d[FOCO[0]])}
        for m in FOCO:
            s = resumen(d[m])
            celdas.append("n/d" if s is None else f"{s['media']:.4f} +-{s['std']:.4f}")
            fila[f"{m}_norm_media"] = "" if s is None else f"{s['media']:.6f}"
            fila[f"{m}_norm_std"] = "" if s is None else f"{s['std']:.6f}"
        t = telemetria.get(rep, {})
        def tf(x, d=1):
            return "n/d" if x is None else f"{x:.{d}f}"
        fila.update(temp_media_c=tf(t.get("temp_media")),
                    temp_max_c=tf(t.get("temp_max")),
                    clock_medio_mhz=tf(t.get("clock_medio"), 0))
        filas_csv.append(fila)
        print(f"{rep:>7} {fila['n_obs']:>6} " + " ".join(f"{c:>22}" for c in celdas) +
              f" {tf(t.get('temp_media')):>13} {tf(t.get('temp_max')):>11} "
              f"{tf(t.get('clock_medio'), 0):>10}")

    # Mitad temprana contra mitad tardia sobre el valor normalizado.
    reps = sorted(por_replica)
    if len(reps) >= 4:
        corte = len(reps) // 2
        tempranas, tardias = reps[:corte], reps[len(reps) - corte:]
        print(f"\nmitad temprana = replicas {tempranas[0]}..{tempranas[-1]} | "
              f"mitad tardia = replicas {tardias[0]}..{tardias[-1]}")
        print(f"{'metrica':<22} {'temprana':>12} {'tardia':>12} {'delta %':>10} "
              f"{'CV% temprana':>14} {'CV% tardia':>12}")
        for m in FOCO:
            a = resumen([v for rep in tempranas for v in por_replica[rep][m]])
            b = resumen([v for rep in tardias for v in por_replica[rep][m]])
            if not a or not b:
                print(f"{m:<22} {'n/d':>12}")
                continue
            delta = 100.0 * (b["media"] - a["media"]) / a["media"] if a["media"] else float("nan")
            print(f"{m:<22} {a['media']:>12.4f} {b['media']:>12.4f} {delta:>10.2f} "
                  f"{a['cv']:>14.2f} {b['cv']:>12.2f}")
        print("\nUn |delta %| pequeno frente al CV% de la celda indica que no hay "
              "deriva sistematica; uno grande apunta a efecto termico o de nodo.")

    # Nodos: si las replicas no cayeron todas en el mismo nodo, el CV mezcla
    # variabilidad temporal con variabilidad de hardware.
    nodos = sorted({r["nodo"] for r in obs if r["nodo"]})
    if len(nodos) > 1:
        print(f"\nAVISO: las observaciones vienen de {len(nodos)} nodos {nodos}: "
              "el CV mezcla variabilidad temporal con variabilidad entre nodos.")
    elif nodos:
        print(f"\nTodas las observaciones en un unico nodo: {nodos[0]}")

    if salida_csv and filas_csv:
        with open(salida_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas_csv[0].keys()))
            w.writeheader()
            w.writerows(filas_csv)
        print(f"\nTabla de deriva escrita en {salida_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campana", required=True, help="directorio Fase_4/<ID>")
    p.add_argument("--replicas-esperadas", type=int, default=15)
    p.add_argument("--anidado", default="5,10,15",
                   help="niveles del analisis anidado (vacio para omitirlo)")
    p.add_argument("--csv-salida", default=None,
                   help="ruta para volcar la tabla de estadisticos (opcional)")
    p.add_argument("--csv-deriva", default=None,
                   help="ruta para volcar la tabla de deriva temporal (opcional)")
    args = p.parse_args()

    campana = args.campana.rstrip("/")
    obs = cargar(campana)
    grupos = agrupar(obs)
    telemetria = leer_telemetria(campana)
    esperadas = 2 * 3 * 2 * args.replicas_esperadas

    print("=" * 110)
    print(f"VARIABILIDAD - campana {os.path.basename(campana)}")
    print(f"Observaciones WMMA: {len(obs)} (esperadas {esperadas}) | "
          f"celdas del diseno: {len(grupos)}/12 | "
          f"replicas presentes: {len(sorted({r['replica'] for r in obs if r['replica']}))}")
    print("Nota: rel_l2/rel_linf en NaN es ESPERADO a ITERS=320/640 y no afecta "
          "a la energia medida.")
    print("=" * 110)

    salida = []
    bloque_estadisticos(grupos, salida)

    niveles = [int(x) for x in args.anidado.split(",") if x.strip()] if args.anidado else []
    if niveles:
        bloque_anidado(obs, niveles, salida)

    bloque_deriva(obs, telemetria, args.csv_deriva)

    print("\n" + "=" * 110)
    print("PANORAMA DE CV% (con todas las replicas disponibles)")
    print("=" * 110)
    print(f"{'precision':<10} {'compensacion':<14} {'iters':>6} " +
          " ".join(f"{m:>20}" for m in FOCO))
    for clave in orden_grupos(grupos):
        celdas = []
        for m in FOCO:
            s = resumen([r.get(m) for r in grupos[clave]])
            celdas.append("n/d" if s is None else f"{s['cv']:.2f}%")
        print(f"{clave[0]:<10} {clave[1]:<14} {clave[2]:>6} " +
              " ".join(f"{c:>20}" for c in celdas))

    peores = []
    for clave, filas in grupos.items():
        for m in FOCO:
            s = resumen([r.get(m) for r in filas])
            if s and s["cv"] == s["cv"]:
                peores.append((s["cv"], m, clave))
    if peores:
        cv, m, clave = max(peores)
        print(f"\nCV% maximo observado: {cv:.2f}% en '{m}' "
              f"({clave[0]}, {clave[1]}, ITERS={clave[2]})")

    if args.csv_salida and salida:
        with open(args.csv_salida, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(salida[0].keys()))
            w.writeheader()
            w.writerows(salida)
        print(f"\nTabla de estadisticos escrita en {args.csv_salida}")


if __name__ == "__main__":
    main()
