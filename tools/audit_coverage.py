#!/usr/bin/env python3
"""tools/audit_coverage.py -- gate de cobertura de la campana corregida.

Lee los CSV YA extraidos (summary_/drift_/energy_ de gemm, conv y stencil) de uno
o varios directorios de resultados, construye la matriz de cobertura
kernel x ruta x tamano x K x comp_scheme x iters y evalua los 8 asserts de
DECISIONS.md (A1-A8). Escribe audit_report.md y sale con codigo != 0 si algun
assert APLICABLE falla. Se ejecuta al final de cada job (--mode job) y al final
de la campana completa sobre la union de todos los directorios (--mode campaign).

Solo lectura sobre los CSV. Funciona tanto con el esquema NUEVO (columnas
device/gpu_valid/comp_scheme/error_evaluable) como con el de la campana 7145
(las deriva con las mismas reglas que Fase_4/tools/extract_csv*.py), de modo que
el gate se puede probar contra 7145: ahi ESPERAMOS que falle.

Estados por assert: PASS, FAIL, N/A (no aplicable a este alcance; en --mode job no
hace fallar, en --mode campaign SI: la campana completa no puede tener N/A).

  A1  >= 1 fila por esquema de compensacion requerido, por kernel
  A2  por configuracion, {iters con error evaluable} ∩ {iters con energia fiable} != vacio
  A3  GEMM N=8192: >= 1 ruta con solucion finita y error evaluable
  A4  0 filas con rel_l2 == 0.0 en rutas que no son la referencia exacta
  A5  0 filas duplicadas exactas tras dedupe (se reporta antes/despues y conflictos)
  A6  fraccion window_reliable=1 >= 0.90 en los pases energeticos
  A7  0 filas con device=cpu y gpu_valid=1
  A8  matriz Stencil alpha=3/16 completa, sin celdas vacias
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
from datetime import datetime

import pandas as pd

# ---------------------------------------------------------------------------
# Parametros pre-registrados (DECISIONS.md S2/S6, campana.env). Si cambian alla,
# cambian aqui: este es el unico sitio del gate donde viven.
# ---------------------------------------------------------------------------
ENERGY_ITERS = {   # pase energetico dedicado, por (kernel, tamano)
    "gemm": {1024: 24000, 2048: 24000, 4096: 500, 8192: 500},
    "conv": {64: 37000, 128: 37000, 256: 2500, 512: 2500},
    "stencil": {4096: 4000, 8192: 1500, 16384: 1500},
}
A1_REQUIRED = {   # esquemas que cada kernel DEBE tener (los de GEMM/Conv son none|local)
    "stencil": {"none", "kahan_local", "spatial"},
    "gemm": {"none", "local"},
    "conv": {"none", "local"},
}
EXACT_REFERENCES = {"GPU_FP64", "CPU_FP64"}          # error 0 legitimo (son la referencia)
A8_ROUTES = ["WMMA_FP16_SP", "WMMA_BF16_SP", "GPU_FP32", "GPU_FP64"]
A8_SIZES = [4096, 8192, 16384]
A8_K = [0, 1, 8, 32]
A8_ALPHA = 0.1875
RELIABLE_MIN = 0.90
NUMERIC_MAX_ITER = 80      # checkpoints del pase numerico de GEMM/Conv (20/40/80); los de energia son >= 500

KERNEL_FILE = {"gemm": "gemm", "conv": "conv", "stencil": "stencil"}


def num(series):
    return pd.to_numeric(series, errors="coerce")


def replica_de(job_id: str) -> str:
    m = re.search(r"-r(\d+)$", str(job_id))
    return m.group(1) if m else "0"


def find_files(dirs, prefix, kernel, recursive):
    out = []
    for d in dirs:
        pat = os.path.join(d, "**" if recursive else "", f"{prefix}_{KERNEL_FILE[kernel]}_*.csv")
        out += glob.glob(pat, recursive=recursive)
    return sorted(set(out))


def read_all(files):
    frames = [pd.read_csv(f, dtype=str) for f in files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Carga y normalizacion por kernel
# ---------------------------------------------------------------------------
def load_chained(kernel, dirs, recursive):
    s = read_all(find_files(dirs, "summary", kernel, recursive))
    d = read_all(find_files(dirs, "drift", kernel, recursive))
    if s.empty:
        return None
    s["size"] = num(s["size"]).astype("Int64")
    s["iters"] = num(s["iters"]).astype("Int64")
    s["K"] = num(s["anchor_every"]).fillna(0).astype(int)
    if "comp_scheme" not in s:
        s["comp_scheme"] = s["route"].str.endswith("_comp").map({True: "local", False: "none"})
    if "device" not in s:
        s["device"] = "gpu"
    if "gpu_valid" not in s:
        s["gpu_valid"] = (num(s["energy_gpu_j"]).notna()).astype(int).astype(str)
    s["t"] = num(s["t_iter_ms"])
    s["wr"] = num(s["window_reliable"])
    s["replica"] = s["job_id"].map(replica_de)
    s["op_mode"] = "n/a"
    s["alpha"] = math.nan
    if not d.empty:
        d["size"] = num(d["size"]).astype("Int64")
        d["iter"] = num(d["iter"]).astype("Int64")
        d["K"] = num(d["anchor_every"]).fillna(0).astype(int)
        d["rel"] = num(d["rel_l2"])
        d["sf"] = num(d["solution_finite"])
        if "comp_scheme" not in d:
            d["comp_scheme"] = d["route"].str.endswith("_comp").map({True: "local", False: "none"})
        if "error_evaluable" not in d:
            d["error_evaluable"] = (d["sf"] == 1).astype(int).astype(str)
        d["ev"] = num(d["error_evaluable"]).fillna(0).astype(int)
        if "motivo_exclusion" not in d:
            d["motivo_exclusion"] = d["ev"].map({1: "ok", 0: "solution_non_finite"})
    return {"S": s, "D": d, "E": None}


def load_stencil(dirs, recursive):
    s = read_all(find_files(dirs, "summary", "stencil", recursive))
    e = read_all(find_files(dirs, "energy", "stencil", recursive))
    if s.empty and e.empty:
        return None
    if not s.empty:
        s["size"] = num(s["nx"]).astype("Int64")
        s["iters"] = num(s["iters"]).astype("Int64")
        s["K"] = num(s["anchor_every"]).fillna(0).astype(int)
        if "comp_scheme" not in s:
            s["comp_scheme"] = s.apply(lambda r: _comp_scheme(r["route"], r["kahan"]), axis=1)
        if "device" not in s:
            s["device"] = s["route"].map(lambda r: "cpu" if str(r).upper().startswith("CPU_") else "gpu")
        if "gpu_valid" not in s:
            s["gpu_valid"] = ((s["device"] == "gpu") & num(s["energy_gpu_j"]).notna()).astype(int).astype(str)
        s["t"] = num(s["t_iter_ms"])
        prim = s.apply(lambda r: r["rel_l2_prop"] if str(r["route"]).upper().startswith("WMMA") else r["rel_l2"], axis=1)
        s["rel"] = num(prim)
        if "error_evaluable" not in s:
            s["error_evaluable"] = s["rel"].notna().astype(int).astype(str)
        s["ev"] = num(s["error_evaluable"]).fillna(0).astype(int)
        s["replica"] = s["job_id"].map(replica_de)
        if "op_mode" not in s:
            s["op_mode"] = "n/a"
        if "alpha" not in s:
            s["alpha"] = math.nan
        s["alpha"] = num(s["alpha"])
    if not e.empty:
        e["size"] = num(e["nx"]).astype("Int64")
        e["iters"] = num(e["iters"]).astype("Int64")
        e["K"] = num(e["anchor_every"]).fillna(0).astype(int)
        if "comp_scheme" not in e:
            e["comp_scheme"] = e.apply(lambda r: _comp_scheme(r["route"], r["kahan"]), axis=1)
        if "device" not in e:
            e["device"] = e["route"].map(lambda r: "cpu" if str(r).upper().startswith("CPU_") else "gpu")
        if "gpu_valid" not in e:
            e["gpu_valid"] = ((e["device"] == "gpu") & e["energy_window_reliable"].notna()).astype(int).astype(str)
        e["wr"] = num(e["energy_window_reliable"])
        e["t_total_s"] = num(e["time_total_s"])
        e["replica"] = e["job_id"].map(replica_de)
    return {"S": s, "D": pd.DataFrame(), "E": e}


def _comp_scheme(route, kahan):
    r = str(route).upper()
    if r.endswith("_SP"):
        return "spatial"
    if r.startswith("WMMA") and kahan == "on":
        return "kahan_local"
    return "none"


# ---------------------------------------------------------------------------
# Filas planas para los asserts
# ---------------------------------------------------------------------------
def flat_rows(kernel, k):
    """Devuelve DataFrame plano de TODAS las filas de medicion del kernel con las
    columnas que necesitan los asserts (una fila por fila de summary)."""
    s = k["S"]
    if s is None or s.empty:
        return pd.DataFrame()
    cols = ["job_id", "replica", "route", "size", "iters", "K", "comp_scheme", "device", "gpu_valid", "op_mode", "alpha"]
    r = s[cols].copy()
    r.insert(0, "kernel", kernel)
    r["t"] = s["t"].values
    r["is_reference"] = r["route"].isin(EXACT_REFERENCES)
    if kernel == "stencil":
        r["rel"] = s["rel"].values
        r["ev"] = s["ev"].values
    return r


def energy_rows(kernel, k):
    """Una fila por medicion de ventana de energia: (route,size,iters,K,comp_scheme,wr,device,gpu_valid)."""
    if kernel == "stencil":
        e = k["E"]
        if e is None or e.empty:
            return pd.DataFrame()
        r = e[["job_id", "replica", "route", "size", "iters", "K", "comp_scheme", "device", "gpu_valid", "wr"]].copy()
    else:
        s = k["S"]
        r = s[["job_id", "replica", "route", "size", "iters", "K", "comp_scheme", "device", "gpu_valid"]].copy()
        r["wr"] = s["wr"].values
    r.insert(0, "kernel", kernel)
    r["dedicated"] = [ENERGY_ITERS.get(kernel, {}).get(int(sz), -1) == int(it) if pd.notna(sz) and pd.notna(it) else False
                      for sz, it in zip(r["size"], r["iters"])]
    return r


def _es_stress(data, kernel, cfg):
    """True si la configuracion Stencil corrio con el operador 'stress' (amplificante)."""
    if kernel != "stencil":
        return False
    s = data["stencil"]["S"]
    if s is None or s.empty or "op_mode" not in s:
        return False
    sub = s[(s["route"] == cfg["route"]) & (s["size"] == cfg["size"]) & (s["K"] == cfg["K"])]
    return bool((sub["op_mode"] == "stress").any())


def error_points(kernel, k):
    """Puntos de error por (route,size,K,comp,iter): finito/evaluable/valor/motivo."""
    if kernel == "stencil":
        s = k["S"]
        if s is None or s.empty:
            return pd.DataFrame()
        p = s[["route", "size", "K", "comp_scheme", "iters", "rel", "ev"]].rename(columns={"iters": "iter"}).copy()
        p["motivo"] = s["motivo_exclusion"].values if "motivo_exclusion" in s else "n/a"
    else:
        d = k["D"]
        if d is None or d.empty:
            return pd.DataFrame()
        p = d[["route", "size", "K", "comp_scheme", "iter", "rel", "ev", "motivo_exclusion"]].rename(
            columns={"motivo_exclusion": "motivo"}).copy()
    p.insert(0, "kernel", kernel)
    p["is_reference"] = p["route"].isin(EXACT_REFERENCES)
    p["ok"] = (p["ev"] == 1) & p["rel"].notna() & (p["is_reference"] | (p["rel"] > 0))
    return p


# ---------------------------------------------------------------------------
# Asserts
# ---------------------------------------------------------------------------
class Res:
    def __init__(self, code, title):
        self.code, self.title = code, title
        self.status, self.detail, self.offenders = "N/A", "", []

    def set(self, status, detail, offenders=None):
        self.status, self.detail, self.offenders = status, detail, (offenders or [])


def run_asserts(data, args):
    results = []
    kernels = list(data)

    # ---- A1 -------------------------------------------------------------
    r = Res("A1", ">= 1 fila por esquema de compensacion requerido, por kernel")
    falta, cuenta = [], []
    for kn, k in data.items():
        have = set(flat_rows(kn, k)["comp_scheme"].dropna().unique())
        need = A1_REQUIRED[kn]
        if kn == "stencil" and args.expect_schemes:
            need = set(args.expect_schemes.split(","))   # job con una sola invocacion (DECISIONS.md S6)
        cuenta.append("%s: %s" % (kn, ", ".join("%s=%d" % (c, int((flat_rows(kn, k)["comp_scheme"] == c).sum())) for c in sorted(need))))
        for c in sorted(need - have):
            falta.append("%s sin filas con comp_scheme=%s" % (kn, c))
    r.set("FAIL" if falta else "PASS", "; ".join(cuenta), falta)
    results.append(r)

    # ---- A2 -------------------------------------------------------------
    r = Res("A2", "por configuracion, {iters con error evaluable} ∩ {iters con energia fiable} no vacio")
    vacias, exentas, total = [], [], 0
    for kn, k in data.items():
        ep = error_points(kn, k)
        er = energy_rows(kn, k)
        if er.empty:
            continue
        if args.mode == "job" and ep.empty:
            continue   # un job solo-energia no puede decidir A2: se evalua sobre la union (--mode campaign)
        er = er[(er["wr"] == 1) & (er["device"] == "gpu")]
        cfgs = er[~er["route"].isin(EXACT_REFERENCES)][["route", "size", "K", "comp_scheme"]].drop_duplicates()
        # configuraciones esperadas = las que tienen alguna medicion de tiempo/error (evita exigir lo no lanzado)
        for _, c in cfgs.iterrows():
            total += 1
            e_it = set(er[(er["route"] == c["route"]) & (er["size"] == c["size"]) & (er["K"] == c["K"])
                          & (er["comp_scheme"] == c["comp_scheme"])]["iters"].dropna().astype(int))
            sub = ep[(ep["route"] == c["route"]) & (ep["size"] == c["size"]) & (ep["K"] == c["K"])
                     & (ep["comp_scheme"] == c["comp_scheme"])] if not ep.empty else ep
            err_it = set(sub[sub["ok"]]["iter"].dropna().astype(int)) if not sub.empty else set()
            if e_it & err_it:
                continue
            etiqueta = "%s/%s/%s/K=%s/%s" % (kn, c["route"], c["size"], c["K"], c["comp_scheme"])
            en_energia = sub[sub["iter"].isin(e_it)] if not sub.empty else sub
            # Exencion fisica: en los operadores AMPLIFICANTES (GEMM c*H con factor sqrt(N),
            # Convolucion x2, Stencil operador "stress") la referencia FP64 o la solucion de 16
            # bits desbordan antes de los iters energeticos. En el difusivo alpha=3/16
            # (contractivo) eso NO se exime: seria un fallo real.
            amplificante = kn in ("gemm", "conv") or _es_stress(data, kn, c)
            if amplificante and en_energia is not None and not en_energia.empty and \
                    en_energia["motivo"].isin(["reference_non_finite", "solution_non_finite"]).any():
                causa = "referencia FP64" if (en_energia["motivo"] == "reference_non_finite").any() else "solucion de 16 bits"
                exentas.append(etiqueta + "  [%s desborda a los iters energeticos %s: operador amplificante]" % (causa, sorted(e_it)))
            elif sub.empty or not sub["ok"].any():
                vacias.append(etiqueta + "  [sin error evaluable en ningun iters]")
            else:
                vacias.append(etiqueta + "  [error medido a iters %s pero energia fiable solo a %s]" % (sorted(err_it)[:4], sorted(e_it)))
    if total == 0:
        r.set("N/A", "sin configuraciones con energia fiable y error medido en el mismo conjunto de resultados")
    else:
        fallan = vacias if args.a2_policy == "explained" else vacias + exentas
        r.set("FAIL" if fallan else "PASS",
              "%d configuraciones evaluadas; %d sin interseccion (%d exentas: desbordamiento de la referencia FP64 o de "
              "la solucion de 16 bits en un operador amplificante; politica=%s)" % (total, len(vacias) + len(exentas), len(exentas), args.a2_policy),
              vacias + ["(exenta) " + x for x in exentas])
    results.append(r)

    # ---- A3 -------------------------------------------------------------
    r = Res("A3", "GEMM N=8192: >= 1 ruta con solucion finita y error evaluable en el horizonte numerico (iter <= %d)" % NUMERIC_MAX_ITER)
    d0 = data["gemm"]["D"] if "gemm" in data else None
    if d0 is None or d0.empty or not ((d0["size"] == 8192) & (d0["iter"] <= NUMERIC_MAX_ITER)).any():
        r.set("N/A", "sin drift numerico de GEMM N=8192 en estos resultados")
    else:
        d = d0[(d0["size"] == 8192) & (d0["iter"] <= NUMERIC_MAX_ITER)]
        d = d[(~d["route"].isin(EXACT_REFERENCES)) & (d["ev"] == 1) & (d["sf"] == 1) & (d["rel"] > 0)]
        rutas = sorted(d["route"].unique())
        r.set("PASS" if rutas else "FAIL", "rutas finitas y evaluables en N=8192: %s" % (rutas or "ninguna"))
    results.append(r)

    # ---- A4 -------------------------------------------------------------
    r = Res("A4", "0 filas con rel_l2 == 0.0 en rutas que no son la referencia exacta")
    ofensores, n_ev = [], 0
    for kn, k in data.items():
        ep = error_points(kn, k)
        if ep.empty:
            continue
        n_ev += len(ep)
        malas = ep[(~ep["is_reference"]) & (ep["rel"] == 0.0)]
        if len(malas):
            g = malas.groupby(["route", "size"]).size().reset_index(name="n")
            ofensores += ["%s/%s/%s: %d filas con rel_l2 == 0.0" % (kn, x["route"], x["size"], x["n"]) for _, x in g.iterrows()]
    r.set("FAIL" if ofensores else ("PASS" if n_ev else "N/A"), "%d puntos de error revisados" % n_ev, ofensores)
    results.append(r)

    # ---- A5 -------------------------------------------------------------
    r = Res("A5", "0 filas duplicadas exactas tras dedupe por (job_id,kernel,route,size,iters,comp_scheme,anchor_every,replica)")
    antes = despues = conflictos = 0
    detalle = []
    key = ["job_id", "kernel", "route", "size", "iters", "comp_scheme", "K", "replica"]
    for kn, k in data.items():
        f = flat_rows(kn, k)
        if f.empty:
            continue
        f = f.assign(t=f["t"].round(9))
        n0 = len(f)
        exact = f.drop_duplicates()
        n1 = len(exact)
        uniq = exact.drop_duplicates(subset=key)
        antes += n0
        despues += len(uniq)
        conf = n1 - len(uniq)
        conflictos += conf
        detalle.append("%s: %d filas -> %d tras dedupe exacto -> %d claves unicas (%d claves con valores distintos = seudo-replicas)"
                       % (kn, n0, n1, len(uniq), conf))
        if len(exact.drop_duplicates(subset=key)) != len(uniq):
            pass
    quedan = 0   # por construccion tras drop_duplicates no quedan exactas
    r.set("PASS" if antes and quedan == 0 else "N/A",
          "antes=%d despues=%d; claves con mediciones distintas (seudo-replicas, no independientes)=%d" % (antes, despues, conflictos),
          detalle)
    results.append(r)

    # ---- A6 -------------------------------------------------------------
    r = Res("A6", "fraccion window_reliable=1 >= %.2f en los pases energeticos" % args.reliable_min)
    tot = rel = 0
    detalle = []
    for kn, k in data.items():
        er = energy_rows(kn, k)
        if er.empty:
            continue
        er = er[(er["dedicated"]) & (er["device"] == "gpu")]
        if er.empty:
            continue
        n, ok = len(er), int((er["wr"] == 1).sum())
        tot, rel = tot + n, rel + ok
        detalle.append("%s: %d/%d = %.3f" % (kn, ok, n, ok / n))
    frac = rel / tot if tot else float("nan")
    r.set("N/A" if not tot else ("PASS" if frac >= args.reliable_min else "FAIL"),
          "%d/%d = %.3f" % (rel, tot, frac) if tot else "sin pases energeticos dedicados", detalle)
    results.append(r)

    # ---- A7 -------------------------------------------------------------
    r = Res("A7", "0 filas con device=cpu y gpu_valid=1")
    malas, n = [], 0
    for kn, k in data.items():
        for nombre, df in (("summary", k["S"]), ("energy", k["E"])):
            if df is None or len(df) == 0:
                continue
            n += len(df)
            m = df[(df["device"] == "cpu") & (df["gpu_valid"].astype(str) == "1")]
            if len(m):
                malas.append("%s/%s: %d filas cpu con gpu_valid=1 (rutas %s)" % (kn, nombre, len(m), sorted(m["route"].unique())))
    r.set("FAIL" if malas else ("PASS" if n else "N/A"), "%d filas revisadas" % n, malas)
    results.append(r)

    # ---- A8 -------------------------------------------------------------
    r = Res("A8", "matriz Stencil alpha=3/16 completa, sin celdas vacias")
    vacias = []
    dif = None
    if args.mode == "job":
        r.set("N/A", "la matriz completa se evalua sobre la union de todos los jobs (--mode campaign)")
        results.append(r)
        return results
    if args.expect_schemes and "spatial" not in args.expect_schemes.split(","):
        r.set("N/A", "este job no corre el esquema spatial (las rutas WMMA_*_SP de la matriz salen de la otra invocacion)")
        results.append(r)
        return results
    if "stencil" in data:
        s = data["stencil"]["S"]
        if s is not None and not s.empty:
            dif = s[(s["op_mode"] == "diffusive") & ((s["alpha"] - A8_ALPHA).abs() < 1e-9)]
    if dif is None or dif.empty:
        r.set("N/A", "no hay filas Stencil con op_mode=diffusive y alpha=0.1875 en estos resultados")
    else:
        e = data["stencil"]["E"]
        for route in A8_ROUTES:
            for size in A8_SIZES:
                win = ENERGY_ITERS["stencil"][size]
                for K in A8_K:
                    causa = []
                    t_ok = ((dif["route"] == route) & (dif["size"] == size) & (dif["K"] == K) & dif["t"].notna()).any()
                    if not t_ok:
                        causa.append("sin T")
                    if e is None or e.empty:
                        causa.append("sin energia")
                    else:
                        e_ok = ((e["route"] == route) & (e["size"] == size) & (e["K"] == K) & (e["iters"] == win) & (e["wr"] == 1)).any()
                        if not e_ok:
                            causa.append("sin energia fiable a %d iters" % win)
                    if route not in EXACT_REFERENCES and route != "GPU_FP32":
                        err_ok = ((dif["route"] == route) & (dif["size"] == size) & (dif["K"] == K) & (dif["ev"] == 1)
                                  & (dif["rel"] > 0) & (dif["iters"] == win)).any()
                        if not err_ok:
                            causa.append("sin error evaluable a %d iters" % win)
                    if causa:
                        vacias.append("%s/%d^2/K=%d: %s" % (route, size, K, ", ".join(causa)))
        total = len(A8_ROUTES) * len(A8_SIZES) * len(A8_K)
        r.set("FAIL" if vacias else "PASS", "%d/%d celdas completas" % (total - len(vacias), total), vacias)
    results.append(r)
    return results


# ---------------------------------------------------------------------------
def matriz_cobertura(data):
    filas = []
    for kn, k in data.items():
        f = flat_rows(kn, k)
        er = energy_rows(kn, k)
        ep = error_points(kn, k)
        if f.empty:
            continue
        g = f.groupby(["route", "size", "K", "comp_scheme", "iters"], dropna=False)
        for (route, size, K, comp, iters), sub in g:
            e_ok = False
            if not er.empty:
                e_ok = bool(((er["route"] == route) & (er["size"] == size) & (er["K"] == K) & (er["comp_scheme"] == comp)
                             & (er["iters"] == iters) & (er["wr"] == 1)).any())
            err_ok = False
            if not ep.empty:
                err_ok = bool(((ep["route"] == route) & (ep["size"] == size) & (ep["K"] == K) & (ep["comp_scheme"] == comp)
                               & (ep["iter"] == iters) & ep["ok"]).any())
            ops = sub[["op_mode", "alpha"]].drop_duplicates()
            op = ";".join("%s%s" % (a, "" if pd.isna(b) else "(alpha=%g)" % b) for a, b in ops.itertuples(index=False))
            filas.append(dict(kernel=kn, route=route, size=size, K=K, comp_scheme=comp, iters=iters,
                              T_disponible=bool(sub["t"].notna().any()), E_fiable=e_ok, error_evaluable_finito=err_ok, operador=op,
                              n_filas=len(sub)))
    return pd.DataFrame(filas)


def escribir_reporte(path, args, data, results, matriz):
    L = ["# audit_report.md -- gate de cobertura", "",
         "- fecha: %s" % datetime.now().isoformat(timespec="seconds"),
         "- modo: `%s`; politica A2: `%s`" % (args.mode, args.a2_policy),
         "- directorios: %s" % ", ".join("`%s`" % d for d in args.results_dir),
         "- kernels con datos: %s" % ", ".join(data) if data else "- kernels con datos: ninguno", "",
         "## Asserts", "", "| assert | estado | detalle |", "|---|---|---|"]
    for r in results:
        L.append("| %s %s | **%s** | %s |" % (r.code, r.title, r.status, r.detail.replace("|", "/")))
    L += ["", "## Detalle por assert (celdas/filas que lo incumplen, max 40)", ""]
    for r in results:
        if r.offenders:
            L.append("### %s -- %s" % (r.code, r.status))
            L += ["- " + o for o in r.offenders[:40]]
            if len(r.offenders) > 40:
                L.append("- ... y %d mas (ver coverage_matrix.csv)" % (len(r.offenders) - 40))
            L.append("")
    # evidencia CPU_FP64
    L += ["## CPU_FP64 -- invocaciones por celda (nx,ny,iters)", ""]
    if "stencil" in data and data["stencil"]["E"] is not None and not data["stencil"]["E"].empty:
        e = data["stencil"]["E"]
        c = e[e["route"] == "CPU_FP64"].groupby(["size", "iters"]).size()
        if len(c):
            L.append("invocaciones=%d en %d celdas; maximo por celda=%d; razon max=%.2f (regla: <= 1)"
                     % (c.sum(), len(c), c.max(), c.max()))
        else:
            L.append("sin filas CPU_FP64")
    else:
        L.append("sin datos de energia de Stencil")
    L += ["", "## Matriz de cobertura (resumen por kernel x comp_scheme)", ""]
    if matriz.empty:
        L.append("(vacia)")
    else:
        res = matriz.groupby(["kernel", "comp_scheme"]).agg(
            configuraciones=("route", "size"), con_T=("T_disponible", "sum"), con_E_fiable=("E_fiable", "sum"),
            con_error_ok=("error_evaluable_finito", "sum")).reset_index()
        L.append("| kernel | comp_scheme | configuraciones | con T | con E fiable | con error evaluable |")
        L.append("|---|---|---|---|---|---|")
        for x in res.itertuples(index=False):
            L.append("| %s | %s | %d | %d | %d | %d |" % (x.kernel, x.comp_scheme, x.configuraciones, x.con_T, x.con_E_fiable, x.con_error_ok))
    L += ["", "Matriz completa: `coverage_matrix.csv` (una fila por kernel/ruta/tamano/K/comp_scheme/iters)."]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Gate de cobertura de la campana corregida (A1-A8).")
    ap.add_argument("--results-dir", nargs="+", required=True, help="directorio(s) con los CSV extraidos")
    ap.add_argument("--recursive", action="store_true", help="incluir subdirectorios (p. ej. variabilidad/r*/)")
    ap.add_argument("--kernels", nargs="+", default=["gemm", "conv", "stencil"], choices=["gemm", "conv", "stencil"])
    ap.add_argument("--out", default="audit_report.md")
    ap.add_argument("--mode", choices=["job", "campaign"], default="job",
                    help="job: los N/A no fallan; campaign: la campana completa no admite N/A")
    ap.add_argument("--a2-policy", choices=["explained", "strict"], default="explained",
                    help="explained: una configuracion sin interseccion cuya causa es que la referencia FP64 desborda "
                         "a los iters energeticos se reporta pero no falla; strict: falla igual")
    ap.add_argument("--reliable-min", type=float, default=RELIABLE_MIN)
    ap.add_argument("--expect-schemes", default="",
                    help="esquemas de compensacion de Stencil que ESTE job debe traer, coma-separados "
                         "(p. ej. 'none,kahan_local' para la invocacion SPATIAL_COMP=off); vacio = los tres")
    args = ap.parse_args()

    data = {}
    for kn in args.kernels:
        k = load_stencil(args.results_dir, args.recursive) if kn == "stencil" else load_chained(kn, args.results_dir, args.recursive)
        if k is not None:
            data[kn] = k
    if not data:
        print("audit_coverage: no encontre CSV en %s" % args.results_dir, file=sys.stderr)
        sys.exit(2)

    results = run_asserts(data, args)
    matriz = matriz_cobertura(data)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    matriz.to_csv(os.path.join(out_dir, "coverage_matrix.csv"), index=False)
    escribir_reporte(args.out, args, data, results, matriz)

    for r in results:
        print("%s %-4s %s" % (r.code, r.status, r.detail[:150]))
    fallan = [r for r in results if r.status == "FAIL" or (args.mode == "campaign" and r.status == "N/A")]
    print("audit_coverage: %s -> %s" % (args.out, "FALLO (%s)" % ", ".join(r.code for r in fallan) if fallan else "OK"))
    sys.exit(1 if fallan else 0)


if __name__ == "__main__":
    main()
