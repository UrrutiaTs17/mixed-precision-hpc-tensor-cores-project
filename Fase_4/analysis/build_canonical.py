#!/usr/bin/env python3
"""Paso 0 -- tabla canonica por kernel + asserts duros + audit_report.md.

Una fila por configuracion fisica (kernel,size,route,format,compensation,
K_efectivo,iters_num). NO modifica ningun CSV/log crudo (solo lectura).
Salida en CONFIG["OUT_DIR"]: canonical_{gemm,conv,stencil}.{csv,parquet*},
replicas_{gemm,conv}.csv, raw_energy_samples.csv, audit_report.md.
(*parquet solo si hay pyarrow/fastparquet; el CSV es la fuente que leen los scripts)
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONFIG, KERNEL_DIR, KEY, OUT  # noqa: E402

ROOT = CONFIG["DATA_ROOT"]
F4 = os.path.join(ROOT, "Fase_4")
LOGS = os.path.join(ROOT, "logs_holder_run")
AUDIT: list[str] = []          # lineas del audit_report
ASSERTS: list[str] = []        # resultados de asserts
KNOWN_ROUTES = {"GPU_FP64", "GPU_FP32", "CPU_FP32", "CPU_FP64", "FP16_none", "FP16_comp",
                "BF16_none", "BF16_comp", "WMMA_FP16_SP", "WMMA_BF16_SP"}


def check(cond: bool, msg: str) -> None:
    ASSERTS.append(f"{'PASS' if cond else 'FAIL'}: {msg}")
    if not cond:
        write_audit(failed=True)
        raise AssertionError(msg)


def read_csvs(pattern: str) -> tuple[pd.DataFrame, list[str]]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True), paths


def route_info(route: str) -> dict:
    if route not in KNOWN_ROUTES:
        raise AssertionError(f"ruta fuera del alcance (FP8/TF32/FP32_SP/...): {route}")
    if route == "GPU_FP64":
        return dict(format="FP64", compensation="none", device="GPU", is_reference=True)
    if route == "GPU_FP32":
        return dict(format="FP32", compensation="none", device="GPU", is_reference=True)
    if route.startswith("CPU_"):
        return dict(format=route[4:], compensation="none", device="CPU", is_reference=True)
    if route.startswith("WMMA_"):
        return dict(format=route.split("_")[1], compensation="spatial", device="GPU", is_reference=False)
    return dict(format=route[:4], compensation="comp" if route.endswith("_comp") else "none",
                device="GPU", is_reference=False)


def add_route_cols(df: pd.DataFrame) -> pd.DataFrame:
    info = pd.DataFrame([route_info(r) for r in df["route"]], index=df.index)
    return pd.concat([df, info], axis=1)


def parse_operator(kernel: str) -> str | float:
    if kernel == "gemm":
        pat, rx = "F4_GEMM*.log", r"^Operador A = (.*)$"
    elif kernel == "stencil":
        pat, rx = "F4_Stencil*.log", r"^Operador\s+: (OP_MODE=\S+ CI_MODE=\S+)"
    else:
        return np.nan
    found = set()
    coef = set()
    for p in sorted(glob.glob(os.path.join(LOGS, pat))):
        with open(p, "r", errors="replace") as fh:
            for line in fh:
                m = re.match(rx, line.strip())
                if m:
                    found.add(m.group(1).strip())
                if kernel == "stencil":
                    c = re.match(r"^Coef vecino / centro\s+: (.*)$", line.strip())
                    if c:
                        coef.add(c.group(1).strip())
    if not found:
        return np.nan
    if kernel == "gemm":   # c y lambda_por_iter dependen de N/pase: se reportan como conjunto observado
        cs = sorted({m.group(1) for f in found for m in [re.search(r"c=(\S+)", f)] if m})
        ls = sorted({m.group(1) for f in found for m in [re.search(r"lambda_por_iter=(\S+)", f)] if m})
        return f"A=c*H (Hadamard de Sylvester); c en {{{','.join(cs)}}}; lambda_por_iter en {{{','.join(ls)}}} (logs F4_GEMM*)"
    check(len(found) == 1, f"{kernel}: un unico operador en los logs F4 ({found})")
    op = next(iter(found))
    if kernel == "stencil":
        check("OP_MODE=stress" in op, "Stencil: OP_MODE=stress en TODOS los logs F4_Stencil*")
        return f"{op}; coef vecino/centro={'|'.join(sorted(coef))}"
    return op


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def collapse_tokens(*tokens: str) -> str:
    return ";".join(t for t in tokens if t)


# ----------------------------------------------------------------------------
# GEMM / Conv (mismo esquema: summary + drift)
# ----------------------------------------------------------------------------
def build_chained(kernel: str) -> pd.DataFrame:
    d_ = KERNEL_DIR[kernel]
    fn = "gemm" if kernel == "gemm" else "conv"
    s, s_paths = read_csvs(f"{F4}/{d_}/results/summary_{fn}_*.csv")
    d, d_paths = read_csvs(f"{F4}/{d_}/results/drift_{fn}_*.csv")
    s = add_route_cols(s)
    numeric = CONFIG["NUMERIC_ITERS"][kernel]
    dedicated = CONFIG["DEDICATED_ITERS"][kernel]
    AUDIT.append(f"\n### {KERNEL_DIR[kernel]}: filas crudas summary={len(s)}, drift={len(d)}")

    # --- Assert A4: window_reliable <=> t_total_s >= 0.5*gpu_segments (pre-filtro)
    s["t_total_s"] = s["t_total_ms"] / 1000.0
    formula = s["t_total_s"] >= 0.5 * s["gpu_segments"]
    rel = s["window_reliable"] == 1
    n_rel, n_unrel = int(rel.sum()), int((~rel).sum())
    ok_rel, ok_unrel = int((rel & formula).sum()), int((~rel & ~formula).sum())
    check(ok_rel == n_rel and ok_unrel == n_unrel,
          f"{kernel}: window_reliable <=> t_total_s>=0.5*gpu_segments ({ok_rel}/{n_rel} fiables, {ok_unrel}/{n_unrel} no fiables)")

    # --- Clasificar pase
    s["pass"] = np.where(s["iters"] == s["size"].map(dedicated), "energy",
                         np.where(s["iters"].isin(numeric), "numeric", "other"))
    check((s["pass"] != "other").all(), f"{kernel}: todo summary pertenece al pase numerico o al dedicado")
    s["K_efectivo"] = np.where(s["route"].str.endswith("_none") | (s["route"] == "GPU_FP64"), 0, s["anchor_every"])
    check(((s["route"].str.endswith("_none") | (s["route"] == "GPU_FP64")) <= (s["anchor_every"] == 0)).all(),
          f"{kernel}: _none y GPU_FP64 reportan anchor_every=0 en el crudo")

    # --- Filas 'fiables accidentales' fuera del pase dedicado (regla 4)
    acc = s[(s["pass"] == "numeric") & rel]
    if kernel == "gemm":
        exp = {(8192, "FP16_comp", 1, 80), (8192, "BF16_comp", 1, 80)}
        got = set(zip(acc["size"], acc["route"], acc["anchor_every"], acc["iters"]))
        check(got == exp, f"gemm: exactamente 2 filas fiables accidentales (N=8192,iters=80,K=1) excluidas -> {sorted(got)}")
    else:
        check(len(acc) == 0, f"conv: 0 filas fiables fuera del pase dedicado (n={len(acc)})")
    AUDIT.append(f"- Filas fiables accidentales fuera del pase dedicado (excluidas de energia): {len(acc)}")

    # --- Energia: solo pase dedicado, reliable=1, GPU-only
    en = s[s["pass"] == "energy"].copy()
    en["e_iter"] = en["energy_gpu_j"] / en["iters"]
    unrel_ded = en[(en["window_reliable"] != 1) | (en["energy_gpu_j"] <= 0)]
    AUDIT.append(f"- Pase dedicado: {len(en)} filas; no fiables/energia<=0: {len(unrel_ded)}")
    en_ok = en[(en["window_reliable"] == 1) & (en["energy_gpu_j"] > 0)]
    e_agg = en_ok.groupby(["size", "route", "K_efectivo"]).agg(
        t_iter_ms_energy=("t_iter_ms", "median"), energy_gpu_j=("energy_gpu_j", "median"),
        energy_gpu_j_per_iter=("e_iter", "median"), n_raw_E=("e_iter", "size"),
        iters_energy=("iters", "first")).reset_index()
    e_agg["energy_reliable"] = True
    raw_samples = en_ok[["size", "route", "K_efectivo", "iters", "t_iter_ms", "e_iter"]].copy()
    raw_samples.insert(0, "kernel", kernel)

    # --- Tiempo numerico (regla 1: pseudo-replicas -> mediana + n_raw)
    nm = s[s["pass"] == "numeric"]
    t_agg = nm.groupby(["size", "route", "K_efectivo", "iters"]).agg(
        t_iter_ms=("t_iter_ms", "median"), gflops=("gflops", "median"), n_raw=("t_iter_ms", "size")).reset_index()
    t_agg = t_agg.rename(columns={"iters": "iters_num"})

    # --- Drift (regla 2): solo checkpoints del pase numerico; colapsar identicos
    d["K_efectivo"] = np.where(d["route"].str.endswith("_none"), 0, d["anchor_every"])
    max_num = max(numeric)
    d_energy = d[d["iter"] > max_num]
    AUDIT.append(f"- Drift: {len(d_energy)} filas de checkpoints del pase de energia descartadas (rel_l2 espurio, regla 5)")
    dn = d[d["iter"] <= max_num]
    g = dn.groupby(["size", "route", "K_efectivo", "iter"])
    spread = (g["rel_l2"].max() - g["rel_l2"].min()).max()
    check(spread == 0 or spread < 1e-15, f"{kernel}: checkpoints duplicados de drift identicos (spread max={spread:.2e})")
    dd = g.agg(rel_l2=("rel_l2", "median"), rel_linf=("rel_linf", "median"),
               solution_finite=("solution_finite", "min"), n_dup=("rel_l2", "size")).reset_index()
    AUDIT.append(f"- Drift numerico: {len(dn)} filas crudas -> {len(dd)} checkpoints unicos (max duplicados={int(dd.n_dup.max())})")

    # --- Ensamblar
    can = t_agg.merge(dd.rename(columns={"iter": "iters_num"}), on=["size", "route", "K_efectivo", "iters_num"], how="left")
    can = can.merge(e_agg, on=["size", "route", "K_efectivo"], how="left")
    can = add_route_cols(can)
    can["kernel"] = kernel
    can["error_horizon"] = can["iters_num"]
    can["error_metric"] = np.where(can["is_reference"], "none(reference)", "rel_l2")
    can["first_nonfinite"] = np.nan
    can["operator"] = parse_operator(kernel)
    can["source_files"] = ";".join(os.path.relpath(p, ROOT) for p in s_paths + d_paths)

    reasons = []
    for _, r in can.iterrows():
        toks = []
        if r["route"] == "GPU_FP64":
            toks.append("reference_no_error_metric")
        elif pd.isna(r["solution_finite"]):
            toks.append("no_drift_row")
        elif r["solution_finite"] == 0:
            toks.append("non_finite")
        if pd.isna(r["energy_reliable"]):
            toks.append("no_dedicated_reliable_energy")
        if kernel == "conv":
            toks.append("operator_not_in_logs")
        reasons.append(collapse_tokens(*toks))
    can["exclusion_reason"] = reasons
    can.loc[can["solution_finite"] == 0, ["rel_l2", "rel_linf"]] = np.nan   # regla 2: NUNCA sentinel 0
    can.loc[can["route"] == "GPU_FP64", ["rel_l2", "rel_linf", "solution_finite"]] = np.nan
    can["energy_reliable"] = can["energy_reliable"].fillna(False).astype(bool)
    can["solution_finite"] = can["solution_finite"].map({1: True, 0: False, 1.0: True, 0.0: False})
    return can, raw_samples


# ----------------------------------------------------------------------------
# Stencil
# ----------------------------------------------------------------------------
def build_stencil() -> pd.DataFrame:
    kernel = "stencil"
    d_ = KERNEL_DIR[kernel]
    s, s_paths = read_csvs(f"{F4}/{d_}/results/summary_stencil_*.csv")
    e, e_paths = read_csvs(f"{F4}/{d_}/results/energy_stencil_*.csv")
    check((s["nx"] == s["ny"]).all() and (e["nx"] == e["ny"]).all(), "stencil: mallas cuadradas (nx==ny)")
    s = add_route_cols(s)
    e = add_route_cols(e)
    numeric = CONFIG["NUMERIC_ITERS"][kernel]
    dedicated = CONFIG["DEDICATED_ITERS"][kernel]
    AUDIT.append(f"\n### Stencil: filas crudas summary={len(s)}, energy={len(e)} (drift_/horizon_/store_ no se consumen: sin K / redundantes)")
    s["size"] = s["nx"]
    e["size"] = e["nx"]
    ref_like = s["route"].isin(["GPU_FP32", "GPU_FP64", "CPU_FP32", "CPU_FP64"])
    s["K_efectivo"] = np.where(ref_like, 0, s["anchor_every"])
    e["K_efectivo"] = np.where(e["route"].isin(["GPU_FP32", "GPU_FP64", "CPU_FP32", "CPU_FP64"]), 0, e["anchor_every"])
    s["pass"] = np.where(s["iters"] == s["size"].map(dedicated), "energy",
                         np.where(s["iters"].isin(numeric), "numeric", "other"))
    check((s["pass"] != "other").all(), "stencil: todo summary pertenece al pase numerico o al dedicado")
    check(np.allclose(e["energy_gpu_j_per_iter"].dropna(), (e["energy_gpu_j"] / e["iters"])[e["energy_gpu_j_per_iter"].notna()], rtol=1e-3),
          "stencil: energy_gpu_j_per_iter == energy_gpu_j/iters")

    # Energia: pase dedicado del energy_stencil, GPU-only, fiable
    ed = e[e["iters"] == e["size"].map(dedicated)].copy()
    ed_gpu = ed[ed["device"] == "GPU"]
    AUDIT.append(f"- Pase dedicado (energy_stencil): {len(ed)} filas, GPU={len(ed_gpu)}; CPU_* fuera de toda metrica GPU")
    unrel = ed_gpu[(ed_gpu["energy_window_reliable"] != 1) | (ed_gpu["energy_gpu_j"] <= 0)]
    AUDIT.append(f"- Pase dedicado GPU no fiable / energia<=0: {len(unrel)}")
    ed_ok = ed_gpu[(ed_gpu["energy_window_reliable"] == 1) & (ed_gpu["energy_gpu_j"] > 0)].copy()
    ed_ok["t_iter_ms_e"] = ed_ok["time_total_s"] / ed_ok["iters"] * 1000.0
    e_agg = ed_ok.groupby(["size", "route", "K_efectivo"]).agg(
        t_iter_ms_energy=("t_iter_ms_e", "median"), energy_gpu_j=("energy_gpu_j", "median"),
        energy_gpu_j_per_iter=("energy_gpu_j_per_iter", "median"), n_raw_E=("energy_gpu_j_per_iter", "size"),
        iters_energy=("iters", "first")).reset_index()
    e_agg["energy_reliable"] = True
    raw_samples = ed_ok[["size", "route", "K_efectivo", "iters", "t_iter_ms_e", "energy_gpu_j_per_iter"]].rename(
        columns={"t_iter_ms_e": "t_iter_ms", "energy_gpu_j_per_iter": "e_iter"})
    raw_samples.insert(0, "kernel", kernel)

    # Numerico: regla 1/3 (referencias -> K=0 unico; mediana + n_raw)
    nm = s[s["pass"] == "numeric"].copy()
    nm["err_src"] = np.where(nm["route"].str.startswith("WMMA_"), nm["rel_l2_prop"], nm["rel_l2"])
    nm["err_src_inf"] = np.where(nm["route"].str.startswith("WMMA_"), nm["rel_linf_prop"], nm["rel_linf"])
    t_agg = nm.groupby(["size", "route", "K_efectivo", "iters"]).agg(
        t_iter_ms=("t_iter_ms", "median"), gflops=("gflops", "median"), n_raw=("t_iter_ms", "size"),
        rel_l2=("err_src", "median"), rel_linf=("err_src_inf", "median"),
        first_nonfinite_run=("first_nonfinite", "max")).reset_index().rename(columns={"iters": "iters_num"})
    # first_nonfinite por K desde la corrida numerica mas larga (regla 6)
    longest = max(numeric)
    fnf = nm[nm["iters"] == longest].groupby(["size", "route", "K_efectivo"])["first_nonfinite"].max().rename("first_nonfinite").reset_index()
    can = t_agg.merge(fnf, on=["size", "route", "K_efectivo"], how="left")
    can = can.merge(e_agg, on=["size", "route", "K_efectivo"], how="left")
    can = add_route_cols(can)
    can["kernel"] = kernel
    can["error_horizon"] = can["iters_num"]
    is_wmma = can["route"].str.startswith("WMMA_")
    can["error_metric"] = np.where(is_wmma, "rel_l2_prop", np.where(can["route"].isin(["GPU_FP64", "CPU_FP64"]), "none(reference)", "rel_l2(context)"))
    fin = (can["first_nonfinite_run"] == -1) & np.isfinite(can["rel_l2"])
    can["solution_finite"] = fin
    # regla 2: no finito -> NaN, nunca 0
    can.loc[~fin, ["rel_l2", "rel_linf"]] = np.nan
    # referencias exactas (GPU_FP64/CPU_FP64): error contra si mismas -> sin metrica
    exact_ref = can["route"].isin(["GPU_FP64", "CPU_FP64"])
    can.loc[exact_ref, ["rel_l2", "rel_linf"]] = np.nan
    can.loc[exact_ref, "solution_finite"] = True
    can["energy_reliable"] = can["energy_reliable"].fillna(False).astype(bool)
    can["operator"] = parse_operator("stencil")
    can["source_files"] = ";".join(os.path.relpath(p, ROOT) for p in s_paths + e_paths)
    reasons = []
    for _, r in can.iterrows():
        toks = []
        if r["route"] in ("GPU_FP64", "CPU_FP64"):
            toks.append("reference_no_error_metric")
        elif not r["solution_finite"]:
            toks.append("non_finite")
        elif r["error_metric"] == "rel_l2(context)":
            toks.append("reference_without_rel_l2_prop_context_only")
        if r["device"] == "CPU":
            toks.append("cpu_route_out_of_gpu_metrics")
        elif not r["energy_reliable"]:
            toks.append("no_dedicated_reliable_energy")
        reasons.append(collapse_tokens(*toks))
    can["exclusion_reason"] = reasons
    # consistencia first_nonfinite de la propia corrida vs corrida mas larga
    own = can[(can["first_nonfinite_run"] != -1) & can["first_nonfinite"].notna()]
    check((own["first_nonfinite_run"] == own["first_nonfinite"]).all(), "stencil: first_nonfinite de cada corrida coincide con el de la corrida mas larga")
    return can, raw_samples


# ----------------------------------------------------------------------------
# Replicas r1..r8 (F3)
# ----------------------------------------------------------------------------
def build_replicas(kernel: str) -> pd.DataFrame:
    d_ = KERNEL_DIR[kernel]
    fn = "gemm" if kernel == "gemm" else "conv"
    s, _ = read_csvs(f"{F4}/{d_}/results/variabilidad/r*/summary_{fn}_*.csv")
    d, _ = read_csvs(f"{F4}/{d_}/results/variabilidad/r*/drift_{fn}_*.csv")
    h = CONFIG["H_CHAINED"]
    s = s[(s["iters"] == h) & s["route"].str.endswith("_comp")].copy()
    s["rep"] = s["job_id"].str.extract(r"-r(\d+)$")[0].astype(int)
    check(s["rep"].nunique() == 8, f"{kernel}: 8 replicas rN en variabilidad")
    check(s.groupby(["rep", "route", "anchor_every"]).size().max() == 1, f"{kernel}: 1 medicion por (rN,ruta,K) a h={h} (n_raw=1)")
    d = d[(d["iter"] == h) & d["route"].str.endswith("_comp")]
    g = d.groupby(["route", "anchor_every"]).agg(rel_l2=("rel_l2", "median"), nuniq=("rel_l2", "nunique"),
                                                 solution_finite=("solution_finite", "min")).reset_index()
    check((g["nuniq"] == 1).all(), f"{kernel}: error determinista identico en las 8 replicas a h={h}")
    out = s[["rep", "route", "anchor_every", "t_iter_ms", "iters"]].merge(
        g[["route", "anchor_every", "rel_l2", "solution_finite"]], on=["route", "anchor_every"], how="left")
    out["format"] = out["route"].str[:4]
    out["kernel"] = kernel
    out["size"] = s["size"].iloc[0]
    out.loc[out["solution_finite"] == 0, "rel_l2"] = np.nan
    out["exclusion_reason"] = np.where(out["solution_finite"] == 0, "non_finite", "")
    out["energy"] = np.nan   # r1..r8 sin ventanas fiables
    return out


# ----------------------------------------------------------------------------
def write_audit(failed: bool = False) -> None:
    lines = ["# audit_report.md -- tabla canonica Fase 4 (job 7145)", "",
             f"Generado por build_canonical.py. Estado: {'FALLO' if failed else 'todos los asserts pasan'}.", "",
             "## Asserts", *[f"- {a}" for a in ASSERTS], "", "## Auditoria", *AUDIT]
    (OUT / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(tables: dict[str, pd.DataFrame]) -> None:
    AUDIT.append("\n## Conteos por kernel x size x ruta (filas canonicas; todos los horizontes)")
    for k, df in tables.items():
        g = df.groupby(["size", "route"]).agg(filas=("route", "size"), finitas=("solution_finite", lambda x: int((x == True).sum())),  # noqa: E712
                                              energia_fiable=("energy_reliable", "sum")).reset_index()
        AUDIT.append(f"\n### {k}\n\n" + g.pipe(md))
        toks = df["exclusion_reason"].str.split(";").explode()
        toks = toks[toks != ""].value_counts()
        AUDIT.append(f"\nMotivos (exclusion_reason) en {k}:\n\n" + toks.to_frame("filas").reset_index().pipe(md))


def main() -> None:
    tables, raws = {}, []
    for k in ("gemm", "conv"):
        can, raw = build_chained(k)
        tables[k] = can
        raws.append(raw)
    can, raw = build_stencil()
    tables["stencil"] = can
    raws.append(raw)

    # -------- Asserts globales (falla dura)
    for k, df in tables.items():
        check(int(df.duplicated(KEY).sum()) == 0, f"{k}: unicidad de la clave fisica {KEY}")
        bad = df[(df["energy_reliable"]) & ~(df["energy_gpu_j_per_iter"] > 0)]
        check(len(bad) == 0, f"{k}: 0 filas con energy_reliable=1 y energia<=0/NaN")
        bad = df[(df["solution_finite"] == False) & df["rel_l2"].notna()]  # noqa: E712
        check(len(bad) == 0, f"{k}: 0 filas con solution_finite=0 y rel_l2 numerico (sin sentinel 0)")
        check(not (df["device"].eq("CPU") & df["energy_reliable"]).any(), f"{k}: ninguna ruta CPU_* con metrica de energia GPU")
        check(set(df["route"]) <= KNOWN_ROUTES and not df["route"].str.contains("SPATIAL|FP8|TF32").any(),
              f"{k}: sin rutas prohibidas (FP8/TF32/GPU_FP32_SP/FP32_SPATIAL)")

    # -------- F1: revalidacion GEMM a h=40
    g = tables["gemm"]
    h = CONFIG["H_CHAINED"]
    g40 = g[g["iters_num"] == h]
    n_cfg = len(g40)
    n_ref = int(g40["is_reference"].sum())
    cand = g40[~g40["is_reference"]]
    n_fin = int((cand["solution_finite"] == True).sum())  # noqa: E712
    nonfin = cand[cand["solution_finite"] == False]  # noqa: E712
    check(n_cfg == 44, f"F1 GEMM h={h}: 44 configuraciones (obtenidas {n_cfg}: {len(cand)} candidatas + {n_ref} GPU_FP64)")
    check(n_fin == 30, f"F1 GEMM h={h}: 30 finitas (obtenidas {n_fin})")
    check(len(nonfin) == 10 and set(nonfin["format"]) == {"FP16"},
          f"F1 GEMM h={h}: 10 no finitas, todas FP16 (obtenidas {len(nonfin)}; N={sorted(nonfin['size'].unique())})")
    AUDIT.append(f"\n## Revalidacion de F1 (GEMM h={h})\n- {n_cfg} configs = {len(cand)} candidatas + {n_ref} GPU_FP64; {n_fin} finitas; "
                 f"{len(nonfin)} FP16 no finitas en N={sorted(nonfin['size'].unique())}. F1 (figura congelada) no se modifica.")

    reps = {k: build_replicas(k) for k in ("gemm", "conv")}
    summarize(tables)

    # -------- Escritura
    cols = ["kernel", "size", "route", "format", "compensation", "K_efectivo", "iters_num", "t_iter_ms", "n_raw", "gflops",
            "t_iter_ms_energy", "energy_gpu_j_per_iter", "energy_reliable", "n_raw_E", "iters_energy", "rel_l2", "rel_linf",
            "solution_finite", "error_metric", "first_nonfinite", "operator", "error_horizon", "device", "is_reference",
            "source_files", "exclusion_reason"]
    parquet_ok = True
    for k, df in tables.items():
        df = df[cols].sort_values(KEY).reset_index(drop=True)
        df.to_csv(OUT / f"canonical_{k}.csv", index=False)
        try:
            df.to_parquet(OUT / f"canonical_{k}.parquet", index=False)
        except Exception as exc:  # sin pyarrow/fastparquet
            parquet_ok = False
            parquet_err = str(exc).splitlines()[0]
    if not parquet_ok:
        AUDIT.append(f"\n> Parquet NO escrito ({parquet_err}); el CSV es la fuente canonica.")
    for k, df in reps.items():
        df.to_csv(OUT / f"replicas_{k}.csv", index=False)
    pd.concat(raws, ignore_index=True).to_csv(OUT / "raw_energy_samples.csv", index=False)
    write_audit()
    print("\n".join(ASSERTS))
    print(f"canonical_* escritos en {OUT}")


if __name__ == "__main__":
    main()
