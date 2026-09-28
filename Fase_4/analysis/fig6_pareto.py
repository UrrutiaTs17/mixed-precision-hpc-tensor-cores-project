#!/usr/bin/env python3
"""Figura 6 -- Pareto 3D (T_iter, E_iter, log10 rel_l2) + proyecciones 2D.

Dominancia independiente por kernel x size x horizonte (nunca entre tamanos).
Solo triples con T, E fiable (pase dedicado) y error finito del MISMO config
fisico. Modo A: solo candidatos de precision reducida (FP16/BF16). Modo B: A +
referencias GPU como contexto/dominadoras (GPU_FP64 = verdad exacta, error
-inf por definicion; GPU_FP32 de Stencil con su rel_l2 medido, solo contexto).
Las referencias nunca cuentan como miembros del frente ni en la cobertura.
Las 3 proyecciones usan exactamente los mismos puntos que el 3D.
"""
from __future__ import annotations

import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CONFIG, FORMAT_COLOR, K_MARKER, KERNEL_LABEL, OUT, SIZE_LABEL, TAB_DIR, horizon,  # noqa: E402
                    isocurves, load_canonical, pareto_mask, save_fig)

FRONT_C, DOM_C = "#CC79A7", "#C8C8C8"
EXPECTED = {   # (kernel, horizon) -> {size: n_triples candidatos}
    ("gemm", 40): {1024: 10, 2048: 5, 4096: 10, 8192: 5},
    ("conv", 40): {64: 5, 128: 5, 256: 5, 512: 5},
    ("stencil", 50): {4096: 4, 8192: 4, 16384: 4},
    ("stencil", 10): {4096: 8, 8192: 8, 16384: 8},
}


def front_layers(objs: np.ndarray) -> np.ndarray:
    """Indice de frente por ordenamiento no dominado (1 = frente)."""
    layer = np.zeros(len(objs), int)
    remaining = np.arange(len(objs))
    k = 1
    while len(remaining):
        m = pareto_mask(objs[remaining])
        if not m.any():
            layer[remaining] = 0
            break
        layer[remaining[m]] = k
        remaining = remaining[~m]
        k += 1
    return layer


def op_label(df: pd.DataFrame) -> str:
    m = re.search(r"OP_MODE=(\w+)", str(df["operator"].dropna().iloc[0])) if df["operator"].notna().any() else None
    return m.group(1) if m else CONFIG["STENCIL_OPERATOR_LABEL"]


def build_points() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(OUT / "raw_energy_samples.csv")
    rng = np.random.default_rng(CONFIG["SEED"])
    allp, excl = [], []
    for (kernel, h), exp in EXPECTED.items():
        df = load_canonical(kernel)
        df = df[(df.iters_num == h) & (df.device == "GPU")].copy()
        df["is_reference"] = df["is_reference"].astype(bool)
        df["fin"] = df["solution_finite"].fillna(False).astype(bool)
        df["T_ms"], df["E_J"] = df["t_iter_ms_energy"], df["energy_gpu_j_per_iter"]
        for size, sub in df.groupby("size"):
            cand = sub[(~sub.is_reference) & sub.energy_reliable & sub.fin & (sub.rel_l2 > 0)].copy()
            assert len(cand) == exp[size], f"cobertura {kernel} h={h} size={size}: {len(cand)} != {exp[size]}"
            for _, r in sub[~sub.index.isin(cand.index) & ~sub.is_reference].iterrows():
                excl.append(dict(kernel=kernel, size=size, horizon=h, route=r.route, K=r.K_efectivo,
                                 reason=r.exclusion_reason or ("energy_unreliable" if not r.energy_reliable else "rel_l2<=0")))
            refs = sub[sub.is_reference & sub.energy_reliable].copy()
            refs["err_obj"] = np.where(refs.route == "GPU_FP64", -np.inf, np.log10(refs.rel_l2.where(refs.rel_l2 > 0)))
            refs = refs[~refs.err_obj.isna()]
            cand["err_obj"] = np.log10(cand.rel_l2)
            # ---- modo A
            objA = cand[["T_ms", "E_J", "err_obj"]].to_numpy(float)
            cand["on_front_A"] = pareto_mask(objA)
            cand["front_index_A"] = front_layers(objA)
            # ---- modo B (candidatos + referencias como dominadoras)
            both = pd.concat([cand.assign(_ref=False), refs.assign(_ref=True)], ignore_index=True)
            objB = both[["T_ms", "E_J", "err_obj"]].to_numpy(float)
            mB, lB = pareto_mask(objB), front_layers(objB)
            cand["on_front_B"] = mB[:len(cand)]
            cand["front_index_B"] = lB[:len(cand)]
            refs["on_front_B"], refs["front_index_B"] = mB[len(cand):], lB[len(cand):]
            # dominado por referencia (solo T,E,err con err de la referencia)
            def dom_by_ref(p):
                return any(np.all(q <= p) and np.any(q < p) for q in refs[["T_ms", "E_J", "err_obj"]].to_numpy(float))
            cand["dominated_by_reference"] = [dom_by_ref(p) for p in objA]
            # ---- bootstrap (solo configs con n_raw_E>=2 se perturban)
            keys = list(zip(cand.route, cand.K_efectivo))
            samples = []
            for route, K in keys:
                s = raw[(raw.kernel == kernel) & (raw["size"] == size) & (raw.route == route) & (raw.K_efectivo == K)]
                samples.append(s[["t_iter_ms", "e_iter"]].to_numpy(float))
            hits = np.zeros(len(cand))
            base = objA.copy()
            for _ in range(CONFIG["BOOTSTRAP_N"]):
                o = base.copy()
                for i, s in enumerate(samples):
                    if len(s) >= 2:
                        pick = s[rng.integers(0, len(s), len(s))]
                        o[i, 0], o[i, 1] = np.median(pick[:, 0]), np.median(pick[:, 1])
                hits += pareto_mask(o)
            cand["boot_frac_nondominated"] = hits / CONFIG["BOOTSTRAP_N"]
            cand["boot_perturbed"] = [len(s) >= 2 for s in samples]
            cand["is_ctx_reference"] = False
            refs["is_ctx_reference"] = True
            allp.append(pd.concat([cand, refs], ignore_index=True))
    pts = pd.concat(allp, ignore_index=True)
    pts["horizon"] = pts["iters_num"]
    return pts, pd.DataFrame(excl)


def marker_for(row) -> str:
    return K_MARKER["ref"] if row["is_ctx_reference"] else K_MARKER[int(row["K_efectivo"])]


def plot_group(pts: pd.DataFrame, kernel: str, h: int, mode: str) -> None:
    # Sin panel 3D: F9 (fig9_pareto2d_color.py) ya es la adecuacion 2D de
    # portada; mantener aqui un 3D "de contexto" ademas de esa adecuacion era
    # redundante (senalado por el usuario). F6 queda en 3 columnas (solo las
    # proyecciones 2D con dominancia real).
    g = pts[(pts.kernel == kernel) & (pts.horizon == h)]
    sizes = sorted(g["size"].unique())
    fig = plt.figure(figsize=(13.5, 3.6 * len(sizes) + 1))
    gs = fig.add_gridspec(len(sizes), 3)
    front_col = f"on_front_{mode}"
    lbl = op_label(load_canonical(kernel)) if kernel == "stencil" else None
    proj = [("T_ms", "E_J", "T por iteración [ms]", "E_GPU por iteración [J]"),
            ("T_ms", "rel_l2", "T por iteración [ms]", "error rel. L2 (h)"),
            ("E_J", "rel_l2", "E_GPU por iteración [J]", "error rel. L2 (h)")]
    for ri, size in enumerate(sizes):
        s = g[g["size"] == size]
        cand = s[~s.is_ctx_reference]
        refs = s[s.is_ctx_reference] if mode == "B" else s.iloc[0:0]
        for ci, (xc, yc, xl, yl) in enumerate(proj):
            ax = fig.add_subplot(gs[ri, ci])
            for _, r in cand.iterrows():
                ax.scatter(r[xc], r[yc], marker=K_MARKER[int(r.K_efectivo)], s=70 if r[front_col] else 45,
                           facecolor=FRONT_C if r[front_col] else DOM_C, edgecolor=FORMAT_COLOR[r["format"]], linewidths=1.8, zorder=3)
                if mode == "B" and r.dominated_by_reference:
                    ax.scatter(r[xc], r[yc], marker="x", s=25, color="k", zorder=4, linewidths=.8)
            for _, r in refs.iterrows():
                if yc == "rel_l2" and r.route == "GPU_FP64":
                    continue   # error exacto por definicion: no representable en escala log
                ax.scatter(r[xc], r[yc], marker="X", s=70, facecolor="none", edgecolor=FORMAT_COLOR[r["format"]], linewidths=1.6, zorder=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xl, fontsize=8)
            ax.set_ylabel(yl, fontsize=8)
            if ci == 0:
                ax.set_title(f"{SIZE_LABEL[kernel]}={size}", fontsize=10, loc="left")
                ref = pd.read_csv(TAB_DIR / "F5_isocurve_levels.csv")
                ref = ref[(ref.kernel == kernel) & (ref["size"] == size)]
                if len(ref):
                    tr = (cand["T_ms"].min() * .8, max(cand["T_ms"].max(), refs["T_ms"].max() if len(refs) else 0) * 1.25)
                    isocurves(ax, tr, ref["EDP_ref_Js"].iloc[0] * 1000.0)   # T en ms -> E = 1000*c/T_ms
    hd = [Line2D([0], [0], marker="o", ls="", mfc=FRONT_C, mec="k", label="frente (T, E, error)"),
          Line2D([0], [0], marker="o", ls="", mfc=DOM_C, mec="k", label="dominado"),
          Line2D([0], [0], color=FORMAT_COLOR["FP16"], lw=3, label="borde FP16"), Line2D([0], [0], color=FORMAT_COLOR["BF16"], lw=3, label="borde BF16")]
    hd += [Line2D([0], [0], marker=K_MARKER[k], ls="", color="k", label=f"K={k}") for k in sorted(set(int(x) for x in cand["K_efectivo"]))]
    if mode == "B":
        hd += [Line2D([0], [0], marker="X", ls="", mfc="none", mec="k", label="referencia GPU (contexto)"),
               Line2D([0], [0], marker="x", ls="", color="k", label="candidato dominado por referencia")]
    hd += [Line2D([0], [0], ls=":", color="#888888", label="isocurvas E=c/T (c=½,1,2 · EDP GPU_FP64)")]
    fig.legend(handles=hd, ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.0))
    name = f"{KERNEL_LABEL[kernel]}" + (f" — operador «{lbl}»" if lbl else "")
    fig.suptitle(f"F6 — Pareto (T, E, error) por tamaño — {name}, h={h}, modo {mode} "
                 f"({'solo precisión reducida' if mode == 'A' else 'con referencias GPU como contexto'})", fontsize=12, y=1.0)
    n_tr = ", ".join(f"{size}: {int((~g[g['size'] == size].is_ctx_reference).sum())}" for size in sizes)
    excl = pd.read_csv(TAB_DIR / "F6_exclusions.csv")
    excl = excl[(excl.kernel == kernel) & (excl.horizon == h)]
    ex_txt = "; ".join(f"{k}={len(v)}" for k, v in excl.groupby(excl["reason"].str.split(";").str[0])) or "ninguna"
    cap = [
        f"F6 ({KERNEL_LABEL[kernel]}, h={h}, modo {mode}). Dominancia sobre [T_iter, E_iter, log10 rel_l2] independiente por tamaño (nunca entre tamaños); "
        f"triples por tamaño: {n_tr}. T y E del pase dedicado (energy_reliable=1, GPU-only); error del pase numérico a h={h}"
        + (" (rel_l2_prop en WMMA)." if kernel == "stencil" else "."),
        f"Exclusiones (motivo: n): {ex_txt}. "
        + ("Modo B: las referencias GPU (GPU_FP64 = error exacto; GPU_FP32 Stencil = contexto) no cuentan como miembros; se marcan los candidatos que dominan." if mode == "B" else "Modo A: solo candidatos FP16/BF16."),
        (f"Operador: Stencil «{lbl}» (provisional hasta la campaña α=3/16); " if kernel == "stencil" else "Operador: "
         + ("GEMM A=c·H (Hadamard/Sylvester)." if kernel == "gemm" else "Conv no registrado en logs.") + " ")
        + f"Robustez bootstrap (n={CONFIG['BOOTSTRAP_N']}) solo perturba configs con n_raw_E≥2 (seudo-réplicas del mismo job, no independientes): ver F6_points.csv.",
    ]
    fig.subplots_adjust(left=.05, right=.985, top=.93, bottom=.07, wspace=.32, hspace=.5)
    fig.text(.01, -.055, cap[0] + "\n" + cap[1], ha="left", va="top", fontsize=8, wrap=True)
    save_fig(fig, f"F6_pareto_{kernel}_h{h}_modo{mode}", cap)


if __name__ == "__main__":
    pts, excl = build_points()
    excl.to_csv(TAB_DIR / "F6_exclusions.csv", index=False)
    cols = ["kernel", "size", "horizon", "route", "format", "compensation", "K_efectivo", "is_ctx_reference", "T_ms", "E_J", "rel_l2",
            "err_obj", "on_front_A", "front_index_A", "on_front_B", "front_index_B", "dominated_by_reference",
            "boot_frac_nondominated", "boot_perturbed", "n_raw_E", "operator"]
    pts[cols].to_csv(TAB_DIR / "F6_points.csv", index=False)
    for (kernel, h) in EXPECTED:
        for mode in ("A", "B"):
            plot_group(pts, kernel, h, mode)
    c = pts[~pts.is_ctx_reference].groupby(["kernel", "horizon", "size"]).agg(
        triples=("route", "size"), frente_A=("on_front_A", "sum"), frente_B=("on_front_B", "sum"))
    print(c)
