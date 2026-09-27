#!/usr/bin/env python3
"""Figura 3 -- efecto de anchor_every (K) en GEMM y Conv (r1..r8, h=40).

Bloques pareados = rN. Solo rutas *_comp, K en {0,1,5}; K=20 (n=1) fuera.
Tiempo: razon geometrica T(K)/T(0) con IC95 t (df=7) sobre log-ratios pareados,
p sin corregir y Holm (familia kernel x formato x K). Error: cociente
puntual determinista, sin IC ni tests. Sin energia.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CONFIG, FORMAT_COLOR, K_MARKER, KERNEL_LABEL, TAB_DIR, foot, holm, save_fig)  # noqa: E402


def compute() -> pd.DataFrame:
    rows = []
    for kernel, fmt in CONFIG["F3_KERNELS_FORMATS"]:
        r = pd.read_csv(os.path.join(CONFIG["OUT_DIR"], f"replicas_{kernel}.csv"))
        r = r[(r["route"] == f"{fmt}_comp") & r["anchor_every"].isin([0] + CONFIG["F3_K_LIST"])]
        t = r.pivot(index="rep", columns="anchor_every", values="t_iter_ms")
        assert t.shape[0] == 8 and not t.isna().any().any(), f"{kernel} {fmt}: se esperan 8 replicas completas"
        e = r.groupby("anchor_every")["rel_l2"].first()
        finite = bool(r["solution_finite"].iloc[0] == 1) if r["solution_finite"].notna().all() else False
        for K in CONFIG["F3_K_LIST"]:
            lr = np.log(t[K] / t[0]).to_numpy()
            n = len(lr)
            m, se = lr.mean(), lr.std(ddof=1) / np.sqrt(n)
            tc = stats.t.ppf(0.975, n - 1)
            p = stats.ttest_1samp(lr, 0.0).pvalue
            rows.append(dict(kernel=kernel, format=fmt, K=K, size=int(r["size"].iloc[0]), horizon=CONFIG["H_CHAINED"],
                             n_pairs=n, df=n - 1, geo_ratio_T=np.exp(m), ci95_lo=np.exp(m - tc * se), ci95_hi=np.exp(m + tc * se),
                             p_raw=p,
                             err_finite=finite,
                             err_ratio=(e[K] / e[0]) if finite and e[0] > 0 else np.nan,
                             err_K0=e[0] if finite else np.nan, err_K=e[K] if finite else np.nan,
                             err_note="" if finite else "non_finite (sin error asignado)"))
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_raw"].tolist())
    df["holm_family_size"] = len(df)
    return df


def plot(df: pd.DataFrame) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    groups = [(k, f) for k, f in CONFIG["F3_KERNELS_FORMATS"]]
    for gi, (kernel, fmt) in enumerate(groups):
        for j, K in enumerate(CONFIG["F3_K_LIST"]):
            x = gi + (j - .5) * .28
            r = df[(df.kernel == kernel) & (df.format == fmt) & (df.K == K)].iloc[0]
            ax.errorbar(x, r.geo_ratio_T, yerr=[[r.geo_ratio_T - r.ci95_lo], [r.ci95_hi - r.geo_ratio_T]],
                        fmt=K_MARKER[K], color=FORMAT_COLOR[fmt], capsize=4, ms=8, lw=1.6)
            ax.annotate(f"p={r.p_raw:.1e}\nHolm={r.p_holm:.1e}", (x, r.ci95_hi), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=6)
            if r.err_finite:
                ax2.plot(x, r.err_ratio, K_MARKER[K], color=FORMAT_COLOR[fmt], ms=8)
                ax2.annotate(f"{r.err_ratio:.5f}", (x, r.err_ratio), textcoords="offset points", xytext=(0, 6),
                             ha="center", fontsize=7)
            else:
                ax2.plot(x, 1.0, K_MARKER[K], mfc="none", color="#999999", ms=8)
        if not df[(df.kernel == kernel) & (df.format == fmt)].err_finite.iloc[0]:
            ax2.annotate("no finito:\nsin error asignado", (gi, 1.0), textcoords="offset points", xytext=(0, -30),
                         ha="center", fontsize=7, color="#666666")
    for a in (ax, ax2):
        a.axhline(1.0, color="k", lw=.8, ls="--")
        a.set_xticks(range(len(groups)), [f"{KERNEL_LABEL[k]}\n{f}" for k, f in groups])
    ax2.yaxis.get_major_formatter().set_useOffset(False)
    ax2.set_ylim(0.9997, 1.0002)
    ax.set(ylabel="T(K) / T(K=0)  (razón geométrica, IC95%)", title="Tiempo por iteración (pareado por rN, n=8)")
    ax2.set(ylabel="E(K) / E(K=0)  (error rel. L2, puntual)", title="Error a h=40 (determinista, sin IC)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker=K_MARKER[K], color="k", ls="", label=f"K={K}") for K in CONFIG["F3_K_LIST"]]
              + [Line2D([0], [0], color=FORMAT_COLOR[f], lw=3, label=f) for f in ("FP16", "BF16")], fontsize=8, ncol=2)
    fig.suptitle("F3 — Costo y efecto del anclaje K en rutas compensadas (GEMM N=1024, Conv HW=64)", fontsize=12, y=1.02)
    caption = [
        "F3. GEMM N=1024 y Conv HW=64, horizonte h=40; rutas *_comp, K∈{0,1,5} (K=20, n=1, excluido); n=8 bloques pareados rN. "
        "Tiempo: razón geométrica T(K)/T(0) con IC95% t (df=7) sobre log-ratios pareados; p sin corregir y Holm sobre 8 comparaciones (kernel×formato×K). "
        "Error: cociente E(K)/E(0) determinista (idéntico en las 8 réplicas), sin IC ni tests; Conv FP16 no finito a h=40: sin error asignado.",
        "Limitaciones: orden de K fijo 0→1→5 dentro de cada réplica (sin aleatorización); rN son bloques pareados, no asignaciones SLURM independientes "
        "(steps srun del mismo holder, mismo nodo A100). Sin energía: r1..r8 no tienen ventanas fiables.",
        "Operador: GEMM = A=c·H (Hadamard/Sylvester); Conv = no registrado en los logs. Exclusiones: K=20 (n=1); Conv FP16 error (no finito).",
    ]
    foot(fig, caption[0] + "\n" + caption[1])
    fig.tight_layout()
    save_fig(fig, "F3_anchor_gemm_conv", caption)


if __name__ == "__main__":
    d = compute()
    d.to_csv(TAB_DIR / "F3_data.csv", index=False)
    plot(d)
    print(d[["kernel", "format", "K", "geo_ratio_T", "ci95_lo", "ci95_hi", "p_raw", "p_holm", "err_ratio", "err_note"]].to_string(index=False))
