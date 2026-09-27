#!/usr/bin/env python3
"""Figura 4 -- speedup vs GPU_FP64 y throughput por tamano.

speedup = t_iter(GPU_FP64) / t_iter(config), mismo kernel, tamano e iters
(pase numerico, h: GEMM/Conv=40, Stencil=50). Sin speedup_cpu ni rutas CPU.
Unidad de trabajo: GFLOP/s (GEMM, Conv) y Mcell-updates/s (Stencil).
Configuraciones no finitas: marcador hueco, NO respaldan afirmaciones de speedup.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CONFIG, FORMAT_COLOR, K_MARKER, KERNEL_LABEL, SIZE_LABEL, TAB_DIR, foot, horizon,  # noqa: E402
                    load_canonical, save_fig)


def table() -> pd.DataFrame:
    out = []
    for kernel in ("gemm", "conv", "stencil"):
        df = load_canonical(kernel)
        h = horizon(kernel)
        x = df[(df.iters_num == h) & (df.device == "GPU")].copy()
        ref = x[x.route == "GPU_FP64"].set_index("size")["t_iter_ms"]
        x["t_ref_fp64_ms"] = x["size"].map(ref)
        x["speedup_vs_gpu_fp64"] = x["t_ref_fp64_ms"] / x["t_iter_ms"]
        if kernel == "stencil":
            x["rate"] = (x["size"] - 2) ** 2 / (x["t_iter_ms"] * 1e3)   # Mcell-updates/s
            x["rate_unit"] = "Mcell-updates/s"
        else:
            x["rate"], x["rate_unit"] = x["gflops"], "GFLOP/s"
        x["horizon"] = h
        x["finite"] = x["solution_finite"].fillna(True).astype(bool)
        out.append(x[["kernel", "size", "route", "format", "compensation", "K_efectivo", "horizon", "t_iter_ms", "n_raw",
                      "t_ref_fp64_ms", "speedup_vs_gpu_fp64", "rate", "rate_unit", "finite", "operator", "exclusion_reason"]])
    return pd.concat(out, ignore_index=True)


def plot(t: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for c, kernel in enumerate(("gemm", "conv", "stencil")):
        d = t[(t.kernel == kernel) & (t.route != "GPU_FP64")]
        for (route, K), g in d.groupby(["route", "K_efectivo"]):
            g = g.sort_values("size")
            fmt = g["format"].iloc[0]
            comp = g["compensation"].iloc[0]
            ls = "-" if comp in ("none", "spatial") else "--"
            marker = K_MARKER[int(K)]
            for r, col in ((0, "speedup_vs_gpu_fp64"), (1, "rate")):
                ax = axes[r, c]
                gf = g[g.finite]
                ax.plot(gf["size"], gf[col], ls=ls, marker=marker, color=FORMAT_COLOR[fmt], lw=1.4, ms=6)
                gn = g[~g.finite]
                ax.plot(gn["size"], gn[col], ls="", marker=marker, mfc="none", color="#777777", ms=7, mew=1.2)
        for r in (0, 1):
            ax = axes[r, c]
            ax.set_xscale("log", base=2)
            sizes = sorted(d["size"].unique())
            ax.set_xticks(sizes, [str(s) for s in sizes])
            ax.set_xlabel(SIZE_LABEL[kernel])
        axes[0, c].axhline(1.0, color="k", lw=.8, ls="--")
        axes[0, c].set_ylabel("speedup = t(GPU_FP64) / t(config)")
        axes[1, c].set_yscale("log")
        axes[1, c].set_ylabel(d["rate_unit"].iloc[0])
        name = KERNEL_LABEL[kernel] + (f" — operador «{CONFIG['STENCIL_OPERATOR_LABEL']}»" if kernel == "stencil" else "")
        axes[0, c].set_title(f"{name} (h={horizon(kernel)})")
    handles = [Line2D([0], [0], color=FORMAT_COLOR[f], lw=3, label=f) for f in ("FP16", "BF16", "FP32")]
    handles += [Line2D([0], [0], color="k", ls="-", label="sin comp. / spatial"), Line2D([0], [0], color="k", ls="--", label="comp")]
    handles += [Line2D([0], [0], marker=K_MARKER[k], color="k", ls="", label=f"K={k}") for k in (0, 1, 5, 8, 20, 32)]
    handles += [Line2D([0], [0], marker="o", mfc="none", color="#777777", ls="", label="no finito (sin speedup útil)")]
    fig.legend(handles=handles, fontsize=8, ncol=7, loc="upper center", bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("F4 — Speedup respecto a GPU_FP64 y throughput según el tamaño", fontsize=13, y=1.0)
    caption = [
        "F4. Speedup = t_iter(GPU_FP64)/t_iter(config) en el pase numérico, mismo kernel, tamaño e iteraciones "
        "(h=40 GEMM/Conv, h=50 Stencil). Throughput: GFLOP/s (GEMM, Conv) y Mcell-updates/s=(nx−2)(ny−2)/t_iter (Stencil). Sin speedup_cpu ni rutas CPU.",
        "Marcadores huecos = configuraciones no finitas al horizonte (FP16 GEMM N=2048/8192; FP16 Conv todos los HW; FP16 Stencil): se muestran pero no sustentan speedup útil. "
        "t_iter = mediana de n_raw mediciones (ver tabla de datos); K=20 solo en GEMM/Conv, K∈{0,1,8,32} en Stencil.",
        "Operador: GEMM A=c·H (Hadamard/Sylvester); Conv no registrado en logs; Stencil = «stress» hasta que exista la campaña α=3/16.",
    ]
    fig.tight_layout()
    fig.text(.01, -.075, caption[0] + "\n" + caption[1], ha="left", va="top", fontsize=8, wrap=True)
    save_fig(fig, "F4_speedup_tamano", caption)


if __name__ == "__main__":
    t = table()
    t.to_csv(TAB_DIR / "F4_data.csv", index=False)
    plot(t)
    print(t.groupby("kernel").agg(filas=("size", "size"), no_finitas=("finite", lambda s: int((~s).sum()))))
