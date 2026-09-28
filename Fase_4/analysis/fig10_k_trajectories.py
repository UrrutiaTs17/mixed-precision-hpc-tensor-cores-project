#!/usr/bin/env python3
"""Figura 10 -- trayectorias pareadas de K, replicas individuales (r1..r8).

Estructura vista en `fase4_graficas_holder_7145_COLAB.ipynb` (F4-01) y
`..._v2.ipynb` (F4_02a/b/c): una linea por replica (no agregada), x=K,
y=t_iter_ms, panel por kernel x formato. Complementa a F7 (que agrega a
media+IC95): aqui se ve el detalle de que las 8 lineas son casi paralelas
(bajo CV%, ya cuantificado en F7) en vez de solo el resumen.

Reusa replicas_{gemm,conv,stencil}.csv (build_canonical.py), sin releer CSV
crudos.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONFIG, KERNEL_LABEL, OUT, foot, save_fig  # noqa: E402

KERNELS = ("gemm", "conv", "stencil")
K_LIST = [0, 1, 5]
REP_COLORS = plt.get_cmap("tab10").colors  # r1..r8 -> 8 colores estables


def plot() -> None:
    data = {k: pd.read_csv(OUT / f"replicas_{k}.csv") for k in KERNELS}
    fmts = {k: sorted(data[k]["format"].unique()) for k in KERNELS}
    ncols = max(len(f) for f in fmts.values())
    fig, axes = plt.subplots(len(KERNELS), ncols, figsize=(4.6 * ncols, 3.6 * len(KERNELS)), squeeze=False)
    for ri, kernel in enumerate(KERNELS):
        d = data[kernel][data[kernel]["anchor_every"].isin(K_LIST)]
        for ci in range(ncols):
            ax = axes[ri][ci]
            if ci >= len(fmts[kernel]):
                ax.set_visible(False)
                continue
            fmt = fmts[kernel][ci]
            g = d[d["format"] == fmt]
            for rep in range(1, 9):
                gr = g[g["rep"] == rep].sort_values("anchor_every")
                if gr.empty:
                    continue
                ax.plot([K_LIST.index(k) for k in gr["anchor_every"]], gr["t_iter_ms"], "-o",
                        color=REP_COLORS[(rep - 1) % 10], ms=4, lw=1.1, alpha=0.85)
            ax.set_xticks(range(len(K_LIST)), [str(k) for k in K_LIST])
            size = int(g["size"].iloc[0]) if len(g) else None
            ax.set_title(f"{KERNEL_LABEL[kernel]} — {fmt} (tamaño={size})", fontsize=9.5)
            if ci == 0:
                ax.set_ylabel("t por iteración (ms)")
            if ri == len(KERNELS) - 1:
                ax.set_xlabel("K (anchor_every)")
    handles = [plt.Line2D([0], [0], color=REP_COLORS[i], marker="o", ms=4, lw=1.1, label=f"r{i+1}") for i in range(8)]
    fig.legend(handles=handles, ncol=8, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.04), title="Réplica")
    fig.suptitle(f"F10 — Trayectorias pareadas del tiempo frente a K, réplicas individuales (h={CONFIG['H_CHAINED']})",
                fontsize=12, y=1.1)
    cap = [
        f"F10. t_iter_ms de cada una de las 8 réplicas SLURM independientes (rN, sin agregar), por kernel×formato, "
        f"K∈{{0,1,5}}, h={CONFIG['H_CHAINED']}. GEMM N=1024, Conv HW=64 (rutas *_comp); Stencil nx=ny=1024 (rutas *_SP).",
        "Complementa a F7 (media±IC95 de las mismas 8 réplicas): aquí se ve que las líneas son casi paralelas entre sí "
        "(dispersión baja, ya cuantificada como CV% en F7), no solapadas por casualidad de escala.",
    ]
    foot(fig, cap[0] + "\n" + cap[1])
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "F10_k_trajectories", cap)


if __name__ == "__main__":
    plot()
    print("F10 escrita.")
