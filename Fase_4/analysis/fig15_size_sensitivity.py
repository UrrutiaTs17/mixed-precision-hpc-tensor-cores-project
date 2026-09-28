#!/usr/bin/env python3
"""Figura 15 -- sensibilidad descriptiva al tamano del problema.

Estructura vista en `fase4_graficas_holder_7145_COLAB_v2.ipynb` (F4-05):
t_iter_ms vs tamano, escala log-log, una linea por ruta representativa, un
panel por kernel. Puramente descriptiva (perfil de escalado), sin ajuste ni
prueba -- complementa a F4 (que muestra speedup vs GPU_FP64 y throughput,
no la curva de tiempo cruda).
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FORMAT_COLOR, KERNEL_LABEL, SIZE_LABEL, TAB_DIR, foot, horizon, load_canonical, save_fig  # noqa: E402

ROUTES = {
    "gemm": [("BF16_comp", "BF16", "BF16 (con compensación)"), ("FP16_comp", "FP16", "FP16 (con compensación)")],
    "conv": [("BF16_comp", "BF16", "BF16 (con compensación)"), ("FP16_comp", "FP16", "FP16 (con compensación)")],
    "stencil": [("GPU_FP32", "FP32", "GPU FP32 (clásico)"), ("WMMA_BF16_SP", "BF16", "WMMA BF16 (espacial)"),
                ("WMMA_FP16_SP", "FP16", "WMMA FP16 (espacial)")],
}


def plot() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, kernel in zip(axes, ("gemm", "conv", "stencil")):
        df = load_canonical(kernel)
        h = horizon(kernel)
        d = df[(df["iters_num"] == h) & (df["K_efectivo"] == 0)]
        for route, fmt, label in ROUTES[kernel]:
            g = d[d["route"] == route].sort_values("size")
            if g.empty:
                continue
            ax.plot(g["size"], g["t_iter_ms"], "-o", color=FORMAT_COLOR[fmt], ms=6, label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(SIZE_LABEL[kernel])
        ax.set_ylabel("Tiempo por iteración (ms)")
        ax.set_title(f"{KERNEL_LABEL[kernel]} (h={h}, K=0)", fontsize=10.5)
        ax.legend(fontsize=7.5)
    fig.suptitle("F15 — Sensibilidad descriptiva al tamaño del problema (producción, job 7145)", fontsize=12)
    cap = [
        "F15. t_iter_ms del pase numérico (K=0) vs. tamaño, escala log-log, un panel por kernel. Puramente descriptivo "
        "(perfil de escalado observado); complementa a F4 (speedup vs. GPU_FP64 y GFLOP/s), que no muestra la curva "
        "de tiempo cruda.",
        "GEMM/Conv: rutas *_comp (con compensación, K=0); Stencil: GPU_FP32 clásico y WMMA_*_SP (espacial, K=0).",
    ]
    foot(fig, cap[0] + "\n" + cap[1])
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_fig(fig, "F15_size_sensitivity", cap)


if __name__ == "__main__":
    plot()
    print("F15 escrita.")
