#!/usr/bin/env python3
"""Figura 9 -- Pareto 2D (T, E) coloreado por error, un panel por tamano.

Estructura vista en el render de 4 notebooks de Colab independientes
(`fase4_analisis_unificado`, `fase4_pareto2d_gemm_N4096_real`,
`fase4_grafica_pareto2d`, `fase4_graficas_holder_7145_COLAB_v2` F4-01a/b/c):
scatter T-E, color=rel_l2 (LogNorm), forma=ruta (precision+compensacion),
estrella=FP64 (referencia, no compite), frente Pareto marcado con linea
punteada, un panel por tamano. Es la vista "de portada" -- mas simple y mas
legible en el cuerpo del texto que la grilla densa de F6 (3 proyecciones +
3D + bootstrap), que sigue siendo la version completa para el apendice.

Reusa `tables/F6_points.csv` (ya calculado por fig6_pareto.py: mismo criterio
de dominancia, mismos puntos) en vez de recalcular -- una sola fuente de
verdad para "que esta en el frente".
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterSciNotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONFIG, KERNEL_LABEL, SIZE_LABEL, TAB_DIR, foot, horizon, save_fig  # noqa: E402

MARKER = {("FP16", "none"): "o", ("FP16", "comp"): "s", ("FP16", "spatial"): "s",
          ("BF16", "none"): "D", ("BF16", "comp"): "^", ("BF16", "spatial"): "^"}
LABEL = {("FP16", "none"): "FP16 — sin compensación", ("FP16", "comp"): "FP16 — con compensación",
          ("FP16", "spatial"): "FP16 — compensación espacial",
          ("BF16", "none"): "BF16 — sin compensación", ("BF16", "comp"): "BF16 — con compensación",
          ("BF16", "spatial"): "BF16 — compensación espacial"}


def plot_kernel(pts: pd.DataFrame, kernel: str, vmin: float, vmax: float) -> None:
    h = horizon(kernel)
    g = pts[(pts.kernel == kernel) & (pts.horizon == h)]
    sizes = sorted(g["size"].unique())
    ncols = 2 if len(sizes) > 1 else 1
    nrows = int(np.ceil(len(sizes) / ncols))
    # constrained_layout (no tight_layout): deja espacio real para el colorbar
    # compartido sin que se superponga a los paneles de la columna derecha.
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8 * ncols, 5.2 * nrows), squeeze=False, constrained_layout=True)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    sc = None
    present = set(zip(g["format"], g["compensation"])) & set(MARKER)
    for i, size in enumerate(sizes):
        ax = axes[i // ncols][i % ncols]
        s = g[g["size"] == size]
        cand = s[~s.is_ctx_reference]
        ref = s[s.is_ctx_reference & (s.route == "GPU_FP64")]
        for (fmt, comp), mk in MARKER.items():
            c = cand[(cand["format"] == fmt) & (cand["compensation"] == comp)]
            if c.empty:
                continue
            sc = ax.scatter(c["T_ms"], c["E_J"], c=c["rel_l2"].clip(lower=vmin), norm=norm, cmap="viridis",
                            marker=mk, s=80, edgecolor="k", linewidths=0.8, zorder=3)
        if len(ref):
            ax.scatter(ref["T_ms"], ref["E_J"], marker="*", s=220, color="#888888", edgecolor="k",
                       linewidths=0.8, zorder=3, label="FP64 (referencia)")
        front = cand[cand["on_front_A"]].sort_values("T_ms")
        if len(front) > 1:
            ax.plot(front["T_ms"], front["E_J"], ls="--", color="k", lw=1.2, zorder=2)
        iters_e = CONFIG["DEDICATED_ITERS"][kernel][size]
        ax.set_title(f"{SIZE_LABEL[kernel]} = {size:,}; T,E a {iters_e:,} iteraciones", fontsize=10)
        ax.set_xlabel("Tiempo por iteración (ms)")
        ax.set_ylabel("Energía GPU por iteración (J)")
    for j in range(len(sizes), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    handles = [Line2D([0], [0], marker=mk, ls="", mfc="none", mec="k", ms=9, label=LABEL[k]) for k, mk in MARKER.items() if k in present]
    handles += [Line2D([0], [0], marker="*", ls="", mfc="#888888", mec="k", ms=14, label="FP64 (referencia)"),
               Line2D([0], [0], ls="--", color="k", label="Frente Pareto (T, E, error)")]
    # Un solo artista "outside" reservado por constrained_layout (el titulo
    # como fig.suptitle() aparte competia por el mismo espacio y se superponia
    # con la leyenda) -- el titulo va como titulo de la leyenda misma.
    leg = fig.legend(handles=handles, ncol=3, fontsize=8.5, loc="outside upper center",
                     title=f"F9 — Pareto tiempo–energía–error — {KERNEL_LABEL[kernel]} (h={h})")
    leg.get_title().set_fontsize(12)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02)
        cbar.set_label("Error relativo L2 (escala log)", fontsize=11)
        # El rango de rel_l2 suele cubrir <1 decada -> el LogLocator por
        # defecto casi no coloca marcas (a veces una sola, como en el render
        # original de Colab). Se fuerzan marcas menores CON etiqueta para que
        # la barra muestre gradiente real, no un bloque casi sin numeros.
        cbar.ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        cbar.ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=12))
        cbar.ax.yaxis.set_major_formatter(LogFormatterSciNotation())
        cbar.ax.yaxis.set_minor_formatter(LogFormatterSciNotation(minor_thresholds=(np.inf, np.inf)))
        cbar.ax.tick_params(which="both", labelsize=8)
    cap = [
        f"F9. {KERNEL_LABEL[kernel]}, h={h}, producción (job 7145). Color = rel_l2 (LogNorm); forma = formato×compensación; "
        "estrella = GPU_FP64 (referencia, no compite por definición). Línea punteada = frente no dominado en (T, E, error) "
        "(modo A: solo candidatos de precisión reducida) — mismos puntos y misma dominancia que F6 (tables/F6_points.csv), "
        "vista simplificada para cuerpo del texto; F6 trae la versión completa (3 proyecciones + 3D + bootstrap) para apéndice.",
        "Solo pase de energía dedicado, energy_reliable=1, GPU-only; error del pase numérico al mismo horizonte h "
        "(no del pase de energía): a iters=24000 (N=1024/2048) hay 18 configuraciones con rel_l2=0 espurio (referencia "
        "FP64 desbordada, defecto ya diagnosticado) — evaluar el error ahí pintaría esas filas como error nulo en vez "
        "de excluirlas.",
    ]
    foot(fig, cap[0] + "\n" + cap[1])
    save_fig(fig, f"F9_pareto2d_{kernel}", cap)


if __name__ == "__main__":
    pts = pd.read_csv(TAB_DIR / "F6_points.csv")
    # Escala de color POR KERNEL (no compartida): un vmin/vmax global entre
    # gemm+conv+stencil aplanaba el contraste de cada panel contra el rango
    # de error de los otros dos kernels (p.ej. el 2e-8 de Stencil K=1
    # comprimia todo el rango de GEMM hacia el amarillo). Bug real, reportado
    # por el usuario al comparar contra el render original de Colab.
    for kernel in ("gemm", "conv", "stencil"):
        finite = pts[(pts.kernel == kernel) & ~pts.is_ctx_reference & pts["rel_l2"].notna() & (pts["rel_l2"] > 0)]["rel_l2"]
        vmin, vmax = finite.min(), finite.max()
        plot_kernel(pts, kernel, vmin, vmax)
        print(f"F9 {kernel}: rango de color rel_l2 = [{vmin:.2e}, {vmax:.2e}]")
