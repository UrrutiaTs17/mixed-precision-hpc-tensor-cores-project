#!/usr/bin/env python3
"""Figura 7 -- variabilidad experimental (8 replicas rN independientes).

Hueco real identificado al revisar los notebooks de Colab (ver el analisis
entregado antes de escribir este script): `replicas_gemm.csv`/`replicas_conv.csv`
ya se extraian en build_canonical.py, pero ninguna figura los usaba, y Stencil
no tenia un extractor de replicas (agregado en build_canonical.py como
build_replicas_stencil()). Esta figura llena ese hueco: dispersion (media, SD,
CV%, IC95) de t_iter_ms entre 8 corridas SLURM independientes (mismo holder,
mismo nodo A100), por kernel x formato x K. Sirve como evidencia empirica de
la seccion de metodologia "Repeticiones y control de variabilidad".

Alcance deliberado: SOLO tiempo. Los CSV de variabilidad no tienen ventanas
de energia fiables (energy_window_reliable=0 en las 3 campanas: corridas
demasiado cortas para pasar el criterio de window_reliable); el error es
determinista (identico en las 8 replicas donde es finito -- ya lo aseguran
los asserts de build_canonical.py), asi que no tiene dispersion que reportar.
No es una prueba de hipotesis (para eso esta F3, que compara log-ratios
pareados por rN); esto es una caracterizacion descriptiva de la dispersion.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONFIG, FORMAT_COLOR, K_MARKER, KERNEL_LABEL, OUT, TAB_DIR, foot, save_fig  # noqa: E402

KERNELS = ("gemm", "conv", "stencil")
K_LIST = [0, 1, 5]   # el unico K compartido por las 3 campanas de replicas


def compute() -> pd.DataFrame:
    rows = []
    for kernel in KERNELS:
        r = pd.read_csv(OUT / f"replicas_{kernel}.csv")
        r = r[r["anchor_every"].isin(K_LIST)]
        for (fmt, K), g in r.groupby(["format", "anchor_every"]):
            t = g["t_iter_ms"].to_numpy()
            n = len(t)
            mean, sd = t.mean(), t.std(ddof=1)
            se = sd / np.sqrt(n)
            tc = stats.t.ppf(0.975, n - 1)
            rows.append(dict(
                kernel=kernel, format=fmt, K=int(K), n=n, size=int(g["size"].iloc[0]),
                horizon=CONFIG["H_CHAINED"],
                t_mean_ms=mean, t_sd_ms=sd, cv_pct=100 * sd / mean if mean else np.nan,
                ci95_lo=mean - tc * se, ci95_hi=mean + tc * se,
                rel_l2=g["rel_l2"].iloc[0], solution_finite=bool(g["solution_finite"].iloc[0] == 1),
            ))
    return pd.DataFrame(rows).sort_values(["kernel", "format", "K"]).reset_index(drop=True)


def plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, kernel in zip(axes, KERNELS):
        g = df[df.kernel == kernel]
        fmts = sorted(g["format"].unique())
        for fi, fmt in enumerate(fmts):
            gf = g[g["format"] == fmt]
            for _, row in gf.iterrows():
                x = K_LIST.index(row.K) + (fi - .5) * .32
                ax.errorbar(x, row.t_mean_ms, yerr=[[row.t_mean_ms - row.ci95_lo], [row.ci95_hi - row.t_mean_ms]],
                            fmt=K_MARKER[row.K], color=FORMAT_COLOR[fmt], capsize=4, ms=8, lw=1.6)
                txt = f"CV={row.cv_pct:.2f}%" + ("\nno finito" if not row.solution_finite else "")
                # fi=0 (primer formato) arriba del punto, fi=1 (segundo) abajo -- evita que las
                # etiquetas de los dos formatos se solapen cuando sus puntos caen muy cerca en y
                va, dy, y0 = ("bottom", 8, row.ci95_hi) if fi == 0 else ("top", -8, row.ci95_lo)
                ax.annotate(txt, (x, y0), textcoords="offset points", xytext=(0, dy),
                            ha="center", va=va, fontsize=6.3, color="#a33" if not row.solution_finite else "#333")
        ax.margins(y=0.28)
        ax.set_xlim(-0.55, len(K_LIST) - 0.45)
        ax.set_xticks(range(len(K_LIST)), [f"K={k}" for k in K_LIST])
        ax.set_xlabel("anchor_every (K)")
        sizes = sorted(g["size"].unique())
        ax.set_title(f"{KERNEL_LABEL[kernel]} (n=8 réplicas, h={CONFIG['H_CHAINED']}, tamaño={sizes[0]})", fontsize=10)
        if kernel == "gemm":
            ax.set_ylabel("t por iteración [ms] (media, IC95%)")
    fig.legend(handles=[Line2D([0], [0], marker=K_MARKER[k], color="k", ls="", label=f"K={k}") for k in K_LIST]
               + [Line2D([0], [0], color=FORMAT_COLOR[f], lw=3, label=f) for f in ("FP16", "BF16")],
               ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("F7 — Variabilidad experimental entre 8 réplicas SLURM independientes (rN)", fontsize=12, y=1.14)
    cap = [
        "F7. Media ± IC95% (t de Student, df=7) de t_iter_ms entre 8 réplicas SLURM independientes (mismo holder compartido, "
        "misma A100), por kernel×formato×K∈{0,1,5}. CV% = 100·SD/media. GEMM N=1024, Conv HW=64 (rutas *_comp, h=40); "
        "Stencil nx=ny=1024 (rutas *_SP, h=40).",
        "Solo tiempo: las réplicas no tienen ventanas de energía fiables (corridas demasiado cortas para "
        "window_reliable) y el error es determinista (idéntico en las 8 réplicas donde es finito, verificado por "
        "assert en build_canonical.py) — sin dispersión que reportar en ninguno de los dos.",
        "Stencil FP16_SP: no finito en las 8 réplicas y los 3 K a h=40 (first_nonfinite=29); el tiempo sí se reporta "
        "(la divergencia no impide medir t_iter_ms), el error no. No es una prueba de hipótesis: para el contraste "
        "pareado K vs. K=0 con corrección de Holm ver F3 (GEMM/Conv) — Stencil no tiene un F3 equivalente todavía.",
    ]
    foot(fig, cap[0] + "\n" + cap[1] + "\n" + cap[2])
    fig.tight_layout()
    save_fig(fig, "F7_variability", cap)


if __name__ == "__main__":
    d = compute()
    d.to_csv(TAB_DIR / "F7_data.csv", index=False)
    plot(d)
    print(d[["kernel", "format", "K", "n", "t_mean_ms", "cv_pct", "ci95_lo", "ci95_hi", "solution_finite"]].to_string(index=False))
