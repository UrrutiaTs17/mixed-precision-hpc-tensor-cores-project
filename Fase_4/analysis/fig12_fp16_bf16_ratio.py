#!/usr/bin/env python3
"""Figura 12 -- FP16 frente a BF16: razon pareada de tiempo (T_FP16/T_BF16).

Estructura vista en `fase4_graficas_holder_7145_COLAB.ipynb` (F4-03): forest
horizontal, fila=condicion (K), punto=razon geometrica T_FP16/T_BF16 con
IC95%, linea de referencia en 1.0, panel por kernel. Mismo par de replicas
rN (t pareado sobre log-ratios, df=7) que F3/F7/F10/F11, ahora comparando
FORMATO a K fijo en vez de K a formato fijo.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONFIG, KERNEL_LABEL, OUT, TAB_DIR, foot, save_fig  # noqa: E402

KERNELS = ("gemm", "conv", "stencil")
K_LIST = [0, 1, 5]


def compute() -> pd.DataFrame:
    rows = []
    for kernel in KERNELS:
        r = pd.read_csv(OUT / f"replicas_{kernel}.csv")
        for K in K_LIST:
            sub = r[r["anchor_every"] == K]
            t = sub.pivot(index="rep", columns="format", values="t_iter_ms")
            if "FP16" not in t.columns or "BF16" not in t.columns:
                continue
            lr = np.log(t["FP16"] / t["BF16"]).dropna().to_numpy()
            n = len(lr)
            if n < 2:
                continue
            m, se = lr.mean(), lr.std(ddof=1) / np.sqrt(n)
            tc = stats.t.ppf(0.975, n - 1)
            rows.append(dict(kernel=kernel, K=K, n_pairs=n, size=int(sub["size"].iloc[0]),
                             horizon=CONFIG["H_CHAINED"],
                             ratio_fp16_bf16=np.exp(m), ci95_lo=np.exp(m - tc * se), ci95_hi=np.exp(m + tc * se)))
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for ax, kernel in zip(axes, KERNELS):
        g = df[df.kernel == kernel].sort_values("K")
        y = np.arange(len(g))[::-1]
        ax.errorbar(g["ratio_fp16_bf16"], y, xerr=[g["ratio_fp16_bf16"] - g["ci95_lo"], g["ci95_hi"] - g["ratio_fp16_bf16"]],
                    fmt="o", color="#D55E00", capsize=3, ms=7)
        ax.axvline(1.0, color="k", lw=1, ls="--")
        ax.set_yticks(y, [f"K={int(k)}" for k in g["K"]], fontsize=9)
        ax.set_xlabel("T(FP16) / T(BF16), IC95%")
        ax.set_title(KERNEL_LABEL[kernel], fontsize=11)
        ax.set_ylim(-0.7, len(g) - 0.3)
    fig.suptitle(f"F12 — FP16 frente a BF16: razón pareada de tiempo (n=8 réplicas, h={CONFIG['H_CHAINED']})", fontsize=12)
    cap = [
        f"F12. Razón geométrica T(FP16)/T(BF16) con IC95% (t pareado sobre log-ratios, df=7, n=8 réplicas rN), por K, "
        f"h={CONFIG['H_CHAINED']}. GEMM N=1024, Conv HW=64 (rutas *_comp); Stencil nx=ny=1024 (rutas *_SP).",
        "Razón < 1: FP16 más rápido que BF16 a ese K. Mismas 8 réplicas que F3/F7/F10/F11; sin corrección de "
        "comparaciones múltiples (3 puntos por kernel).",
    ]
    foot(fig, cap[0] + "\n" + cap[1])
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_fig(fig, "F12_fp16_bf16_ratio", cap)


if __name__ == "__main__":
    d = compute()
    d.to_csv(TAB_DIR / "F12_data.csv", index=False)
    plot(d)
    print(d.to_string(index=False))
