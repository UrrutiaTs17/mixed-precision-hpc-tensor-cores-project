#!/usr/bin/env python3
"""Figura 11 -- forest plot del efecto de K (razon geometrica T(K)/T(K=0)).

Estructura vista en `fase4_graficas_holder_7145_COLAB.ipynb` (F4-02): forest
horizontal, fila=condicion exacta, punto=razon geometrica con IC95%, linea de
referencia en 1.0, un panel por kernel. Mismo estadistico que F3 (t pareado
sobre log-ratios, n=8 replicas rN, df=7) -- para GEMM/Conv se REUSA
`tables/F3_data.csv` en vez de recalcular; Stencil no tiene un F3 equivalente
(fig3_anchor.py solo cubre GEMM/Conv), asi que aqui se calcula esa parte con
el mismo metodo, sobre replicas_stencil.csv.
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

K_LIST = CONFIG["F3_K_LIST"]  # [1, 5]


def stencil_ratios() -> pd.DataFrame:
    r = pd.read_csv(OUT / "replicas_stencil.csv")
    rows = []
    for fmt, g in r.groupby("format"):
        t = g.pivot(index="rep", columns="anchor_every", values="t_iter_ms")
        for K in K_LIST:
            if K not in t.columns or 0 not in t.columns:
                continue
            lr = np.log(t[K] / t[0]).dropna().to_numpy()
            n = len(lr)
            if n < 2:
                continue
            m, se = lr.mean(), lr.std(ddof=1) / np.sqrt(n)
            tc = stats.t.ppf(0.975, n - 1)
            rows.append(dict(kernel="stencil", format=fmt, K=K, size=int(g["size"].iloc[0]),
                             horizon=CONFIG["H_CHAINED"], n_pairs=n,
                             geo_ratio_T=np.exp(m), ci95_lo=np.exp(m - tc * se), ci95_hi=np.exp(m + tc * se)))
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> None:
    kernels = ("gemm", "conv", "stencil")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, kernel in zip(axes, kernels):
        g = df[df.kernel == kernel].sort_values(["format", "K"])
        labels = [f"{r.format}, K={int(r.K)}/K=0" for r in g.itertuples()]
        y = np.arange(len(g))[::-1]
        ax.errorbar(g["geo_ratio_T"], y, xerr=[g["geo_ratio_T"] - g["ci95_lo"], g["ci95_hi"] - g["geo_ratio_T"]],
                    fmt="o", color="#0072B2", capsize=3, ms=7)
        ax.axvline(1.0, color="k", lw=1, ls="--")
        ax.set_yticks(y, labels, fontsize=9)
        ax.set_xlabel("Razón geométrica de tiempo T(K)/T(K=0), IC95%")
        ax.set_title(KERNEL_LABEL[kernel], fontsize=11)
        ax.set_ylim(-0.7, len(g) - 0.3)
    fig.suptitle(f"F11 — Efecto de K sobre el tiempo: razón geométrica por condición exacta (n=8 réplicas, h={CONFIG['H_CHAINED']})",
                fontsize=12)
    cap = [
        "F11. Razón geométrica T(K)/T(K=0) con IC95% (t pareado sobre log-ratios, df=7, n=8 réplicas rN independientes), "
        f"K∈{K_LIST}, h={CONFIG['H_CHAINED']}. GEMM/Conv: mismo dato que F3 (tables/F3_data.csv), vista horizontal; "
        "Stencil: mismo método aplicado aquí por primera vez sobre replicas_stencil.csv (sin equivalente F3 todavía).",
        "Sin corrección de comparaciones múltiples en este panel (para eso, ver p_holm en F3_data.csv/F11_data.csv). "
        "GEMM N=1024, Conv HW=64 (rutas *_comp); Stencil nx=ny=1024 (rutas *_SP).",
    ]
    foot(fig, cap[0] + "\n" + cap[1])
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "F11_k_forest", cap)


if __name__ == "__main__":
    f3 = pd.read_csv(TAB_DIR / "F3_data.csv")
    gc = f3[["kernel", "format", "K", "size", "horizon", "n_pairs", "geo_ratio_T", "ci95_lo", "ci95_hi"]]
    stc = stencil_ratios()
    df = pd.concat([gc, stc], ignore_index=True)
    df.to_csv(TAB_DIR / "F11_data.csv", index=False)
    plot(df)
    print(df.to_string(index=False))
