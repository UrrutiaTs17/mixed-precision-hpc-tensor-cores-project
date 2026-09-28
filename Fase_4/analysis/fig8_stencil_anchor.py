#!/usr/bin/env python3
"""Figura 8 -- efecto del anclaje FP64 (K) en Stencil, produccion (job 7145).

Hueco real identificado al revisar los notebooks de Colab: F3 (fig3_anchor.py)
solo cubre GEMM/Conv (K en {1,5}, replicas rN); Stencil nunca tuvo una figura
de efecto de K, aunque el barrido K={0,1,8,32} SI existe en produccion para
WMMA_FP16_SP/WMMA_BF16_SP x {4096,8192,16384}^2 (verificado en canonical_stencil.csv).

Diseno (misma idea que la "Opcion B" documentada en los Colabs de referencia,
mas defendible que un scatter tiempo-error unico): dos paneles normalizados a
K=0 por tamano -- (a) costo temporal T(K)/T(K=0), (b) efecto numerico
error(K)/error(K=0) con error = rel_l2_prop (estado propagado en precision
reducida, mismo criterio que build_stencil()/fig6_pareto.py).

Sin CI ni test de hipotesis: a diferencia de F3/F7 (replicas rN independientes),
aqui cada punto es UNA fila de produccion por (tamano,K) -- no hay repeticion
independiente que soporte un intervalo. Es lectura descriptiva, igual que el
panel de error de F3.

Solo BF16: WMMA_FP16_SP diverge antes del horizonte primario (h=50) en las 3
mallas y los 4 K -- incluido K=32, el anclaje mas frecuente del barrido, lo
que ya es en si mismo un hallazgo (el anclaje no rescata a FP16 en este
horizonte con el operador "stress"). Se documenta en vez de excluirse en
silencio: tabla F8_data.csv incluye tambien FP16 en h=10 (donde SI es finito)
como referencia secundaria.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONFIG, FORMAT_COLOR, KERNEL_LABEL, SIZE_LABEL, TAB_DIR, foot, load_canonical, save_fig  # noqa: E402

K_LIST = [0, 1, 8, 32]
K_MARKER8 = {0: "o", 1: "s", 8: "D", 32: "v"}


def compute() -> pd.DataFrame:
    df = load_canonical("stencil")
    sp = df[(df["compensation"] == "spatial")].copy()
    rows = []
    for h in (CONFIG["H_STENCIL_PRIMARY"], CONFIG["H_STENCIL_SECONDARY"]):
        hh = sp[sp["iters_num"] == h]
        for fmt in ("FP16", "BF16"):
            g = hh[hh["format"] == fmt]
            for size in sorted(g["size"].unique()):
                gs = g[g["size"] == size].set_index("K_efectivo")
                if 0 not in gs.index or gs.loc[0, "t_iter_ms"] != gs.loc[0, "t_iter_ms"]:
                    continue
                t0 = gs.loc[0, "t_iter_ms"]
                e0 = gs.loc[0, "rel_l2"]
                fin0 = bool(gs.loc[0, "solution_finite"])
                for K in K_LIST:
                    if K not in gs.index:
                        continue
                    r = gs.loc[K]
                    finite = bool(r["solution_finite"])
                    rows.append(dict(
                        format=fmt, size=int(size), horizon=h, K=K,
                        t_iter_ms=r["t_iter_ms"], t_ratio=r["t_iter_ms"] / t0 if t0 == t0 else np.nan,
                        rel_l2=r["rel_l2"], err_ratio=(r["rel_l2"] / e0) if (finite and fin0 and e0 and e0 == e0) else np.nan,
                        solution_finite=finite,
                    ))
    return pd.DataFrame(rows).sort_values(["format", "horizon", "size", "K"]).reset_index(drop=True)


def plot(df: pd.DataFrame) -> None:
    h_primary = CONFIG["H_STENCIL_PRIMARY"]
    d = df[(df.horizon == h_primary) & (df.format == "BF16")]
    sizes = sorted(d["size"].unique())
    fig, axes = plt.subplots(2, len(sizes), figsize=(4.6 * len(sizes), 8), sharex="col")
    for ci, size in enumerate(sizes):
        g = d[d["size"] == size].set_index("K")
        ax_t, ax_e = axes[0, ci], axes[1, ci]
        xs = [K_LIST.index(K) for K in g.index]
        ax_t.plot(xs, g["t_ratio"], "-o", color=FORMAT_COLOR["BF16"], ms=9, lw=1.6)
        for x, K in zip(xs, g.index):
            ax_t.annotate(f"{g.loc[K, 't_ratio']:.3f}×", (x, g.loc[K, "t_ratio"]), textcoords="offset points",
                          xytext=(0, 7), ha="center", fontsize=7.5)
        ax_e.plot(xs, g["err_ratio"], "-o", color=FORMAT_COLOR["BF16"], ms=9, lw=1.6)
        for x, K in zip(xs, g.index):
            v = g.loc[K, "err_ratio"]
            ax_e.annotate(f"{v:.2e}×" if v == v else "n/d", (x, v if v == v else 1.0), textcoords="offset points",
                          xytext=(0, 7), ha="center", fontsize=7.5)
        for ax in (ax_t, ax_e):
            ax.axhline(1.0, color="k", lw=.8, ls="--")
            ax.set_xticks(range(len(K_LIST)), [f"K={k}" for k in K_LIST])
        ax_e.set_yscale("log")
        ax_t.set_title(f"{SIZE_LABEL['stencil']}={size}", fontsize=11)
        if ci == 0:
            ax_t.set_ylabel("(a) T(K) / T(K=0)")
            ax_e.set_ylabel("(b) error(K) / error(K=0)\n(rel_l2_prop, escala log)")
        fp16 = df[(df["format"] == "FP16") & (df["horizon"] == h_primary) & (df["size"] == size)]
        n_nonfinite = int((~fp16["solution_finite"]).sum())
        if n_nonfinite:
            ax_t.annotate(f"WMMA_FP16_SP: no finito\nen los {n_nonfinite} K (h={h_primary})", (0.5, 0.05),
                          xycoords="axes fraction", ha="center", fontsize=7, color="#a33")
    fig.suptitle(f"F8 — Efecto del anclaje FP64 (K) en Stencil WMMA-BF16, h={h_primary} (producción, job 7145)", fontsize=12, y=1.0)
    cap = [
        f"F8. Stencil, formato BF16 (WMMA_BF16_SP), h={h_primary}, producción (job 7145), sin réplicas ni IC (un solo "
        "valor de producción por tamaño×K). (a) T(K)/T(K=0): costo temporal relativo. (b) error(K)/error(K=0) con "
        "error=rel_l2_prop (estado propagado en precisión reducida), escala log.",
        "WMMA_FP16_SP no se grafica: no finito en las tres mallas y los 4 K (incluido K=32) al horizonte primario "
        f"h={h_primary} — el anclaje no rescata a FP16 en este horizonte. Sí es finito a h={CONFIG['H_STENCIL_SECONDARY']} "
        "(ver F8_data.csv, columna horizon).",
        "Operador: Stencil «stress» (provisional hasta la campaña α=3/16 pre-registrada). Sin corrección de "
        "comparaciones múltiples (a diferencia de F3/F7): no hay réplicas independientes que soporten un test.",
    ]
    foot(fig, cap[0] + "\n" + cap[1] + "\n" + cap[2])
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_fig(fig, "F8_stencil_anchor", cap)


if __name__ == "__main__":
    d = compute()
    d.to_csv(TAB_DIR / "F8_data.csv", index=False)
    plot(d)
    print(d.to_string(index=False))
