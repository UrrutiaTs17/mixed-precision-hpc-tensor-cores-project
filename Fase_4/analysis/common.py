"""Utilidades compartidas (paleta F1/F2, guardado, Pareto, Holm, isocurvas)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG, KERNEL_DIR  # noqa: E402

# Mismo lenguaje visual que F1/F2 (analisis_fase4/construir_figuras_fase4.py)
FORMAT_COLOR = {"FP16": "#0072B2", "BF16": "#D55E00", "FP32": "#009E73", "FP64": "#4D4D4D"}
K_MARKER = {"ref": "X", 0: "o", 1: "s", 5: "^", 8: "D", 20: "P", 32: "v"}
KERNEL_LABEL = {"gemm": "GEMM", "conv": "Convolución", "stencil": "Stencil"}
SIZE_LABEL = {"gemm": "N", "conv": "HW", "stencil": "nx=ny"}
KEY = ["kernel", "size", "route", "format", "compensation", "K_efectivo", "iters_num"]

OUT = Path(CONFIG["OUT_DIR"])
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
for _d in (OUT, FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def horizon(kernel: str, secondary: bool = False) -> int:
    if kernel == "stencil":
        return CONFIG["H_STENCIL_SECONDARY" if secondary else "H_STENCIL_PRIMARY"]
    return CONFIG["H_CHAINED"]


def load_canonical(kernel: str) -> pd.DataFrame:
    df = pd.read_csv(OUT / f"canonical_{kernel}.csv")
    for c in ("energy_reliable", "solution_finite", "is_reference"):
        df[c] = df[c].map({True: True, False: False, "True": True, "False": False}).astype("boolean")
    df["exclusion_reason"] = df["exclusion_reason"].fillna("")
    return df


def save_fig(fig, name: str, caption: list[str]) -> None:
    import matplotlib.pyplot as plt
    fig.savefig(FIG_DIR / f"{name}.png", dpi=CONFIG["DPI"], bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    (FIG_DIR / f"{name}.caption.txt").write_text("\n".join(caption) + "\n", encoding="utf-8")


def foot(fig, text: str) -> None:
    fig.text(.01, -.02, text, ha="left", va="top", fontsize=8, wrap=True)


def pareto_mask(objs: np.ndarray) -> np.ndarray:
    """True = no dominado (minimizar todas las columnas). Filas con NaN -> False."""
    objs = np.asarray(objs, float)
    ok = ~np.isnan(objs).any(axis=1) & ~np.isposinf(objs).any(axis=1)
    mask = np.zeros(len(objs), bool)
    for i in np.where(ok)[0]:
        p = objs[i]
        dominated = False
        for j in np.where(ok)[0]:
            if j != i and np.all(objs[j] <= p) and np.any(objs[j] < p):
                dominated = True
                break
        mask[i] = not dominated
    return mask


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def isocurves(ax, t_range, edp_ref: float, mults=None, color="#888888"):
    """Isocurvas E = c/T (c = mult*EDP_ref) para la proyeccion T-E."""
    mults = mults or CONFIG["ISOCURVE_MULTIPLIERS"]
    t = np.geomspace(*t_range, 100)
    for m in mults:
        ax.plot(t, m * edp_ref / t, ls=":", lw=1, color=color, zorder=0)
        ax.annotate(f"{m:g}·EDP$_{{FP64}}$", xy=(t[-1], m * edp_ref / t[-1]), fontsize=6,
                    color=color, ha="right", va="bottom")
