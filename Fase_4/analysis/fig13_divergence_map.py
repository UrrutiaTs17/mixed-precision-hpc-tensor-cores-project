#!/usr/bin/env python3
"""Figura 13 -- mapa de divergencia GEMM/Convolution (primera iteracion no finita).

Estructura vista en `fase4_analisis_holder_7145_colab_v2.ipynb` (Figura 3):
scatter simple, x=tamano, y=primera iteracion no finita, un panel por kernel.
`canonical_{gemm,conv}.csv` NO trae `first_nonfinite` (build_chained() lo deja
en NaN a proposito -- GEMM/Conv no tienen una columna equivalente lista, a
diferencia de Stencil); se calcula aqui, leyendo directo `drift_*.csv` crudo
(mismo glob que build_canonical.read_csvs), sin tocar el canonico.
"""
from __future__ import annotations

import glob
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONFIG, FORMAT_COLOR, KERNEL_DIR, KERNEL_LABEL, SIZE_LABEL, TAB_DIR, foot, save_fig  # noqa: E402

ROOT = CONFIG["DATA_ROOT"]
F4 = os.path.join(ROOT, "Fase_4")
KERNELS = ("gemm", "conv")


def first_nonfinite(kernel: str) -> pd.DataFrame:
    fn = "gemm" if kernel == "gemm" else "conv"
    paths = sorted(glob.glob(f"{F4}/{KERNEL_DIR[kernel]}/results/drift_{fn}_*.csv"))
    d = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    d["format"] = d["route"].str[:4]
    d["K_efectivo"] = d["anchor_every"].where(~d["route"].str.endswith("_none"), 0)
    nf = d[d["solution_finite"] == 0]
    if nf.empty:
        return pd.DataFrame(columns=["kernel", "size", "route", "format", "K_efectivo", "first_nonfinite"])
    g = nf.groupby(["size", "route", "format", "K_efectivo"])["iter"].min().rename("first_nonfinite").reset_index()
    g.insert(0, "kernel", kernel)
    return g


def plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, kernel in zip(axes, KERNELS):
        g = df[df.kernel == kernel]
        if g.empty:
            ax.text(0.5, 0.5, "sin filas no finitas en el pase numérico\n(0 divergencias observadas)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9, color="#666")
        else:
            for fmt, gg in g.groupby("format"):
                ax.scatter(gg["size"], gg["first_nonfinite"], color=FORMAT_COLOR.get(fmt, "#888"), s=90,
                          edgecolor="k", linewidths=0.8, label=fmt, zorder=3)
        ax.set_xlabel(SIZE_LABEL[kernel])
        ax.set_ylabel("Primera iteración no finita")
        ax.set_title(KERNEL_LABEL[kernel], fontsize=11)
        if not g.empty:
            ax.legend(fontsize=8)
    fig.suptitle("F13 — Mapa de divergencia: primera iteración no finita (producción, job 7145)", fontsize=12)
    cap = [
        "F13. Primera iteración con solution_finite=0 en drift_{gemm,conv}_*.csv (pase numérico, todas las K), por "
        "tamaño×formato. Un tamaño/formato ausente del panel no divergió en ninguna K dentro del pase numérico.",
        "Calculado directamente de los CSV crudos (no del canónico: build_chained() deja first_nonfinite=NaN "
        "porque GEMM/Conv no traían esta columna lista, a diferencia de Stencil).",
    ]
    foot(fig, cap[0] + "\n" + cap[1])
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "F13_divergence_map", cap)


if __name__ == "__main__":
    df = pd.concat([first_nonfinite(k) for k in KERNELS], ignore_index=True)
    df.to_csv(TAB_DIR / "F13_data.csv", index=False)
    plot(df)
    print(df.to_string(index=False) if len(df) else "F13: 0 filas no finitas en GEMM/Conv (pase numérico).")
