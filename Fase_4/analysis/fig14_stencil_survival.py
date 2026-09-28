#!/usr/bin/env python3
"""Figura 14 -- supervivencia numerica observada de Stencil (produccion).

Estructura vista en `fase4_graficas_holder_7145_COLAB.ipynb` (F4-05): por
malla, un marcador de "evento" (circulo relleno, en la iteracion
first_nonfinite) si la solucion diverge dentro del horizonte numerico
probado, o un marcador de "censurado/no observado" (triangulo hueco) si
sigue finita en el ultimo checkpoint probado (h=120) -- terminologia de
analisis de supervivencia: el evento no se observo DENTRO de la ventana, no
que nunca vaya a ocurrir.

Deliberadamente NO usa `horizon_stencil_7145.csv` (que sondea mas alla de
h=120): DECISIONS.md (item 3 del pipeline) excluye esa fuente de este
pipeline porque compara contra otro objeto y no trae K -- por eso BF16 (que
nunca diverge dentro de h<=120 en canonical_stencil.csv) se marca como
censurado en h=120, no con un evento inventado mas alla de lo que este
pipeline consume.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONFIG, FORMAT_COLOR, TAB_DIR, foot, load_canonical, save_fig  # noqa: E402

H_MAX = max(CONFIG["NUMERIC_ITERS"]["stencil"])  # 120: horizonte censurado


def compute() -> pd.DataFrame:
    df = load_canonical("stencil")
    sp = df[(df["compensation"] == "spatial") & (df["K_efectivo"] == 0)]
    rows = []
    for (fmt, size), g in sp.groupby(["format", "size"]):
        fnf = int(g["first_nonfinite"].iloc[0])
        event = 1 <= fnf <= H_MAX
        rows.append(dict(format=fmt, size=int(size), iteration=fnf if event else H_MAX, event=event))
    return pd.DataFrame(rows).sort_values(["size", "format"])


def plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sizes = sorted(df["size"].unique())
    xpos = {s: i for i, s in enumerate(sizes)}
    for fmt, g in df.groupby("format"):
        dx = -0.08 if fmt == df["format"].unique()[0] else 0.08
        ev = g[g["event"]]
        ce = g[~g["event"]]
        ax.scatter([xpos[s] + dx for s in ev["size"]], ev["iteration"], marker="o", s=110,
                  color=FORMAT_COLOR[fmt], edgecolor="k", linewidths=0.8, zorder=3,
                  label=f"{fmt}: evento" if len(ev) else None)
        ax.scatter([xpos[s] + dx for s in ce["size"]], ce["iteration"], marker="v", s=110,
                  facecolor="none", edgecolor=FORMAT_COLOR[fmt], linewidths=1.6, zorder=3,
                  label=f"{fmt}: censurado (≥{H_MAX})" if len(ce) else None)
    ax.set_xticks(range(len(sizes)), [str(s) for s in sizes])
    ax.set_xlabel("Malla nx = ny")
    ax.set_ylabel("Iteración del primer no finito / horizonte")
    ax.legend(fontsize=8.5, ncol=2)
    fig.suptitle("F14 — Supervivencia numérica observada de Stencil (producción, K=0)", fontsize=12)
    cap = [
        f"F14. Stencil, K=0, producción (job 7145): círculo relleno = evento (first_nonfinite, WMMA_FP16_SP diverge "
        "en iter=29 en las 3 mallas); triángulo hueco = censurado/no observado (WMMA_BF16_SP sigue finito en el "
        f"último checkpoint numérico probado, h={H_MAX}).",
        "No usa horizon_stencil_*.csv (sondeo más allá de h=120): excluido de este pipeline por diseño "
        "(DECISIONS.md, no trae K); BF16 se reporta como censurado en h=120, no con un evento fuera de esa ventana.",
    ]
    foot(fig, cap[0] + "\n" + cap[1])
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "F14_stencil_survival", cap)


if __name__ == "__main__":
    d = compute()
    d.to_csv(TAB_DIR / "F14_data.csv", index=False)
    plot(d)
    print(d.to_string(index=False))
