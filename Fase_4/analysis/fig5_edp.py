#!/usr/bin/env python3
"""Figura 5 -- EDP por iteracion (E_iter x T_iter), potencia media y filtro de tolerancia.

Solo energy_reliable=1, pase dedicado, GPU-only (energy_gpu_j). T_iter sale de la
MISMA ventana que E (t_iter_ms_energy) para que P=E/T sea coherente. Se normaliza
contra GPU_FP64 (mismo kernel/tamano). Filtro (error finito y rel_l2<=eps) ANTES
de rankear. Spearman rho(T,E) por kernel = contexto (EDP es metrica compuesta
complementaria, no una dimension independiente).
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
from common import (CONFIG, FORMAT_COLOR, K_MARKER, KERNEL_LABEL, SIZE_LABEL, TAB_DIR, horizon,  # noqa: E402
                    load_canonical, save_fig)

EPS = CONFIG["EPS_REL_L2"]


def build() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    rows, iso, rho = [], [], {}
    for kernel in ("gemm", "conv", "stencil"):
        df = load_canonical(kernel)
        g = df[(df.device == "GPU") & df.energy_reliable].copy()
        g["T_iter_s"] = g["t_iter_ms_energy"] / 1000.0
        g["E_iter_J"] = g["energy_gpu_j_per_iter"]
        g["EDP_iter_Js"] = g["E_iter_J"] * g["T_iter_s"]
        g["P_mean_W"] = g["E_iter_J"] / g["T_iter_s"]
        # rho(T,E): una fila por config fisica (independiente del horizonte), pase dedicado
        u = g.drop_duplicates(["size", "route", "K_efectivo"])
        rho[kernel] = dict(all=stats.spearmanr(u["T_iter_s"], u["E_iter_J"])[0],
                           within_size=np.nanmean([stats.spearmanr(x["T_iter_s"], x["E_iter_J"])[0]
                                                   for _, x in u.groupby("size") if len(x) > 2]),
                           n=len(u))
        ref = g[g.route == "GPU_FP64"].drop_duplicates("size").set_index("size")
        g["EDP_ref_fp64"] = g["size"].map(ref["EDP_iter_Js"])
        g["EDP_norm"] = g["EDP_iter_Js"] / g["EDP_ref_fp64"]
        g["P_ref_W"] = g["size"].map(ref["P_mean_W"])
        for s_, r in ref.iterrows():
            iso.append(dict(kernel=kernel, size=s_, T_ref_s=r["T_iter_s"], E_ref_J=r["E_iter_J"], EDP_ref_Js=r["EDP_iter_Js"]))
        hs = [horizon(kernel)] + ([horizon(kernel, True)] if kernel == "stencil" else [])
        g = g[g.iters_num.isin(hs) & (g.iters_num.isin(hs))]
        fin = g["solution_finite"].fillna(False).astype(bool)
        g["is_reference"] = g["is_reference"].astype(bool)
        g["passes_filter"] = (~g["is_reference"]) & fin & (g["rel_l2"] <= EPS)
        g["fail_reason"] = np.where(g["is_reference"], "reference(baseline)",
                                    np.where(~fin, "non_finite", np.where(g["rel_l2"] > EPS, f"rel_l2>eps({EPS:g})", "")))
        rows.append(g)
    t = pd.concat(rows, ignore_index=True)
    # ranking (solo tras filtrar)
    rk = t[t.passes_filter].copy()
    rk["rank_EDP"] = rk.groupby(["kernel", "size", "iters_num"])["EDP_iter_Js"].rank(method="min")
    rk = rk.sort_values(["kernel", "iters_num", "size", "rank_EDP"])
    # sensibilidad a eps
    sens = []
    for eps in CONFIG["EPS_SENSITIVITY"]:
        c = t[(~t.is_reference) & t["solution_finite"].fillna(False).astype(bool) & (t["rel_l2"] <= eps)]
        for (k, s_, h), x in c.groupby(["kernel", "size", "iters_num"]):
            b = x.sort_values("EDP_iter_Js").iloc[0]
            sens.append(dict(eps=eps, kernel=k, size=s_, horizon=h, n_pass=len(x), best_route=b["route"], best_K=b["K_efectivo"],
                             best_EDP_norm=b["EDP_norm"], best_rel_l2=b["rel_l2"]))
    return t, rk, pd.DataFrame(sens), {"rho": rho, "iso": pd.DataFrame(iso)}


def plot(t: pd.DataFrame, rho: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for c, kernel in enumerate(("gemm", "conv", "stencil")):
        d = t[(t.kernel == kernel) & (t.iters_num == horizon(kernel)) & (~t.is_reference | (t.route == "GPU_FP32"))]
        for (route, K), g in d.groupby(["route", "K_efectivo"]):
            g = g.sort_values("size")
            fmt = g["format"].iloc[0]
            ls = "--" if g["compensation"].iloc[0] == "comp" else "-"
            m = K_MARKER[int(K)]
            for r, col in ((0, "EDP_norm"), (1, "P_mean_W")):
                ax = axes[r, c]
                ok = g[g.passes_filter | g.is_reference]
                ax.plot(ok["size"], ok[col], ls=ls, marker=m, color=FORMAT_COLOR[fmt], lw=1.4, ms=6)
                bad = g[~(g.passes_filter | g.is_reference)]
                ax.plot(bad["size"], bad[col], ls="", marker=m, mfc="none", color="#777777", ms=7, mew=1.2)
        for r in (0, 1):
            ax = axes[r, c]
            ax.set_xscale("log", base=2)
            ss = sorted(d["size"].unique())
            ax.set_xticks(ss, [str(s) for s in ss])
            ax.set_xlabel(SIZE_LABEL[kernel])
        axes[0, c].axhline(1.0, color="k", lw=.8, ls="--")
        axes[0, c].set_yscale("log")
        axes[0, c].set_ylabel("EDP_iter / EDP_iter(GPU_FP64)")
        axes[1, c].set_ylabel("Potencia media P = E_iter / T_iter [W]")
        rr = rho[kernel]
        name = KERNEL_LABEL[kernel] + (f" — «{CONFIG['STENCIL_OPERATOR_LABEL']}»" if kernel == "stencil" else "")
        axes[0, c].set_title(f"{name} (h={horizon(kernel)}, ε={EPS:g})\nSpearman ρ(T,E)={rr['all']:.3f} (n={rr['n']})", fontsize=10)
    hd = [Line2D([0], [0], color=FORMAT_COLOR[f], lw=3, label=f) for f in ("FP16", "BF16", "FP32")]
    hd += [Line2D([0], [0], color="k", ls="-", label="sin comp./spatial"), Line2D([0], [0], color="k", ls="--", label="comp")]
    hd += [Line2D([0], [0], marker=K_MARKER[k], color="k", ls="", label=f"K={k}") for k in (0, 1, 5, 8, 20, 32)]
    hd += [Line2D([0], [0], marker="o", mfc="none", color="#777777", ls="", label="excluido por filtro (no finito / >ε)")]
    fig.legend(handles=hd, fontsize=8, ncol=7, loc="upper center", bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("F5 — EDP por iteración normalizado contra GPU_FP64 y potencia media", fontsize=13, y=1.0)
    cap = [
        f"F5. EDP_iter=E_iter·T_iter con energy_reliable=1, pase dedicado (GEMM 24000/500, Conv 37000/2500, Stencil 4000/1500/1500 iteraciones), solo energía GPU; "
        f"T e E de la misma ventana; normalizado por GPU_FP64 (mismo kernel/tamaño). Filtro ANTES del ranking: error finito a h y rel_l2≤ε={EPS:g} (ε en CONFIG; sensibilidad en F5_eps_sensitivity.csv).",
        "Spearman ρ(T,E) por kernel (contexto; EDP es métrica compuesta complementaria, no una dimensión independiente): "
        + ", ".join(f"{KERNEL_LABEL[k]}={rho[k]['all']:.3f}" for k in rho) + ". Potencia P=E/T por ruta en el panel inferior.",
        "Operador: GEMM A=c·H; Conv no registrado en logs; Stencil «stress». Excluidos: FP16 no finito (GEMM N=2048/8192, Conv todos, Stencil), rutas CPU_*, "
        "2 filas GEMM N=8192 iters=80 K=1 (fiables accidentales, fuera del pase dedicado). Las referencias GPU sin rel_l2_prop (Stencil GPU_FP32) son contexto.",
    ]
    fig.tight_layout()
    fig.text(.01, -.075, cap[0] + "\n" + cap[1], ha="left", va="top", fontsize=8, wrap=True)
    save_fig(fig, "F5_edp_potencia", cap)


if __name__ == "__main__":
    t, rk, sens, extra = build()
    cols = ["kernel", "size", "route", "format", "compensation", "K_efectivo", "iters_num", "T_iter_s", "E_iter_J", "EDP_iter_Js",
            "EDP_norm", "P_mean_W", "rel_l2", "solution_finite", "passes_filter", "fail_reason", "n_raw_E", "operator"]
    t[cols].to_csv(TAB_DIR / "F5_data.csv", index=False)
    rk[cols + ["rank_EDP"]].to_csv(TAB_DIR / "F5_ranking.csv", index=False)
    sens.to_csv(TAB_DIR / "F5_eps_sensitivity.csv", index=False)
    extra["iso"].to_csv(TAB_DIR / "F5_isocurve_levels.csv", index=False)
    pd.DataFrame(extra["rho"]).T.to_csv(TAB_DIR / "F5_spearman_T_E.csv")
    plot(t, extra["rho"])
    print(pd.DataFrame(extra["rho"]).T)
    print(rk.groupby(["kernel", "iters_num", "size"]).first()[["route", "K_efectivo", "EDP_norm"]])
