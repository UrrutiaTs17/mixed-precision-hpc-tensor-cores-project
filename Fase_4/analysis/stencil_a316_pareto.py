#!/usr/bin/env python3
"""Pareto de Stencil con el operador difusivo alpha=3/16 (campana 7757).

Con `stress` (7145) no existe ventana donde T, E y error sean validos a la vez
(todas las precisiones desbordan antes del pase de energia). Con el operador
difusivo nada desborda, asi que T, E y error salen de la MISMA corrida del pase
de energia (en_S: 4096 x 4000 it; en_L: 8192 x 1500 it) -- no hace falta unir
pases distintos.

Entrada: CONFIG["STENCIL_A316_ROOT"]/{spk,off}_{en_S,en_L}/{summary,energy}_stencil_*.csv
Salida:  tables/SA316_points.csv, tables/SA316_exclusions.csv,
         figures/SA316_pareto_TE.png (vista F9) y SA316_pareto_proyecciones.png

Candidatos: rutas WMMA (FP16/BF16) de GPU, con su comp_scheme real
(spatial | none | kahan_local) y su K. Error = rel_l2_prop (estado propagado),
como en el resto del pipeline de Stencil. Referencias (GPU_FP64 exacta,
GPU_FP32 contexto) nunca cuentan como miembros del frente.
"""
from __future__ import annotations

import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONFIG, TAB_DIR, foot, pareto_mask, save_fig  # noqa: E402

ROOT = CONFIG["STENCIL_A316_ROOT"]
ITERS = CONFIG["STENCIL_A316_ENERGY_ITERS"]
MARKER = {("FP16", "spatial"): "s", ("BF16", "spatial"): "^", ("FP16", "none"): "o", ("BF16", "none"): "D",
          ("FP16", "kahan_local"): "P", ("BF16", "kahan_local"): "X"}
LABEL = {"spatial": "compensación espacial", "none": "sin compensación", "kahan_local": "Kahan local"}
KEYS = ["nx", "iters", "kahan", "route", "anchor_every"]


def load_energy_passes() -> pd.DataFrame:
    frames = []
    for d in sorted(glob.glob(os.path.join(ROOT, "*_en_[SL]"))):
        grupo = os.path.basename(d).split("_")[0]
        s = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(d, "summary_stencil_*.csv"))], ignore_index=True)
        e = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(d, "energy_stencil_*.csv"))], ignore_index=True)
        assert not s.duplicated(KEYS).any() and not e.duplicated(KEYS).any(), f"{d}: filas repetidas por {KEYS}"
        ecols = ["energy_gpu_j", "energy_gpu_j_per_iter", "energy_window_reliable", "time_total_s"]
        m = s.drop(columns=[c for c in ecols if c in s.columns]).merge(
            e[KEYS + ecols], on=KEYS, how="left", validate="one_to_one")
        m["grupo"], m["paso"] = grupo, os.path.basename(d)
        frames.append(m)
    if not frames:
        raise FileNotFoundError(f"sin pases de energia en {ROOT}/*_en_[SL]")
    df = pd.concat(frames, ignore_index=True)
    assert set(df["op_mode"]) == {"diffusive"} and np.allclose(df["alpha"], 0.1875), "operador distinto de difusivo alpha=3/16"
    df = df[df["nx"].isin(ITERS) & (df["iters"] == df["nx"].map(ITERS))].copy()
    df["size"] = df["nx"]
    return df


def classify(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df[df["device"] == "gpu"].copy()
    df["is_candidate"] = df["route"].str.startswith("WMMA_")
    df["format"] = df["route"].str.extract(r"(FP16|BF16|FP32|FP64)")[0]
    df["compensation"] = np.where(df["is_candidate"], df["comp_scheme"], "none")
    df["K_efectivo"] = np.where(df["is_candidate"], df["anchor_every"], 0)
    df["T_ms"] = df["time_total_s"] / df["iters"] * 1000.0
    df["E_J"] = df["energy_gpu_j_per_iter"]
    df["rel_l2"] = np.where(df["is_candidate"], df["rel_l2_prop"], df["rel_l2"])
    reasons = []
    for _, r in df.iterrows():
        t = []
        if r["route"] != "GPU_FP64" and (r["first_nonfinite"] != -1 or not np.isfinite(r["rel_l2"])):
            t.append("non_finite")
        if r.get("error_evaluable", 1) != 1 and r["route"] != "GPU_FP64":
            t.append(f"error_no_evaluable({r.get('motivo_exclusion', '')})")
        if r["energy_window_reliable"] != 1 or not (r["energy_gpu_j"] > 0):
            t.append("energy_window_unreliable")
        if r["gpu_valid"] != 1:
            t.append("gpu_invalid")
        reasons.append(";".join(t))
    df["exclusion_reason"] = reasons
    # Referencias corren una vez por invocacion (cada K, cada pasada KAHAN, cada
    # grupo): son pseudo-replicas de la MISMA configuracion -> mediana + n_raw.
    ok = df[df["exclusion_reason"] == ""]
    key = ["size", "route", "format", "compensation", "K_efectivo"]
    cand = ok[ok.is_candidate]
    dup = cand[cand.duplicated(key, keep=False)]
    assert dup.empty, f"candidatos repetidos por {key}:\n{dup[key + ['paso']]}"
    refs = ok[~ok.is_candidate].groupby(key).agg(
        T_ms=("T_ms", "median"), E_J=("E_J", "median"), rel_l2=("rel_l2", "median"), n_raw=("T_ms", "size")).reset_index()
    cand = cand[key + ["T_ms", "E_J", "rel_l2", "paso"]].assign(n_raw=1)
    pts = pd.concat([cand.assign(is_ctx_reference=False), refs.assign(is_ctx_reference=True)], ignore_index=True)
    excl = df[df["exclusion_reason"] != ""][["size", "paso", "route", "compensation", "K_efectivo", "exclusion_reason"]]
    return pts, excl


def fronts(pts: pd.DataFrame) -> pd.DataFrame:
    out = []
    for size, g in pts.groupby("size"):
        g = g.copy()
        c = ~g.is_ctx_reference
        g["err_obj"] = np.where(g.route == "GPU_FP64", -np.inf, np.log10(g.rel_l2.where(g.rel_l2 > 0)))
        g["on_front_A"] = False
        g.loc[c, "on_front_A"] = pareto_mask(g.loc[c, ["T_ms", "E_J", "err_obj"]].to_numpy(float))
        # dominado por una referencia (GPU_FP64 con error -inf, GPU_FP32 con su error medido)
        ref = g[g.is_ctx_reference][["T_ms", "E_J", "err_obj"]].to_numpy(float)
        g["dominated_by_reference"] = [bool(c_) and any(np.all(q <= p) and np.any(q < p) for q in ref)
                                       for c_, p in zip(c, g[["T_ms", "E_J", "err_obj"]].to_numpy(float))]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def colorbar(fig, sc, axes) -> None:
    cbar = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("Error relativo L2 del estado propagado (escala log)", fontsize=10)
    cbar.ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    cbar.ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    cbar.ax.tick_params(which="both", labelsize=8)


def draw_candidates(ax, cand, x, y, norm, color_by_err=True):
    sc = None
    for (fmt, comp), mk in MARKER.items():
        c = cand[(cand["format"] == fmt) & (cand["compensation"] == comp)]
        if c.empty:
            continue
        kw = dict(c=c["rel_l2"], norm=norm, cmap="viridis") if color_by_err else dict(color="#bbbbbb")
        sc = ax.scatter(c[x], c[y], marker=mk, s=80, edgecolor="k", linewidths=0.8, zorder=3, **kw)
        for _, r in c.iterrows():
            ax.annotate(f"K={int(r.K_efectivo)}", (r[x], r[y]), xytext=(5, 3), textcoords="offset points", fontsize=7)
    front = cand[cand["on_front_A"]].sort_values(x)
    if len(front) > 1:
        ax.plot(front[x], front[y], ls="--", color="k", lw=1.2, zorder=2)
    elif len(front) == 1:
        ax.scatter(front[x], front[y], s=260, facecolor="none", edgecolor="k", lw=1.2, zorder=2)
    return sc


def legend_handles(pts):
    present = set(zip(pts["format"], pts["compensation"])) & set(MARKER)
    h = [Line2D([0], [0], marker=mk, ls="", mfc="none", mec="k", ms=9, label=f"{k[0]} — {LABEL[k[1]]}")
         for k, mk in MARKER.items() if k in present]
    h += [Line2D([0], [0], marker="*", ls="", mfc="#888888", mec="k", ms=14, label="GPU FP64 (referencia)"),
          Line2D([0], [0], marker="h", ls="", mfc="#009E73", mec="k", ms=10, label="GPU FP32 (contexto)"),
          Line2D([0], [0], ls="--", color="k", label="Frente Pareto (T, E, error)")]
    return h


def caption_common() -> str:
    return ("Operador difusivo α=3/16, condición inicial monomodo (campaña 7757). T, E y error salen de la MISMA corrida del "
            "pase de energía (4096²: 4000 it; 8192²: 1500 it; ventana de energía fiable, GPU). Error = rel_l2 del estado "
            "propagado frente a FP64, al final de esa ventana: para entonces la solución de referencia ha decaído varios "
            "órdenes de magnitud, de modo que errores relativos del orden de la unidad (sin ancla) indican que el residuo de "
            "redondeo supera la solución remanente, no una divergencia (todo es finito). Frente = no dominado en (T, E, error) "
            "solo entre candidatos de precisión reducida; las referencias no compiten.")


def plot_te(pts: pd.DataFrame) -> None:
    sizes = sorted(pts["size"].unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(6.8 * len(sizes), 5.4), squeeze=False, constrained_layout=True)
    cand_all = pts[~pts.is_ctx_reference]
    norm = LogNorm(vmin=cand_all.rel_l2.min(), vmax=cand_all.rel_l2.max())
    sc = None
    for ax, size in zip(axes[0], sizes):
        g = pts[pts["size"] == size]
        sc = draw_candidates(ax, g[~g.is_ctx_reference], "T_ms", "E_J", norm) or sc
        for route, mk, col, ms in (("GPU_FP64", "*", "#888888", 220), ("GPU_FP32", "h", "#009E73", 120)):
            r = g[g.route == route]
            ax.scatter(r["T_ms"], r["E_J"], marker=mk, s=ms, color=col, edgecolor="k", lw=0.8, zorder=3)
        ax.set_title(f"nx=ny = {size:,}; T, E y error a {ITERS[size]:,} iteraciones", fontsize=10)
        ax.set_xlabel("Tiempo por iteración (ms)")
        ax.set_ylabel("Energía GPU por iteración (J)")
    leg = fig.legend(handles=legend_handles(pts), ncol=4, fontsize=8, loc="outside upper center",
                     title="Pareto tiempo–energía–error — Stencil, operador difusivo α=3/16")
    leg.get_title().set_fontsize(12)
    if sc is not None:
        colorbar(fig, sc, axes)
    foot(fig, caption_common())
    save_fig(fig, "SA316_pareto_TE", [caption_common()])


def plot_projections(pts: pd.DataFrame) -> None:
    sizes = sorted(pts["size"].unique())
    fig, axes = plt.subplots(len(sizes), 2, figsize=(12.5, 4.6 * len(sizes)), squeeze=False, constrained_layout=True)
    cand_all = pts[~pts.is_ctx_reference]
    norm = LogNorm(vmin=cand_all.rel_l2.min(), vmax=cand_all.rel_l2.max())
    for row, size in zip(axes, sizes):
        g = pts[pts["size"] == size]
        cand = g[~g.is_ctx_reference]
        for ax, x, xl in ((row[0], "T_ms", "Tiempo por iteración (ms)"), (row[1], "E_J", "Energía GPU por iteración (J)")):
            draw_candidates(ax, cand, x, "rel_l2", norm, color_by_err=False)
            f32 = g[g.route == "GPU_FP32"]
            ax.scatter(f32[x], f32["rel_l2"], marker="h", s=120, color="#009E73", edgecolor="k", lw=0.8, zorder=3)
            f64 = g[g.route == "GPU_FP64"]
            for v in f64[x]:
                ax.axvline(v, color="#888888", ls=":", lw=1)
            ax.set_yscale("log")
            ax.set_xlabel(xl)
            ax.set_ylabel("Error relativo L2 (estado propagado)")
            ax.set_title(f"nx=ny = {size:,} ({ITERS[size]:,} it)" + ("; línea gris = GPU FP64" if x == "T_ms" else ""), fontsize=10)
    fig.legend(handles=legend_handles(pts), ncol=4, fontsize=8, loc="outside upper center",
               title="Proyecciones tiempo–error y energía–error — Stencil, α=3/16").get_title().set_fontsize(12)
    foot(fig, caption_common())
    save_fig(fig, "SA316_pareto_proyecciones", [caption_common()])


if __name__ == "__main__":
    raw = load_energy_passes()
    pts, excl = classify(raw)
    pts = fronts(pts)
    pts.to_csv(TAB_DIR / "SA316_points.csv", index=False)
    excl.to_csv(TAB_DIR / "SA316_exclusions.csv", index=False)
    for size, g in pts.groupby("size"):
        c = g[~g.is_ctx_reference]
        f = c[c.on_front_A].sort_values("T_ms")
        print(f"nx={size}: {len(c)} candidatos, {len(f)} en el frente -> "
              + ", ".join(f"{r.format}/{r.compensation}/K={int(r.K_efectivo)}" for r in f.itertuples()))
    print(f"excluidas: {len(excl)}" + (" -> " + "; ".join(sorted(set(excl.exclusion_reason))) if len(excl) else ""))
    plot_te(pts)
    plot_projections(pts)
