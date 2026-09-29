#!/usr/bin/env python3
"""Pareto de Stencil con el operador difusivo alpha=3/16 (campana 7757).

Con `stress` (7145) no existe ventana donde T, E y error sean validos a la vez
(todas las precisiones desbordan antes del pase de energia). Con el operador
difusivo nada desborda.

Ejes (misma regla que Convolucion: el error a un horizonte COMUN para todos los
candidatos; T y E como tasas por iteracion de una ventana de energia fiable):
  - error: rel_l2 (rel_l2_prop en WMMA) al horizonte estandar H(size)
    (4096: 4000 it; 8192: 1500 it), de cualquier pasada que llegue a H
    (en_*, num_*: es determinista, se verifica que coincidan);
  - T, E: de la pasada de energia con energy_window_reliable=1. Las rutas WMMA
    abren un tramo NVML extra (exigen >= 1.0 s de ventana): a 4096 x 4000 it,
    K=0 dura ~0.84 s y no es fiable, por eso existen en_Slargo (8000 it) y
    en_Llargo (3000 it). Si hay varias ventanas fiables, se usa la mas larga.

Entrada: CONFIG["STENCIL_A316_ROOT"]/<paso>/{summary,energy}_stencil_*.csv
         (se ignora spk_num_corta: log contaminado; usar spk_num_corta_limpio)
Salida:  tables/SA316_points.csv, tables/SA316_exclusions.csv,
         figures/SA316_pareto_TE.png (vista F9) y SA316_pareto_proyecciones.png

Candidatos: rutas WMMA (FP16/BF16) de GPU, con su comp_scheme real
(spatial | none | kahan_local) y su K. Referencias (GPU_FP64 exacta, GPU_FP32
contexto) nunca cuentan como miembros del frente.
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
K_OFFSET = {0: (10, -16), 1: (-34, -6), 2: (10, -4), 32: (14, 16), 64: (18, 4), 128: (14, -12)}


def load_steps() -> pd.DataFrame:
    frames = []
    for d in sorted(glob.glob(os.path.join(ROOT, "*_*"))):
        paso = os.path.basename(d)
        if not os.path.isdir(d) or paso == "spk_num_corta":
            continue
        sp = glob.glob(os.path.join(d, "summary_stencil_*.csv"))
        ep = glob.glob(os.path.join(d, "energy_stencil_*.csv"))
        if not sp or not ep:
            continue
        s = pd.concat([pd.read_csv(p) for p in sp], ignore_index=True)
        e = pd.concat([pd.read_csv(p) for p in ep], ignore_index=True)
        assert not s.duplicated(KEYS).any() and not e.duplicated(KEYS).any(), f"{d}: filas repetidas por {KEYS}"
        ecols = ["energy_gpu_j", "energy_gpu_j_per_iter", "energy_window_reliable", "time_total_s"]
        m = s.drop(columns=[c for c in ecols if c in s.columns]).merge(
            e[KEYS + ecols], on=KEYS, how="left", validate="one_to_one")
        m["paso"] = paso
        m["es_energia"] = "_en_" in f"_{paso.split('_', 1)[1]}"
        frames.append(m)
    if not frames:
        raise FileNotFoundError(f"sin pasos en {ROOT}")
    df = pd.concat(frames, ignore_index=True)
    assert set(df["op_mode"]) == {"diffusive"} and np.allclose(df["alpha"], 0.1875), "operador distinto de difusivo alpha=3/16"
    df = df[df["nx"].isin(ITERS) & (df["device"] == "gpu")].copy()
    df["size"] = df["nx"]
    df["is_candidate"] = df["route"].str.startswith("WMMA_")
    df["format"] = df["route"].str.extract(r"(FP16|BF16|FP32|FP64)")[0]
    df["compensation"] = np.where(df["is_candidate"], df["comp_scheme"], "none")
    df["K_efectivo"] = np.where(df["is_candidate"], df["anchor_every"], 0)
    df["err"] = np.where(df["is_candidate"], df["rel_l2_prop"], df["rel_l2"])
    return df


KEY = ["size", "route", "format", "compensation", "K_efectivo"]


def classify(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # ---- error al horizonte estandar H(size), de cualquier pasada
    h = df[df["iters"] == df["size"].map(ITERS)]
    spread = h.groupby(KEY)["err"].agg(lambda x: np.nanmax(x) - np.nanmin(x) if x.notna().any() else 0.0)
    rel = spread / h.groupby(KEY)["err"].median().abs().replace(0, np.nan)
    assert (rel.fillna(0) < 1e-6).all(), f"error no determinista entre pasadas:\n{rel[rel >= 1e-6]}"
    err = h.groupby(KEY).agg(rel_l2=("err", "median"), first_nonfinite=("first_nonfinite", "max"),
                             error_evaluable=("error_evaluable", "min"),
                             err_src=("paso", lambda x: ",".join(sorted(set(x))))).reset_index()
    # ---- T, E: ventana de energia fiable (la mas larga por configuracion)
    en = df[df["es_energia"]].copy()
    en["fiable"] = (en["energy_window_reliable"] == 1) & (en["energy_gpu_j"] > 0) & (en["gpu_valid"] == 1)
    en["T_ms"] = en["time_total_s"] / en["iters"] * 1000.0
    en["E_J"] = en["energy_gpu_j_per_iter"]
    ok = en[en["fiable"]]
    longest = ok.groupby(KEY)["iters"].transform("max")
    ok = ok[ok["iters"] == longest]
    te = ok.groupby(KEY).agg(T_ms=("T_ms", "median"), E_J=("E_J", "median"), n_raw=("T_ms", "size"),
                             iters_energia=("iters", "first"),
                             te_src=("paso", lambda x: ",".join(sorted(set(x))))).reset_index()
    allk = pd.concat([err[KEY], en[KEY]]).drop_duplicates()
    pts = allk.merge(err, on=KEY, how="left").merge(te, on=KEY, how="left")
    reasons = []
    for _, r in pts.iterrows():
        t = []
        if pd.isna(r["rel_l2"]) and r["route"] != "GPU_FP64":
            t.append("sin_error_al_horizonte" if pd.isna(r["first_nonfinite"]) else "non_finite")
        elif r["route"] != "GPU_FP64" and (r["first_nonfinite"] != -1 or r["error_evaluable"] != 1):
            t.append("non_finite_o_no_evaluable")
        if pd.isna(r["T_ms"]):
            t.append("sin_ventana_energia_fiable")
        reasons.append(";".join(t))
    pts["exclusion_reason"] = reasons
    pts["is_ctx_reference"] = ~pts["route"].str.startswith("WMMA_")
    excl = pts[pts["exclusion_reason"] != ""].copy()
    pts = pts[pts["exclusion_reason"] == ""].drop(columns="exclusion_reason")
    return pts, excl


def fronts(pts: pd.DataFrame) -> pd.DataFrame:
    out = []
    for size, g in pts.groupby("size"):
        g = g.copy()
        c = ~g.is_ctx_reference
        g["err_obj"] = np.where(g.route == "GPU_FP64", -np.inf, np.log10(g.rel_l2.where(g.rel_l2 > 0)))
        g["rel_l2"] = g["rel_l2"].astype(float)
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
        if fmt != "BF16" or comp != "spatial":
            continue   # FP16 y BF16 espaciales se superponen: una sola etiqueta por K
        for _, r in c.iterrows():
            dx, dy = K_OFFSET.get(int(r.K_efectivo), (8, 0))
            ax.annotate(f"K={int(r.K_efectivo)}", (r[x], r[y]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.4, color="#666666"))
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
    h += [Line2D([0], [0], marker="*", ls="", mfc="none", mec="#555555", ms=16, label="GPU FP64 (referencia)"),
          Line2D([0], [0], marker="h", ls="", mfc="#009E73", mec="k", ms=10, label="GPU FP32 (contexto)"),
          Line2D([0], [0], ls="--", color="k", label="Frente Pareto (T, E, error)")]
    return h


def caption_common(pts: pd.DataFrame, excl: pd.DataFrame) -> str:
    ven = ", ".join(f"{size}²: {sorted(set(int(v) for v in g.iters_energia.dropna()))} it"
                    for size, g in pts.groupby("size"))
    txt = ("Operador difusivo α=3/16, condición inicial monomodo (campaña 7757). Error = rel_l2 del estado propagado "
           "frente a FP64 al horizonte común de cada malla (4096²: 4000 it; 8192²: 1500 it), igual para todos los "
           "candidatos. T y E = tasas por iteración de la ventana de energía fiable más larga disponible (" + ven + "; "
           "las rutas WMMA exigen ≥1 s de ventana NVML). A 4000 it (4096²) la solución de referencia ha decaído varios "
           "órdenes de magnitud: errores relativos del orden de la unidad sin ancla indican que el residuo de redondeo "
           "supera la solución remanente, no una divergencia (todo es finito). Frente = no dominado en (T, E, error) solo "
           "entre candidatos de precisión reducida; GPU FP64/FP32 se muestran como referencia y no compiten.")
    pend = pendientes(pts)
    if pend:
        txt += " PRELIMINAR: faltan candidatos (" + pend + "); el frente puede cambiar al incorporarlos."
    return txt


def pendientes(pts: pd.DataFrame) -> str:
    """Esquemas de compensacion esperados sin ningun candidato admitido, por malla."""
    faltan = []
    for size in sorted(ITERS):
        have = set(pts[(pts["size"] == size) & ~pts.is_ctx_reference]["compensation"])
        miss = [LABEL[c] for c in ("spatial", "none", "kahan_local") if c not in have]
        if miss:
            faltan.append(f"{size}²: " + ", ".join(miss))
    return "; ".join(faltan)


def plot_te(pts: pd.DataFrame, excl: pd.DataFrame) -> None:
    sizes = sorted(pts["size"].unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(6.8 * len(sizes), 5.4), squeeze=False, constrained_layout=True)
    cand_all = pts[~pts.is_ctx_reference]
    norm = LogNorm(vmin=cand_all.rel_l2.min(), vmax=cand_all.rel_l2.max())
    sc = None
    for ax, size in zip(axes[0], sizes):
        g = pts[pts["size"] == size]
        sc = draw_candidates(ax, g[~g.is_ctx_reference], "T_ms", "E_J", norm) or sc
        r = g[g.route == "GPU_FP64"]
        ax.scatter(r["T_ms"], r["E_J"], marker="*", s=420, facecolor="none", edgecolor="#555555", lw=1.2, zorder=4)
        r = g[g.route == "GPU_FP32"]
        ax.scatter(r["T_ms"], r["E_J"], marker="h", s=120, color="#009E73", edgecolor="k", lw=0.8, zorder=3)
        ax.set_title(f"nx=ny = {size:,}; error a {ITERS[size]:,} iteraciones", fontsize=10)
        ax.set_xlabel("Tiempo por iteración (ms)")
        ax.set_ylabel("Energía GPU por iteración (J)")
    leg = fig.legend(handles=legend_handles(pts), ncol=4, fontsize=8, loc="outside upper center",
                     title="Pareto tiempo–energía–error — Stencil, operador difusivo α=3/16"
                           + (" — PRELIMINAR" if pendientes(pts) else ""))
    leg.get_title().set_fontsize(12)
    if sc is not None:
        colorbar(fig, sc, axes)
    foot(fig, caption_common(pts, excl))
    save_fig(fig, "SA316_pareto_TE", [caption_common(pts, excl)])


def plot_projections(pts: pd.DataFrame, excl: pd.DataFrame) -> None:
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
               title="Proyecciones tiempo–error y energía–error — Stencil, α=3/16"
               + (" — PRELIMINAR" if pendientes(pts) else "")).get_title().set_fontsize(12)
    foot(fig, caption_common(pts, excl))
    save_fig(fig, "SA316_pareto_proyecciones", [caption_common(pts, excl)])


if __name__ == "__main__":
    raw = load_steps()
    pts, excl = classify(raw)
    pts = fronts(pts)
    pts.to_csv(TAB_DIR / "SA316_points.csv", index=False)
    excl.to_csv(TAB_DIR / "SA316_exclusions.csv", index=False)
    for r in excl.itertuples():
        print(f"  excluida: nx={r.size} {r.route}/{r.compensation}/K={int(r.K_efectivo)} -> {r.exclusion_reason}")
    for size, g in pts.groupby("size"):
        c = g[~g.is_ctx_reference]
        f = c[c.on_front_A].sort_values("T_ms")
        print(f"nx={size}: {len(c)} candidatos, {len(f)} en el frente -> "
              + ", ".join(f"{r.format}/{r.compensation}/K={int(r.K_efectivo)}" for r in f.itertuples()))
    print(f"excluidas: {len(excl)}" + (" -> " + "; ".join(sorted(set(excl.exclusion_reason))) if len(excl) else ""))
    plot_te(pts, excl)
    plot_projections(pts, excl)
