#!/usr/bin/env python3
"""Genera figuras publicables desde analysis_out/*.csv.

Las figuras son derivados reproducibles: los CSV analiticos siguen siendo la
fuente tabular y este script no modifica datos crudos ni tablas existentes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def read_csv(out: Path, name: str) -> pd.DataFrame:
    path = out / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def label(row: pd.Series) -> str:
    return f"{row.get('format', '')}-{row.get('comp', '')}-K{row.get('anchor_every', 0)}"


def performance(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return
    work = df.copy()
    work["label"] = work.apply(label, axis=1)
    for kernel, group in work.groupby("kernel"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for name, part in group.groupby("label"):
            part = part.sort_values("size")
            axes[0].plot(part["size"], part["gflops"], marker="o", label=name)
            axes[1].plot(part["size"], part["t_iter_ms"], marker="o", label=name)
        axes[0].set_title(f"{kernel.upper()}: rendimiento")
        axes[0].set_xlabel("Tamaño del problema")
        axes[0].set_ylabel("GFLOP/s")
        axes[1].set_title(f"{kernel.upper()}: tiempo por iteración")
        axes[1].set_xlabel("Tamaño del problema")
        axes[1].set_ylabel("ms/iteración")
        for axis in axes:
            axis.set_xscale("log", base=2)
            axis.grid(True, alpha=0.25)
        axes[1].legend(fontsize=7, loc="best")
        save(fig, out, f"performance_{kernel}.png")


def accuracy(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return
    for kernel, group in df.groupby("kernel"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        group = group.copy()
        group["label"] = group.apply(label, axis=1)
        for name, part in group.groupby("label"):
            part = part.sort_values("size")
            axes[0].plot(part["size"], part["rel_l2_max"], marker="o", label=name)
            axes[1].plot(part["size"], part["rel_linf_max"], marker="o", label=name)
        axes[0].set_title(f"{kernel.upper()}: error relativo L2 máximo")
        axes[1].set_title(f"{kernel.upper()}: error relativo Linf máximo")
        for axis in axes:
            axis.set_xlabel("Tamaño del problema")
            axis.set_yscale("log")
            axis.grid(True, alpha=0.25)
        axes[0].set_ylabel("Error relativo")
        axes[1].set_ylabel("Error relativo")
        axes[1].legend(fontsize=7, loc="best")
        save(fig, out, f"accuracy_{kernel}.png")


def energy(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return
    valid = df[df["window_reliable"].astype(str).isin(["1", "True", "true"])].copy()
    if valid.empty:
        valid = df.copy()
        title_suffix = " (ventanas disponibles; revisar confiabilidad)"
    else:
        title_suffix = " (ventanas confiables)"
    valid["edp"] = valid["t_iter_ms"] * valid["energy_gpu_j"]
    valid = valid[(valid["t_iter_ms"] > 0) & (valid["energy_gpu_j"] > 0) & (valid["edp"] > 0)]
    for kernel, group in valid.groupby("kernel"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        group = group.copy()
        group["label"] = group.apply(label, axis=1)
        for name, part in group.groupby("label"):
            part = part.sort_values("size")
            axes[0].plot(part["size"], part["energy_gpu_j"], marker="o", label=name)
            axes[1].plot(part["size"], part["edp"], marker="o", label=name)
        axes[0].set_title(f"{kernel.upper()}: energía{title_suffix}")
        axes[1].set_title(f"{kernel.upper()}: EDP{title_suffix}")
        for axis in axes:
            axis.set_xlabel("Tamaño del problema")
            axis.set_yscale("log")
            axis.grid(True, alpha=0.25)
        axes[0].set_ylabel("J")
        axes[1].set_ylabel("J·ms")
        axes[1].legend(fontsize=7, loc="best")
        save(fig, out, f"energy_edp_{kernel}.png")


def anchor(df: pd.DataFrame, out: Path) -> None:
    if df.empty or "anchor_every" not in df:
        return
    for kernel, group in df.groupby("kernel"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for (fmt, comp), part in group.groupby(["format", "comp"]):
            part = part.sort_values("anchor_every")
            name = f"{fmt}-{comp}"
            axes[0].plot(part["anchor_every"], part["t_iter_ms"], marker="o", label=name)
            axes[1].plot(part["anchor_every"], part["gflops"], marker="o", label=name)
        axes[0].set_title(f"{kernel.upper()}: costo del ancla")
        axes[0].set_ylabel("ms/iteración")
        axes[1].set_title(f"{kernel.upper()}: rendimiento frente a K")
        axes[1].set_ylabel("GFLOP/s")
        for axis in axes:
            axis.set_xlabel("anchor_every (K)")
            axis.grid(True, alpha=0.25)
        axes[1].legend(fontsize=7, loc="best")
        save(fig, out, f"anchor_effect_{kernel}.png")


def pareto(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return
    for kernel, group in df.groupby("kernel"):
        clean = group.dropna(subset=["t_iter_ms", "energy_gpu_j_per_iter", "rel_l2"]).copy()
        clean = clean[(clean["t_iter_ms"] > 0) & (clean["energy_gpu_j_per_iter"] > 0) & (clean["rel_l2"] > 0)]
        if clean.empty:
            continue
        fig = plt.figure(figsize=(9, 7))
        axis = fig.add_subplot(111, projection="3d")
        front = clean[clean["on_front"].astype(str).isin(["True", "true", "1"])]
        axis.scatter(clean["t_iter_ms"], clean["energy_gpu_j_per_iter"], clean["rel_l2"], alpha=0.35, label="No dominante")
        axis.scatter(front["t_iter_ms"], front["energy_gpu_j_per_iter"], front["rel_l2"], s=45, label="Frente Pareto")
        axis.set_title(f"{kernel.upper()}: frente de Pareto 3D")
        axis.set_xlabel("ms/iteración")
        axis.set_ylabel("J/iteración")
        axis.set_zlabel("Error relativo L2")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_zscale("log")
        axis.legend()
        save(fig, out, f"pareto_3d_{kernel}.png")


def coverage(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return
    fig, axis = plt.subplots(figsize=(8, 4))
    work = df.copy()
    axis.bar(work["kernel"], work["summary_rows"], label="Summary")
    axis.bar(work["kernel"], work["drift_rows"], bottom=work["summary_rows"], label="Drift")
    axis.set_title("Cobertura de datos por kernel")
    axis.set_ylabel("Número de filas")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    save(fig, out, "coverage_by_kernel.png")


def oom(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return
    counts = df.assign(short_phase=df["phase"].str.replace(r"\s+\(.*", "", regex=True)).groupby("short_phase").size()
    fig, axis = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=axis, color="#b23a48")
    axis.set_title("Eventos OOM por fase")
    axis.set_ylabel("Eventos")
    axis.set_xlabel("Fase")
    axis.grid(axis="y", alpha=0.25)
    save(fig, out, "oom_events.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default=str(ROOT / "analysis_out"))
    parser.add_argument("--outdir", default=str(ROOT / "analysis_figures"))
    args = parser.parse_args()
    indir = Path(args.indir)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    performance(read_csv(indir, "objective_performance_energy.csv"), out)
    accuracy(read_csv(indir, "objective_accuracy_drift.csv"), out)
    energy(read_csv(indir, "objective_performance_energy.csv"), out)
    anchor(read_csv(indir, "objective_anchor_effect.csv"), out)
    pareto(read_csv(indir, "objective_pareto_long.csv"), out)
    coverage(read_csv(indir, "objective_coverage.csv"), out)
    oom(read_csv(indir, "objective_oom_events.csv"), out)
    print(f"[figures] output={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
