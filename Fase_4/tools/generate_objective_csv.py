#!/usr/bin/env python3
"""Genera tablas analiticas para verificar los objetivos de la campana.

No modifica CSV crudos. Produce tablas largas y resumidas listas para
pandas/seaborn/Excel: rendimiento, exactitud, energia, ancla, Pareto,
cobertura y fallos OOM.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def files(pattern: str) -> list[str]:
    return sorted(glob.glob(str(ROOT / pattern)))


def numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df

def normalize_stencil_summary(frame: pd.DataFrame, path: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["job_id"] = frame["job_id"]
    out["kernel"] = "stencil"
    out["size"] = pd.to_numeric(frame["nx"], errors="coerce")
    out["format"] = frame["route"].astype(str).str.extract(r"(FP16|BF16|FP32|FP64)", expand=False).fillna("NA")
    out["comp"] = frame["kahan"].map(lambda value: "on" if str(value).lower() == "on" else "off")
    out["anchor_every"] = pd.to_numeric(frame.get("anchor_every", 0), errors="coerce").fillna(0).astype(int)
    out["route"] = frame["route"]
    out["iters"] = pd.to_numeric(frame["iters"], errors="coerce")
    out["t_iter_ms"] = pd.to_numeric(frame["t_iter_ms"], errors="coerce")
    out["t_total_ms"] = pd.to_numeric(frame["t_total_ms"], errors="coerce")
    out["gflops"] = pd.to_numeric(frame["gflops"], errors="coerce")
    out["energy_gpu_j"] = pd.to_numeric(frame["energy_gpu_j"], errors="coerce")
    out["window_reliable"] = pd.NA
    out["gpu_segments"] = pd.NA
    out["source_csv"] = os.path.relpath(path, ROOT)
    return out

def normalize_stencil_drift(frame: pd.DataFrame, path: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["job_id"] = frame["job_id"]
    out["kernel"] = "stencil"
    out["size"] = pd.to_numeric(frame["nx"], errors="coerce")
    out["format"] = frame["route"].astype(str).str.extract(r"(FP16|BF16|FP32|FP64)", expand=False).fillna("NA")
    out["comp"] = frame["kahan"].map(lambda value: "on" if str(value).lower() == "on" else "off")
    out["anchor_every"] = pd.to_numeric(frame.get("anchor_every", 0), errors="coerce").fillna(0).astype(int)
    out["route"] = frame["route"]
    out["iters"] = pd.to_numeric(frame["iters"], errors="coerce")
    out["iter"] = pd.to_numeric(frame["iter"], errors="coerce")
    out["rel_l2"] = pd.to_numeric(frame["rel_l2"], errors="coerce")
    out["rel_linf"] = pd.to_numeric(
        frame.get("rel_linf", pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    out["solution_finite"] = pd.to_numeric(frame.get("solution_finite"), errors="coerce")
    out["source_csv"] = os.path.relpath(path, ROOT)
    return out


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[pd.DataFrame] = []
    drifts: list[pd.DataFrame] = []
    for path in files("Fase_3/Stencil/results/summary_stencil_*.csv") + files("Fase_4/Stencil/results/summary_stencil_*.csv"):
        summaries.append(normalize_stencil_summary(pd.read_csv(path), path))
    for path in files("Fase_3/Stencil/results/drift_stencil_*.csv") + files("Fase_4/Stencil/results/drift_stencil_*.csv"):
        drifts.append(normalize_stencil_drift(pd.read_csv(path), path))
    for kernel in ("gemm", "conv"):
        for path in files(f"Fase_3/**/results/summary_{kernel}_*.csv") + files(
            f"Fase_4/**/results/summary_{kernel}_*.csv"
        ):
            frame = pd.read_csv(path)
            frame["source_csv"] = os.path.relpath(path, ROOT)
            summaries.append(frame)
        for path in files(f"Fase_3/**/results/drift_{kernel}_*.csv") + files(
            f"Fase_4/**/results/drift_{kernel}_*.csv"
        ):
            frame = pd.read_csv(path)
            frame["source_csv"] = os.path.relpath(path, ROOT)
            drifts.append(frame)

    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    drift = pd.concat(drifts, ignore_index=True) if drifts else pd.DataFrame()
    summary = numeric(summary, ["size", "iters", "anchor_every", "t_iter_ms", "t_total_ms", "gflops", "energy_gpu_j", "window_reliable", "gpu_segments"])
    drift = numeric(drift, ["size", "iters", "iter", "anchor_every", "rel_l2", "rel_linf"])

    pareto_frames = []
    for path in files("pareto_out/pareto_*.csv"):
        frame = pd.read_csv(path)
        frame["source_csv"] = os.path.relpath(path, ROOT)
        pareto_frames.append(frame)
    pareto = pd.concat(pareto_frames, ignore_index=True) if pareto_frames else pd.DataFrame()

    guidelines = []
    for path in files("pareto_out/guideline_*.csv"):
        frame = pd.read_csv(path)
        frame["source_csv"] = os.path.relpath(path, ROOT)
        guidelines.append(frame)
    guideline = pd.concat(guidelines, ignore_index=True) if guidelines else pd.DataFrame()
    return summary, drift, pareto, guideline


def write_coverage(summary: pd.DataFrame, drift: pd.DataFrame, pareto: pd.DataFrame, out: Path) -> None:
    rows = []
    for kernel in ("gemm", "conv", "stencil"):
        s = summary[summary.get("kernel", pd.Series(dtype=str)) == kernel] if not summary.empty else pd.DataFrame()
        d = drift[drift.get("kernel", pd.Series(dtype=str)) == kernel] if not drift.empty else pd.DataFrame()
        p = pareto[pareto.get("kernel", pd.Series(dtype=str)) == kernel] if not pareto.empty else pd.DataFrame()
        rows.append({
            "kernel": kernel,
            "summary_rows": len(s),
            "drift_rows": len(d),
            "pareto_rows": len(p),
            "sizes": " ".join(map(str, sorted(s["size"].dropna().unique()))) if "size" in s else "",
            "formats": " ".join(sorted(s["format"].dropna().astype(str).unique())) if "format" in s else "",
            "anchor_levels": " ".join(map(str, sorted(s["anchor_every"].dropna().unique()))) if "anchor_every" in s else "",
            "status": "available" if len(s) else "missing",
        })
    pd.DataFrame(rows).to_csv(out / "objective_coverage.csv", index=False)


def write_performance(summary: pd.DataFrame, out: Path) -> None:
    columns = ["job_id", "kernel", "size", "format", "comp", "anchor_every", "route", "iters", "t_iter_ms", "t_total_ms", "gflops", "energy_gpu_j", "window_reliable", "gpu_segments", "source_csv"]
    frame = summary[[c for c in columns if c in summary]].copy() if not summary.empty else pd.DataFrame(columns=columns)
    frame.to_csv(out / "objective_performance_energy.csv", index=False)


def write_accuracy(summary: pd.DataFrame, drift: pd.DataFrame, out: Path) -> None:
    if drift.empty:
        pd.DataFrame().to_csv(out / "objective_accuracy_drift.csv", index=False)
        return
    group = [c for c in ["job_id", "kernel", "size", "format", "comp", "anchor_every", "route"] if c in drift]
    accuracy = drift.groupby(group, dropna=False).agg(
        drift_rows=("iter", "count"),
        first_iter=("iter", "min"),
        last_iter=("iter", "max"),
        rel_l2_mean=("rel_l2", "mean"),
        rel_l2_max=("rel_l2", "max"),
        rel_linf_mean=("rel_linf", "mean"),
        rel_linf_max=("rel_linf", "max"),
        finite_fraction=("solution_finite", "mean") if "solution_finite" in drift else ("rel_l2", lambda x: np.nan),
    ).reset_index()
    accuracy.to_csv(out / "objective_accuracy_drift.csv", index=False)


def write_anchor(summary: pd.DataFrame, out: Path) -> None:
    if summary.empty or "anchor_every" not in summary:
        pd.DataFrame().to_csv(out / "objective_anchor_effect.csv", index=False)
        return
    keys = [c for c in ["kernel", "size", "format", "comp", "iters", "route"] if c in summary]
    cols = keys + ["anchor_every", "t_iter_ms", "gflops", "energy_gpu_j", "window_reliable"]
    frame = summary[[c for c in cols if c in summary]].copy()
    frame.to_csv(out / "objective_anchor_effect.csv", index=False)


def write_pareto(pareto: pd.DataFrame, guideline: pd.DataFrame, out: Path) -> None:
    pareto.to_csv(out / "objective_pareto_long.csv", index=False)
    guideline.to_csv(out / "objective_guideline_long.csv", index=False)


def write_oom(out: Path) -> None:
    rows = []
    for path in files("oom_failures_*.log"):
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                rows.append({
                    "source_log": os.path.relpath(path, ROOT),
                    "timestamp": fields[0] if len(fields) > 0 else "",
                    "phase": fields[1] if len(fields) > 1 else "",
                    "exit_code": fields[2] if len(fields) > 2 else "",
                    "script": fields[3] if len(fields) > 3 else "",
                })
    pd.DataFrame(rows, columns=["source_log", "timestamp", "phase", "exit_code", "script"]).to_csv(out / "objective_oom_events.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(ROOT / "analysis_out"))
    args = parser.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    summary, drift, pareto, guideline = load_inputs()
    write_performance(summary, out)
    write_accuracy(summary, drift, out)
    write_anchor(summary, out)
    write_pareto(pareto, guideline, out)
    write_coverage(summary, drift, pareto, out)
    write_oom(out)
    print(f"[objective_analysis] summary={len(summary)} drift={len(drift)} pareto={len(pareto)}")
    print(f"[objective_analysis] outputs={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
