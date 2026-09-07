#!/usr/bin/env python3
"""Fase_4/tools/pareto_front.py

Etapa 9 del plan de finalizacion: Frente de Pareto 3D (tiempo, energia,
error) -- UN FRENTE POR KERNEL, no uno combinado (correccion explicita de
la Etapa 9: GEMM, Convolucion y Stencil resuelven problemas distintos, nadie
elige "el kernel que minimiza el EDP" -- fusionar los tres en un solo frente
de dominancia produciria una directriz sin sentido de ingenieria). Opera
sobre la tabla normalizada de common_analysis.py, igual que run_statistics.py
-- leer ese modulo primero si no se ha leido.

Depende de la Etapa 8 (dataset valido de error + energia SIMULTANEOS): cada
punto del frente necesita t_iter_ms, energy_gpu_j_per_iter Y rel_l2 en la
MISMA fila -- si tu campana corrio RUN_KIND=energy (sin checkpoints, error
no medido) y RUN_KIND=numeric (con checkpoints, energia no confiable) por
separado, como recomienda Fase_4/Stencil/run_stencil_tc.sbatch, hay que
UNIRLAS primero por clave de configuracion antes de pasarlas aqui -- este
script asume que ya llegan unidas (ver merge_energy_into_stencil en
common_analysis.py, que ya hace esa union para Stencil; GEMM/Conv miden
error y energia en la MISMA corrida, no necesitan esa union).

USO:
    python3 pareto_front.py --results-dir results/ --outdir pareto_out/
    python3 pareto_front.py --stencil-summary "results/summary_stencil_*.csv" \\
        --stencil-energy "results/energy_stencil_*.csv" --outdir pareto_out/

Por cada kernel presente en los datos produce:
    pareto_<kernel>.csv       -- todas las configuraciones, con columna
                                  `on_front` (True/False) marcando el frente
    pareto_<kernel>.png       -- dispersion 3D (tiempo, energia, error) con
                                  el frente resaltado
    guideline_<kernel>.csv    -- para una rejilla de tolerancias de error,
                                  que configuracion (formato/tratamiento/K)
                                  minimiza el EDP dentro de esa tolerancia

SIN DATOS REALES DE PACCA: probado con `python3 pareto_front.py --self-test`
sobre datos sinteticos -- valida que el codigo no crashea y que la logica de
dominancia/guia es internamente consistente, NO que las cifras tengan
sentido fisico real.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_analysis import Dataset, discover_from_results_dir, load_dataset  # noqa: E402


def _aggregate_replicas(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (kernel, format, treatment, anchor_every, size, iters):
    promedia entre replicas (job_id distintos) -- el frente de Pareto
    compara CONFIGURACIONES, no corridas individuales; promediar reduce el
    ruido de una sola replica antes de decidir dominancia."""
    keys = ["kernel", "format", "treatment", "anchor_every", "size", "iters"]
    agg = (df.groupby(keys, dropna=False)
           .agg(t_iter_ms=("t_iter_ms", "mean"),
                energy_gpu_j_per_iter=("energy_gpu_j_per_iter", "mean"),
                rel_l2=("rel_l2", "mean"),
                n_replicas=("job_id", "nunique"))
           .reset_index())
    return agg


def pareto_front_3d(df: pd.DataFrame) -> pd.Series:
    """Dominancia de Pareto en 3D, minimizando las tres columnas
    (t_iter_ms, energy_gpu_j_per_iter, rel_l2). Un punto A domina a B si A
    es <= B en las tres y < en al menos una. O(n^2) -- el numero de
    configuraciones DISTINTAS por kernel (no de replicas) es chico
    (formatos x tratamientos x valores de K x tamanos), nunca va a
    justificar un algoritmo de frente de Pareto mas sofisticado aqui.
    Devuelve una Serie booleana alineada con el indice de df."""
    cols = ["t_iter_ms", "energy_gpu_j_per_iter", "rel_l2"]
    valid = df.dropna(subset=cols)
    values = valid[cols].to_numpy()
    on_front = np.ones(len(valid), dtype=bool)
    for i in range(len(valid)):
        if not on_front[i]:
            continue
        for j in range(len(valid)):
            if i == j:
                continue
            dominates_i = np.all(values[j] <= values[i]) and np.any(values[j] < values[i])
            if dominates_i:
                on_front[i] = False
                break
    result = pd.Series(False, index=df.index)
    result.loc[valid.index] = on_front
    return result


def guideline_table(df: pd.DataFrame, tolerances: "list[float]") -> pd.DataFrame:
    """Para cada tolerancia de rel_l2 en `tolerances`, entre las
    configuraciones cuyo rel_l2 <= tolerancia, cual minimiza
    (t_iter_ms * energy_gpu_j_per_iter) -- el EDP (Energy-Delay Product) por
    iteracion. Devuelve una fila por tolerancia; "sin_configuracion_valida"
    si ninguna configuracion cumple esa tolerancia."""
    rows = []
    for tol in tolerances:
        candidates = df.dropna(subset=["rel_l2", "t_iter_ms", "energy_gpu_j_per_iter"])
        candidates = candidates[candidates["rel_l2"] <= tol]
        if candidates.empty:
            rows.append({"tolerancia_rel_l2": tol, "format": "NA", "treatment": "NA",
                         "anchor_every": "NA", "edp": float("nan"),
                         "nota": "sin_configuracion_valida"})
            continue
        edp = candidates["t_iter_ms"] * candidates["energy_gpu_j_per_iter"]
        best_idx = edp.idxmin()
        best = candidates.loc[best_idx]
        rows.append({
            "tolerancia_rel_l2": tol,
            "format": best["format"],
            "treatment": best["treatment"],
            "anchor_every": best["anchor_every"],
            "size": best["size"],
            "edp": float(edp.loc[best_idx]),
            "rel_l2": float(best["rel_l2"]),
            "nota": "",
        })
    return pd.DataFrame(rows)


def _default_tolerances(rel_l2: pd.Series) -> "list[float]":
    finite = rel_l2.dropna()
    finite = finite[finite > 0]
    if finite.empty:
        return [1e-2, 1e-4, 1e-6]
    # Rejilla logaritmica entre el error mas chico y el mas grande
    # observados -- 6 puntos, suficiente para trazar donde el ancla deja de
    # aportar frente a compensacion sin ancla (la pregunta que pide la
    # Etapa 9) sin necesitar que el usuario adivine que tolerancias tienen
    # sentido para SU campana.
    lo, hi = float(finite.min()), float(finite.max())
    if lo == hi:
        return [lo]
    return list(np.geomspace(lo, hi, num=6))


def plot_pareto_3d(df: pd.DataFrame, kernel: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")  # sin display -- este script corre en un .sbatch/servidor
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sub = df.dropna(subset=["t_iter_ms", "energy_gpu_j_per_iter", "rel_l2"])
    off_front = sub[~sub["on_front"]]
    on_front = sub[sub["on_front"]]
    ax.scatter(off_front["t_iter_ms"], off_front["energy_gpu_j_per_iter"],
               off_front["rel_l2"], c="gray", alpha=0.4, label="dominado")
    ax.scatter(on_front["t_iter_ms"], on_front["energy_gpu_j_per_iter"],
               on_front["rel_l2"], c="red", s=60, label="frente de Pareto")
    ax.set_xlabel("t_iter_ms")
    ax.set_ylabel("energy_gpu_j_per_iter")
    ax.set_zlabel("rel_l2")
    positive = sub[sub["rel_l2"] > 0]["rel_l2"]
    if not positive.empty:
        # zlim explicito ANTES de set_zscale: si se deja que matplotlib
        # autoescale un eje log con datos que rozan/incluyen 0 (posible con
        # K grande, rel_l2 puede caer a ruido de punto flotante ~1e-16),
        # emite un UserWarning de "non-positive zlim ignorado" -- fijarlo a
        # mano sobre solo los valores positivos lo evita sin descartar
        # ningun punto del grafico.
        ax.set_zlim(positive.min(), positive.max())
        ax.set_zscale("log")
    ax.set_title(f"Frente de Pareto 3D -- {kernel}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_pareto_per_kernel(df: pd.DataFrame, outdir: str) -> None:
    agg = _aggregate_replicas(df)
    if agg.empty:
        print("[pareto_front] Sin filas con datos suficientes -- nada que graficar.", file=sys.stderr)
        return

    for kernel, kdf in agg.groupby("kernel"):
        kdf = kdf.copy()
        kdf["on_front"] = pareto_front_3d(kdf)

        csv_path = os.path.join(outdir, f"pareto_{kernel}.csv")
        kdf.sort_values(["on_front", "rel_l2"], ascending=[False, True]).to_csv(csv_path, index=False)
        print(f"[pareto_front] {kernel}: {kdf['on_front'].sum()}/{len(kdf)} "
              f"configuraciones en el frente -> {csv_path}")

        try:
            png_path = os.path.join(outdir, f"pareto_{kernel}.png")
            plot_pareto_3d(kdf, kernel, png_path)
            print(f"[pareto_front] {kernel}: visualizacion -> {png_path}")
        except ImportError as exc:
            print(f"[pareto_front] {kernel}: matplotlib no disponible, se omite "
                  f"la visualizacion ({exc}).", file=sys.stderr)

        tolerances = _default_tolerances(kdf["rel_l2"])
        guide = guideline_table(kdf, tolerances)
        guide_path = os.path.join(outdir, f"guideline_{kernel}.csv")
        guide.to_csv(guide_path, index=False)
        print(f"[pareto_front] {kernel}: tabla de directrices -> {guide_path}")
        print(guide.to_string(index=False))


# =============================================================================
# CLI
# =============================================================================

def _build_dataset_from_args(args: argparse.Namespace) -> Dataset:
    if args.results_dir:
        found = discover_from_results_dir(args.results_dir)
        return load_dataset(
            stencil_summary=found["stencil_summary"], stencil_energy=found["stencil_energy"],
            gemm_summary=found["gemm_summary"], gemm_drift=found["gemm_drift"],
            conv_summary=found["conv_summary"], conv_drift=found["conv_drift"],
        )
    return load_dataset(
        stencil_summary=args.stencil_summary, stencil_energy=args.stencil_energy,
        gemm_summary=args.gemm_summary, gemm_drift=args.gemm_drift,
        conv_summary=args.conv_summary, conv_drift=args.conv_drift,
    )


def _self_test() -> None:
    import tempfile

    rng = np.random.default_rng(7)
    rows_summary, rows_energy = [], []
    job_id = 0
    for fmt, base_err, base_t in [("wmma_fp16_sp", 1e-3, 1.0), ("wmma_bf16_sp", 1e-2, 1.3)]:
        for k in (0, 1, 2, 4, 8, 16):
            for _rep in range(3):
                job_id += 1
                rel_l2 = (base_err if k == 0 else base_err * (0.5 ** k)) * (1 + rng.normal(0, 0.02))
                rel_l2 = max(rel_l2, 1e-16)
                t_iter = base_t * (1 + 0.08 * k) * (1 + rng.normal(0, 0.02))
                rows_summary.append(dict(
                    job_id=job_id, kernel="stencil", nx=1024, ny=1024, iters=20, kahan="off",
                    route=fmt, t_iter_ms=t_iter, t_total_ms=t_iter * 20, gflops=900,
                    speedup_cpu="", speedup_fp32="", t_kernel_ms="", t_convert_ms="",
                    t_checkpoint_ms="", rel_l2=rel_l2, rel_linf=rel_l2 * 2, max_abs="",
                    rel_l2_prop="", rel_linf_prop="", first_nonfinite="", store_rel_norm="",
                    store_rel_max_guarded="", store_excluded_count="", store_eval_iter="",
                    energy_gpu_j="", energy_cpu_j="", energy_total_j="", edp_j_s="",
                    joules_per_gflop="", onset_checkpoint="", anchor_every=k,
                ))
                rows_energy.append(dict(
                    job_id=job_id, kernel="stencil", nx=1024, ny=1024, iters=20, kahan="off",
                    route=fmt, energy_gpu_j_per_iter=(0.5 + 0.05 * k) * (1 + rng.normal(0, 0.02)),
                    energy_window_reliable=1,
                ))

    with tempfile.TemporaryDirectory() as tmp:
        p_sum = os.path.join(tmp, "summary_stencil_1.csv")
        p_en = os.path.join(tmp, "energy_stencil_1.csv")
        pd.DataFrame(rows_summary).to_csv(p_sum, index=False)
        pd.DataFrame(rows_energy).to_csv(p_en, index=False)

        outdir = os.path.join(tmp, "out")
        os.makedirs(outdir, exist_ok=True)
        ds = load_dataset(stencil_summary=[p_sum], stencil_energy=[p_en])
        run_pareto_per_kernel(ds.df, outdir)
        produced = sorted(os.listdir(outdir))
        print(f"\n[self-test] Archivos producidos en {outdir}: {produced}")
        assert any(f.startswith("pareto_stencil.csv") for f in produced)
        assert any(f.startswith("guideline_stencil.csv") for f in produced)
        pareto_df = pd.read_csv(os.path.join(outdir, "pareto_stencil.csv"))
        assert pareto_df["on_front"].any(), "el frente de Pareto salio vacio"
    print("\nSELF-TEST OK: pareto_front.py corre end-to-end sobre datos sinteticos "
          "sin excepciones y produce un frente no vacio.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Etapa 9 del plan: Frente de Pareto 3D (tiempo, energia, error), un frente por kernel.")
    parser.add_argument("--results-dir", help="Autodetecta CSV por prefijo en este directorio.")
    parser.add_argument("--stencil-summary", nargs="*", default=[])
    parser.add_argument("--stencil-energy", nargs="*", default=[])
    parser.add_argument("--gemm-summary", nargs="*", default=[])
    parser.add_argument("--gemm-drift", nargs="*", default=[])
    parser.add_argument("--conv-summary", nargs="*", default=[])
    parser.add_argument("--conv-drift", nargs="*", default=[])
    parser.add_argument("--outdir", default="pareto_out")
    parser.add_argument("--self-test", action="store_true",
                         help="Ignora el resto de flags: genera datos sinteticos y corre el pipeline end-to-end.")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    os.makedirs(args.outdir, exist_ok=True)
    dataset = _build_dataset_from_args(args)
    if dataset.df.empty:
        raise SystemExit("No se cargo ninguna fila -- revisa las rutas/patrones de CSV pasados.")

    print(f"[pareto_front] {len(dataset.df)} filas normalizadas, "
          f"kernels presentes: {sorted(dataset.df['kernel'].unique())}")
    run_pareto_per_kernel(dataset.df, args.outdir)


if __name__ == "__main__":
    main()
