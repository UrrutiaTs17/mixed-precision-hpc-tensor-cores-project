#!/usr/bin/env python3
"""Fase_4/tools/common_analysis.py

Capa de normalizacion compartida entre run_statistics.py (Etapa 7 del plan:
ANOVA/t-test) y pareto_front.py (Etapa 9: Frente de Pareto 3D). Los tres
kernels emiten esquemas de CSV DISTINTOS (ver Fase_3/tools/README.md sobre
por que extract_csv.py de Stencil y extract_csv_chained.py de GEMM/Conv no
comparten codigo) -- este modulo es el UNICO lugar que sabe traducir ambos
esquemas a una tabla larga comun, para que run_statistics.py y
pareto_front.py no tengan que repetir esa logica cada uno por su lado.

Columnas de la tabla normalizada (una fila = una configuracion, replica
identificada por job_id):
    job_id, kernel, format, treatment, anchor_every, size, iters,
    route, rel_l2, rel_linf, t_iter_ms, energy_gpu_j_per_iter,
    energy_window_reliable

NO intenta hacer "size" comparable ENTRE kernels (GEMM.n, Convolucion.hw y
Stencil.nx no son la misma magnitud fisica) -- el propio plan pide un Pareto
POR kernel, no uno combinado (Etapa 9, correccion sobre el frente unico), asi
que ninguna funcion de este modulo asume que "size" se pueda comparar entre
kernels. Se preserva tal cual (numerico) para poder graficar/agrupar DENTRO
de un kernel.

Nada de esto se ha corrido contra datos reales de PACCA (no hay GPU en este
entorno de desarrollo) -- se disenio para ser correcto contra el esquema de
columnas que ya documentan Fase_3/tools/README.md y Fase_4/tools/README.md,
y se probo con CSV sinteticos (ver el bloque `if __name__ == "__main__"`
de este archivo, que genera un ejemplo minimo y lo normaliza como smoke
test). Antes de confiar en un resultado real, correr ese smoke test y
verificar a mano un par de filas del CSV real contra la tabla normalizada
que produce.
"""
from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

# Tratamientos validos, en el orden en que tiene sentido compararlos
# (de menos a mas agresivo). "reference" NO es un tratamiento -- son las
# rutas de referencia (gpu_fp64, cpu_fp64, gpu_fp32) que sirven como
# denominador de error/tiempo, nunca como nivel del factor "tratamiento"
# (ver correccion #1c del plan: mezclar referencia y tratamiento en el mismo
# factor confunde "efecto de la compensacion" con "efecto de usar la
# libreria en vez de WMMA propio").
TREATMENT_NONE = "none"
TREATMENT_KAHAN_LOCAL = "kahan_local"
TREATMENT_SPATIAL = "spatial"
TREATMENT_COMP = "comp"          # GEMM/Convolucion: unica compensacion que tienen
TREATMENT_REFERENCE = "reference"

STENCIL_REFERENCE_ROUTES = {"gpu_fp32", "gpu_fp64", "cpu_fp64",
                             "ncu_gpu_fp32", "ncu_gpu_fp64", "ncu_cpu_fp64"}

NORMALIZED_COLUMNS = [
    "job_id", "kernel", "format", "treatment", "anchor_every", "size",
    "iters", "route", "rel_l2", "rel_linf", "t_iter_ms",
    "energy_gpu_j_per_iter", "energy_window_reliable",
]


def _derive_treatment_stencil(route: str, kahan: str) -> str:
    route_l = (route or "").strip().lower()
    if route_l in STENCIL_REFERENCE_ROUTES:
        return TREATMENT_REFERENCE
    if route_l.endswith("_sp"):
        return TREATMENT_SPATIAL
    if str(kahan).strip().lower() == "on":
        return TREATMENT_KAHAN_LOCAL
    return TREATMENT_NONE


def _derive_format_stencil(route: str) -> str:
    route_l = (route or "").strip().lower()
    if "fp16" in route_l:
        return "FP16"
    if "bf16" in route_l:
        return "BF16"
    if "fp64" in route_l:
        return "FP64"
    if "fp32" in route_l:
        return "FP32"
    return "NA"


def _derive_from_chained_route(route: str) -> "tuple[str, str]":
    # route = "<FORMATO>_none" o "<FORMATO>_comp" (ver
    # Fase_3/tools/extract_csv_chained.py, route_format()).
    fmt, _, suffix = (route or "").rpartition("_")
    fmt = fmt or route or "NA"
    treatment = TREATMENT_COMP if suffix == "comp" else TREATMENT_NONE
    return fmt.upper(), treatment


def load_stencil_summary(paths: Iterable[str]) -> pd.DataFrame:
    """Carga y normaliza uno o mas summary_stencil_*.csv (extract_csv.py)."""
    frames = []
    for path in paths:
        df = pd.read_csv(path, dtype=str)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    raw = pd.concat(frames, ignore_index=True)

    out = pd.DataFrame()
    out["job_id"] = raw["job_id"]
    out["kernel"] = "stencil"
    out["format"] = raw["route"].apply(_derive_format_stencil)
    out["treatment"] = [
        _derive_treatment_stencil(r, k) for r, k in zip(raw["route"], raw["kahan"])
    ]
    out["anchor_every"] = pd.to_numeric(raw.get("anchor_every", 0), errors="coerce").fillna(0).astype(int)
    # nx==ny en todas las campanas de este proyecto (mallas cuadradas) --
    # si algun dia deja de serlo, esto silenciosamente usa solo nx; falla
    # de forma visible en el smoke test (columna size no coincidiria con lo
    # esperado), no en produccion silenciosa.
    out["size"] = pd.to_numeric(raw["nx"], errors="coerce")
    out["iters"] = pd.to_numeric(raw["iters"], errors="coerce")
    out["route"] = raw["route"]
    out["rel_l2"] = pd.to_numeric(raw.get("rel_l2"), errors="coerce")
    out["rel_linf"] = pd.to_numeric(raw.get("rel_linf"), errors="coerce")
    out["t_iter_ms"] = pd.to_numeric(raw.get("t_iter_ms"), errors="coerce")
    # energy_gpu_j_per_iter vive en el CSV de energia (energy_*.csv), NO en
    # el de summary -- summary solo trae energy_gpu_j (total, no por iter).
    # Quien quiera energia debe pasar tambien los energy_*.csv (ver
    # load_stencil_energy) y unirlos por (job_id, route) -- este loader deja
    # la columna en NaN si no se le paso ese CSV, en vez de inventar un
    # valor dividiendo aqui mismo (evita duplicar la logica de filtrado de
    # ventana confiable que ya vive en extract_csv.py).
    out["energy_gpu_j_per_iter"] = float("nan")
    out["energy_window_reliable"] = pd.NA
    return out[NORMALIZED_COLUMNS]


def load_stencil_energy(paths: Iterable[str]) -> pd.DataFrame:
    """Carga energy_stencil_*.csv y devuelve (job_id, route, energy_gpu_j_per_iter,
    energy_window_reliable) para unir con load_stencil_summary()."""
    frames = [pd.read_csv(p, dtype=str) for p in paths]
    if not frames:
        return pd.DataFrame(columns=["job_id", "route", "energy_gpu_j_per_iter", "energy_window_reliable"])
    raw = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame()
    out["job_id"] = raw["job_id"]
    out["route"] = raw["route"]
    out["energy_gpu_j_per_iter"] = pd.to_numeric(raw.get("energy_gpu_j_per_iter"), errors="coerce")
    out["energy_window_reliable"] = raw.get("energy_window_reliable")
    return out


def load_chained_summary(paths: Iterable[str], kernel: str) -> pd.DataFrame:
    """Carga y normaliza uno o mas summary_{gemm,conv}_*.csv (extract_csv_chained.py).

    A diferencia de Stencil, aqui `window_reliable`/`energy_gpu_j` YA vienen
    en el mismo summary (ver Fase_3/GEMM/README.md, esquema CSV_SUMMARY) --
    no hace falta un segundo CSV de energia.
    """
    frames = [pd.read_csv(p, dtype=str) for p in paths]
    if not frames:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    raw = pd.concat(frames, ignore_index=True)

    out = pd.DataFrame()
    out["job_id"] = raw["job_id"]
    out["kernel"] = kernel
    fmt_treat = [_derive_from_chained_route(r) for r in raw["route"]]
    out["format"] = [ft[0] for ft in fmt_treat]
    out["treatment"] = [ft[1] for ft in fmt_treat]
    out["anchor_every"] = pd.to_numeric(raw.get("anchor_every", 0), errors="coerce").fillna(0).astype(int)
    out["size"] = pd.to_numeric(raw["size"], errors="coerce")
    out["iters"] = pd.to_numeric(raw["iters"], errors="coerce")
    out["route"] = raw["route"]
    out["rel_l2"] = float("nan")  # summary de GEMM/Conv no trae rel_l2 -- viene en drift_*.csv
    out["rel_linf"] = float("nan")
    out["t_iter_ms"] = pd.to_numeric(raw.get("t_iter_ms"), errors="coerce")
    energy_j = pd.to_numeric(raw.get("energy_gpu_j"), errors="coerce")
    iters_num = pd.to_numeric(raw.get("iters"), errors="coerce")
    out["energy_gpu_j_per_iter"] = energy_j / iters_num
    out["energy_window_reliable"] = raw.get("window_reliable")
    return out[NORMALIZED_COLUMNS]


def load_chained_drift(paths: Iterable[str], kernel: str) -> pd.DataFrame:
    """Carga drift_{gemm,conv}_*.csv -- (job_id, route, size, iter, rel_l2,
    rel_linf) para unir con load_chained_summary() por (job_id, route, size).
    Se queda con la ULTIMA fila de drift por (job_id, route, size) -- el
    error final de la corrida, comparable con el t_iter_ms/energia agregados
    de todo el barrido de iteraciones que reporta el summary."""
    frames = [pd.read_csv(p, dtype=str) for p in paths]
    if not frames:
        return pd.DataFrame(columns=["job_id", "route", "size", "rel_l2", "rel_linf"])
    raw = pd.concat(frames, ignore_index=True)
    raw["iter"] = pd.to_numeric(raw["iter"], errors="coerce")
    raw = raw.sort_values("iter").groupby(["job_id", "route", "size"], as_index=False).last()
    out = pd.DataFrame()
    out["job_id"] = raw["job_id"]
    out["route"] = raw["route"]
    out["size"] = pd.to_numeric(raw["size"], errors="coerce")
    out["rel_l2"] = pd.to_numeric(raw["rel_l2"], errors="coerce")
    out["rel_linf"] = pd.to_numeric(raw["rel_linf"], errors="coerce")
    return out


def merge_energy_into_stencil(summary: pd.DataFrame, energy: pd.DataFrame) -> pd.DataFrame:
    if energy.empty:
        return summary
    merged = summary.drop(columns=["energy_gpu_j_per_iter", "energy_window_reliable"]).merge(
        energy, on=["job_id", "route"], how="left")
    return merged[NORMALIZED_COLUMNS]


def merge_drift_into_chained(summary: pd.DataFrame, drift: pd.DataFrame) -> pd.DataFrame:
    if drift.empty:
        return summary
    merged = summary.drop(columns=["rel_l2", "rel_linf"]).merge(
        drift.drop(columns=["size"]).rename(columns={}),
        on=["job_id", "route"], how="left")
    # El merge de arriba no incluye "size" en las claves porque drift.size
    # es redundante con summary.size (misma corrida) y a veces llega como
    # string/num distinto tras el groupby -- se preserva la columna size
    # ORIGINAL de summary, no la de drift.
    return merged[NORMALIZED_COLUMNS]


@dataclass
class Dataset:
    """Tabla normalizada mas metadatos utiles para reportar cuantas replicas
    hay por celda del diseno antes de correr cualquier prueba estadistica."""
    df: pd.DataFrame

    def cells(self, extra_keys: "list[str] | None" = None) -> pd.DataFrame:
        keys = ["kernel", "format", "treatment", "anchor_every", "size", "iters"]
        # dict.fromkeys en vez de un simple "+=": preserva el orden y evita
        # una columna duplicada si extra_keys repite una clave ya presente
        # (p.ej. un llamador pasando extra_keys=["format"] sin saber que ya
        # esta incluida por defecto -- reset_index() fallaria con "cannot
        # insert X, already exists" en vez de deduplicar en silencio).
        if extra_keys:
            keys = list(dict.fromkeys(keys + list(extra_keys)))
        return (self.df.groupby(keys, dropna=False)
                .agg(n_replicas=("job_id", "nunique"))
                .reset_index()
                .sort_values(keys))

    def warn_low_replicas(self, min_replicas: int = 2, extra_keys: "list[str] | None" = None) -> None:
        cells = self.cells(extra_keys)
        low = cells[cells["n_replicas"] < min_replicas]
        if not low.empty:
            print(f"AVISO: {len(low)} celda(s) del diseno tienen menos de "
                  f"{min_replicas} replica(s) (job_id distintos) -- la prueba "
                  "estadistica sobre esas celdas no es confiable:", file=sys.stderr)
            print(low.to_string(index=False), file=sys.stderr)


def _expand_globs(patterns: Iterable[str]) -> "list[str]":
    paths: list[str] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])
    return [p for p in paths if os.path.isfile(p)]


def discover_from_results_dir(results_dir: str) -> dict:
    """Autodetecta los CSV de un directorio de resultados (el que llenan los
    .sbatch al final de cada corrida) por prefijo de nombre de archivo --
    ver los patrones drift_/summary_/energy_ que ya usan extract_csv.py y
    extract_csv_chained.py."""
    def _glob(prefix, kernel):
        return sorted(glob.glob(os.path.join(results_dir, f"{prefix}_{kernel}_*.csv")))

    return {
        "stencil_summary": _glob("summary", "stencil"),
        "stencil_energy": _glob("energy", "stencil"),
        "gemm_summary": _glob("summary", "gemm"),
        "gemm_drift": _glob("drift", "gemm"),
        "conv_summary": _glob("summary", "conv"),
        "conv_drift": _glob("drift", "conv"),
    }


def load_dataset(
    stencil_summary: Optional[Iterable[str]] = None,
    stencil_energy: Optional[Iterable[str]] = None,
    gemm_summary: Optional[Iterable[str]] = None,
    gemm_drift: Optional[Iterable[str]] = None,
    conv_summary: Optional[Iterable[str]] = None,
    conv_drift: Optional[Iterable[str]] = None,
) -> Dataset:
    """Punto de entrada principal: recibe listas de rutas/patrones glob por
    kernel y devuelve un Dataset con la tabla larga normalizada y unida
    (drift+summary+energia), lista para run_statistics.py/pareto_front.py."""
    frames = []

    stencil_summary = _expand_globs(stencil_summary or [])
    stencil_energy = _expand_globs(stencil_energy or [])
    if stencil_summary:
        s = load_stencil_summary(stencil_summary)
        e = load_stencil_energy(stencil_energy)
        frames.append(merge_energy_into_stencil(s, e))

    gemm_summary = _expand_globs(gemm_summary or [])
    gemm_drift = _expand_globs(gemm_drift or [])
    if gemm_summary:
        s = load_chained_summary(gemm_summary, "gemm")
        d = load_chained_drift(gemm_drift, "gemm")
        frames.append(merge_drift_into_chained(s, d))

    conv_summary = _expand_globs(conv_summary or [])
    conv_drift = _expand_globs(conv_drift or [])
    if conv_summary:
        s = load_chained_summary(conv_summary, "conv")
        d = load_chained_drift(conv_drift, "conv")
        frames.append(merge_drift_into_chained(s, d))

    if not frames:
        return Dataset(pd.DataFrame(columns=NORMALIZED_COLUMNS))
    df = pd.concat(frames, ignore_index=True)
    return Dataset(df)


if __name__ == "__main__":
    # Smoke test SIN datos reales de PACCA: construye CSV sinteticos con el
    # esquema exacto que producen extract_csv.py/extract_csv_chained.py y
    # verifica que load_dataset() los normaliza sin excepciones y con los
    # valores esperados. No valida que el FENOMENO fisico tenga sentido
    # (eso exige datos reales) -- solo que el codigo no crashea contra el
    # esquema documentado.
    import io
    import tempfile

    stencil_summary_csv = """job_id,kernel,nx,ny,iters,kahan,route,t_iter_ms,t_total_ms,gflops,speedup_cpu,speedup_fp32,t_kernel_ms,t_convert_ms,t_checkpoint_ms,rel_l2,rel_linf,max_abs,rel_l2_prop,rel_linf_prop,first_nonfinite,store_rel_norm,store_rel_max_guarded,store_excluded_count,store_eval_iter,energy_gpu_j,energy_cpu_j,energy_total_j,edp_j_s,joules_per_gflop,onset_checkpoint,anchor_every
1,stencil,1024,1024,20,off,wmma_fp16_sp,1.2,24.0,900.1,,,,,,1e-05,2e-05,,,,,,,,,,,,,,0
2,stencil,1024,1024,20,off,wmma_fp16_sp,1.3,26.0,880.0,,,,,,1.1e-05,2.1e-05,,,,,,,,,,,,,,0
"""
    gemm_summary_csv = """job_id,kernel,size,format,comp,anchor_every,route,iters,t_iter_ms,t_total_ms,gflops,energy_gpu_j,window_reliable,gpu_segments
9,gemm,1024,FP16,on,5,FP16_comp,20,1.8,36.0,1300.2,14.1,1,1
"""
    gemm_drift_csv = """job_id,kernel,size,format,comp,anchor_every,route,iter,rel_l2,rel_linf,solution_finite
9,gemm,1024,FP16,on,5,FP16_comp,20,3.1e-05,7.2e-05,1
"""

    with tempfile.TemporaryDirectory() as tmp:
        p_ssum = os.path.join(tmp, "summary_stencil_1.csv")
        p_gsum = os.path.join(tmp, "summary_gemm_9.csv")
        p_gdrift = os.path.join(tmp, "drift_gemm_9.csv")
        for path, content in [(p_ssum, stencil_summary_csv), (p_gsum, gemm_summary_csv),
                               (p_gdrift, gemm_drift_csv)]:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)

        ds = load_dataset(stencil_summary=[p_ssum], gemm_summary=[p_gsum], gemm_drift=[p_gdrift])
        print(ds.df.to_string(index=False))
        assert len(ds.df) == 3, f"esperaba 3 filas, salieron {len(ds.df)}"
        assert set(ds.df["kernel"]) == {"stencil", "gemm"}
        gemm_row = ds.df[ds.df["kernel"] == "gemm"].iloc[0]
        assert gemm_row["treatment"] == TREATMENT_COMP
        assert gemm_row["anchor_every"] == 5
        assert abs(gemm_row["rel_l2"] - 3.1e-05) < 1e-12
        ds.warn_low_replicas(min_replicas=2)
        print("\nSMOKE TEST OK: common_analysis.py normaliza ambos esquemas sin excepciones.")
