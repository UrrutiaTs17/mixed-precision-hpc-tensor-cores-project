#!/usr/bin/env python3
"""Fase_4/tools/run_statistics.py

Etapa 7 del plan de finalizacion ("Plan de Precision Mixta", documento
normativo): estadistica inferencial sobre las campanas de variabilidad
(replicas) de los tres kernels. Opera sobre la tabla normalizada que produce
common_analysis.py -- lee este archivo junto con ese modulo, no por separado.

Dos analisis, correspondientes a los dos incisos de la Etapa 7:

1. ANOVA FACTORIAL (formato x tratamiento x horizonte) + post-hoc Tukey HSD
   para el contraste kahan_local vs. spatial vs. none -- el que motiva el
   objetivo 3. Solo aplica a STENCIL: es el unico kernel con tres niveles de
   tratamiento (kahan_local es exclusivo de Stencil, ver el comentario de
   cabecera de common_analysis.py sobre por que la ruta librer¡a-TC
   encadenada sin compensacion es una comparacion aparte, correccion #1c, no
   un cuarto nivel de este mismo factor).

2. EFECTO DEL ANCLA FP64 (--anchor-every K) -- en los TRES kernels, con
   "kernel" como factor adicional cuando hay datos de mas de uno. K se trata
   como variable ORDINAL (regresion sobre log(K+1)) ADEMAS de (no en vez de)
   un ANOVA categorico sobre los niveles de K -- ver la correccion explicita
   de la Etapa 7 sobre este punto en el documento de plan.

USO:
    # Autodetectar CSV en un directorio de resultados (o varios):
    python3 run_statistics.py --results-dir results/ --outdir stats_out/

    # O apuntar a archivos/patrones concretos:
    python3 run_statistics.py \\
        --stencil-summary "results/summary_stencil_*.csv" \\
        --stencil-energy  "results/energy_stencil_*.csv" \\
        --gemm-summary    "results/summary_gemm_*.csv" \\
        --gemm-drift      "results/drift_gemm_*.csv" \\
        --conv-summary    "results/summary_conv_*.csv" \\
        --conv-drift      "results/drift_conv_*.csv" \\
        --outdir stats_out/

SIN DATOS REALES DE PACCA (nada se ha corrido en GPU en este entorno de
desarrollo), este script se probo con datos SINTETICOS generados a proposito
(ver `python3 run_statistics.py --self-test`) para verificar que no crashea
y que produce las tablas esperadas -- NO que las conclusiones tengan sentido
fisico, eso exige datos reales. Antes de citar cualquier resultado de este
script en la tesis, correlo contra una campana real y verifica a mano que
las celdas del diseno tienen las replicas que esperabas (ver
Dataset.warn_low_replicas en common_analysis.py, que este script llama
automaticamente y advierte por stderr si alguna celda tiene menos de 2).
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_analysis import (  # noqa: E402
    TREATMENT_COMP,
    TREATMENT_KAHAN_LOCAL,
    TREATMENT_NONE,
    TREATMENT_SPATIAL,
    Dataset,
    discover_from_results_dir,
    load_dataset,
)

METRICS = ["rel_l2", "t_iter_ms", "energy_gpu_j_per_iter"]


def _build_factorial_formula(metric: str, df: pd.DataFrame, factors: "list[str]") -> "str | None":
    """Construye 'metric ~ C(f1) * C(f2) * ...' usando SOLO los factores que
    tienen >=2 niveles distintos en `df`. Un factor con un solo nivel (p.ej.
    una campana de humo con un unico valor de `iters`) no aporta ninguna
    variacion que explicar, y peor: statsmodels.anova_lm(typ=2) revienta con
    "must have at least one row in constraint matrix" al intentar construir
    el test F de un termino sin grados de libertad -- un crash de bajo nivel
    en vez de un mensaje entendible. Se filtra ANTES de llegar ahi. Devuelve
    None si queda menos de un factor (nada que modelar)."""
    varying = [f for f in factors if df[f].nunique() >= 2]
    dropped = [f for f in factors if f not in varying]
    if dropped:
        print(f"[{metric}] factor(es) sin variacion en estos datos (un solo nivel), "
              f"excluidos del modelo: {dropped}", file=sys.stderr)
    if not varying:
        return None
    return metric + " ~ " + " * ".join(f"C({f})" for f in varying)


def _enough_residual_dof(model, label: str) -> bool:
    """True si el modelo ajustado tiene grados de libertad residuales
    positivos. Con pocas filas y un formula con varias interacciones
    (C(format)*C(treatment)*C(iters), o C(anchor_every)*C(kernel)), el
    diseno puede quedar EXACTAMENTE saturado (tantos parametros como
    observaciones) o incluso sin suficientes filas -- ahi df_resid es 0 o
    negativo, los coeficientes salen inf/NaN, y anova_lm() revienta con un
    error de algebra lineal (QR sobre una matriz con NaN) en vez de un
    mensaje entendible. Se detecta ANTES de llamar a anova_lm() para poder
    avisar con contexto (cuantas filas, que formula) en vez de propagar la
    excepcion cruda de statsmodels/scipy."""
    if model.df_resid is None or model.df_resid < 1:
        print(f"[{label}] datos insuficientes para este modelo: "
              f"{int(model.nobs)} observacion(es), {model.df_resid} grados de "
              "libertad residuales -- hacen falta mas replicas o menos "
              "niveles de factor para que el ANOVA sea estimable. Se omite.",
              file=sys.stderr)
        return False
    return True


def _require_statsmodels():
    try:
        import statsmodels  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Falta 'statsmodels' (ANOVA factorial + Tukey HSD lo necesitan; "
            "scipy.stats no trae Tukey HSD). Instalar via environment.yml o "
            "'pip install statsmodels' -- ver REQUIREMENTS.md."
        ) from exc


# =============================================================================
# 1. ANOVA factorial (formato x tratamiento x horizonte) -- solo Stencil.
# =============================================================================

def anova_treatment_stencil(df: pd.DataFrame, outdir: str) -> None:
    _require_statsmodels()
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    stencil = df[(df["kernel"] == "stencil") &
                 (df["treatment"].isin([TREATMENT_NONE, TREATMENT_KAHAN_LOCAL, TREATMENT_SPATIAL]))].copy()
    if stencil.empty:
        print("[anova_treatment_stencil] Sin filas de Stencil con tratamiento "
              "none/kahan_local/spatial -- nada que analizar.", file=sys.stderr)
        return

    Dataset(stencil).warn_low_replicas(min_replicas=2, extra_keys=["format"])

    for metric in METRICS:
        sub = stencil.dropna(subset=[metric])
        if sub.empty or sub[metric].nunique() < 2:
            print(f"[anova_treatment_stencil] '{metric}': datos insuficientes "
                  "(vacio o sin variabilidad) -- se omite.", file=sys.stderr)
            continue

        # "iters" es el proxy de "horizonte" (ver comentario de cabecera):
        # el esquema de CSV no trae una columna "horizonte" explicita: cada
        # valor DISTINTO de iters presente en la campana ES un nivel de
        # horizonte, tal como se disenio el barrido en el .sbatch.
        formula = _build_factorial_formula(metric, sub, ["format", "treatment", "iters"])
        if formula is None:
            print(f"[anova_treatment_stencil] '{metric}': ningun factor tiene "
                  "variacion -- se omite.", file=sys.stderr)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # celdas desbalanceadas son normales aqui
            model = ols(formula, data=sub).fit()
        if not _enough_residual_dof(model, f"anova_treatment_stencil:{metric}"):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            table = anova_lm(model, typ=2)

        anova_path = os.path.join(outdir, f"anova_stencil_{metric}.csv")
        table.to_csv(anova_path)
        print(f"[anova_treatment_stencil] '{metric}': ANOVA factorial -> {anova_path}")
        print(table.to_string())

        # Tukey HSD POR FORMATO: si hubiera interaccion formato x tratamiento
        # (que la tabla ANOVA de arriba ya reporta), agrupar todos los
        # formatos en un solo Tukey HSD mezclaria ese efecto con el que
        # interesa (kahan_local vs spatial vs none). Un Tukey HSD por
        # formato evita esa confusion.
        for fmt, group in sub.groupby("format"):
            if group["treatment"].nunique() < 2:
                continue
            tukey = pairwise_tukeyhsd(endog=group[metric].to_numpy(),
                                       groups=group["treatment"].to_numpy(), alpha=0.05)
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:],
                                     columns=tukey._results_table.data[0])
            tukey_path = os.path.join(outdir, f"tukey_stencil_{metric}_{fmt}.csv")
            tukey_df.to_csv(tukey_path, index=False)
            print(f"[anova_treatment_stencil] Tukey HSD '{metric}' formato={fmt} -> {tukey_path}")
            print(tukey)


# =============================================================================
# 2. Efecto del ancla FP64 (K = anchor_every) -- los tres kernels.
# =============================================================================

def anchor_k_analysis(df: pd.DataFrame, outdir: str) -> None:
    _require_statsmodels()
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    # Solo tiene sentido sobre filas donde el ancla PUDO estar activa (rutas
    # con compensacion: spatial en Stencil, comp en GEMM/Conv) -- la ruta
    # "none"/sin compensacion nunca corre con ancla (el binario lo rechaza),
    # asi que incluirla aqui solo agregaria una columna de K=0 identica a la
    # de "spatial"/"comp" sin K activo, sin aportar informacion nueva.
    sub = df[df["treatment"].isin([TREATMENT_SPATIAL, TREATMENT_COMP])].copy()
    sub = sub.dropna(subset=["anchor_every"])
    if sub.empty:
        print("[anchor_k_analysis] Sin filas con tratamiento spatial/comp -- "
              "nada que analizar.", file=sys.stderr)
        return

    n_kernels = sub["kernel"].nunique()
    Dataset(sub).warn_low_replicas(min_replicas=2, extra_keys=["kernel"] if n_kernels > 1 else None)

    # log(K+1): K=0 (ancla deshabilitada) queda en log(1)=0 en vez de -inf,
    # sin necesitar tratarlo como un caso aparte -- +1 es la transformacion
    # estandar para variables que incluyen cero en una escala logaritmica.
    sub["log_k"] = np.log1p(sub["anchor_every"].astype(float))

    for metric in METRICS:
        m = sub.dropna(subset=[metric])
        if m.empty or m[metric].nunique() < 2:
            print(f"[anchor_k_analysis] '{metric}': datos insuficientes -- se omite.", file=sys.stderr)
            continue

        # (a) ORDINAL: regresion sobre log(K+1) -- la pregunta real del
        # objetivo 4 es como cambia el resultado a medida que K CRECE, no
        # solo si los niveles difieren entre si (eso es (b), abajo). Con mas
        # de un kernel en los datos, "kernel" entra como termino aditivo (no
        # interaccion completa por defecto, para no perder grados de
        # libertad si hay pocas replicas por kernel) mas la interaccion
        # kernel x log_k, que es la que responde si el ancla ayuda distinto
        # segun el kernel (ver la nota de la etapa 6 del plan sobre FP64/TC
        # tener una brecha de costo muy distinta en GEMM que en Stencil).
        # nunique() sobre `m` (el subconjunto de ESTA metrica), no sobre
        # `sub`/n_kernels: dropna(metric) puede dejar un solo kernel aunque
        # el dataset completo tenga varios.
        kernel_varies_here = m["kernel"].nunique() > 1
        formula_ordinal = f"{metric} ~ log_k * C(kernel)" if kernel_varies_here else f"{metric} ~ log_k"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ols(formula_ordinal, data=m).fit()
        if not _enough_residual_dof(model, f"anchor_k_analysis:{metric}:ordinal"):
            continue
        coef_path = os.path.join(outdir, f"anchor_k_ordinal_{metric}.csv")
        model.params.to_frame("coef").join(model.pvalues.to_frame("p_value")).to_csv(coef_path)
        print(f"[anchor_k_analysis] '{metric}': regresion ordinal sobre log(K+1) -> {coef_path}")
        print(model.summary().tables[1])

        # (b) CATEGORICO: ANOVA sobre los niveles de K tal cual (sin asumir
        # una tendencia log-lineal) -- complementa a (a), no lo reemplaza.
        if m["anchor_every"].nunique() < 2:
            continue
        formula_cat = (f"{metric} ~ C(anchor_every) * C(kernel)" if kernel_varies_here
                        else f"{metric} ~ C(anchor_every)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_cat = ols(formula_cat, data=m).fit()
        if not _enough_residual_dof(model_cat, f"anchor_k_analysis:{metric}:categorico"):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            table_cat = anova_lm(model_cat, typ=2)
        cat_path = os.path.join(outdir, f"anchor_k_categorical_{metric}.csv")
        table_cat.to_csv(cat_path)
        print(f"[anchor_k_analysis] '{metric}': ANOVA categorico sobre K -> {cat_path}")
        print(table_cat.to_string())


# =============================================================================
# CLI
# =============================================================================

def _build_dataset_from_args(args: argparse.Namespace) -> Dataset:
    if args.results_dir:
        found = discover_from_results_dir(args.results_dir)
        for key, paths in found.items():
            print(f"[run_statistics] autodetectado {key}: {len(paths)} archivo(s)")
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
    """Genera datos sinteticos (con replicas y ruido, a diferencia del smoke
    test minimo de common_analysis.py) y corre ambos analisis end-to-end --
    ver el comentario de cabecera del archivo sobre que SI y que NO valida
    esto."""
    import tempfile

    rng = np.random.default_rng(42)
    rows_summary = []
    rows_energy = []
    treatments = [("none", "off"), ("kahan_local", "on")]
    job_id = 1000
    for fmt_route in ["wmma_fp16", "wmma_bf16"]:
        for treat_route, kahan in treatments:
            route = fmt_route if treat_route != "spatial" else fmt_route + "_sp"
            for iters in (10, 100):
                for _rep in range(5):
                    job_id += 1
                    base_err = 1e-3 if "fp16" in fmt_route else 1e-2
                    err_mult = {"none": 1.0, "kahan_local": 0.95, "spatial": 1e-5}[treat_route]
                    rel_l2 = base_err * err_mult * (1 + rng.normal(0, 0.05))
                    t_iter = (1.0 if "fp16" in fmt_route else 1.3) * (1 + rng.normal(0, 0.05))
                    rows_summary.append(dict(
                        job_id=job_id, kernel="stencil", nx=1024, ny=1024, iters=iters,
                        kahan=kahan, route=route, t_iter_ms=t_iter, t_total_ms=t_iter * iters,
                        gflops=900, speedup_cpu="", speedup_fp32="", t_kernel_ms="",
                        t_convert_ms="", t_checkpoint_ms="", rel_l2=rel_l2, rel_linf=rel_l2 * 2,
                        max_abs="", rel_l2_prop="", rel_linf_prop="", first_nonfinite="",
                        store_rel_norm="", store_rel_max_guarded="", store_excluded_count="",
                        store_eval_iter="", energy_gpu_j="", energy_cpu_j="", energy_total_j="",
                        edp_j_s="", joules_per_gflop="", onset_checkpoint="", anchor_every=0,
                    ))
                    rows_energy.append(dict(
                        job_id=job_id, kernel="stencil", nx=1024, ny=1024, iters=iters,
                        kahan=kahan, route=route,
                        energy_gpu_j_per_iter=0.6 * (1 + rng.normal(0, 0.05)),
                        energy_window_reliable=1,
                    ))
        # tratamiento spatial, con barrido de K (ancla) -- para el analisis (2)
        for k in (0, 1, 4, 16):
            for _rep in range(4):
                job_id += 1
                rel_l2 = (1e-5 if k == 0 else 1e-15 * (k ** 0.1)) * (1 + rng.normal(0, 0.05))
                t_iter = (1.0 + 0.05 * k) * (1 + rng.normal(0, 0.05))
                route = fmt_route + "_sp"
                rows_summary.append(dict(
                    job_id=job_id, kernel="stencil", nx=1024, ny=1024, iters=20,
                    kahan="off", route=route, t_iter_ms=t_iter, t_total_ms=t_iter * 20,
                    gflops=900, speedup_cpu="", speedup_fp32="", t_kernel_ms="",
                    t_convert_ms="", t_checkpoint_ms="", rel_l2=rel_l2, rel_linf=rel_l2 * 2,
                    max_abs="", rel_l2_prop="", rel_linf_prop="", first_nonfinite="",
                    store_rel_norm="", store_rel_max_guarded="", store_excluded_count="",
                    store_eval_iter="", energy_gpu_j="", energy_cpu_j="", energy_total_j="",
                    edp_j_s="", joules_per_gflop="", onset_checkpoint="", anchor_every=k,
                ))
                rows_energy.append(dict(
                    job_id=job_id, kernel="stencil", nx=1024, ny=1024, iters=20,
                    kahan="off", route=route,
                    energy_gpu_j_per_iter=(0.6 + 0.03 * k) * (1 + rng.normal(0, 0.05)),
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
        print(f"[self-test] {len(ds.df)} filas normalizadas.")
        anova_treatment_stencil(ds.df, outdir)
        anchor_k_analysis(ds.df, outdir)
        produced = sorted(os.listdir(outdir))
        print(f"\n[self-test] Archivos producidos en {outdir}: {produced}")
        assert produced, "el self-test no produjo ningun archivo de salida"
    print("\nSELF-TEST OK: run_statistics.py corre end-to-end sobre datos sinteticos "
          "sin excepciones y produce tablas ANOVA/Tukey/regresion no vacias.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Etapa 7 del plan: ANOVA factorial + Tukey HSD + efecto ordinal del ancla FP64.")
    parser.add_argument("--results-dir", help="Autodetecta CSV por prefijo (drift_/summary_/energy_) en este directorio.")
    parser.add_argument("--stencil-summary", nargs="*", default=[])
    parser.add_argument("--stencil-energy", nargs="*", default=[])
    parser.add_argument("--gemm-summary", nargs="*", default=[])
    parser.add_argument("--gemm-drift", nargs="*", default=[])
    parser.add_argument("--conv-summary", nargs="*", default=[])
    parser.add_argument("--conv-drift", nargs="*", default=[])
    parser.add_argument("--outdir", default="stats_out")
    parser.add_argument("--self-test", action="store_true",
                         help="Ignora el resto de flags: genera datos sinteticos y corre ambos analisis end-to-end.")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    os.makedirs(args.outdir, exist_ok=True)
    dataset = _build_dataset_from_args(args)
    if dataset.df.empty:
        raise SystemExit("No se cargo ninguna fila -- revisa las rutas/patrones de CSV pasados.")

    print(f"[run_statistics] {len(dataset.df)} filas normalizadas, "
          f"kernels presentes: {sorted(dataset.df['kernel'].unique())}")

    anova_treatment_stencil(dataset.df, args.outdir)
    anchor_k_analysis(dataset.df, args.outdir)


if __name__ == "__main__":
    main()
