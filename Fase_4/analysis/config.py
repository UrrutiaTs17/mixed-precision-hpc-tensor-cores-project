"""Fase_4/analysis/config.py -- UNICO bloque CONFIG del pipeline.

Colab: sube esta carpeta (o `%cd` a ella), edita DATA_ROOT/OUT_DIR aqui (o
exporta PACCA_DATA_ROOT / PACCA_ANALYSIS_OUT) y ejecuta `python run_all.py`
o cada script por separado. Ningun otro archivo contiene rutas.
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
CONFIG = {
    # Rutas (unico lugar donde se configuran)
    "DATA_ROOT": os.environ.get(
        "PACCA_DATA_ROOT", "/home/Willy/Documentos/Resultados_PACCA/campana_holder_20260917"),
    "OUT_DIR": os.environ.get("PACCA_ANALYSIS_OUT", str(HERE / "out")),
    "JOB_TAG": "7145",

    # Horizontes de error (iteracion a la que se evalua el error)
    "H_CHAINED": 40,             # GEMM y Conv
    "H_STENCIL_PRIMARY": 50,
    "H_STENCIL_SECONDARY": 10,

    # Iteraciones de cada pase (verificadas contra los CSV en build_canonical)
    "NUMERIC_ITERS": {"gemm": [20, 40, 80], "conv": [20, 40, 80],
                      "stencil": [10, 50, 100, 120]},
    "DEDICATED_ITERS": {          # pase dedicado de energia, por tamano
        "gemm": {1024: 24000, 2048: 24000, 4096: 500, 8192: 500},
        "conv": {64: 37000, 128: 37000, 256: 2500, 512: 2500},
        "stencil": {4096: 4000, 8192: 1500, 16384: 1500},
    },

    # Tolerancia de error para F5 (rel_l2 <= EPS). Default: ver DECISIONS.md
    "EPS_REL_L2": 1e-2,
    "EPS_SENSITIVITY": [1e-3, 3e-3, 1e-2, 3e-2],

    # F3
    "F3_KERNELS_FORMATS": [("gemm", "FP16"), ("gemm", "BF16"), ("conv", "FP16"), ("conv", "BF16")],
    "F3_K_LIST": [1, 5],         # K=0 es la referencia del par; K=20 (n=1) excluido
    "F3_ALPHA": 0.05,

    # F5 / F6
    "ISOCURVE_MULTIPLIERS": [0.5, 1.0, 2.0],   # E = c/T, c = mult * EDP_FP64
    "BOOTSTRAP_N": 2000,
    "SEED": 20260919,

    # Campana Stencil alpha=3/16 (holder 7757, 2026-09-28): directorio con un
    # subdirectorio por paso ({spk,off}_{num_corta_limpio,num_S,num_L,en_S,en_L}).
    # Solo 4096 y 8192 (16384 retirado por decision del responsable, MANIFIESTO.md).
    "STENCIL_A316_ROOT": os.environ.get(
        "PACCA_STENCIL_A316_ROOT", "/home/Willy/Documentos/Resultados_PACCA/campana_v2_a316/fase4_v2"),
    "STENCIL_A316_ENERGY_ITERS": {4096: 4000, 8192: 1500},

    # Salida
    "DPI": 300,
    "STENCIL_OPERATOR_LABEL": "stress",       # etiqueta de titulo hasta que exista campana alpha=3/16
}

KERNEL_DIR = {"gemm": "GEMM", "conv": "Convolution", "stencil": "Stencil"}
