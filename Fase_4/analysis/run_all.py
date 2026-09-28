#!/usr/bin/env python3
"""Ejecuta el pipeline en orden; aborta si build_canonical (asserts) falla.
Colab: `%run run_all.py` tras editar config.py (unico bloque CONFIG)."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for step in ("build_canonical.py", "fig3_anchor.py", "fig4_speedup.py", "fig5_edp.py", "fig6_pareto.py",
            "fig7_variability.py", "fig8_stencil_anchor.py", "fig9_pareto2d_color.py", "fig10_k_trajectories.py",
            "fig11_k_forest.py", "fig12_fp16_bf16_ratio.py", "fig13_divergence_map.py", "fig14_stencil_survival.py",
            "fig15_size_sensitivity.py"):
    # fig9 depende de tables/F6_points.csv (fig6); fig11 depende de tables/F3_data.csv (fig3) -- por eso van despues.
    print(f"\n=== {step} ===", flush=True)
    r = subprocess.run([sys.executable, str(HERE / step)])
    if r.returncode != 0:
        sys.exit(f"FALLO en {step}")
