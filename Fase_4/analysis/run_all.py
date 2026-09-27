#!/usr/bin/env python3
"""Ejecuta el pipeline en orden; aborta si build_canonical (asserts) falla.
Colab: `%run run_all.py` tras editar config.py (unico bloque CONFIG)."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for step in ("build_canonical.py", "fig3_anchor.py", "fig4_speedup.py", "fig5_edp.py", "fig6_pareto.py"):
    print(f"\n=== {step} ===", flush=True)
    r = subprocess.run([sys.executable, str(HERE / step)])
    if r.returncode != 0:
        sys.exit(f"FALLO en {step}")
