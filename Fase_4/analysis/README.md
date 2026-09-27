# Fase_4/analysis — pipeline de análisis (Figuras 3–6)

`python run_all.py` → `build_canonical.py` (asserts duros + `out/audit_report.md`) → `fig3_anchor.py` → `fig4_speedup.py` → `fig5_edp.py` → `fig6_pareto.py`.
Rutas solo en `config.py` (bloque `CONFIG`; también `PACCA_DATA_ROOT` / `PACCA_ANALYSIS_OUT`). Solo lectura sobre los CSV/logs crudos. Requiere pandas, numpy, scipy, matplotlib (sin statsmodels). Colab: subir esta carpeta, editar `config.py`, `%run run_all.py`.
Salidas en `out/`: `canonical_*.csv`, `replicas_*.csv`, `raw_energy_samples.csv`, `audit_report.md`, `tables/`, `figures/`. Decisiones y pendientes: `DECISIONS.md`.
