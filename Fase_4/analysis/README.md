# Fase_4/analysis — pipeline de análisis (Figuras 3–8)

`python run_all.py` → `build_canonical.py` (asserts duros + `out/audit_report.md`) → `fig3_anchor.py` → `fig4_speedup.py` → `fig5_edp.py` → `fig6_pareto.py` → `fig7_variability.py` → `fig8_stencil_anchor.py`.
Rutas solo en `config.py` (bloque `CONFIG`; también `PACCA_DATA_ROOT` / `PACCA_ANALYSIS_OUT`). Solo lectura sobre los CSV/logs crudos. Requiere pandas, numpy, scipy, matplotlib (sin statsmodels). Colab: subir esta carpeta, editar `config.py`, `%run run_all.py`.
Salidas en `out/`: `canonical_*.csv`, `replicas_*.csv` (incluye `replicas_stencil.csv`), `raw_energy_samples.csv`, `audit_report.md`, `tables/`, `figures/`. Decisiones y pendientes: `DECISIONS.md`.

**F7 — Variabilidad (8 réplicas rN, los 3 kernels).** Media/SD/CV%/IC95 de `t_iter_ms` entre réplicas SLURM independientes, por kernel×formato×K∈{0,1,5}. Solo tiempo (las réplicas no tienen ventanas de energía fiables y el error es determinista). Necesitó `build_replicas_stencil()` nuevo en `build_canonical.py` (Stencil no tenía extractor de réplicas; sus candidatos son rutas `*_SP`, no `*_comp`).

**F8 — Efecto del anclaje K en Stencil (producción, sin réplicas).** Único kernel sin una figura de efecto de K hasta ahora, aunque el barrido K∈{0,1,8,32} ya existía en producción. Dos paneles normalizados a K=0 por malla: costo temporal y error (`rel_l2_prop`), solo WMMA_BF16_SP (WMMA_FP16_SP no es finito a h=50 en ningún K, incluido K=32 — documentado en la figura, no excluido en silencio).

Ambas figuras salieron de revisar los notebooks de Colab de `~/Escritorio` (ver bitácora de la conversación): eran los dos huecos reales sin cubrir por F3–F6; el resto de ideas de esos notebooks ya estaba cubierto (o mejor) por F3–F6.
