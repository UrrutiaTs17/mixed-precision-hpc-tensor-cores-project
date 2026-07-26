# Stencil CSV extraction

`extract_csv.py` reads the stdout log produced by the Fase 3 Stencil sbatch
jobs and extracts `CSV_*` tokens into five analysis files:

```bash
python3 tools/extract_csv.py \
    --input results/run_${SLURM_JOB_ID}.log \
    --outdir results \
    --job-id "${SLURM_JOB_ID}" \
    --kernel stencil
```

The `CSV_ONSET` token is folded into `summary_<kernel>_<job>.csv` as
`onset_checkpoint`. This value is checkpoint-granular, not the exact overflow
iteration. It is therefore an upper bound on the measured horizon: the onset is
detected at the first checkpoint after the real divergence occurred.

`CSV_ENERGY` is written to `energy_<kernel>_<job>.csv` and its five summary
energy columns are merged into `summary_<kernel>_<job>.csv`. `NaN` in an energy
column means that NVML or RAPL was unavailable, disabled at compile time, or
the corresponding capture failed. It is not an interpolated or smoothed value.
RAPL counters are cumulative and may show only small raw changes for very
short benchmarks.

Historical logs that only contain `CSV_DRIFT`, `CSV_REGION`, and `CSV_ONSET`
are accepted. In that case the extractor writes drift rows and partial summary
rows, while horizon, store, and energy files contain only their headers.
