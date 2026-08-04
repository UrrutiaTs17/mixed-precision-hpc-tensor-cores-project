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

## Which WMMA route labels to expect

Both sbatch scripts default to **spatial** compensation (`SPATIAL_COMP=on`), so
a run launched with no `--export` produces the `WMMA_FP16_SP` / `WMMA_BF16_SP`
routes. The default was inverted because spatial compensation is the only one
of the three policies that mitigates the error: local Kahan (`--kahan on`) is
indistinguishable from no compensation (`rel_l2` matches `--kahan off` to the
4th significant digit, horizon unchanged at FP16=28 / BF16=138) at ~1.97x the
per-iteration cost, while spatial compensation drops `rel_l2` by 5-6 orders of
magnitude and moves the BF16 horizon from 138 to 142 — the same horizon as
classic FP32 — at ~1.56x.

The local-Kahan routes (`WMMA_FP16` / `WMMA_BF16`, with `kahan` off and on) are
unchanged and still reachable; they are just no longer the default path:

```bash
sbatch --export=ALL,SPATIAL_COMP=off run_stencil_horizon.sbatch
```

`SPATIAL_COMP=on` forces `KAHAN_LIST=off`, because the binary rejects
`--kahan on` together with `--spatial-comp on` (alternative policies, not
stackable layers). No CSV column schema changed: only which route rows a
default run emits.
