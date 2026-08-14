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

### `energy_gpu_j_per_iter` and `energy_window_reliable`

NVML's energy counter is fed by the onboard power sensor, which refreshes every
~20-25 ms, so the counter advances in discrete jumps rather than continuously.
A delta taken over a window that spans only one or two refreshes carries a
quantization error of the same order as the value itself — measured in job
5153, where 12 GPU windows of 10-125 ms produced a hard floor of 3.842 J,
three readings of exactly 0 J, and 2x swings in implied power for the same
kernel. The fix is procedural, not instrumental: measure over long windows and
normalize per iteration.

`energy_window_reliable` is `1` when the GPU energy window is long enough for
that error to fall to ~5% or less, namely
`time_total_s >= 0.500 s * <number of energy segments>` (checkpointing splits
the window into several segments, and each one carries its own quantization
error). `energy_gpu_j_per_iter` is `energy_gpu_j / iters` — the quantity that
is comparable across runs with different `ITERS`, and therefore the one used
for GPU-vs-GPU comparisons between formats.

The extractor **excludes unreliable rows from any downstream average** by
blanking `energy_gpu_j_per_iter` to `NaN` when `energy_window_reliable` is not
`1` or when the NVML read was invalid. `energy_gpu_j` and
`energy_window_reliable` are left raw so the discard can be audited, and the
number of rows dropped per reason is printed on stdout. Both columns are `NaN`
on `CPU_FP32` rows: that route sets `gpu_valid` without reading NVML, so there
is no GPU window to judge.

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
