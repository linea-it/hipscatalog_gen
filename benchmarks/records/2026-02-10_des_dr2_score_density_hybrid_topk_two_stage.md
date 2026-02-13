# Benchmark Record: DES_DR2 score_density_hybrid (top-k two-stage)

Date: 2026-02-10
Author: Luigi Silva / hipscatalog-gen local run logs
Goal: compare `score_density_hybrid` stage-1 runtime before vs after the two-stage exact top-k optimization.

## Context

- Input catalog: `/data/public/des/dr2/secondary/catalogs/main/hats`
- Selection mode: `score_density_hybrid`
- Score column: `MAG_AUTO_R_DERED`
- level_limit: `11`
- Input rows: `691483608`
- Cluster hardware (user report): each node with `128 GB RAM` and `56 cores`

## Configuration used in both runs

Same configuration before and after top-k optimization.
Key runtime parameters:

```yaml
cluster:
  mode: slurm
  n_workers: 10
  threads_per_worker: 8
  memory_per_worker: "96GB"
  low_memory_mode: true
  slurm:
    queue: "cpu"
    account: "hpc-bpglsst"
    job_extra_directives:
      - "--partition=cpu"
      - "--time=04:00:00"
    diagnostics_mode: global
```

Other relevant settings:
- input.format: `hats`
- algorithm.selection_mode: `score_density_hybrid`
- algorithm.level_limit: `11`
- algorithm.score_density_hybrid.score_column: `MAG_AUTO_R_DERED`

### Exact config values used (verbatim fields, comments removed)

```yaml
input:
  paths:
    - "/data/public/des/dr2/secondary/catalogs/main/hats"
  format: hats

columns:
  ra: RA
  dec: DEC
  keep:
    - COADD_OBJECT_ID
    - RA
    - DEC
    - FLAGS_I
    - EXTENDED_CLASS_COADD
    - MAG_AUTO_G_DERED
    - MAG_AUTO_R_DERED
    - MAG_AUTO_I_DERED
    - MAG_AUTO_Z_DERED
    - MAG_AUTO_Y_DERED
    - MAGERR_AUTO_G
    - MAGERR_AUTO_R
    - MAGERR_AUTO_I
    - MAGERR_AUTO_Z
    - MAGERR_AUTO_Y

algorithm:
  selection_mode: "score_density_hybrid"
  level_limit: 11
  selection_defaults:
    hist_nbins: 2048
    adaptive_range: "complete"
    order_desc: false
    keep_invalid_values: true
    tie_column: null
  mag_global:
    mag_column: r_cModelMag_dered
    flux_column: null
    mag_offset: null
    mag_min: null
    mag_max: null
    adaptive_range: "complete"
    hist_nbins: 2048
    n_1: null
    n_2: null
    n_3: null
    order_desc: false
    keep_invalid_values: true
    tie_column: null
  score_global:
    adaptive_range: "complete"
    hist_nbins: 2048
    order_desc: false
    keep_invalid_values: false
    tie_column: null
  score_density_hybrid:
    score_column: MAG_AUTO_R_DERED
    adaptive_range: "complete"
    order_desc: false
    keep_invalid_values: true
    tie_column: null

cluster:
  mode: slurm
  n_workers: 10
  threads_per_worker: 8
  memory_per_worker: "96GB"
  low_memory_mode: true
  slurm:
    queue: "cpu"
    account: "hpc-bpglsst"
    job_extra_directives:
      - "--partition=cpu"
      - "--time=04:00:00"
    diagnostics_mode: global

output:
  out_dir: "/scratch/users/luigi.silva/hipscatalog_gen/outputs/DES_DR2"
  cat_name: "DES_DR2"
  target: "0 0"
  creator_did: "ivo://PRIVATE_USER/DES_DR2"
  obs_title: "DES DR2 catalog"
```

## Compared behavior

- Before: stage-1 used a single global `groupby(...).apply(topk)` over all candidate rows.
- After: stage-1 uses exact two-stage top-k:
  1. per-partition local top-k prune
  2. global top-k reduce on pruned rows

Both paths keep the same ordering semantics (`score`, `tie`, `RA`, `DEC`) and the same `k` per tile.

## Raw log excerpts used

### Before (baseline)

- Stage-1 targets logged: `20:16:15.679`
- `[DEPTH 1] done in 00:06:29.250` at `20:22:44.929`
- `[DEPTH 2] done in 00:03:24.082` at `20:26:09.012`
- `[DEPTH 3] done in 00:03:08.526` at `20:29:17.538`
- Next stage started: `20:30:19.995`

### After (optimized)

- Stage-1 targets logged: `20:58:26.521`
- `[DEPTH 1] done in 00:02:19.846` at `21:00:46.367`
- `[DEPTH 2] done in 00:02:28.409` at `21:03:14.776`
- `[DEPTH 3] done in 00:02:32.150` at `21:05:46.927`
- Next stage started: `21:06:58.251`

## Runtime comparison

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| Depth 1 duration | 6m 29.250s | 2m 19.846s | -64.1% |
| Depth 2 duration | 3m 24.082s | 2m 28.409s | -27.3% |
| Depth 3 duration | 3m 08.526s | 2m 32.150s | -19.3% |
| Stage-1 wall time (`targets` -> `DEPTH 3 done`) | 13m 01.859s | 7m 20.406s | -43.7% |

## Correctness checks

- Stage-1 target totals remained the same across runs:
  - depth 1: `1391`
  - depth 2: `3764`
  - depth 3: `11374`
- Tile/row writes for stage-1 remained consistent in logs:
  - depth 1: `tiles_written=16`, `rows_written=1391`
  - depth 2: `tiles_written=42`, `rows_written=3764`
  - depth 3: `tiles_written=132`, `rows_written=11374`

## Stability observations

- Largest gain occurred at depth 1, where candidate volume and shuffle pressure are highest.
- Before memory/scaling fixes, runs showed worker restarts and communication failures.
- With the current config and current code path, compared runs progressed through densmaps and stage-1 depths without fatal worker churn in the shared logs.

## Reproducibility checklist

To reproduce this benchmark:
1. Use the same DES DR2 HATS input path and the same config above.
2. Capture logs from pipeline start through `[DEPTH 3] done`.
3. Record:
   - stage-1 start timestamp (`stage 1 targets` log line)
   - per-depth duration from `[DEPTH n] done in ...`
   - stage-1 end timestamp (`[DEPTH 3] done` line)
4. Compare against this table.

## Notes

- In this run window, densmap times were higher (~53-55s each) than in a previous window (~48-49s each), likely due to cluster variability.
- Densmap-stage optimization benchmark is documented separately in:
  - `benchmarks/records/2026-02-10_des_dr2_densmaps_finest_derive.md`
