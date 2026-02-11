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

## Config used in both runs

The same YAML configuration was used before and after the top-k change.
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

Full config snapshot (as provided in run notes):
- input.format: `hats`
- columns.keep: `COADD_OBJECT_ID, RA, DEC, FLAGS_I, EXTENDED_CLASS_COADD, MAG_AUTO_*_DERED, MAGERR_AUTO_*`
- algorithm.selection_mode: `score_density_hybrid`
- algorithm.score_density_hybrid.score_column: `MAG_AUTO_R_DERED`
- algorithm.score_density_hybrid.adaptive_range: `complete`
- algorithm.score_density_hybrid.keep_invalid_values: `true`
- output.cat_name: `DES_DR2`
- output.out_dir: `/scratch/users/luigi.silva/hipscatalog_gen/outputs/DES_DR2`

## Compared commits / behavior

- Before: stage-1 used a single global `groupby(...).apply(topk)` over all candidate rows.
- After: stage-1 uses exact two-stage top-k:
  1. per-partition local top-k prune
  2. global top-k reduce on pruned rows

Both paths keep the same ordering semantics (`score`, `tie`, `RA`, `DEC`) and the same `k` per tile.

## Raw timestamps extracted from logs

### Before (pre two-stage top-k)

- Stage-1 targets logged: `20:16:15.679`
- `[DEPTH 1] done in 00:06:29.250` at `20:22:44.929`
- `[DEPTH 2] done in 00:03:24.082` at `20:26:09.012`
- `[DEPTH 3] done in 00:03:08.526` at `20:29:17.538`
- Next stage started: `20:30:19.995`

### After (post two-stage top-k)

- Stage-1 targets logged: `20:58:26.521`
- `[DEPTH 1] done in 00:02:19.846` at `21:00:46.367`
- `[DEPTH 2] done in 00:02:28.409` at `21:03:14.776`
- `[DEPTH 3] done in 00:02:32.150` at `21:05:46.927`
- Next stage started: `21:06:58.251`

## Runtime comparison (stage-1)

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| Depth 1 duration | 6m 29.250s | 2m 19.846s | -64.1% |
| Depth 2 duration | 3m 24.082s | 2m 28.409s | -27.3% |
| Depth 3 duration | 3m 08.526s | 2m 32.150s | -19.3% |
| Stage-1 wall time (`targets` -> `DEPTH 3 done`) | 13m 01.859s | 7m 20.406s | -43.7% |

Notes:
- Largest gain occurred at depth 1, where candidate volume and shuffle pressure are highest.
- Tile counts and selected-row counts remained consistent across runs:
  - depth 1: selected 1391
  - depth 2: selected 3764
  - depth 3: selected 11374

## Densmap timing notes (same config, different run windows)

Observed `densmap_o*` times were slightly higher in the second run (~53-55s vs ~48-49s each).
This record keeps stage-1 optimization and densmap timing as separate observations because
they were measured in different run windows and can vary with cluster load.

## Stability observations

- Before these memory/scaling fixes, runs showed worker restarts and communication failures.
- With the current config and current code path, the compared runs progressed through densmaps
  and stage-1 depths without fatal worker churn in the shared logs.

## Reproducibility checklist

To reproduce this benchmark:
1. Use the same DES DR2 HATS input path and the same config above.
2. Capture logs from pipeline start through `[DEPTH 3] done`.
3. Record:
   - stage-1 start timestamp (`stage 1 targets` log line)
   - per-depth duration from `[DEPTH n] done in ...`
   - stage-1 end timestamp (`[DEPTH 3] done` line)
4. Compare against this table.
