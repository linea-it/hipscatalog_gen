# Benchmark Record: DES_DR2 densmaps (single-pass finest + derive)

Date: 2026-02-10
Author: Luigi Silva / hipscatalog-gen local run logs
Goal: compare densmap stage runtime before vs after replacing per-depth source scans with:
- one source pass at `o11`
- exact NESTED hierarchical derivation for `o10..o0`

## Context

- Input catalog: `/data/public/des/dr2/secondary/catalogs/main/hats`
- Selection mode: `score_density_hybrid`
- level_limit: `11`
- Input rows: `691483608`
- Cluster hardware (user report): each node with `128 GB RAM` and `56 cores`

## Configuration used in both runs

Same configuration before and after densmap optimization.
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

- Before: densmaps computed directly from source data for each depth (`o0..o11`), i.e. 12 source passes.
- After: compute only `o11` from source data once, derive `o10..o0` by exact NESTED parent-child aggregation.
- Both paths write the same `densmap_o*.fits` products for all depths.

## Raw log excerpts used

### Before (baseline)

- Start densmaps: `20:46:49.284` (`Computing densmap_o0.fits (1/12)...`)
- Per-depth source compute+write around `~52-55s` each from `o0` to `o11`
- Final write: `20:57:31.134` (`Wrote densmap_o11.fits in 00:00:54.830`)

### After (optimized)

- Start densmaps: `21:23:26.857` (`Computing densmap_o11.fits (single source pass)...`)
- Finest compute done: `21:24:20.340` (`Computed densmap_o11.fits in 00:00:53.481`)
- Derivations:
  - `o10 <- o11`: `00:00:00.289`
  - `o9 <- o10`: `00:00:00.063`
  - `o8 <- o9`: `00:00:00.014`
  - `o7 <- o8`: `00:00:00.004`
  - `o6..o0`: `~0.000-0.001s` each
- Final write: `21:24:22.631` (`Wrote densmap_o11.fits in 00:00:01.454`)

## Runtime comparison

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| Densmaps stage wall time | 10m 41.850s | 55.774s | -91.3% |
| Source passes over catalog | 12 | 1 | -91.7% |
| Derived lower depths | no | yes (`o10..o0`) | exact NESTED aggregation |

Derived speedup factor (stage wall time):
- `641.850 / 55.774 = 11.5x`

## Correctness checks

Validation performed on generated files (`outputs/test_densmaps`):
- depths present: `o0..o11`
- shape per depth matches `12 * 4^depth`
- total counts invariant across all depths (`691483608`)
- parent-child consistency passes for all consecutive pairs:
  - `o0 <- o1`, `o1 <- o2`, ..., `o10 <- o11`

Conclusion: derived densmaps are internally consistent and preserve counts.

## Stability observations

- No additional source-scan fan-out per depth is required in the optimized path.
- The densmaps stage completes in under 1 minute in the captured run window.

## Reproducibility checklist

To reproduce this benchmark:
1. Use the same DES DR2 HATS input path and the same config above.
2. Capture logs from densmap stage start through final densmap write.
3. Record:
   - densmap stage start timestamp
   - densmap stage end timestamp
   - per-depth compute/derive/write durations from log lines
4. Compare against this table.

## Notes

- This benchmark isolates densmap stage behavior only.
- Stage-1 selection benchmark (top-k two-stage optimization) is documented separately in:
  - `benchmarks/records/2026-02-10_des_dr2_score_density_hybrid_topk_two_stage.md`
