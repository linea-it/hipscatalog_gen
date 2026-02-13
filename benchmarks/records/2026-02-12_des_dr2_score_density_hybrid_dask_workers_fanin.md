# Benchmark Record: DES_DR2 score_density_hybrid (Dask workers + bounded fan-in merge)

Date: 2026-02-12
Author: Luigi Silva / hipscatalog-gen run logs
Goal: compare stage-2 runtime before vs after today's changes:
- abandon local `ThreadPoolExecutor` bucket processing path
- run bucket merge/write on Dask workers (`Client.submit`)
- keep bounded fan-in safety path to avoid `EMFILE` under high fragment fan-out

## Context

- Input catalog: `/data/public/des/dr2/secondary/catalogs/main/hats`
- Selection mode: `score_density_hybrid`
- Score column: `MAG_AUTO_R_DERED`
- level_limit: `11`
- Input rows: `691483608`
- Cluster (both references): SLURM, `n_workers=20`, `threads_per_worker=8`, `memory_per_worker=32GB`

## Compared behavior

- Before (reference 2026-02-11/12):
  - stage-2 bucket processing driven by local ThreadPool path
  - aggressive incremental compaction path
  - logs like `ThreadPoolExecutor setup ... compaction_mode=auto`
- After (reference 2026-02-12):
  - stage-2 bucket processing submitted to workers (`dask bucket submit`)
  - bounded fan-in reduction path active when needed (FD-safe merge)
  - driver remains orchestration-focused

## Raw log anchors used

### Before

- depth 8 done in `00:16:32.311`
- depth 9 done in `00:51:34.197`
- depth 10 done in `02:13:32.386`

### After

- depth 8 done in `00:02:40.434`
- depth 9 done in `00:05:08.977`
- depth 10 done in `00:13:22.640`
- depth 11 (new run) completed in `00:45:53.568`

## Runtime comparison (depth 4-10)

| Depth | Before | After | Delta |
|---|---:|---:|---:|
| 4 | 00:01:10.208 | 00:01:31.016 | +29.6% |
| 5 | 00:01:16.955 | 00:01:33.582 | +21.6% |
| 6 | 00:02:07.603 | 00:01:37.898 | -23.3% |
| 7 | 00:05:16.651 | 00:01:57.243 | -63.0% |
| 8 | 00:16:32.311 | 00:02:40.434 | -83.8% |
| 9 | 00:51:34.197 | 00:05:08.977 | -90.0% |
| 10 | 02:13:32.386 | 00:13:22.640 | -90.0% |

Aggregate (`depth 4..10`):
- Before: `03:31:30.311`
- After: `00:27:51.790`
- Overall delta: `-86.8%` (about `7.6x` faster)

## Scientific consistency checks from logs

For compared depths, selection/write counts were unchanged between runs:

- depth 4: `selected=39028`, `tiles_written=460`, `rows_written=39028`
- depth 5: `selected=142121`, `tiles_written=1715`, `rows_written=142121`
- depth 6: `selected=536398`, `tiles_written=6521`, `rows_written=536398`
- depth 7: `selected=2088623`, `tiles_written=25423`, `rows_written=2088623`
- depth 8: `selected=8221248`, `tiles_written=100280`, `rows_written=8221248`
- depth 9: `selected=32611602`, `tiles_written=398210`, `rows_written=32611602`
- depth 10: `selected=129793653`, `tiles_written=1585946`, `rows_written=129793653`

Interpretation: benchmark shows strong runtime gain without changing logged scientific totals.

## Fan-in / file-fragment observations

- Before depth 10 compaction summary: `files_in=200488 files_out=128 rounds_total=256` (very aggressive collapse).
- After depth 10 fan-in safety summary: `files_in=200488 files_out=1664 rounds_total=128` (less aggressive collapse).

Even with more output fragments post-reduction, worker-side execution plus bounded fan-in path yielded much lower wall-clock at high depths.

## Operational notes

- New run includes environment warnings not directly tied to selection correctness/performance:
  - Astropy `XDG_CONFIG_HOME` precedence warning.
  - Dask runtime warning about IP detection fallback (`8.8.8.8` unreachable).
- Stage-1 (depths 1-3) in this run window was slower than the reference window; this benchmark focuses on stage-2 path changes (depth >= 4).

## Reproducibility checklist

1. Use the same DES DR2 HATS input and `score_density_hybrid` config.
2. Keep cluster shape comparable (`20 x 8 x 32GB`).
3. Capture per-depth `done in ...` lines for depths 4..10.
4. Compare:
   - durations,
   - `selected`,
   - `tiles_written`,
   - `rows_written`,
   - fan-in/compaction summary lines.
