# Changelog

## Unreleased

- Fix densmap scalability for large catalogs by replacing dense per-partition aggregation with sparse histogram reduction in a bounded fan-in tree, preventing oversized gather tasks at high depths.
- Compute only the finest densmap from source data and derive lower orders by exact NESTED parent-child aggregation (4 children -> 1 parent), reducing repeated catalog scans; keep per-depth progress logs (`Computing/Derived/Wrote densmap_o*.fits`).
- Optimize `score_density_hybrid` stage-1 per-tile top-k with an exact two-stage strategy (local prune + global reduce), reducing shuffle volume and improving runtime on large catalogs.
- Add `score_density_hybrid.density_up_to_depth` (default `3`) to control how far stage-1 density selection runs before switching to score-based stage-2.
- Make stage-2 depth writing (`depth >= 3`, no Allsky) streaming-based with bucketed temporary fragments, avoiding `depth_ddf.compute()` materialization on the driver and reducing distributed-filesystem metadata pressure.
- Run stage-2 bucket processing on distributed workers (`Client.submit`) so compute/IO stay on workers and the driver remains orchestration-only.
- Require an active `dask.distributed` client for streamed stage-2 writes; fail fast when absent instead of silently degrading to local execution.
- Auto-tune merge fan-in per worker task using `RLIMIT_NOFILE` and worker concurrency, and bound fan-in rounds to prevent `EMFILE` (`Too many open files`) during high-depth bucket merges.
- Keep stage-2 k-way merge on a single bounded fan-in safety path, simplifying behavior while preserving robustness under high fragment fan-out.
- Reuse selection-stage per-depth write stats for final output counts (`telemetry`/`properties`) and remove slow full-TSV recount fallback; pipeline now fails fast if required intermediate stats are missing/invalid.
- Add startup observability logs for cluster runtime (local/SLURM resources + directives) and stage-2 streaming execution (worker count, bucket count, fan-in reduction summary).
- Fix distributed compatibility warning by reading worker concurrency from `Worker.state.nthreads` (with fallback for older versions), avoiding `FutureWarning` on new `distributed`.
- Remove pandas `FutureWarning` in local top-k pruning by avoiding partition-level `DataFrameGroupBy.apply`.
- Detailed run benchmarks for these optimizations are tracked in:
  - `benchmarks/records/2026-02-10_des_dr2_score_density_hybrid_topk_two_stage.md`
  - `benchmarks/records/2026-02-10_des_dr2_densmaps_finest_derive.md`
  - `benchmarks/records/2026-02-12_des_dr2_score_density_hybrid_dask_workers_fanin.md`

## 0.2.0

- Merge Dependabot updates (GitHub workflows and dependency version limits).
- Add score/magnitude column details to `process.log`.
- Add output row count to `process.log`.
- Improve generated `properties` file content.
- Improve generated `arguments` file content.
- Add `index.html` preview file generation in outputs.
- Fix MOC order generation bug (`Moc.fits`/`Moc.json`) for compatibility with current `mocpy` signatures.

## 0.1.1

- Fix `score_density_hybrid` stage-1 de-duplication for LSDB catalogs by deriving unique IDs from pixel metadata and partition context.
- Add tests for unique ID generation in Dask and LSDB paths.
- Pin `sphinx-rtd-theme>=3.0,<4` to avoid Sphinx 7+ theme incompatibility; update docs for mag_global hist_peak clipping.

## 0.1.0

- First publishable release of `hipscatalog-gen`.
- Three selection modes: `mag_global`, `score_global`, `score_density_hybrid`, each with normalize/prepare/run stages via a mode registry.
- Structured pipeline with immutable context, per-stage telemetry (`telemetry.json`), and optional JSON logs (`process.jsonl`).
- CLI: `--config` to run, plus `--list-modes`, `--check-config`, `--telemetry` (summary of telemetry.json), and `--json-logs`.
- Outputs: HiPS tiles/Allsky, density maps, MOC, metadata, logs, and consolidated counts in `telemetry.json` (no separate input/output counts files).
- Config validation (common + per-mode), schema for telemetry bundled in the package.
