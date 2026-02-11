# Changelog

## Unreleased

- Fix densmap scalability for large catalogs by replacing dense per-partition aggregation with sparse histogram reduction in a bounded fan-in tree, preventing oversized gather tasks at high depths.
- Compute only the finest densmap from source data and derive lower orders by exact NESTED parent-child aggregation (4 children -> 1 parent), reducing repeated catalog scans; keep per-depth progress logs (`Computing/Derived/Wrote densmap_o*.fits`).
- Optimize `score_density_hybrid` stage-1 per-tile top-k with an exact two-stage strategy (local prune + global reduce), reducing shuffle volume and improving runtime on large catalogs.
- Make stage-2 depth writing (`depth >= 3`, no Allsky) streaming-based to avoid `depth_ddf.compute()` materialization on the driver; preserves tile/output logic while reducing scheduler/driver memory pressure on very large runs.
- Reduce stage-2 spill metadata pressure by switching from per-tile temporary fragments to bucketed temporary fragments, keeping exact output semantics while improving distributed-filesystem throughput.
- Remove pandas `FutureWarning` in local top-k pruning by avoiding partition-level `DataFrameGroupBy.apply`.
- Detailed run benchmarks for these optimizations are tracked in:
  - `benchmarks/records/2026-02-10_des_dr2_score_density_hybrid_topk_two_stage.md`
  - `benchmarks/records/2026-02-10_des_dr2_densmaps_finest_derive.md`

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
