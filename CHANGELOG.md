# Changelog

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
