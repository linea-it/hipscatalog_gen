# Pipeline overview

High-level stages
-----------------
- `prepare_input`: expand paths, validate RA/DEC, adjust partitions/persistence per cluster settings.
- `input_total`: count rows after validation.
- `normalize_selection`: create internal columns and compute mode parameters (ranges, sentinels) without mutating `cfg`.
- `prepare_<mode>`: apply the range filter using normalized parameters and optionally persist.
- `densmaps`: compute density maps for all depths and write FITS files.
- `static_products`: write MOC, metadata.xml, and arguments.
- `run_<mode>`: slice by depth/score/magnitude and write tiles + Allsky when applicable.
- `counts` / `properties`: write counts and HiPS properties.

Telemetry (`telemetry.json`)
----------------------------
- Written to `output.out_dir/telemetry.json` at the end of the run. Schema in `docs/telemetry.schema.json`; includes `schema_version` for forward-compatibility.
- Fields: `selection_mode`, `level_limit`, `moc_order`, `input_rows`, `output_rows`, `total_duration_s`, per-stage durations under `stages.{stage}.duration_s`, and counts under `counts` (input/output totals and per-depth).
- Schema snapshot:

```json
{
  "selection_mode": "mag_global",
  "level_limit": 6,
  "moc_order": 6,
  "input_rows": 123456,
  "output_rows": 98765,
  "total_duration_s": 12.34,
  "stages": {
    "prepare_input": {"duration_s": 0.5},
    "input_total": {"duration_s": 0.1},
    "normalize_selection": {"duration_s": 0.2},
    "prepare_mag_global": {"duration_s": 1.0},
    "densmaps": {"duration_s": 2.5},
    "static_products": {"duration_s": 0.3},
    "run_mag_global": {"duration_s": 6.0},
    "counts": {"duration_s": 0.1},
    "properties": {"duration_s": 0.1}
  },
  "counts": {
    "input": {"total": 123456},
    "output": {
      "total": 98765,
      "depth_totals": {"1": 10000, "2": 15000, "3": 20000, "4": 25000, "5": 28765},
      "depths": {"1": {"0": 5000, "1": 5000}}
    }
  }
}
```

How to add a new mode
---------------------
1) Implement `normalize_<mode>(ddf, cfg, diag_ctx, log_fn, persist_ddfs, avoid_computes)` returning `(ddf, params)` with any mode-specific fields, without changing `cfg`.
2) Implement `prepare_<mode>(ddf, cfg, diag_ctx, log_fn, params, persist_ddfs, avoid_computes)` to apply initial filters using `params`.
3) Implement `run_<mode>(remainder_ddf, densmaps, keep_cols, ra_col, dec_col, cfg, out_dir, diag_ctx, log_fn, avoid_computes, params)` for the final selection and writing.
4) Add an entry to `MODE_REGISTRY` (`src/hipscatalog_gen/pipeline/modes.py`), including a `validate_fn` that surfaces config issues early.
5) If the mode needs specific parameters, create a dataclass in `src/hipscatalog_gen/pipeline/params.py` to carry them between phases.

Mode scaffold (template)
------------------------
```python
from dataclasses import dataclass
from hipscatalog_gen.pipeline.params import YourModeParams

def validate_your_mode_cfg(cfg): ...

def normalize_your_mode(ddf, cfg, diag_ctx, log_fn, persist_ddfs, avoid_computes):
    # compute params; avoid mutating cfg
    params = YourModeParams(...)
    return ddf, params

def prepare_your_mode(ddf, cfg, diag_ctx, log_fn, params, persist_ddfs, avoid_computes):
    # filter/annotate ddf using params
    return ddf_filtered

def run_your_mode(remainder_ddf, densmaps, keep_cols, ra_col, dec_col, cfg, out_dir, diag_ctx, log_fn, avoid_computes, params):
    # slice/write tiles
    ...
```

Register in `MODE_REGISTRY` with `validate_fn`, `normalize_fn`, `prepare_fn`, and `run_fn`.

Best practices
--------------
- Validate configuration early in `validate_fn` with clear messages.
- Avoid mutating `cfg` inside normalize/prepare/run; return everything in `params`.
- Use the structured logger (`log_fn`) for depth/stage-aware messages.
- Reuse existing helpers (`maybe_persist_ddf`, `assign_level_edges`, etc.) to avoid hard-to-follow paths.
