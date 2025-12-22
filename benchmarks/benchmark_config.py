"""Lightweight benchmark to keep the ASV pipeline exercised.

This avoids "No benchmarks selected" failures in CI by timing a minimal
config load + validation round-trip.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hipscatalog_gen.config import load_config
from hipscatalog_gen.pipeline.validation import validate_common_cfg

_MINIMAL_CFG = """\
input:
  paths: ["tests/data/sample.parquet"]
  format: parquet
  header: true
columns:
  ra: RA
  dec: DEC
algorithm:
  selection_mode: mag_global
  level_limit: 4
  moc_order: 4
  mag_global:
    mag_column: "__mag__"
    mag_min: 17.0
    mag_max: 20.0
    hist_nbins: 4
cluster:
  mode: local
  n_workers: 1
  threads_per_worker: 1
  memory_per_worker: "1GB"
output:
  out_dir: "/tmp/hips"
  cat_name: "bench"
  target: "0 0"
"""


class ConfigBenchmark:
    """ASV benchmark for configuration parsing + validation."""

    def setup(self):
        """Prepare a temporary config file."""
        self._tmp = tempfile.TemporaryDirectory()
        self.cfg_path = Path(self._tmp.name) / "bench.yaml"
        self.cfg_path.write_text(_MINIMAL_CFG, encoding="utf-8")

    def teardown(self):
        """Clean up temporary files."""
        self._tmp.cleanup()

    def time_load_and_validate_config(self):
        """Time loading + validation of a minimal config."""
        cfg = load_config(self.cfg_path)
        validate_common_cfg(cfg)
