from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from dask import dataframe as dd
from hipscatalog_gen.score_density_hybrid.pipeline import (
    _redistribute_by_density,
    prepare_score_density_hybrid,
)


@pytest.fixture
def diag_ctx():
    """Returns a no-op diagnostic context."""
    return lambda name: nullcontext()


@pytest.fixture
def log_capture():
    """Collects log messages emitted by the pipeline."""
    logs: list[str] = []

    def _log_fn(msg: str, always: bool = False) -> None:
        logs.append(msg)

    return logs, _log_fn


def _cfg(**algo_kwargs) -> SimpleNamespace:
    """Creates a minimal config namespace for score_density_hybrid tests."""
    algo_defaults = dict(
        sdh_score_hist_nbins=4,
        sdh_score_adaptive_range="complete",
        level_limit=3,
        order_desc=True,
        sdh_density_weight=0.0,
        sdh_density_weight_levels=None,
        sdh_coverage_order=2,
        sdh_n_1=None,
        sdh_n_2=None,
        sdh_n_3=None,
    )
    algo_defaults.update(algo_kwargs)
    return SimpleNamespace(algorithm=SimpleNamespace(**algo_defaults))


def test_prepare_hist_peak_resolves_missing_bound(diag_ctx, log_capture):
    """score_min missing, hist_peak fills it from histogram peak."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"score": [0.1, 0.2, 0.2, 0.9]})
    ddf = dd.from_pandas(pdf, npartitions=2)
    cfg = _cfg(
        sdh_score_column="score",
        sdh_score_max=1.0,
        sdh_score_adaptive_range="hist_peak",
    )

    ddf_sel = prepare_score_density_hybrid(ddf, cfg, diag_ctx, log_fn)
    _ = ddf_sel.compute()

    assert cfg.algorithm.sdh_score_min is not None
    assert cfg.algorithm.sdh_score_min <= 0.2
    assert cfg.algorithm.sdh_score_max == 1.0
    assert any("histogram peak" in msg for msg in logs)


def test_density_redistribution_prefers_denser_tiles(log_capture):
    """Density weights should allocate more rows to denser tiles when weight > 0."""
    _, log_fn = log_capture
    counts_ref = np.zeros(48, dtype=np.int64)
    counts_ref[0] = 100
    counts_ref[1] = 10

    pdf = pd.DataFrame(
        {
            "__ipix__": [0, 0, 0, 1, 1, 1],
            "__score__": [6, 5, 4, 3, 2, 1],
            "val": range(6),
        }
    )

    redistributed = _redistribute_by_density(
        pdf,
        counts_ref=counts_ref,
        weight=1.0,
        order_desc=True,
        seed=42,
        log_fn=log_fn,
    )

    counts_out = redistributed["__ipix__"].value_counts().to_dict()
    assert counts_out.get(0, 0) > counts_out.get(1, 0)
