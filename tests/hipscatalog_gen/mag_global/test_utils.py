import numpy as np
import pandas as pd
import pytest
from dask import dataframe as dd
from dask import delayed
from hipscatalog_gen.mag_global.utils import _quantile_from_histogram, compute_mag_histogram_ddf


def test_compute_mag_histogram_ddf_counts_basic():
    """Tests that the histogram counts and totals match a simple input."""
    pdf = pd.DataFrame({"mag": [1.0, 2.0, 3.0, np.nan, np.inf]})
    ddf = dd.from_pandas(pdf, npartitions=2)

    hist, edges, n_total = compute_mag_histogram_ddf(
        ddf_like=ddf,
        mag_col="mag",
        mag_min=1.0,
        mag_max=3.0,
        nbins=2,
    )

    assert n_total == 3
    assert hist.tolist() == [1, 2]  # [1, 2) and [2, 3]
    assert edges.tolist() == [1.0, 2.0, 3.0]


def test_compute_mag_histogram_ddf_ignores_invalid_partitions():
    """Tests that empty/NaN-only partitions yield zero counts and zero total."""
    pdf = pd.DataFrame({"mag": [np.nan, np.inf, -np.inf]})
    ddf = dd.from_pandas(pdf, npartitions=1)

    hist, edges, n_total = compute_mag_histogram_ddf(
        ddf_like=ddf,
        mag_col="mag",
        mag_min=-5.0,
        mag_max=5.0,
        nbins=4,
    )

    assert n_total == 0
    assert np.all(hist == 0)
    assert len(edges) == 5


@pytest.mark.parametrize(
    ("q", "expected"),
    [
        (-0.5, 10.0),  # clipped to 0
        (0.0, 10.0),
        (0.51, 30.0),  # first bin where CDF crosses q
        (1.0, 50.0),
        (2.0, 50.0),  # clipped to 1
    ],
)
def test_quantile_from_histogram_clips_and_inverts(q, expected):
    """Tests that histogram inversion clips q and returns the correct edge."""
    cdf = np.array([0.2, 0.5, 0.8, 1.0], dtype="float64")
    edges = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype="float64")

    assert _quantile_from_histogram(cdf, edges, q) == expected


def test_quantile_from_histogram_empty_returns_first_edge():
    """Tests that an empty CDF defaults to the leftmost edge."""
    edges = np.array([1.0, 2.0, 3.0], dtype="float64")
    assert _quantile_from_histogram(np.array([], dtype="float64"), edges, 0.3) == 1.0


def test_compute_mag_histogram_ddf_handles_none_partition():
    """Tests that None partitions are treated like empty partitions (all zeros)."""

    class DummyDDF:
        def __init__(self, col_name: str):
            self.columns = [col_name]

        def __getitem__(self, _):
            return self

        def to_delayed(self):
            return [delayed(lambda: None)()]

    ddf = DummyDDF("mag")
    hist, edges, n_total = compute_mag_histogram_ddf(
        ddf_like=ddf,
        mag_col="mag",
        mag_min=0.0,
        mag_max=1.0,
        nbins=1,
    )

    assert n_total == 0
    assert hist.tolist() == [0]
    assert len(edges) == 2
