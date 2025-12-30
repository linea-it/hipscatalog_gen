"""Score quantile helpers for selection (regression tests)."""

import numpy as np

from hipscatalog_gen.selection.score import _quantile_from_histogram


def test_quantile_interpolates_inside_populated_bin():
    """Quantiles spread inside a single populated bin (no edge collapse)."""
    cdf = np.array([0.0, 1.0, 1.0], dtype="float64")
    edges = np.array([0.0, 1.0, 2.0, 3.0], dtype="float64")

    qs = [0.1, 0.5, 0.9]
    values = [_quantile_from_histogram(cdf, edges, q) for q in qs]

    assert values[0] < values[1] < values[2]
    assert all(1.0 <= v <= 2.0 for v in values)


def test_quantile_progresses_through_flat_regions():
    """Flat CDF plateaus still yield increasing thresholds."""
    cdf = np.array([0.0, 0.0, 0.5, 0.5, 1.0], dtype="float64")
    edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype="float64")

    low = _quantile_from_histogram(cdf, edges, 0.25)
    mid = _quantile_from_histogram(cdf, edges, 0.5)
    high = _quantile_from_histogram(cdf, edges, 0.75)

    assert low < mid < high
    assert 2.0 <= low <= 5.0
    assert 2.0 <= mid <= 5.0
    assert 3.0 <= high <= 5.0


"""Score quantile helpers for selection (regression tests)."""
