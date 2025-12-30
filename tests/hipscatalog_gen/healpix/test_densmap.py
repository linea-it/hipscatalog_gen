"""Unit tests for HEALPix helpers and density map aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dask import dataframe as dd
from hipscatalog_gen.healpix import densmap


def _hist_for_values(ra, dec, depth):
    """Helper to compute expected histogram via ipix_for_depth."""
    ipix = densmap.ipix_for_depth(np.asarray(ra), np.asarray(dec), depth)
    npix = densmap.hp.nside2npix(1 << depth)
    return np.bincount(ipix, minlength=npix).astype(np.int64)


def test_ipix_for_depth_scalar_and_array():
    """ipix_for_depth returns int64 arrays and matches healpy for arrays."""
    out_scalar = densmap.ipix_for_depth(np.array(0.0), np.array(0.0), depth=1)
    assert out_scalar.shape == (1,)
    assert out_scalar.dtype == np.int64

    ra = np.array([0.0, 120.0, 240.0])
    dec = np.array([0.0, 45.0, -30.0])
    expected = densmap.hp.ang2pix(1 << 1, np.deg2rad(90.0 - dec), np.deg2rad(ra), nest=True)
    out = densmap.ipix_for_depth(ra, dec, depth=1)
    np.testing.assert_array_equal(out, expected.astype(np.int64))


def test_densmap_from_radec_partitions():
    """densmap_for_depth computes histogram from RA/DEC when no HEALPix index."""
    pdf = pd.DataFrame({"RA": [0.0, 0.0, 120.0, 240.0], "DEC": [0.0, 0.0, 45.0, -30.0]})
    ddf = dd.from_pandas(pdf, npartitions=2)
    expected = _hist_for_values(pdf["RA"].to_numpy(), pdf["DEC"].to_numpy(), depth=1)

    delayed = densmap.densmap_for_depth_delayed(ddf, "RA", "DEC", depth=1)
    result = delayed.compute()
    np.testing.assert_array_equal(result, expected)

    # Wrapper returns same values
    result_eager = densmap.densmap_for_depth(ddf, "RA", "DEC", depth=1)
    np.testing.assert_array_equal(result_eager, expected)


def test_densmap_radec_with_non_healpix_index():
    """Even with a named index that is not HEALPix, RA/DEC path is used."""
    pdf = pd.DataFrame({"RA": [10.0, 20.0], "DEC": [0.0, 10.0]})
    pdf.index = pd.Index([5, 6], name="not_healpix")
    ddf = dd.from_pandas(pdf, npartitions=1)
    expected = _hist_for_values(pdf["RA"].to_numpy(), pdf["DEC"].to_numpy(), depth=1)
    result = densmap.densmap_for_depth(ddf, "RA", "DEC", depth=1)
    np.testing.assert_array_equal(result, expected)


def test_densmap_with_healpix_index_fast_path():
    """When a HEALPix index is present at higher order, densmap shifts indices."""
    base_order = 3
    depth = 2
    # Index values correspond to order=3; shifting by 2*(3-2)=2.
    ipix_base = np.array([0, 1, 8, 15], dtype=np.int64)
    pdf = pd.DataFrame(
        {"RA": [0.0] * len(ipix_base), "DEC": [0.0] * len(ipix_base)},
        index=pd.Index(ipix_base, name=f"_healpix_{base_order}"),
    )
    ddf = dd.from_pandas(pdf, npartitions=2)

    expected_ipix = (ipix_base >> (2 * (base_order - depth))).astype(np.int64)
    npix = densmap.hp.nside2npix(1 << depth)
    expected = np.bincount(expected_ipix, minlength=npix).astype(np.int64)

    result = densmap.densmap_for_depth(ddf, "RA", "DEC", depth=depth)
    np.testing.assert_array_equal(result, expected)


def test_densmap_empty_partitions_and_no_partitions(monkeypatch):
    """Handles empty partitions and objects without partitions."""
    # Empty Dask DataFrame -> histogram of zeros.
    empty_pdf = pd.DataFrame({"RA": pd.Series([], dtype="float64"), "DEC": pd.Series([], dtype="float64")})
    empty_ddf = dd.from_pandas(empty_pdf, npartitions=1)
    zeros = densmap.densmap_for_depth(empty_ddf, "RA", "DEC", depth=1)
    assert zeros.sum() == 0

    # Object whose to_delayed returns nothing exercises the no-partitions branch.
    class NoParts:
        def to_delayed(self):
            return []

    # Avoid _get_meta_df trying to inspect NoParts internals.
    monkeypatch.setattr(densmap, "_get_meta_df", lambda *_: pd.DataFrame(index=pd.Index([], name=None)))
    no_parts = densmap.densmap_for_depth_delayed(NoParts(), "RA", "DEC", depth=1).compute()
    np.testing.assert_array_equal(no_parts, np.zeros(densmap.hp.nside2npix(1 << 1), dtype=np.int64))
