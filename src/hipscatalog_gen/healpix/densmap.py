"""HEALPix pixel computations and density map aggregation."""

from __future__ import annotations

from typing import Any, List, Sequence, cast

import healpy as hp
import numpy as np
import pandas as pd
from dask import delayed as _delayed
from numpy.typing import NDArray

from ..utils import _HEALPIX_INDEX_RE, _get_meta_df

__all__ = [
    "ipix_for_depth",
    "densmap_for_depth_delayed",
    "densmap_for_depth",
]


# =============================================================================
# HEALPix helpers and density maps
# =============================================================================


def ipix_for_depth(ra_deg: np.ndarray, dec_deg: np.ndarray, depth: int) -> NDArray[np.int64]:
    """Return HEALPix NESTED pixel indices for a given depth (order).

    Ensures the return type is always a 1D numpy.ndarray[int64], even if Healpy
    would otherwise return a scalar for scalar inputs.

    Args:
        ra_deg: Right ascension in degrees.
        dec_deg: Declination in degrees.
        depth: HEALPix order (depth).

    Returns:
        Numpy array (1D) with NESTED pixel indices (dtype=int64).
    """
    nside = 1 << depth
    theta = np.deg2rad(90.0 - dec_deg)  # colatitude
    phi = np.deg2rad(ra_deg)  # longitude
    pix = hp.ang2pix(nside, theta, phi, nest=True)  # escalar ou array
    arr = np.atleast_1d(np.asarray(pix, dtype=np.int64))
    return cast(NDArray[np.int64], arr)


def densmap_for_depth_delayed(ddf: Any, ra_col: str, dec_col: str, depth: int):
    """Build a delayed HEALPix density map at a given depth.

    For HATS/LSDB catalogs with a HEALPix nested index named "_healpix_<order>",
    the density map is derived by bit-shifting that index to the requested
    depth. For other inputs, pixel indices are computed from RA/DEC.

    Args:
        ddf: Dask-like collection or LSDB catalog.
        ra_col: RA column name (degrees).
        dec_col: DEC column name (degrees).
        depth: HEALPix order (depth).

    Returns:
        Dask delayed object that evaluates to a 1D numpy array of counts.
    """
    nside = 1 << depth
    npix = hp.nside2npix(nside)

    # Detect whether this looks like a HATS/LSDB catalog with a HEALPix index.
    meta = _get_meta_df(ddf)
    idx_name = getattr(meta.index, "name", None)
    base_order = None

    if idx_name:
        m = _HEALPIX_INDEX_RE.match(str(idx_name))
        if m:
            base_order = int(m.group(1))

    def _empty_sparse() -> tuple[np.ndarray, np.ndarray]:
        """Return an empty sparse histogram representation."""
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    def _part_hist_sparse(pdf: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return sparse histogram counts for one partition.

        The returned tuple is ``(ipix_unique, counts)`` where both arrays are
        ``int64``. This avoids materializing one dense ``npix`` vector per
        partition, which scales poorly for high depths.
        """
        if pdf is None or len(pdf) == 0:
            return _empty_sparse()

        # Fast path: HEALPix nested index available and depth <= base_order.
        if base_order is not None and pdf.index.name == idx_name and depth <= base_order:
            ipix_base = pdf.index.to_numpy(dtype=np.int64, copy=False)
            shift = 2 * (base_order - depth)
            ip = (ipix_base >> shift).astype(np.int64)
        else:
            # Generic path: compute HEALPix indices from RA/DEC.
            ip = ipix_for_depth(
                pdf[ra_col].to_numpy(),
                pdf[dec_col].to_numpy(),
                depth,
            )

        if ip.size == 0:
            return _empty_sparse()

        uniq, cnt = np.unique(ip, return_counts=True)
        return uniq.astype(np.int64), cnt.astype(np.int64)

    def _merge_sparse_chunks(
        chunks: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Merge many sparse histograms into one sparse histogram."""
        if not chunks:
            return _empty_sparse()

        if len(chunks) == 1:
            ip0, c0 = chunks[0]
            if ip0.size == 0:
                return _empty_sparse()
            return ip0.astype(np.int64, copy=False), c0.astype(np.int64, copy=False)

        ip_parts: List[np.ndarray] = []
        c_parts: List[np.ndarray] = []
        for ip_arr, cnt_arr in chunks:
            if ip_arr.size == 0:
                continue
            ip_parts.append(np.asarray(ip_arr, dtype=np.int64))
            c_parts.append(np.asarray(cnt_arr, dtype=np.int64))

        if not ip_parts:
            return _empty_sparse()

        ip_all = np.concatenate(ip_parts)
        c_all = np.concatenate(c_parts)

        order = np.argsort(ip_all, kind="mergesort")
        ip_sorted = ip_all[order]
        c_sorted = c_all[order]

        starts = np.flatnonzero(np.r_[True, ip_sorted[1:] != ip_sorted[:-1]])
        c_merged = np.add.reduceat(c_sorted, starts).astype(np.int64)
        ip_merged = ip_sorted[starts].astype(np.int64)
        return ip_merged, c_merged

    def _reduce_sparse_tree(
        leaves: List[Any],  # delayed sparse tuples
        fanin: int = 8,
    ):
        """Reduce sparse histograms with bounded fan-in to avoid giant gather tasks."""
        if not leaves:
            return _delayed(_empty_sparse)()

        current = leaves
        while len(current) > 1:
            nxt: List[Any] = []
            for i in range(0, len(current), fanin):
                chunk = current[i : i + fanin]
                nxt.append(_delayed(_merge_sparse_chunks)(chunk))
            current = nxt
        return current[0]

    def _sparse_to_dense(sparse_pair: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Convert sparse histogram to dense HEALPix counts vector."""
        ipix, counts = sparse_pair
        dense = np.zeros(npix, dtype=np.int64)
        if ipix.size > 0:
            dense[ipix] = counts
        return dense

    # One sparse histogram per partition.
    part_delayed = ddf.to_delayed()
    hists_sparse = [_delayed(_part_hist_sparse)(p) for p in part_delayed]

    if len(hists_sparse) == 0:
        # Still return a delayed object for consistency.
        return _delayed(lambda: np.zeros(npix, dtype=np.int64))()

    sparse_total = _reduce_sparse_tree(hists_sparse, fanin=8)
    total_dense = _delayed(_sparse_to_dense)(sparse_total)
    return total_dense


def densmap_for_depth(ddf: Any, ra_col: str, dec_col: str, depth: int) -> np.ndarray:
    """Compute a HEALPix density map for a given depth immediately.

    This is a simple wrapper around `densmap_for_depth_delayed(...).compute()`.

    Args:
        ddf: Dask-like collection or LSDB catalog.
        ra_col: RA column name (degrees).
        dec_col: DEC column name (degrees).
        depth: HEALPix order (depth).

    Returns:
        Numpy array with counts per pixel.
    """
    return densmap_for_depth_delayed(ddf, ra_col, dec_col, depth).compute()
