from __future__ import annotations

from typing import Any, List, cast

import healpy as hp
import numpy as np
import pandas as pd
from dask import delayed as _delayed
from numpy.typing import NDArray

from .utils import _HEALPIX_INDEX_RE, _get_meta_df

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

    def _part_hist(pdf: pd.DataFrame) -> np.ndarray:
        if pdf is None or len(pdf) == 0:
            return np.zeros(npix, dtype=np.int64)

        # Fast path: HEALPix nested index available and depth <= base_order.
        if base_order is not None and pdf.index.name == idx_name and depth <= base_order:
            ipix_base = pdf.index.to_numpy()
            shift = 2 * (base_order - depth)
            ip = (ipix_base >> shift).astype(np.int64)
        else:
            # Generic path: compute HEALPix indices from RA/DEC.
            ip = ipix_for_depth(
                pdf[ra_col].to_numpy(),
                pdf[dec_col].to_numpy(),
                depth,
            )

        return np.bincount(ip, minlength=npix).astype(np.int64)

    # One histogram per partition.
    part_delayed = ddf.to_delayed()
    hists = [_delayed(_part_hist)(p) for p in part_delayed]

    if len(hists) == 0:
        # Still return a delayed object for consistency.
        return _delayed(lambda: np.zeros(npix, dtype=np.int64))()

    def _sum_vecs(vecs: List[np.ndarray]) -> np.ndarray:
        arr = np.sum(vecs, axis=0, dtype=np.int64)
        # Ensure ndarray[int64] for the type checker
        return cast(NDArray[np.int64], np.asarray(arr, dtype=np.int64))

    total = _delayed(_sum_vecs)(hists)
    return total


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
