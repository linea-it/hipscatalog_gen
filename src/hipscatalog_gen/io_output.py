from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.io.votable import from_table
from astropy.io.votable import writeto as vot_writeto
from astropy.table import Table
from mocpy import MOC

from .config import OutputCfg
from .utils import _mkdirs, _now_str, _write_text

__all__ = [
    "TSVTileWriter",
    "write_properties",
    "write_arguments",
    "write_metadata_xml",
    "write_moc",
    "write_densmap_fits",
    "finalize_write_tiles",
    "build_header_line_from_keep",
]


# =============================================================================
# Tile writer (TSV) and Allsky helpers
# =============================================================================


class TSVTileWriter:
    """Helper for writing HiPS catalogue tiles in TSV format."""

    def __init__(self, out_dir: Path, depth: int, header_line: str):
        """Initialize tile writer for a given depth.

        Args:
            out_dir: HiPS root output directory.
            depth: HiPS order (NorderX).
            header_line: Header line for TSV tiles (without completeness line).
        """
        self.depth = depth
        self.out_dir = out_dir
        self.norder_dir = out_dir / f"Norder{depth}"
        _mkdirs(self.norder_dir)
        self.header_line = header_line

    def _dir_for_ipix(self, ipix: int) -> Path:
        """Return directory path for a given HEALPix pixel."""
        base = (ipix // 10_000) * 10_000
        p = self.norder_dir / f"Dir{base}"
        _mkdirs(p)
        return p

    def allsky_tmp(self) -> Path:
        """Return temporary path for Allsky.tsv."""
        return self.norder_dir / ".Allsky.tsv"

    def allsky_path(self) -> Path:
        """Return final path for Allsky.tsv."""
        return self.norder_dir / "Allsky.tsv"

    def cell_tmp(self, ipix: int) -> Path:
        """Return temporary path for a given Npix tile."""
        return self._dir_for_ipix(ipix) / f".Npix{ipix}.tsv"

    def cell_path(self, ipix: int) -> Path:
        """Return final path for a given Npix tile."""
        return self._dir_for_ipix(ipix) / f"Npix{ipix}.tsv"


def finalize_write_tiles(
    out_dir: Path,
    depth: int,
    header_line: str,
    ra_col: str,
    dec_col: str,
    counts: np.ndarray,
    selected: pd.DataFrame,
    order_desc: bool,
    allsky_collect: bool = False,
) -> tuple[Dict[int, int], Optional[pd.DataFrame]]:
    """Write one TSV per HEALPix cell and build optional Allsky dataframe.

    The function:
        * Writes a completeness header line.
        * Writes a single header line per tile.
        * Writes rows in the same column order as the header.
        * Uses atomic rename to avoid partial tiles.

    Args:
        out_dir: HiPS root output directory.
        depth: HiPS order.
        header_line: Header line for tiles.
        ra_col: RA column name (unused here but kept for interface stability).
        dec_col: DEC column name (unused here but kept for interface stability).
        counts: Densmap counts for this depth.
        selected: DataFrame with selected rows for this depth.
        order_desc: Whether scores were sorted in descending order (unused here).
        allsky_collect: If True, return a concatenated Allsky dataframe.

    Returns:
        Tuple (written, allsky_df) where:
            written: Mapping ipix -> number of rows written.
            allsky_df: DataFrame with all rows (if allsky_collect=True), else None.
    """
    writer = TSVTileWriter(out_dir, depth, header_line)
    npix = len(counts)
    written: Dict[int, int] = {}
    allsky_rows: List[List[str]] = []

    header_cols = header_line.strip("\n").split("\t")
    internal = {"__ipix__", "__score__", "__icov__"}
    tile_cols = [c for c in header_cols if c not in internal and c in selected.columns]

    if selected is None or len(selected) == 0 or len(tile_cols) == 0:
        return {}, None

    for pid, g in selected.groupby("__ipix__"):
        ip = int(pid)
        if ip < 0 or ip >= npix:
            continue

        n_src_cell = int(counts[ip])

        g_tile = g[tile_cols].copy()
        n_written = int(len(g_tile))
        written[ip] = n_written
        n_remaining = max(0, n_src_cell - n_written)

        completeness_header = f"# Completeness = {n_remaining} / {n_src_cell}\n"

        final_path = writer.cell_path(ip)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = final_path.with_name(f".Npix{ip}.tsv.tmp")

        # 1) Write completeness + header.
        with tmp.open("w", encoding="utf-8", newline="") as f:
            f.write(completeness_header)
            f.write(header_line)

        # Sanitize string columns.
        obj_cols = g_tile.select_dtypes(include=["object", "string"]).columns
        if len(obj_cols) > 0:
            g_tile[obj_cols] = g_tile[obj_cols].replace({r"[\t\r\n]": " "}, regex=True)

        # 2) Append rows.
        g_tile.to_csv(
            tmp,
            sep="\t",
            index=False,
            header=False,
            mode="a",
            encoding="utf-8",
            lineterminator="\n",
        )

        os.replace(tmp, final_path)

        if allsky_collect and n_written > 0:
            allsky_rows.extend(g_tile.values.tolist())

    allsky_df: Optional[pd.DataFrame] = None
    if allsky_collect and allsky_rows:
        allsky_df = pd.DataFrame(allsky_rows, columns=tile_cols)

    return written, allsky_df


def build_header_line_from_keep(keep_cols: List[str]) -> str:
    """Build header line from a list of column names."""
    return "\t".join([str(c) for c in keep_cols]) + "\n"


# =============================================================================
# Metadata writers (properties, metadata.xml, MOC, densmaps, arguments)
# =============================================================================


def write_properties(
    out_dir: Path,
    output_cfg: OutputCfg,
    level_limit: int,
    n_src: int,
    tile_format: str = "tsv",
) -> None:
    """Write HiPS 'properties' file for a catalogue HiPS.

    Args:
        out_dir: HiPS root output directory.
        output_cfg: Output configuration object.
        level_limit: Deepest HiPS order.
        n_src: Total number of catalogue sources.
        tile_format: Tile format string (usually "tsv").
    """
    now_iso = _now_str()
    label = output_cfg.cat_name

    creator_did = output_cfg.creator_did or f"ivo://PRIVATE_USER/{label}"
    obs_title = output_cfg.obs_title or label
    publisher_id = "ivo://PRIVATE_USER"

    buf: List[str] = []
    buf.append("# Generated by the hipscatalog_gen tool (inspired by the CDS HiPS catalog tools).\n")
    buf.append(f"# hips_release_date generated at {now_iso} (UTC)\n")

    # Mandatory HiPS 1.4 keywords.
    buf.append(f"creator_did     = {creator_did}\n")
    buf.append(f"obs_title       = {obs_title}\n")
    buf.append("dataproduct_type  = catalog\n")
    buf.append("hips_version    = 1.4\n")
    buf.append(f"hips_release_date = {now_iso}\n")
    buf.append("hips_status     = public master unclonable\n")
    buf.append("hips_frame      = equatorial\n")
    buf.append(f"hips_order      = {level_limit}\n")
    buf.append(f"hips_tile_format  = {tile_format}\n")

    # Recommended/extra keywords.
    buf.append(f"publisher_id    = {publisher_id}\n")
    buf.append(f"hips_service_url  = {str(out_dir).rstrip('/')}/{label}\n")
    buf.append("hips_builder    = hipscatalog_gen\n")
    buf.append(f"hips_cat_nrows  = {n_src}\n")

    # Optional initial view center.
    try:
        ra0, dec0 = output_cfg.target.split()
    except Exception:
        ra0, dec0 = "0", "0"
    buf.append(f"hips_initial_ra = {ra0}\n")
    buf.append(f"hips_initial_dec = {dec0}\n")

    # Legacy fields kept for compatibility.
    buf.append("# Deprecated but still kept for compatibility with some clients\n")
    buf.append(f"label={label}\n")
    buf.append("coordsys=C\n")

    _write_text(out_dir / "properties", "".join(buf))


def write_arguments(out_dir: Path, args_text: str) -> None:
    """Write command-line arguments used to run the pipeline."""
    _write_text(out_dir / "arguments", args_text)


def write_metadata_xml(
    out_dir: Path,
    columns: List[tuple[str, str, Optional[str]]],
    ra_idx: int,
    dec_idx: int,
) -> None:
    """Write VOTable metadata (metadata.xml and Metadata.xml).

    Marks RA/DEC columns with appropriate UCDs.

    Args:
        out_dir: HiPS root output directory.
        columns: List of (name, dtype, ucd) tuples.
        ra_idx: Index of RA column in `columns`.
        dec_idx: Index of DEC column in `columns`.
    """
    table = Table()
    for name, dtype, _ucd in columns:  # _ucd intentionally unused here
        # Ensure consistent typing for mypy by declaring a neutral dtype
        np_dt: np.dtype[Any]
        dts = str(dtype)
        if dts.startswith("float"):
            np_dt = np.dtype("float64")
        elif dts.startswith("int"):
            np_dt = np.dtype("int64")
        else:
            np_dt = np.dtype("U1")
        table[name] = np.array([], dtype=np_dt)

    vot = from_table(table)
    res = vot.resources[0]
    vtab = res.tables[0]

    for i, field in enumerate(vtab.fields):
        field.name = columns[i][0]
        if i == ra_idx:
            field.ucd = "pos.eq.ra;meta.main"
        elif i == dec_idx:
            field.ucd = "pos.eq.dec;meta.main"
        else:
            field.ucd = columns[i][2] or field.ucd

    path_lower = str(out_dir / "metadata.xml")
    path_upper = str(out_dir / "Metadata.xml")
    try:
        vot_writeto(vot, path_lower)
    except TypeError:
        with open(path_lower, "wb") as fh:
            vot.to_xml(fh)
    try:
        vot_writeto(vot, path_upper)
    except TypeError:
        with open(path_upper, "wb") as fh:
            vot.to_xml(fh)


def write_moc(out_dir: Path, level_coverage: int, dens_counts: np.ndarray) -> None:
    """Build and write MOC from densmap counts.

    Outputs both FITS (Moc.fits) and JSON (Moc.json) representations.

    Args:
        out_dir: HiPS root output directory.
        level_coverage: HEALPix order used for coverage.
        dens_counts: Densmap counts at the coverage order.
    """
    order = int(level_coverage)
    ipix = np.flatnonzero(dens_counts > 0)

    if ipix.size == 0:
        moc = MOC.empty(order)
    else:
        ipix_list = [int(x) for x in np.asarray(ipix, dtype=np.int64).tolist()]
        nside = 1 << order

        moc = None
        last_err: Optional[Exception] = None
        candidates = [
            lambda: MOC.from_healpix_cells(order, ipix_list, True),
            lambda: MOC.from_healpix_cells(order, ipix_list),
            lambda: MOC.from_healpix_cells(nside, ipix_list, order, True),
            lambda: MOC.from_healpix_cells(nside, ipix_list, order),
            lambda: MOC.from_healpix_cells(ipix_list, order, True),
            lambda: MOC.from_healpix_cells(ipix_list, order),
            lambda: MOC.from_healpix_cells(nside=nside, ipix=ipix_list, max_depth=order, nested=True),
            lambda: MOC.from_healpix_cells(nside=nside, ipix=ipix_list, max_depth=order),
        ]
        for builder in candidates:
            try:
                moc = builder()
                break
            except Exception as e:  # noqa: PERF203
                last_err = e
                continue
        if moc is None:
            raise RuntimeError(
                "Failed to build MOC with your mocpy version. "
                f"Last error: {type(last_err).__name__}: {last_err}"
            )

    fits_path = out_dir / "Moc.fits"
    try:
        try:
            moc.save(str(fits_path), "fits")
        except TypeError:
            moc.save(str(fits_path), format="fits")
    except Exception:
        moc.write(fits_path, overwrite=True)

    json_path = out_dir / "Moc.json"
    data = moc.serialize(format="json")
    with json_path.open("w", encoding="utf-8") as f:
        if isinstance(data, str):
            f.write(data)
        elif isinstance(data, bytes):
            f.write(data.decode("utf-8"))
        else:
            json.dump(data, f)


def write_densmap_fits(out_dir: Path, depth: int, counts: np.ndarray) -> None:
    """Write densmap_o<depth>.fits for depths < 13.

    Args:
        out_dir: HiPS root output directory.
        depth: HiPS order (depth).
        counts: Counts per pixel at this depth.
    """
    if depth >= 13:
        return
    hdu0 = fits.PrimaryHDU()
    col = fits.Column(name="VALUE", array=counts.astype(np.int64), format="K")
    hdu1 = fits.BinTableHDU.from_columns([col])
    fits.HDUList([hdu0, hdu1]).writeto(out_dir / f"densmap_o{depth}.fits", overwrite=True)
