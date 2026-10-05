"""Planning and vectorized execution for derived photometric columns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..config import Config, PhotometryCfg
from ..utils import _get_dask_base, _get_meta_df, _score_deps

_MAG_ERROR_FACTOR = 2.5 / np.log(10.0)


@dataclass(frozen=True, slots=True)
class PhotometryPlan:
    """Physical dependencies and derived schemas for one pipeline run."""

    source_columns: tuple[str, ...]
    replaced_columns: tuple[str, ...]
    derived_columns: tuple[str, ...]
    output_columns: tuple[str, ...]


def _append_unique(target: list[str], value: str | None) -> None:
    """Append a non-empty string once while preserving order."""
    if value and value not in target:
        target.append(value)


def build_photometry_plan(
    cfg: Config,
    available_columns: Sequence[str],
) -> PhotometryPlan | None:
    """Validate physical dependencies and compile derived column names."""
    photometry = cfg.photometry
    if photometry is None:
        return None

    available = list(available_columns)
    available_set = set(available)
    sources: list[str] = []
    replaced: list[str] = []
    derived: list[str] = []
    outputs: list[str] = []
    dered = photometry.dereddening

    for measurement in photometry.measurements:
        _append_unique(sources, measurement.flux_column)
        _append_unique(sources, measurement.flux_error_column)
        _append_unique(replaced, measurement.flux_column)
        _append_unique(replaced, measurement.flux_error_column)
        _append_unique(derived, measurement.mag_column)
        _append_unique(derived, measurement.mag_error_column)

        observed = [measurement.mag_column]
        if measurement.mag_error_column:
            observed.append(measurement.mag_error_column)

        if dered.enabled:
            corrected = [measurement.mag_column + dered.output_suffix]
            _append_unique(derived, corrected[0])
            if measurement.mag_error_column:
                corrected.append(measurement.mag_error_column + dered.output_suffix)
                _append_unique(derived, corrected[-1])
            if dered.keep_observed_magnitudes:
                outputs.extend(observed)
            outputs.extend(corrected)
        else:
            outputs.extend(observed)

    if dered.enabled:
        _append_unique(sources, dered.ebv_column)

    missing = [name for name in sources if name not in available_set]
    if missing:
        raise KeyError(f"Photometry source column(s) not found in input: {missing}.")

    collisions = [name for name in derived if name in available_set]
    if collisions:
        raise ValueError(
            "Photometry output column(s) already exist in the input: "
            f"{collisions}. Choose different output names."
        )

    protected = {str(cfg.columns.ra), str(cfg.columns.dec)}
    protected_collisions = protected.intersection(derived)
    if protected_collisions:
        raise ValueError(
            f"Photometry outputs cannot overwrite coordinate columns: {sorted(protected_collisions)}."
        )

    return PhotometryPlan(
        source_columns=tuple(sources),
        replaced_columns=tuple(replaced),
        derived_columns=tuple(derived),
        output_columns=tuple(outputs),
    )


def _numeric_array(series: pd.Series) -> np.ndarray:
    """Convert a Series to float64 with non-numeric values represented by NaN."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.to_numpy(dtype="float64", na_value=np.nan)


def _transform_partition(pdf: pd.DataFrame, photometry: PhotometryCfg) -> pd.DataFrame:
    """Derive every configured band in one vectorized partition pass."""
    if pdf.empty:
        out = pdf.copy()
        for measurement in photometry.measurements:
            out[measurement.mag_column] = pd.Series([], dtype="float64")
            if measurement.mag_error_column:
                out[measurement.mag_error_column] = pd.Series([], dtype="float64")
            if photometry.dereddening.enabled:
                suffix = photometry.dereddening.output_suffix
                out[measurement.mag_column + suffix] = pd.Series([], dtype="float64")
                if measurement.mag_error_column:
                    out[measurement.mag_error_column + suffix] = pd.Series([], dtype="float64")
        return out

    out = pdf.copy()
    dered = photometry.dereddening
    ebv: np.ndarray | None = None
    valid_ebv: np.ndarray | None = None
    if dered.enabled:
        if dered.ebv_column is None:  # pragma: no cover - config validation
            raise ValueError("Dereddening requires ebv_column.")
        ebv = _numeric_array(out[dered.ebv_column])
        valid_ebv = np.isfinite(ebv)

    for measurement in photometry.measurements:
        flux = _numeric_array(out[measurement.flux_column])
        valid_flux = np.isfinite(flux) & (flux > 0.0)
        magnitude = np.full(len(out), np.nan, dtype="float64")
        magnitude[valid_flux] = photometry.mag_offset - 2.5 * np.log10(flux[valid_flux])
        out[measurement.mag_column] = magnitude

        magnitude_error: np.ndarray | None = None
        if measurement.flux_error_column and measurement.mag_error_column:
            flux_error = _numeric_array(out[measurement.flux_error_column])
            valid_error = valid_flux & np.isfinite(flux_error) & (flux_error >= 0.0)
            magnitude_error = np.full(len(out), np.nan, dtype="float64")
            magnitude_error[valid_error] = _MAG_ERROR_FACTOR * flux_error[valid_error] / flux[valid_error]
            out[measurement.mag_error_column] = magnitude_error

        if dered.enabled:
            if ebv is None or valid_ebv is None:  # pragma: no cover - defensive
                raise RuntimeError("Missing E(B-V) values for dereddening.")
            corrected = np.full(len(out), np.nan, dtype="float64")
            valid_corrected = valid_flux & valid_ebv
            corrected[valid_corrected] = (
                magnitude[valid_corrected] - dered.coefficients[measurement.band] * ebv[valid_corrected]
            )
            out[measurement.mag_column + dered.output_suffix] = corrected
            if measurement.mag_error_column and magnitude_error is not None:
                corrected_error = np.where(valid_ebv, magnitude_error, np.nan)
                out[measurement.mag_error_column + dered.output_suffix] = corrected_error

    return out


def working_photometry_columns(
    cfg: Config,
    available_columns: Sequence[str],
    output_columns: Sequence[str],
) -> list[str]:
    """Return final output plus selection-only columns needed downstream."""
    if cfg.photometry is None:
        return list(output_columns)

    algo = cfg.algorithm
    mode = str(algo.selection_mode).lower()
    available = list(available_columns)
    internal: list[str] = []
    if mode == "score_global":
        internal.extend(_score_deps(str(algo.score_column or ""), available))
        tie_column = algo.score_tie_column or algo.tie_column
    elif mode == "score_density_hybrid":
        internal.extend(_score_deps(str(algo.sdh_score_column or ""), available))
        tie_column = algo.sdh_tie_column or algo.tie_column
    else:
        for name in (algo.mag_column, algo.flux_column):
            if name and name in available:
                internal.append(name)
        tie_column = algo.mag_tie_column or algo.tie_column
    if tie_column and tie_column in available:
        internal.append(tie_column)

    return list(dict.fromkeys([*output_columns, *internal]))


def apply_photometry(ddf_like: Any, cfg: Config) -> Any:
    """Attach configured magnitudes lazily using one task per input partition."""
    photometry = cfg.photometry
    if photometry is None:
        return ddf_like

    meta = _get_meta_df(ddf_like).copy()
    for measurement in photometry.measurements:
        meta[measurement.mag_column] = pd.Series([], dtype="float64")
        if measurement.mag_error_column:
            meta[measurement.mag_error_column] = pd.Series([], dtype="float64")
        if photometry.dereddening.enabled:
            suffix = photometry.dereddening.output_suffix
            meta[measurement.mag_column + suffix] = pd.Series([], dtype="float64")
            if measurement.mag_error_column:
                meta[measurement.mag_error_column + suffix] = pd.Series([], dtype="float64")

    if hasattr(ddf_like, "map_partitions"):
        return ddf_like.map_partitions(_transform_partition, photometry, meta=meta)

    base = _get_dask_base(ddf_like, require_map_partitions=True)
    return base.map_partitions(_transform_partition, photometry, meta=meta)
