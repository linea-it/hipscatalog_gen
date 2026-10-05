"""Tests for lazy flux-to-magnitude and dereddening transformations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dask import dataframe as dd

from hipscatalog_gen.config import (
    AlgoOpts,
    ClusterCfg,
    ColumnsCfg,
    Config,
    DereddeningCfg,
    InputCfg,
    OutputCfg,
    PhotometryCfg,
    PhotometryMeasurementCfg,
    load_config_from_dict,
)
from hipscatalog_gen.io.input import _build_input_ddf
from hipscatalog_gen.photometry import (
    apply_photometry,
    build_photometry_plan,
    working_photometry_columns,
)


def _config(
    *,
    mode: str = "score_global",
    score: str = "r_psfMag_dered",
    replace_fluxes: bool = True,
) -> Config:
    """Create a minimal configuration with one transformed band."""
    measurement = PhotometryMeasurementCfg(
        band="r",
        flux_column="r_psfFlux",
        flux_error_column="r_psfFluxErr",
        mag_column="r_psfMag",
        mag_error_column="r_psfMagErr",
    )
    photometry = PhotometryCfg(
        mag_offset=31.4,
        measurements=[measurement],
        replace_fluxes=replace_fluxes,
        dereddening=DereddeningCfg(
            enabled=True,
            ebv_column="ebv",
            coefficients={"r": 2.7},
            output_suffix="_dered",
        ),
    )
    return Config(
        input=InputCfg(paths=["input.parquet"], format="parquet", header=True),
        columns=ColumnsCfg(ra="ra", dec="dec", keep=[]),
        algorithm=AlgoOpts(
            selection_mode=mode,
            level_limit=4,
            moc_order=4,
            score_column=score if mode == "score_global" else None,
            sdh_score_column=score if mode == "score_density_hybrid" else None,
            mag_column=score if mode == "mag_global" else None,
        ),
        cluster=ClusterCfg(
            mode="local",
            n_workers=1,
            threads_per_worker=1,
            memory_per_worker="1GB",
        ),
        output=OutputCfg(out_dir="out", cat_name="cat", target="0 0"),
        photometry=photometry,
    )


def test_flux_to_magnitude_and_dereddening_are_vectorized_and_lazy():
    """All numerical policies are applied independently in each Dask partition."""
    pdf = pd.DataFrame(
        {
            "ra": [1.0, 2.0, 3.0, 4.0],
            "dec": [0.0, 0.0, 0.0, 0.0],
            "r_psfFlux": [100.0, 0.0, 10.0, 10.0],
            "r_psfFluxErr": [10.0, 1.0, -1.0, 2.0],
            "ebv": [0.1, 0.2, 0.3, np.nan],
        }
    )
    source = dd.from_pandas(pdf, npartitions=2)
    transformed = apply_photometry(source, _config())

    assert transformed.npartitions == source.npartitions
    result = transformed.compute()
    expected_mag = 31.4 - 2.5 * np.log10(100.0)
    expected_err = (2.5 / np.log(10.0)) * 10.0 / 100.0
    assert result.loc[0, "r_psfMag"] == pytest.approx(expected_mag)
    assert result.loc[0, "r_psfMagErr"] == pytest.approx(expected_err)
    assert result.loc[0, "r_psfMag_dered"] == pytest.approx(expected_mag - 0.27)
    assert result.loc[0, "r_psfMagErr_dered"] == pytest.approx(expected_err)

    assert np.isnan(result.loc[1, "r_psfMag"])
    assert np.isnan(result.loc[1, "r_psfMagErr"])
    assert np.isfinite(result.loc[2, "r_psfMag"])
    assert np.isnan(result.loc[2, "r_psfMagErr"])
    assert np.isnan(result.loc[3, "r_psfMag_dered"])
    assert np.isnan(result.loc[3, "r_psfMagErr_dered"])


def test_photometry_plan_separates_sources_outputs_and_selection_columns():
    """Fluxes remain read dependencies but are omitted from the final schema."""
    cfg = _config()
    available = ["ra", "dec", "r_psfFlux", "r_psfFluxErr", "ebv", "quality"]
    plan = build_photometry_plan(cfg, available)
    assert plan is not None
    assert plan.source_columns == ("r_psfFlux", "r_psfFluxErr", "ebv")
    assert plan.replaced_columns == ("r_psfFlux", "r_psfFluxErr")
    assert plan.output_columns == ("r_psfMag_dered", "r_psfMagErr_dered")

    working = working_photometry_columns(
        cfg,
        [*available, *plan.derived_columns],
        ["ra", "dec", *plan.output_columns],
    )
    assert working == ["ra", "dec", "r_psfMag_dered", "r_psfMagErr_dered"]


def test_photometry_plan_rejects_missing_sources_and_output_collisions():
    """Schema errors are raised before any distributed computation starts."""
    cfg = _config()
    with pytest.raises(KeyError, match="r_psfFluxErr"):
        build_photometry_plan(cfg, ["ra", "dec", "r_psfFlux", "ebv"])

    with pytest.raises(ValueError, match="already exist"):
        build_photometry_plan(
            cfg,
            ["ra", "dec", "r_psfFlux", "r_psfFluxErr", "ebv", "r_psfMag"],
        )


def test_expand_config_and_validation():
    """Regular survey schemas can be configured with compact band templates."""
    raw = {
        "input": {"paths": ["input.parquet"]},
        "columns": {"ra": "ra", "dec": "dec"},
        "photometry": {
            "mag_offset": 31.4,
            "expand": {
                "bands": ["g", "r"],
                "flux_template": "{band}_psfFlux",
                "flux_error_template": "{band}_psfFluxErr",
                "mag_template": "{band}_psfMag",
                "mag_error_template": "{band}_psfMagErr",
            },
            "dereddening": {
                "enabled": True,
                "ebv_column": "ebv",
                "coefficients": {"g": 3.64, "r": 2.7},
            },
        },
        "algorithm": {
            "selection_mode": "score_global",
            "level_limit": 4,
            "score_global": {"score_column": "r_psfMag_dered"},
        },
        "cluster": {},
        "output": {"out_dir": "out", "cat_name": "cat"},
    }
    cfg = load_config_from_dict(raw)
    assert cfg.photometry is not None
    assert [m.flux_column for m in cfg.photometry.measurements] == [
        "g_psfFlux",
        "r_psfFlux",
    ]

    raw["photometry"]["dereddening"]["coefficients"].pop("r")
    with pytest.raises(ValueError, match="Missing extinction coefficient"):
        load_config_from_dict(raw)


def test_parquet_projection_separates_read_and_output_columns(tmp_path):
    """Parquet reads only physical dependencies while exposing virtual outputs."""
    pdf = pd.DataFrame(
        {
            "ra": [1.0],
            "dec": [2.0],
            "r_psfFlux": [100.0],
            "r_psfFluxErr": [10.0],
            "ebv": [0.1],
            "unused": [999],
        }
    )
    path = tmp_path / "input.parquet"
    pdf.to_parquet(path, index=False)
    cfg = _config()

    raw, ra_name, dec_name, output_columns = _build_input_ddf([str(path)], cfg)
    assert (ra_name, dec_name) == ("ra", "dec")
    assert list(raw.columns) == ["ra", "dec", "r_psfFlux", "r_psfFluxErr", "ebv"]
    assert output_columns == [
        "ra",
        "dec",
        "r_psfMag_dered",
        "r_psfMagErr_dered",
    ]

    transformed = apply_photometry(raw, cfg)
    working = working_photometry_columns(cfg, transformed.columns, output_columns)
    result = transformed[working].compute()
    assert list(result.columns) == output_columns
    assert "r_psfFlux" not in result.columns


@pytest.mark.parametrize("mode", ["mag_global", "score_global", "score_density_hybrid"])
def test_derived_magnitude_is_available_to_every_selection_mode(mode):
    """Every selection strategy retains its configured virtual ranking column."""
    cfg = _config(mode=mode)
    physical = ["ra", "dec", "r_psfFlux", "r_psfFluxErr", "ebv"]
    plan = build_photometry_plan(cfg, physical)
    assert plan is not None
    available = [*physical, *plan.derived_columns]
    working = working_photometry_columns(
        cfg,
        available,
        ["ra", "dec", *plan.output_columns],
    )
    assert "r_psfMag_dered" in working


def test_hats_transformation_preserves_native_catalog():
    """Projection and partition transforms stay native on current LSDB."""
    catalog_path = (
        Path(__file__).resolve().parents[2] / "data" / "des_dr2_small_sample_collection" / "catalog"
    )
    raw = {
        "input": {"paths": [str(catalog_path)], "format": "hats"},
        "columns": {"ra": "RA", "dec": "DEC", "keep": []},
        "photometry": {
            "mag_offset": 30.0,
            "measurements": [
                {
                    "band": "g",
                    "flux_column": "FLUX_AUTO_G",
                    "flux_error_column": "FLUXERR_AUTO_G",
                    "mag_column": "MAG_TEST_G",
                    "mag_error_column": "MAGERR_TEST_G",
                }
            ],
        },
        "algorithm": {
            "selection_mode": "score_global",
            "level_limit": 4,
            "score_global": {"score_column": "MAG_TEST_G"},
        },
        "cluster": {},
        "output": {"out_dir": "out", "cat_name": "cat"},
    }
    cfg = load_config_from_dict(raw)
    catalog, _, _, output_columns = _build_input_ddf([str(catalog_path)], cfg)
    transformed = apply_photometry(catalog, cfg)
    working = working_photometry_columns(cfg, transformed.columns, output_columns)
    final = transformed[working]

    for value in (catalog, transformed, final):
        assert type(value).__module__.startswith("lsdb.")
        assert hasattr(value, "map_partitions")
        assert hasattr(value, "to_dask_dataframe")
    assert list(final.columns) == output_columns
