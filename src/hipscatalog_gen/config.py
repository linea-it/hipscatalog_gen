from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import yaml  # type: ignore[import-untyped]

__all__ = [
    "AlgoOpts",
    "ColumnsCfg",
    "InputCfg",
    "ClusterCfg",
    "OutputCfg",
    "Config",
    "load_config",
    "load_config_from_dict",
    "display_available_configs",
]


@dataclass
class AlgoOpts:
    """Algorithm options for HiPS selection and density profiles.

    Common settings (all modes):
        selection_mode: High-level strategy ("mag_global", "score_global", or
        "score_density_hybrid").
        level_limit: Maximum HiPS order (NorderL).
        moc_order: HiPS order used for the MOC (defaults to level_limit).

    mag_global mode:
        mag_column / flux_column + mag_offset: Magnitude source (column or flux+offset).
        mag_min / mag_max / mag_adaptive_range / mag_hist_nbins: Magnitude bounds logic.
        n_1 / n_2 / n_3: Optional approximate targets for depths 1–3.
        mg_order_desc: Set True to prefer higher magnitudes/scores (default False → lower is better).

    score_global mode:
        score_column / score_min / score_max / score_adaptive_range / score_hist_nbins.
        score_n_1 / score_n_2 / score_n_3: Optional approximate targets for depths 1–3.
        sg_order_desc: Set True to prefer higher scores (default False → lower is better).

    score_density_hybrid mode:
        sdh_score_column / sdh_score_min / sdh_score_max / sdh_score_adaptive_range / sdh_score_hist_nbins.
        sdh_n_1 / sdh_n_2 / sdh_n_3: optional fixed totals for depths 1–3.
        sdh_density_bias_n1 / sdh_density_bias_n2 / sdh_density_bias_n3: density bias per depth (0.0–1.0).
        sdh_order_desc: Set True to prefer higher scores (default False → lower is better).
    """

    # Common settings
    selection_mode: str
    level_limit: int  # maximum HiPS order (NorderL)
    moc_order: int  # HiPS order for the MOC
    mg_order_desc: bool = False  # False → ascending (lower is better)

    # mag_global mode (precedence: mag_column or flux_column+offset)
    mag_column: Optional[str] = None
    flux_column: Optional[str] = None
    mag_offset: Optional[float] = None
    mag_min: Optional[float] = None
    mag_max: Optional[float] = None
    mag_adaptive_range: str = "complete"
    mag_hist_nbins: int = 512
    n_1: Optional[int] = None
    n_2: Optional[int] = None
    n_3: Optional[int] = None

    # score_global mode
    score_column: Optional[str] = None
    score_min: Optional[float] = None
    score_max: Optional[float] = None
    score_adaptive_range: str = "complete"
    score_hist_nbins: int = 512
    score_n_1: Optional[int] = None
    score_n_2: Optional[int] = None
    score_n_3: Optional[int] = None
    sg_order_desc: bool = False

    # score_density_hybrid mode
    sdh_score_column: Optional[str] = None
    sdh_score_min: Optional[float] = None
    sdh_score_max: Optional[float] = None
    sdh_score_adaptive_range: str = "complete"
    sdh_score_hist_nbins: int = 512
    sdh_n_1: Optional[int] = None
    sdh_n_2: Optional[int] = None
    sdh_n_3: Optional[int] = None
    sdh_density_bias_n1: float = 1.0
    sdh_density_bias_n2: float = 1.0
    sdh_density_bias_n3: float = 1.0
    sdh_order_desc: bool = False


@dataclass
class ColumnsCfg:
    """Column mapping for RA/DEC and extra fields."""

    ra: str  # RA column name (or index for ASCII without header)
    dec: str  # DEC column name
    keep: Optional[List[str]] = None  # optional explicit list of columns to keep


@dataclass
class InputCfg:
    """Input catalog configuration."""

    paths: List[str]  # list of glob patterns for files
    format: str  # "parquet" | "csv" | "tsv"
    header: bool  # header row present for CSV/TSV
    ascii_format: Optional[str] = None  # optional hint ("CSV" or "TSV")


@dataclass
class ClusterCfg:
    """Dask cluster configuration."""

    mode: str  # "local" | "slurm"
    n_workers: int
    threads_per_worker: int
    memory_per_worker: str  # e.g. "8GB"
    slurm: Optional[Dict] = None
    low_memory_mode: bool = True
    persist_ddfs: bool = False
    avoid_computes_wherever_possible: bool = True
    diagnostics_mode: str = "global"  # "per_step" | "global" | "off"


@dataclass
class OutputCfg:
    """Output HiPS catalog configuration."""

    out_dir: str
    cat_name: str
    target: str
    creator_did: Optional[str] = None
    obs_title: Optional[str] = None
    overwrite: bool = False


@dataclass
class Config:
    """Top-level configuration container for the HiPS pipeline."""

    input: InputCfg
    columns: ColumnsCfg
    algorithm: AlgoOpts
    cluster: ClusterCfg
    output: OutputCfg


_CONFIG_HELP_TEXT = """
HiPS catalog pipeline configuration reference
=============================================

Top-level sections
------------------
input      [required]
columns    [required]
algorithm  [required]
cluster    [required]
output     [required]

input
-----
paths         [required] list[str]
    Glob patterns for input files (Parquet/CSV/TSV/HATS).
format        [optional, default="parquet"]
    One of: "parquet", "csv", "tsv", "hats".
header        [optional, default=True]
    Whether CSV/TSV files include a header row.
ascii_format  [optional, default=None]
    Optional hint for ASCII input ("CSV" or "TSV").

columns
-------
ra    [required] str
    RA column name.
dec   [required] str
    DEC column name.
keep  [optional, default=None] list[str] or null
    Controls which columns are kept in the HiPS tiles:
      - Not set / null (default):
          Keep all input columns. RA, DEC, score expression dependencies
          (sg_score_column / score_column or sdh_score_column)
          and mg_mag_column (when used) are moved to the beginning of the
          output column order.
      - Empty list []:
          Keep only the minimal set required by the pipeline:
          RA, DEC, score expression dependencies and mg_mag_column (when used).
      - Non-empty list:
          Keep the minimal set (RA, DEC, score deps, mg_mag_column if any)
          plus the explicitly listed columns.

algorithm
---------
selection_mode         [required]
    High-level selection strategy. Must be one of:
      - "mag_global" → global magnitude-complete selection.
      - "score_global" → global selection using an arbitrary score/column.
      - "score_density_hybrid" → hybrid score selection with density-driven depths 1–3.
level_limit            [required] int
    Maximum HiPS order (NorderL). Must be in [4, 11].
moc_order              [optional, default=level_limit] int
    HiPS order used for the MOC.

mag_global mode (prefix mg_)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mg_mag_column          [required in mag_global mode] str
    Magnitude column used when selection_mode == "mag_global".
mg_flux_column         [required if mg_mag_column is absent in mag_global mode] str
    Flux column used to derive magnitudes when mg_mag_column is not provided.
mg_mag_offset          [required when using mg_flux_column] float
    Offset applied to the flux→magnitude conversion:
        mag = -2.5 * log10(flux) + mg_mag_offset
mg_mag_min             [optional, default=None] float
    Lower bound of the magnitude range. If omitted, global minimum clipped to >= -2.
mg_mag_max             [optional, default=None] float
    Upper bound of the magnitude range. If omitted, estimated from the histogram peak.
mg_mag_adaptive_range  [optional, default="complete"] str
    How to fill missing mag_min/mag_max:
      - "complete"  → use global min/max when a bound is missing.
      - "hist_peak" → use the histogram peak (bin center) for the missing bound.
mg_mag_hist_nbins      [optional, default=512] int
    Number of bins in the global magnitude histogram.
mg_n_1, mg_n_2, mg_n_3 [optional, default=None] int
    Approximate global target counts for depths 1–3. Must be provided in order.
mg_order_desc          [optional, default=False] bool
    If False, lower magnitudes/scores are better; if True, higher are better.

score_global mode (prefix sg_)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sg_score_column        [required in score_global mode] str
    Column or expression evaluated globally.
sg_score_min           [optional, default=None] float
    Lower bound of the score range. If omitted:
      * sg_score_adaptive_range = "complete" → global minimum.
      * sg_score_adaptive_range = "hist_peak" → histogram peak (bin center).
sg_score_max           [optional, default=None] float
    Upper bound of the score range. If omitted:
      * sg_score_adaptive_range = "complete" → global maximum.
      * sg_score_adaptive_range = "hist_peak" → histogram peak (bin center).
sg_score_adaptive_range [optional, default="complete"] str
    When a bound is missing, how to auto-complete it: "complete" or "hist_peak".
sg_score_hist_nbins    [optional, default=512] int
    Number of bins in the global score histogram.
sg_n_1, sg_n_2, sg_n_3 [optional, default=None] int
    Approximate global target counts for depths 1–3. Must be provided in order.
sg_order_desc          [optional, default=False] bool
    If False, lower scores are better; if True, higher are better.

score_density_hybrid mode (prefix sdh_)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sdh_score_column        [required in score_density_hybrid mode] str
    Column or expression evaluated globally.
sdh_score_min / sdh_score_max [optional, default=None] float
    Optional bounds for the score range. When missing, filled according to
    sdh_score_adaptive_range ("complete" or "hist_peak"), analogously to sg_*.
sdh_score_adaptive_range [optional, default="complete"] str
    How to complete missing score_min/score_max: "complete" or "hist_peak".
sdh_score_hist_nbins    [optional, default=512] int
    Number of bins in the global score histogram.
sdh_n_1 / sdh_n_2 / sdh_n_3 [optional, default=None] int
    Optional fixed totals for depths 1–3 (must be provided in order).
sdh_density_bias_n1 / sdh_density_bias_n2 / sdh_density_bias_n3
    [optional, defaults: 0.1, 0.3, 0.5] float in [0, 1]
    Density bias per depth when distributing targets across tiles.
sdh_order_desc         [optional, default=False] bool
    If False, lower scores are better; if True, higher are better.

cluster
-------
mode                     [optional, default="local"]
    Cluster mode: "local" or "slurm".
n_workers                [optional, default=3] int
    Number of Dask workers.
threads_per_worker       [optional, default=1] int
    Threads per worker.
memory_per_worker        [optional, default="2GB"] str
    Memory per worker (e.g. "8GB").
slurm                    [optional, default=None] dict
    Additional SLURM options when mode == "slurm".
low_memory_mode          [optional, default=True] bool
    Convenience toggle for memory/throughput:
      * True  → persist_ddfs=False, avoid_computes_wherever_possible=True
      * False → persist_ddfs=True, avoid_computes_wherever_possible=False
diagnostics_mode         [optional, default="global"]
    Dask diagnostics mode: "per_step", "global" or "off".

output
------
out_dir      [required] str
    Output directory where the HiPS hierarchy will be written.
cat_name     [required] str
    Catalog name used in metadata and directory naming.
target       [optional, default="0 0"] str
    Target coordinates (RA DEC) for metadata.
creator_did  [optional, default=None] str
    Dataset identifier for the creator, used in metadata.
obs_title    [optional, default=None] str
    Human-readable title for the observation/catalog, used in metadata.
overwrite    [optional, default=False] bool
    If True and output.out_dir already exists, delete its contents before writing.


Examples
========

Example: minimal configuration (dict)
-------------------------------------
This is the smallest valid configuration you can pass to
``load_config_from_dict()``::

    cfg = {
        "input": {
            "paths": ["/path/to/catalog/*.parquet"],
        },
        "columns": {
            "ra": "ra",
            "dec": "dec",
        },
        "algorithm": {
            "selection_mode": "mag_global",
            "level_limit": 10,
            "mg_mag_column": "mag_r",
        },
        "cluster": {},
        "output": {
            "out_dir": "/path/to/output",
            "cat_name": "MyCatalog"
        }
    }


Example: minimal configuration (YAML)
-------------------------------------
This is the smallest valid YAML file you can pass to ``load_config()``::

    input:
      paths:
        - "/path/to/catalog/*.parquet"

    columns:
      ra: "ra"
      dec: "dec"

    algorithm:
      selection_mode: "mag_global"
      level_limit: 10
      mg_mag_column: "mag_r"

    cluster: {}

    output:
      out_dir: "/path/to/output"
      cat_name: "MyCatalog"
""".strip()


def display_available_configs() -> None:
    """Display a concise reference of all configuration options.

    This prints a structured summary of all available configuration keys,
    grouped by top-level section (input, columns, algorithm, cluster, output),
    indicating which parameters are required, which are optional, and the
    default values for optional parameters.

    This function is intended for interactive use, e.g.:

        from hipscatalog_gen.config import display_available_configs
        display_available_configs()
    """
    print(_CONFIG_HELP_TEXT)


def _build_config_from_mapping(y: Mapping[str, Any]) -> Config:
    """Internal helper to build a Config from a raw mapping."""
    algo = y["algorithm"]

    def _get_mode_value(mapping: Mapping[str, Any], key: str, default=None):
        return mapping.get(key, default)

    # ------------------------------------------------------------------
    # Common settings (all modes)
    # ------------------------------------------------------------------
    raw_selection_mode = algo.get("selection_mode")
    if raw_selection_mode is None:
        raise ValueError(
            "Missing required parameter: algorithm.selection_mode. "
            "Set it to 'mag_global', 'score_global' or 'score_density_hybrid' "
            "in the configuration."
        )
    selection_mode = str(raw_selection_mode).lower()
    allowed_modes = {"mag_global", "score_global", "score_density_hybrid"}
    if selection_mode not in allowed_modes:
        raise ValueError(
            f"algorithm.selection_mode must be one of {sorted(allowed_modes)} (got {selection_mode!r})."
        )

    level_limit = int(algo["level_limit"])
    raw_moc_order = algo.get("moc_order")
    moc_order = level_limit if raw_moc_order is None else int(raw_moc_order)
    if moc_order > level_limit:
        moc_order = level_limit

    # ------------------------------------------------------------------
    # mag_global mode
    # ------------------------------------------------------------------
    n_1_raw = _get_mode_value(algo, "mg_n_1")
    n_2_raw = _get_mode_value(algo, "mg_n_2")
    n_3_raw = _get_mode_value(algo, "mg_n_3")

    # ------------------------------------------------------------------
    # score_global mode
    # ------------------------------------------------------------------
    score_n_1_raw = _get_mode_value(algo, "sg_n_1")
    score_n_2_raw = _get_mode_value(algo, "sg_n_2")
    score_n_3_raw = _get_mode_value(algo, "sg_n_3")

    # ------------------------------------------------------------------
    # score_density_hybrid mode
    # ------------------------------------------------------------------
    sdh_n_1_raw = _get_mode_value(algo, "sdh_n_1")
    sdh_n_2_raw = _get_mode_value(algo, "sdh_n_2")
    sdh_n_3_raw = _get_mode_value(algo, "sdh_n_3")

    # Enforce prefix rule: n_2 requires n_1, n_3 requires n_1 and n_2.
    if n_2_raw is not None and n_1_raw is None:
        raise ValueError(
            "algorithm.mg_n_2 is set but algorithm.mg_n_1 is missing. "
            "These controls must be provided in order: mg_n_1, then mg_n_2, then mg_n_3."
        )
    if n_3_raw is not None and (n_1_raw is None or n_2_raw is None):
        raise ValueError(
            "algorithm.mg_n_3 is set but algorithm.mg_n_1 and algorithm.mg_n_2 are not "
            "both defined. These controls must be provided in order: mg_n_1, mg_n_2, mg_n_3."
        )
    if score_n_2_raw is not None and score_n_1_raw is None:
        raise ValueError(
            "algorithm.sg_n_2 is set but algorithm.sg_n_1 is missing. "
            "These controls must be provided in order: sg_n_1, then sg_n_2, then sg_n_3."
        )
    if score_n_3_raw is not None and (score_n_1_raw is None or score_n_2_raw is None):
        raise ValueError(
            "algorithm.sg_n_3 is set but algorithm.sg_n_1 and algorithm.sg_n_2 are not "
            "both defined. These controls must be provided in order: sg_n_1, sg_n_2, sg_n_3."
        )
    if sdh_n_2_raw is not None and sdh_n_1_raw is None:
        raise ValueError(
            "algorithm.sdh_n_2 is set but algorithm.sdh_n_1 is missing. "
            "These controls must be provided in order: sdh_n_1, then sdh_n_2, then sdh_n_3."
        )
    if sdh_n_3_raw is not None and (sdh_n_1_raw is None or sdh_n_2_raw is None):
        raise ValueError(
            "algorithm.sdh_n_3 is set but algorithm.sdh_n_1 and algorithm.sdh_n_2 are not "
            "both defined. These controls must be provided in order: sdh_n_1, sdh_n_2, sdh_n_3."
        )

    def _to_int_or_none(x, name: str) -> Optional[int]:
        if x is None:
            return None
        try:
            v = int(x)
        except Exception as err:
            raise ValueError(f"algorithm.{name} must be an integer, got {x!r}.") from err
        if v < 0:
            raise ValueError(f"algorithm.{name} must be non-negative, got {v}.")
        return v

    n_1 = _to_int_or_none(n_1_raw, "mg_n_1")
    n_2 = _to_int_or_none(n_2_raw, "mg_n_2")
    n_3 = _to_int_or_none(n_3_raw, "mg_n_3")
    score_n_1 = _to_int_or_none(score_n_1_raw, "sg_n_1")
    score_n_2 = _to_int_or_none(score_n_2_raw, "sg_n_2")
    score_n_3 = _to_int_or_none(score_n_3_raw, "sg_n_3")
    sdh_n_1 = _to_int_or_none(sdh_n_1_raw, "sdh_n_1")
    sdh_n_2 = _to_int_or_none(sdh_n_2_raw, "sdh_n_2")
    sdh_n_3 = _to_int_or_none(sdh_n_3_raw, "sdh_n_3")

    cfg = Config(
        input=InputCfg(
            paths=y["input"]["paths"],
            format=y["input"].get("format", "parquet"),
            header=y["input"].get("header", True),
            ascii_format=y["input"].get("ascii_format"),
        ),
        columns=ColumnsCfg(
            ra=y["columns"]["ra"],
            dec=y["columns"]["dec"],
            keep=y["columns"].get("keep"),
        ),
        algorithm=AlgoOpts(
            # Common settings
            selection_mode=selection_mode,
            level_limit=level_limit,
            moc_order=moc_order,
            mg_order_desc=bool(_get_mode_value(algo, "mg_order_desc", False)),
            # mag_global mode
            mag_column=_get_mode_value(algo, "mg_mag_column"),
            flux_column=_get_mode_value(algo, "mg_flux_column"),
            mag_offset=_get_mode_value(algo, "mg_mag_offset"),
            mag_min=_get_mode_value(algo, "mg_mag_min"),
            mag_max=_get_mode_value(algo, "mg_mag_max"),
            mag_adaptive_range=_get_mode_value(algo, "mg_mag_adaptive_range", "complete"),
            mag_hist_nbins=int(_get_mode_value(algo, "mg_mag_hist_nbins", 512)),
            n_1=n_1,
            n_2=n_2,
            n_3=n_3,
            # score_global mode
            score_column=_get_mode_value(algo, "sg_score_column"),
            score_min=_get_mode_value(algo, "sg_score_min"),
            score_max=_get_mode_value(algo, "sg_score_max"),
            score_adaptive_range=_get_mode_value(algo, "sg_score_adaptive_range", "complete"),
            score_hist_nbins=int(_get_mode_value(algo, "sg_score_hist_nbins", 512)),
            score_n_1=score_n_1,
            score_n_2=score_n_2,
            score_n_3=score_n_3,
            sg_order_desc=bool(_get_mode_value(algo, "sg_order_desc", False)),
            # score_density_hybrid mode
            sdh_score_column=_get_mode_value(algo, "sdh_score_column"),
            sdh_score_min=_get_mode_value(algo, "sdh_score_min"),
            sdh_score_max=_get_mode_value(algo, "sdh_score_max"),
            sdh_score_adaptive_range=_get_mode_value(algo, "sdh_score_adaptive_range", "complete"),
            sdh_score_hist_nbins=int(_get_mode_value(algo, "sdh_score_hist_nbins", 512)),
            sdh_n_1=sdh_n_1,
            sdh_n_2=sdh_n_2,
            sdh_n_3=sdh_n_3,
            sdh_density_bias_n1=float(_get_mode_value(algo, "sdh_density_bias_n1", 1.0)),
            sdh_density_bias_n2=float(_get_mode_value(algo, "sdh_density_bias_n2", 1.0)),
            sdh_density_bias_n3=float(_get_mode_value(algo, "sdh_density_bias_n3", 1.0)),
            sdh_order_desc=bool(_get_mode_value(algo, "sdh_order_desc", False)),
        ),
        cluster=ClusterCfg(
            mode=y["cluster"].get("mode", "local"),
            n_workers=int(y["cluster"].get("n_workers", 3)),
            threads_per_worker=int(y["cluster"].get("threads_per_worker", 1)),
            memory_per_worker=str(y["cluster"].get("memory_per_worker", "2GB")),
            slurm=y["cluster"].get("slurm"),
            low_memory_mode=bool(y["cluster"].get("low_memory_mode", True)),
            persist_ddfs=False,  # set below after resolving low_memory_mode
            avoid_computes_wherever_possible=True,  # set below after resolving low_memory_mode
            diagnostics_mode=y["cluster"].get("diagnostics_mode", "global"),
        ),
        output=OutputCfg(
            out_dir=y["output"]["out_dir"],
            cat_name=y["output"]["cat_name"],
            target=y["output"].get("target", "0 0"),
            creator_did=y["output"].get("creator_did"),
            obs_title=y["output"].get("obs_title"),
            overwrite=bool(y["output"].get("overwrite", False)),
        ),
    )

    # Resolve memory/compute policy
    lmm = bool(y["cluster"].get("low_memory_mode", True))
    # Allow explicit overrides if present, otherwise derive from low_memory_mode.
    cfg.cluster.persist_ddfs = bool(y["cluster"].get("persist_ddfs", not lmm))
    cfg.cluster.avoid_computes_wherever_possible = bool(
        y["cluster"].get("avoid_computes_wherever_possible", lmm)
    )
    cfg.cluster.low_memory_mode = lmm

    # ------------------------------------------------------------------
    # mag_global-specific validation (mag_column vs flux_column)
    # ------------------------------------------------------------------
    algo = cfg.algorithm
    mag_col = getattr(algo, "mag_column", None)
    flux_col = getattr(algo, "flux_column", None)
    if mag_col and flux_col:
        raise ValueError(
            "mag_global configuration: mg_mag_column and mg_flux_column are mutually exclusive. "
            "Please set only one of them."
        )
    if str(algo.selection_mode).lower() == "mag_global":
        if not mag_col and not flux_col:
            raise ValueError(
                "selection_mode='mag_global' requires either mg_mag_column " "or mg_flux_column to be set."
            )
        if flux_col and algo.mag_offset is None:
            raise ValueError(
                "selection_mode='mag_global' with mg_flux_column requires "
                "mg_mag_offset to be defined for the flux→magnitude conversion."
            )

        mag_range_mode = str(getattr(algo, "mag_adaptive_range", "complete")).lower()
        if mag_range_mode not in {"complete", "hist_peak"}:
            raise ValueError(
                "selection_mode='mag_global' requires mg_mag_adaptive_range to be "
                "either 'complete' or 'hist_peak'."
            )
        # Normalize to lowercase for downstream consumers.
        algo.mag_adaptive_range = mag_range_mode
    if str(algo.selection_mode).lower() == "score_global":
        if not getattr(algo, "score_column", None):
            raise ValueError("selection_mode='score_global' requires algorithm.score_column to be set.")
        score_range_mode = str(getattr(algo, "score_adaptive_range", "complete") or "complete").lower()
        if score_range_mode not in ("complete", "hist_peak"):
            raise ValueError("algorithm.score_adaptive_range must be either 'complete' or 'hist_peak'.")
        if int(getattr(algo, "score_hist_nbins", 512)) <= 0:
            raise ValueError("algorithm.score_hist_nbins must be a positive integer.")
    if str(algo.selection_mode).lower() == "score_density_hybrid":
        if not getattr(algo, "sdh_score_column", None):
            raise ValueError(
                "selection_mode='score_density_hybrid' requires algorithm.sdh_score_column to be set."
            )
        score_range_mode = str(getattr(algo, "sdh_score_adaptive_range", "complete") or "complete").lower()
        if score_range_mode not in ("complete", "hist_peak"):
            raise ValueError("algorithm.sdh_score_adaptive_range must be either 'complete' or 'hist_peak'.")
        algo.sdh_score_adaptive_range = score_range_mode
        if int(getattr(algo, "sdh_score_hist_nbins", 512)) <= 0:
            raise ValueError("algorithm.sdh_score_hist_nbins must be a positive integer.")
        for name in ("sdh_density_bias_n1", "sdh_density_bias_n2", "sdh_density_bias_n3"):
            val = float(getattr(algo, name, 0.0))
            if val < 0.0 or val > 1.0:
                raise ValueError(f"algorithm.{name} must be in [0, 1]. Got {val}.")
            setattr(algo, name, val)

    return cfg


def load_config(path: str) -> Config:
    """Load configuration from a YAML file.

    The YAML structure must follow the sections described in
    ``display_available_configs()``. For an overview of all available
    configuration keys (required vs optional, and defaults), call:

        from hipscatalog_gen.config import display_available_configs
        display_available_configs()

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed Config instance.

    Raises:
        ValueError: If algorithm options are inconsistent.
    """
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    return _build_config_from_mapping(y)


def load_config_from_dict(cfg_dict: Mapping[str, Any]) -> Config:
    """Build configuration from an in-memory mapping.

    This is useful in interactive environments (e.g., notebooks) where the
    configuration is defined directly as a Python dict instead of a YAML
    file. The mapping must follow the same structure described in
    ``display_available_configs()``. For a summary of all configuration
    keys, call:

        from hipscatalog_gen.config import display_available_configs
        display_available_configs()

    Args:
        cfg_dict: Mapping with the same structure expected in the YAML file.

    Returns:
        Parsed Config instance.

    Raises:
        ValueError: If algorithm options are inconsistent.
    """
    return _build_config_from_mapping(cfg_dict)
