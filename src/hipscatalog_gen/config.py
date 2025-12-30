"""Configuration parsing and validation for hipscatalog-gen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

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
        order_desc/tie_column/keep_invalid_values: global defaults for ordering, tie-breakers,
        and handling NaN/Inf (sentinel mapping only for adaptive_range=complete).

    mag_global block:
        mag_column or flux_column+mag_offset; mag_min/max; adaptive_range; hist_nbins;
        optional n_1/n_2/n_3 targets (or k_1/k_2/k_3 as “per active tile” aliases);
        order_desc/tie_column/keep_invalid_values (fall back to selection_defaults).

    score_global block:
        score_column; score_min/max; adaptive_range; hist_nbins;
        optional n_1/n_2/n_3 (or k_1/k_2/k_3) targets; order_desc/tie_column/keep_invalid_values
        (fall back to selection_defaults).

    score_density_hybrid block:
        score_column; score_min/max; adaptive_range; hist_nbins;
        optional n_1/n_2/n_3 (or k_1/k_2/k_3) targets; density_bias_n1/n2/n3;
        order_desc/tie_column/keep_invalid_values (fall back to selection_defaults).
    """

    # Common settings
    selection_mode: str
    level_limit: int  # maximum HiPS order (NorderL)
    moc_order: int  # HiPS order for the MOC
    order_desc: bool = False  # global default for ordering (False → ascending/lower is better)
    tie_column: str | None = None  # optional tie-breaker (falls back to RA/DEC)
    mg_order_desc: bool = False  # per-mode override

    # mag_global mode (precedence: mag_column or flux_column+offset)
    mag_column: str | None = None
    flux_column: str | None = None
    mag_offset: float | None = None
    mag_min: float | None = None
    mag_max: float | None = None
    mag_adaptive_range: str = "complete"
    mag_hist_nbins: int = 2048
    mag_keep_invalid_values: bool = False  # map NaN/Inf to sentinel when True (complete mode only)
    mag_tie_column: str | None = None  # optional tie-breaker for mag_global
    n_1: int | None = None
    n_2: int | None = None
    n_3: int | None = None
    k_1: int | None = None  # optional “per active tile” alias for n_1
    k_2: int | None = None  # optional “per active tile” alias for n_2
    k_3: int | None = None  # optional “per active tile” alias for n_3

    # score_global mode
    score_column: str | None = None
    score_min: float | None = None
    score_max: float | None = None
    score_adaptive_range: str = "complete"
    score_hist_nbins: int = 2048
    score_keep_invalid_values: bool = False  # keep NaN/Inf with sentinel (complete mode only)
    score_tie_column: str | None = None  # optional tie-breaker for score_global
    score_n_1: int | None = None
    score_n_2: int | None = None
    score_n_3: int | None = None
    score_k_1: int | None = None  # optional “per active tile” alias for score_n_1
    score_k_2: int | None = None  # optional “per active tile” alias for score_n_2
    score_k_3: int | None = None  # optional “per active tile” alias for score_n_3
    sg_order_desc: bool = False  # per-mode override

    # score_density_hybrid mode
    sdh_score_column: str | None = None
    sdh_score_min: float | None = None
    sdh_score_max: float | None = None
    sdh_score_adaptive_range: str = "complete"
    sdh_score_hist_nbins: int = 2048
    sdh_keep_invalid_values: bool = False  # keep NaN/Inf with sentinel (complete mode only)
    sdh_tie_column: str | None = None  # optional tie-breaker for score_density_hybrid
    sdh_n_1: int | None = None
    sdh_n_2: int | None = None
    sdh_n_3: int | None = None
    sdh_k_1: int | None = None  # optional “per active tile” alias for sdh_n_1
    sdh_k_2: int | None = None  # optional “per active tile” alias for sdh_n_2
    sdh_k_3: int | None = None  # optional “per active tile” alias for sdh_n_3
    sdh_density_bias_n1: float = 1.0
    sdh_density_bias_n2: float = 1.0
    sdh_density_bias_n3: float = 1.0
    sdh_order_desc: bool = False  # per-mode override


@dataclass
class ColumnsCfg:
    """Column mapping for RA/DEC and extra fields."""

    ra: str  # RA column name (or index for ASCII without header)
    dec: str  # DEC column name
    keep: List[str] | None = None  # optional explicit list of columns to keep


@dataclass
class InputCfg:
    """Input catalog configuration."""

    paths: List[str]  # list of glob patterns for files
    format: str  # "parquet" | "csv" | "tsv"
    header: bool  # header row present for CSV/TSV
    ascii_format: str | None = None  # optional hint ("CSV" or "TSV")


@dataclass
class ClusterCfg:
    """Dask cluster configuration."""

    mode: str  # "local" | "slurm"
    n_workers: int
    threads_per_worker: int
    memory_per_worker: str  # e.g. "8GB"
    slurm: Dict | None = None
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
    creator_did: str | None = None
    obs_title: str | None = None
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
          Keep all input columns; RA/DEC, score expression deps, and mag/flux
          (if mag_global) are ordered first.
      - Empty list []:
          Keep only the essential set: RA, DEC, score deps, and mag/flux
          (if mag_global).
      - Non-empty list:
          Keep the essential set plus all explicitly listed columns (filtered
          by availability).

algorithm (block-based)
-----------------------
selection_mode         [required]
    "mag_global" | "score_global" | "score_density_hybrid".
level_limit            [required] int
    Maximum HiPS order (NorderL). Must be in [4, 11].
moc_order              [optional, default=level_limit] int
    HiPS order used for the MOC.
selection_defaults     [optional] dict
    Shared defaults for all modes. Recognized keys:
      - hist_nbins        (int, default 2048)
      - adaptive_range    (\"complete\" | \"hist_peak\", default \"complete\")
      - order_desc        (bool, default False)
      - density_bias_n1/n2/n3 (float, default 1.0 for SDH)

mag_global block
^^^^^^^^^^^^^^^^
mag_global.mag_column        [required if flux_column absent] str
mag_global.flux_column       [required if mag_column absent] str
mag_global.mag_offset        [required when flux_column is set] float
mag_global.mag_min/max       [optional] float
mag_global.adaptive_range    [optional, default=selection_defaults.adaptive_range or \"complete\"]
mag_global.hist_nbins        [optional, default=selection_defaults.hist_nbins or 2048]
mag_global.k_1/k_2/k_3       [optional] int, \"per active tile\" aliases for n_*
mag_global.n_1/n_2/n_3       [optional] int (must be provided in order)
mag_global.order_desc        [optional, default=selection_defaults.order_desc or False]

score_global block
^^^^^^^^^^^^^^^^^^
score_global.score_column    [required] str (column or expression)
score_global.score_min/max   [optional] float
score_global.adaptive_range  [optional, default=selection_defaults.adaptive_range or \"complete\"]
score_global.hist_nbins      [optional, default=selection_defaults.hist_nbins or 2048]
score_global.k_1/k_2/k_3     [optional] int, \"per active tile\" aliases for n_*
score_global.n_1/n_2/n_3     [optional] int (must be provided in order)
score_global.order_desc      [optional, default=selection_defaults.order_desc or False]

score_density_hybrid block
^^^^^^^^^^^^^^^^^^^^^^^^^^
score_density_hybrid.score_column   [required] str (column or expression)
score_density_hybrid.score_min/max  [optional] float
score_density_hybrid.adaptive_range [optional, default=selection_defaults.adaptive_range or \"complete\"]
score_density_hybrid.hist_nbins     [optional, default=selection_defaults.hist_nbins or 2048]
score_density_hybrid.k_1/k_2/k_3    [optional] int, \"per active tile\" aliases for n_*
score_density_hybrid.n_1/n_2/n_3    [optional] int (must be provided in order)
score_density_hybrid.density_bias_n1/n2/n3 [optional, default=selection_defaults.density_bias_n* or 1.0]
    float in [0,1]
score_density_hybrid.order_desc     [optional, default=selection_defaults.order_desc or False]

cluster
-------
mode                     [optional, default="local"]
    Cluster mode: "local" or "slurm".
n_workers                [optional, default=3] int
threads_per_worker       [optional, default=1] int
memory_per_worker        [optional, default="2GB"] str
slurm                    [optional, default=None] dict
low_memory_mode          [optional, default=True] bool
    True  → persist_ddfs=False, avoid_computes_wherever_possible=True
    False → persist_ddfs=True, avoid_computes_wherever_possible=False
diagnostics_mode         [optional, default="global"]
    "per_step" | "global" | "off".

output
------
out_dir      [required] str
cat_name     [required] str
target       [optional, default="0 0"] str
creator_did  [optional, default=None] str
obs_title    [optional, default=None] str
overwrite    [optional, default=False] bool

Examples
========

Minimal configuration (YAML)
----------------------------

    input:
      paths: ["/path/to/catalog/*.parquet"]
    columns:
      ra: "ra"
      dec: "dec"
    algorithm:
      selection_mode: "mag_global"
      level_limit: 10
      mag_global:
        mag_column: "mag_r"
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

    defaults = algo.get("selection_defaults", {}) or {}
    mag_cfg = algo.get("mag_global", {}) or {}
    score_cfg = algo.get("score_global", {}) or {}
    sdh_cfg = algo.get("score_density_hybrid", {}) or {}

    def _to_int_or_none(x, name: str) -> int | None:
        """Convert to int when set, enforcing non-negative inputs."""
        if x is None:
            return None
        try:
            v = int(x)
        except Exception as err:
            raise ValueError(f"algorithm.{name} must be an integer, got {x!r}.") from err
        if v < 0:
            raise ValueError(f"algorithm.{name} must be non-negative, got {v}.")
        return v

    def _require_order(vals: List[int | None], names: List[str]) -> List[int | None]:
        """Ensure values are provided in order without gaps."""
        for idx, val in enumerate(vals):
            if val is not None and any(v is None for v in vals[:idx]):
                prev = names[idx - 1]
                raise ValueError(f"algorithm.{names[idx]} is set but algorithm.{prev} is missing.")
        return vals

    def _hist_nbins(block: Mapping[str, Any]) -> int:
        """Resolve histogram bin count with per-mode override."""
        return int(block.get("hist_nbins", defaults.get("hist_nbins", 2048)))

    def _keep_invalid(block: Mapping[str, Any]) -> bool:
        """Resolve keep_invalid_values flag with per-mode override."""
        return bool(block.get("keep_invalid_values", defaults.get("keep_invalid_values", False)))

    def _tie_col(block: Mapping[str, Any]) -> str | None:
        """Resolve tie-breaker column with per-mode override."""
        return block.get("tie_column", defaults.get("tie_column", None))

    def _adaptive(block: Mapping[str, Any]) -> str:
        """Resolve adaptive_range mode with per-mode override."""
        return str(block.get("adaptive_range", defaults.get("adaptive_range", "complete"))).lower()

    def _order_desc(block: Mapping[str, Any]) -> bool:
        """Resolve ordering direction with per-mode override."""
        return bool(block.get("order_desc", defaults.get("order_desc", False)))

    order_desc_global = _order_desc(defaults)

    def _order_desc_mode(block: Mapping[str, Any]) -> bool:
        """Resolve per-mode order_desc with fallback to global default."""
        return bool(block.get("order_desc", order_desc_global))

    # Target validation
    n_vals = _require_order(
        [mag_cfg.get("n_1"), mag_cfg.get("n_2"), mag_cfg.get("n_3")],
        ["mag_global.n_1", "mag_global.n_2", "mag_global.n_3"],
    )
    k_vals = _require_order(
        [mag_cfg.get("k_1"), mag_cfg.get("k_2"), mag_cfg.get("k_3")],
        ["mag_global.k_1", "mag_global.k_2", "mag_global.k_3"],
    )
    sg_vals = _require_order(
        [score_cfg.get("n_1"), score_cfg.get("n_2"), score_cfg.get("n_3")],
        ["score_global.n_1", "score_global.n_2", "score_global.n_3"],
    )
    sg_k_vals = _require_order(
        [score_cfg.get("k_1"), score_cfg.get("k_2"), score_cfg.get("k_3")],
        ["score_global.k_1", "score_global.k_2", "score_global.k_3"],
    )
    sdh_vals = _require_order(
        [sdh_cfg.get("n_1"), sdh_cfg.get("n_2"), sdh_cfg.get("n_3")],
        ["score_density_hybrid.n_1", "score_density_hybrid.n_2", "score_density_hybrid.n_3"],
    )
    sdh_k_vals = _require_order(
        [sdh_cfg.get("k_1"), sdh_cfg.get("k_2"), sdh_cfg.get("k_3")],
        ["score_density_hybrid.k_1", "score_density_hybrid.k_2", "score_density_hybrid.k_3"],
    )

    n_1, n_2, n_3 = (_to_int_or_none(v, f"mag_global.n_{i + 1}") for i, v in enumerate(n_vals))
    k_1, k_2, k_3 = (_to_int_or_none(v, f"mag_global.k_{i + 1}") for i, v in enumerate(k_vals))
    score_n_1, score_n_2, score_n_3 = (
        _to_int_or_none(v, f"score_global.n_{i + 1}") for i, v in enumerate(sg_vals)
    )
    score_k_1, score_k_2, score_k_3 = (
        _to_int_or_none(v, f"score_global.k_{i + 1}") for i, v in enumerate(sg_k_vals)
    )
    sdh_n_1, sdh_n_2, sdh_n_3 = (
        _to_int_or_none(v, f"score_density_hybrid.n_{i + 1}") for i, v in enumerate(sdh_vals)
    )
    sdh_k_1, sdh_k_2, sdh_k_3 = (
        _to_int_or_none(v, f"score_density_hybrid.k_{i + 1}") for i, v in enumerate(sdh_k_vals)
    )

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
            order_desc=order_desc_global,
            tie_column=_tie_col(defaults),
            mg_order_desc=_order_desc_mode(mag_cfg),
            mag_tie_column=_tie_col(mag_cfg),
            # mag_global mode
            mag_column=mag_cfg.get("mag_column"),
            flux_column=mag_cfg.get("flux_column"),
            mag_offset=mag_cfg.get("mag_offset"),
            mag_min=mag_cfg.get("mag_min"),
            mag_max=mag_cfg.get("mag_max"),
            mag_adaptive_range=_adaptive(mag_cfg),
            mag_hist_nbins=_hist_nbins(mag_cfg),
            mag_keep_invalid_values=_keep_invalid(mag_cfg),
            k_1=k_1,
            k_2=k_2,
            k_3=k_3,
            n_1=n_1,
            n_2=n_2,
            n_3=n_3,
            # score_global mode
            score_column=score_cfg.get("score_column"),
            score_min=score_cfg.get("score_min"),
            score_max=score_cfg.get("score_max"),
            score_adaptive_range=_adaptive(score_cfg),
            score_hist_nbins=_hist_nbins(score_cfg),
            score_keep_invalid_values=_keep_invalid(score_cfg),
            score_tie_column=_tie_col(score_cfg),
            score_k_1=score_k_1,
            score_k_2=score_k_2,
            score_k_3=score_k_3,
            score_n_1=score_n_1,
            score_n_2=score_n_2,
            score_n_3=score_n_3,
            sg_order_desc=_order_desc_mode(score_cfg),
            # score_density_hybrid mode
            sdh_score_column=sdh_cfg.get("score_column"),
            sdh_score_min=sdh_cfg.get("score_min"),
            sdh_score_max=sdh_cfg.get("score_max"),
            sdh_score_adaptive_range=_adaptive(sdh_cfg),
            sdh_score_hist_nbins=_hist_nbins(sdh_cfg),
            sdh_keep_invalid_values=_keep_invalid(sdh_cfg),
            sdh_tie_column=_tie_col(sdh_cfg),
            sdh_k_1=sdh_k_1,
            sdh_k_2=sdh_k_2,
            sdh_k_3=sdh_k_3,
            sdh_n_1=sdh_n_1,
            sdh_n_2=sdh_n_2,
            sdh_n_3=sdh_n_3,
            sdh_density_bias_n1=float(sdh_cfg.get("density_bias_n1", defaults.get("density_bias_n1", 1.0))),
            sdh_density_bias_n2=float(sdh_cfg.get("density_bias_n2", defaults.get("density_bias_n2", 1.0))),
            sdh_density_bias_n3=float(sdh_cfg.get("density_bias_n3", defaults.get("density_bias_n3", 1.0))),
            sdh_order_desc=_order_desc_mode(sdh_cfg),
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
            "mag_global configuration: mag_global.mag_column and mag_global.flux_column are mutually "
            "exclusive. Please set only one of them."
        )
    if str(algo.selection_mode).lower() == "mag_global":
        if not mag_col and not flux_col:
            raise ValueError(
                "selection_mode='mag_global' requires either mag_global.mag_column "
                "or mag_global.flux_column to be set."
            )
        if flux_col and algo.mag_offset is None:
            raise ValueError(
                "selection_mode='mag_global' with mag_global.flux_column requires "
                "mag_global.mag_offset to be defined for the flux→magnitude conversion."
            )

        mag_range_mode = str(getattr(algo, "mag_adaptive_range", "complete")).lower()
        if mag_range_mode not in {"complete", "hist_peak"}:
            raise ValueError(
                "selection_mode='mag_global' requires mag_global.adaptive_range to be "
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
        if int(getattr(algo, "score_hist_nbins", 2048)) <= 0:
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
        if int(getattr(algo, "sdh_score_hist_nbins", 2048)) <= 0:
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
