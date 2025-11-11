from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml


__all__ = [
    "AlgoOpts",
    "ColumnsCfg",
    "InputCfg",
    "ClusterCfg",
    "OutputCfg",
    "Config",
    "load_config",
]


@dataclass
class AlgoOpts:
    """Algorithm options for HiPS selection and density profiles.

    Attributes:
        selection_mode: High-level selection strategy ("coverage" or "mag_global").
        level_limit: Maximum HiPS order (NorderL).
        level_coverage: Coverage / MOC order (lC).
        order_desc: If True, sort score in descending order.
        coverage_order: HEALPix order used for coverage cells (__icov__).
        k_per_cov_per_level: Optional per-depth overrides of k per coverage cell.
        targets_total_per_level: Optional per-depth total caps (rows per level).
        tie_buffer: Score tie buffer near the selection cut.
        density_mode: Depth profile mode ("constant", "linear", "exp", "log").
        k_per_cov_initial: Initial expected rows per coverage cell at depth 1.
        targets_total_initial: Initial expected total rows at depth 1.
        density_exp_base: Base used when density_mode == "exp".
        density_bias_mode: Density bias mode ("none", "proportional", "inverse").
        density_bias_exponent: Strength of the density bias.
        fractional_mode: How to handle fractional k ("random" or "score").
        fractional_mode_logic: Scope of fractional logic ("auto", "local", "global").
        use_hats_as_coverage: If True, use HATS/LSDB partitions as coverage cells.
        mag_column: Magnitude column used in mag_global mode.
        mag_min: Lower bound of the magnitude range in mag_global mode.
        mag_max: Upper bound of the magnitude range in mag_global mode.
        mag_hist_nbins: Number of bins in the global magnitude histogram.
        n_1: Approximate global target for depth 1 in mag_global mode.
        n_2: Approximate global target for depth 2 in mag_global mode.
        n_3: Approximate global target for depth 3 in mag_global mode.
    """

    # Selection mode:
    #   "coverage"  → coverage-based selection
    #   "mag_global" → global magnitude-complete selection
    selection_mode: str

    # HiPS / coverage geometry
    level_limit: int            # maximum HiPS order (NorderL)
    level_coverage: int         # MOC / coverage order (lC)
    order_desc: bool            # False → ascending score (lower is better)
    coverage_order: int         # HEALPix order for __icov__ coverage cells

    # Optional per-level overrides for k (hard overrides by depth).
    # Example: {3: 0.6, 4: 1.2}
    k_per_cov_per_level: Optional[Dict[int, float]] = None

    # Optional global total caps per level (rows per level).
    targets_total_per_level: Optional[Dict[int, int]] = None

    # Score tie buffer near the cut (helps avoid artifacts).
    tie_buffer: int = 10

    # -------------------------
    # Density profile controls
    # -------------------------
    # How k (or total targets) varies with depth:
    #   "constant" → same base value at all depths
    #   "linear"   → increases linearly with depth
    #   "exp"      → increases exponentially with depth
    #   "log"      → increases ~log(depth)
    density_mode: str = "constant"

    # Expected rows per coverage cell (__icov__) at depth 1
    # (base of the density profile in per-coverage mode).
    k_per_cov_initial: float = 1.0

    # Expected total number of rows at depth 1 (base of the total-target profile).
    # When not None, the per-depth density is derived as:
    #   T_desired(depth) → k_desired(depth) = T_desired / N_cov
    # and k_per_cov_initial is ignored by the density profile.
    targets_total_initial: Optional[float] = None

    # Base used only when density_mode == "exp".
    density_exp_base: float = 2.0

    # Optional density bias based on coverage density (densmap at coverage_order).
    # density_bias_mode:
    #   "none"         → no bias (default)
    #   "proportional" → k_per_cov increases with density
    #   "inverse"      → k_per_cov increases in sparse regions
    density_bias_mode: str = "none"
    density_bias_exponent: float = 1.0

    # How to handle the fractional part of k:
    #   "random" → random +1 decisions
    #   "score"  → score-based decisions
    fractional_mode: str = "random"

    # Scope of the fractional logic:
    #   "auto"   → random → local, score → global (backward-compatible)
    #   "local"  → per coverage cell (__icov__)
    #   "global" → union of all coverage cells at this depth
    fractional_mode_logic: str = "auto"

    # When True (and input.format == "hats"), use HATS/LSDB partitions
    # themselves as coverage cells (__icov__), instead of HEALPix cells.
    use_hats_as_coverage: bool = False

    # -------------------------
    # mag_global selection controls
    # -------------------------
    mag_column: Optional[str] = None
    mag_min: Optional[float] = None
    mag_max: Optional[float] = None

    # Number of bins used for the global magnitude histogram in mag_global mode.
    mag_hist_nbins: int = 256

    # Optional approximate total targets for the first HiPS orders (depths 1–3)
    # in mag_global mode. These are global target counts per depth before
    # magnitude slicing. They must be provided in order: n_1, then n_2, then n_3.
    n_1: Optional[int] = None
    n_2: Optional[int] = None
    n_3: Optional[int] = None


@dataclass
class ColumnsCfg:
    """Column mapping for RA/DEC, score and extra fields."""

    ra: str                     # RA column name (or index for ASCII without header)
    dec: str                    # DEC column name
    score: str                  # score expression or column used for ranking
    keep: Optional[List[str]] = None  # optional explicit list of columns to keep


@dataclass
class InputCfg:
    """Input catalog configuration."""

    paths: List[str]            # list of glob patterns for files
    format: str                 # "parquet" | "csv" | "tsv"
    header: bool                # header row present for CSV/TSV
    ascii_format: Optional[str] = None  # optional hint ("CSV" or "TSV")


@dataclass
class ClusterCfg:
    """Dask cluster configuration."""

    mode: str                   # "local" | "slurm"
    n_workers: int
    threads_per_worker: int
    memory_per_worker: str      # e.g. "8GB"
    slurm: Optional[Dict] = None
    persist_ddfs: bool = False
    avoid_computes_wherever_possible: bool = True
    diagnostics_mode: str = "per_step"  # "per_step" | "global" | "off"


@dataclass
class OutputCfg:
    """Output HiPS catalog configuration."""

    out_dir: str
    cat_name: str
    target: str
    creator_did: Optional[str] = None
    obs_title: Optional[str] = None


@dataclass
class Config:
    """Top-level configuration container for the HiPS pipeline."""

    input: InputCfg
    columns: ColumnsCfg
    algorithm: AlgoOpts
    cluster: ClusterCfg
    output: OutputCfg


def load_config(path: str) -> Config:
    """Load configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed Config instance.

    Raises:
        ValueError: If algorithm options are inconsistent.
    """
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    algo = y["algorithm"]

    # Coverage / MOC orders
    level_limit = int(algo["level_limit"])
    raw_level_coverage = algo.get("level_coverage")
    raw_coverage_order = algo.get("coverage_order")

    # If only one of level_coverage / coverage_order is provided, use it for the other.
    if raw_level_coverage is None and raw_coverage_order is None:
        # Default: use level_limit for both if neither is explicitly given.
        raw_level_coverage = level_limit
        raw_coverage_order = level_limit
    elif raw_level_coverage is None:
        raw_level_coverage = raw_coverage_order
    elif raw_coverage_order is None:
        raw_coverage_order = raw_level_coverage

    level_coverage = int(raw_level_coverage)
    coverage_order = int(raw_coverage_order)

    # Density / selection parameters
    density_mode = algo.get("density_mode", "constant")

    # Mutually exclusive initial parameters:
    #   * k_per_cov_initial     → base expected rows per coverage cell (depth 1)
    #   * targets_total_initial → base expected total rows per level (depth 1)
    raw_k_per_cov_initial = algo.get("k_per_cov_initial", None)
    raw_targets_total_initial = algo.get("targets_total_initial", None)

    if raw_k_per_cov_initial is not None and raw_targets_total_initial is not None:
        raise ValueError(
            "algorithm.k_per_cov_initial and algorithm.targets_total_initial "
            "are mutually exclusive. Please define only one of them in the YAML."
        )

    if raw_k_per_cov_initial is not None:
        k_per_cov_initial = float(raw_k_per_cov_initial)
        targets_total_initial = None
    elif raw_targets_total_initial is not None:
        targets_total_initial = float(raw_targets_total_initial)
        # k_per_cov_initial is not used when targets_total_initial is set,
        # but we keep a harmless default for completeness / compatibility.
        k_per_cov_initial = 1.0
    else:
        # Default behaviour: per-coverage profile with k_per_cov_initial = 1.0
        # and no total-target profile.
        k_per_cov_initial = 1.0
        targets_total_initial = None

    # Approximate fixed totals for mag_global selection (depths 1–3).
    n_1_raw = algo.get("n_1", None)
    n_2_raw = algo.get("n_2", None)
    n_3_raw = algo.get("n_3", None)

    # Enforce prefix rule: n_2 requires n_1, n_3 requires n_1 and n_2.
    if n_2_raw is not None and n_1_raw is None:
        raise ValueError(
            "algorithm.n_2 is set but algorithm.n_1 is missing. "
            "These controls must be provided in order: n_1, then n_2, then n_3."
        )
    if n_3_raw is not None and (n_1_raw is None or n_2_raw is None):
        raise ValueError(
            "algorithm.n_3 is set but algorithm.n_1 and algorithm.n_2 are not "
            "both defined. These controls must be provided in order: n_1, n_2, n_3."
        )

    def _to_int_or_none(x, name: str) -> Optional[int]:
        if x is None:
            return None
        try:
            v = int(x)
        except Exception:
            raise ValueError(f"algorithm.{name} must be an integer, got {x!r}.")
        if v < 0:
            raise ValueError(f"algorithm.{name} must be non-negative, got {v}.")
        return v

    n_1 = _to_int_or_none(n_1_raw, "n_1")
    n_2 = _to_int_or_none(n_2_raw, "n_2")
    n_3 = _to_int_or_none(n_3_raw, "n_3")

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
            score=y["columns"]["score"],
            keep=y["columns"].get("keep"),
        ),
        algorithm=AlgoOpts(
            selection_mode=str(algo.get("selection_mode", "coverage")).lower(),
            level_limit=level_limit,
            level_coverage=level_coverage,
            order_desc=bool(algo.get("order_desc", False)),
            coverage_order=coverage_order,
            # Per-level overrides for k (float values).
            k_per_cov_per_level=(
                {int(k): float(v) for k, v in algo.get("k_per_cov_per_level", {}).items()}
                if isinstance(algo.get("k_per_cov_per_level"), dict)
                else None
            ),
            # Per-level total caps (int values).
            targets_total_per_level=(
                {int(k): int(v) for k, v in algo.get("targets_total_per_level", {}).items()}
                if isinstance(algo.get("targets_total_per_level"), dict)
                else None
            ),
            tie_buffer=int(algo.get("tie_buffer", 10)),
            density_mode=density_mode,
            k_per_cov_initial=k_per_cov_initial,
            targets_total_initial=targets_total_initial,
            density_exp_base=float(algo.get("density_exp_base", 2.0)),
            density_bias_mode=algo.get("density_bias_mode", "none"),
            density_bias_exponent=float(algo.get("density_bias_exponent", 1.0)),
            fractional_mode=algo.get("fractional_mode", "random"),
            fractional_mode_logic=algo.get("fractional_mode_logic", "auto"),
            use_hats_as_coverage=bool(algo.get("use_hats_as_coverage", False)),
            mag_column=algo.get("mag_column"),
            mag_min=algo.get("mag_min"),
            mag_max=algo.get("mag_max"),
            mag_hist_nbins=int(algo.get("mag_hist_nbins", 256)),
            n_1=n_1,
            n_2=n_2,
            n_3=n_3,
        ),
        cluster=ClusterCfg(
            mode=y["cluster"].get("mode", "local"),
            n_workers=int(y["cluster"].get("n_workers", 4)),
            threads_per_worker=int(y["cluster"].get("threads_per_worker", 1)),
            memory_per_worker=str(y["cluster"].get("memory_per_worker", "4GB")),
            slurm=y["cluster"].get("slurm"),
            persist_ddfs=bool(y["cluster"].get("persist_ddfs", False)),
            avoid_computes_wherever_possible=bool(
                y["cluster"].get("avoid_computes_wherever_possible", True)
            ),
            diagnostics_mode=y["cluster"].get("diagnostics_mode", "per_step"),
        ),
        output=OutputCfg(
            out_dir=y["output"]["out_dir"],
            cat_name=y["output"]["cat_name"],
            target=y["output"].get("target", "0 0"),
            creator_did=y["output"].get("creator_did"),
            obs_title=y["output"].get("obs_title"),
        ),
    )

    # Align level_coverage if user set it above level_limit.
    if cfg.algorithm.level_coverage > cfg.algorithm.level_limit:
        cfg.algorithm.level_coverage = cfg.algorithm.level_limit

    return cfg
