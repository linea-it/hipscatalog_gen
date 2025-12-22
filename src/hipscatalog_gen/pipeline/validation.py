"""Validation helpers for configuration blocks across selection modes."""

from __future__ import annotations

from typing import Any


def _validate_nk_pairs(label: str, cfg: Any, prefix: str = "") -> None:
    """Ensure mutually exclusive n_i/k_i pairs."""
    for depth in (1, 2, 3):
        n_val = getattr(cfg, f"{prefix}n_{depth}", None)
        k_val = getattr(cfg, f"{prefix}k_{depth}", None)
        if (n_val is not None) and (k_val is not None):
            raise ValueError(f"{label}: both n_{depth} and k_{depth} are set; choose one.")
        if n_val is not None and int(n_val) < 0:
            raise ValueError(f"{label}: n_{depth} must be non-negative.")
        if k_val is not None and float(k_val) < 0:
            raise ValueError(f"{label}: k_{depth} must be non-negative.")


def validate_mag_global_cfg(cfg: Any) -> None:
    """Surface mag_global config issues early."""
    algo = cfg.algorithm
    mag_col_cfg = getattr(algo, "mag_column", None)
    flux_col_cfg = getattr(algo, "flux_column", None)
    if mag_col_cfg and flux_col_cfg:
        raise ValueError("mag_global: mag_column and flux_column are mutually exclusive.")
    if flux_col_cfg and getattr(algo, "mag_offset", None) is None:
        raise ValueError("mag_global: flux_column requires algorithm.mag_offset to be set.")
    if int(getattr(algo, "mag_hist_nbins", 0)) <= 0:
        raise ValueError("mag_global: mag_hist_nbins must be positive.")
    _validate_nk_pairs("mag_global", algo, prefix="")


def validate_score_global_cfg(cfg: Any) -> None:
    """Surface score_global config issues early."""
    algo = cfg.algorithm
    if not getattr(algo, "score_column", None):
        raise ValueError("score_global: algorithm.score_column is required.")
    if int(getattr(algo, "score_hist_nbins", getattr(algo, "mag_hist_nbins", 0))) <= 0:
        raise ValueError("score_global: score_hist_nbins must be positive.")
    _validate_nk_pairs("score_global", algo, prefix="score_")


def validate_score_density_hybrid_cfg(cfg: Any) -> None:
    """Surface score_density_hybrid config issues early."""
    algo = cfg.algorithm
    score_expr = getattr(algo, "sdh_score_column", getattr(algo, "score_column", None))
    if not score_expr:
        raise ValueError("score_density_hybrid: algorithm.sdh_score_column (or score_column) is required.")
    if int(getattr(algo, "sdh_score_hist_nbins", getattr(algo, "score_hist_nbins", 0))) <= 0:
        raise ValueError("score_density_hybrid: sdh_score_hist_nbins must be positive.")
    _validate_nk_pairs("score_density_hybrid", algo, prefix="sdh_")


def validate_common_cfg(cfg: Any) -> None:
    """Cross-field validation for cluster/output and shared algorithm settings."""
    algo = cfg.algorithm
    # level_limit vs moc_order
    lM = int(getattr(algo, "level_limit", 0))
    moc_order = int(getattr(algo, "moc_order", lM))
    if moc_order > lM:
        raise ValueError(f"algorithm.moc_order ({moc_order}) must be <= level_limit ({lM}).")
    if lM <= 0:
        raise ValueError("algorithm.level_limit must be positive.")

    # Cluster required fields
    cluster = cfg.cluster
    if getattr(cluster, "mode", None) not in {"local", "slurm"}:
        raise ValueError("cluster.mode must be 'local' or 'slurm'.")
    for field_name in ("n_workers", "threads_per_worker", "memory_per_worker"):
        if getattr(cluster, field_name, None) is None:
            raise ValueError(f"cluster.{field_name} is required.")

    # Output required fields
    output = cfg.output
    for field_name in ("out_dir", "cat_name", "target"):
        val = getattr(output, field_name, None)
        if val is None or str(val).strip() == "":
            raise ValueError(f"output.{field_name} must be set.")
