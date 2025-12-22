"""Selection helpers for HEALPix slicing, histograms, and score handling."""

from .common import add_ipix_column, reduce_topk_by_group_dask, targets_per_tile
from .levels import assign_level_edges
from .score import add_score_column, compute_histogram_ddf, compute_score_histogram_ddf, resolve_value_range
from .slicing import select_by_score_slices, select_by_value_slices

__all__ = [
    "add_ipix_column",
    "assign_level_edges",
    "reduce_topk_by_group_dask",
    "targets_per_tile",
    "add_score_column",
    "compute_score_histogram_ddf",
    "resolve_value_range",
    "compute_histogram_ddf",
    "select_by_score_slices",
    "select_by_value_slices",
]
