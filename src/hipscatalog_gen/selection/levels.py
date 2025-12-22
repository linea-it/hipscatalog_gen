"""Assign targets per HEALPix level and depth."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .score import _quantile_from_histogram

__all__ = ["assign_level_edges"]


def assign_level_edges(
    densmaps: Dict[int, np.ndarray],
    depths_sel: List[int],
    fixed_targets: Dict[int, float],
    cdf_hist: np.ndarray,
    score_edges_hist: np.ndarray,
    score_min: float,
    score_max: float,
    n_tot_score: float,
    log_fn,
    label: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cumulative targets per depth and corresponding score edges."""
    weights_list: List[float] = []
    for d in depths_sel:
        counts_d = densmaps[d]
        tiles_active = int((counts_d > 0).sum())
        weights_list.append(max(1, tiles_active))

    weights = np.asarray(weights_list, dtype="float64")
    T = np.zeros_like(weights, dtype="float64")

    fixed_norm: Dict[int, float] = {}
    for d, val in fixed_targets.items():
        if d not in depths_sel:
            continue
        if int(val) < 0:
            raise ValueError(f"{label}: fixed target for depth {d} must be non-negative (got {val}).")
        fixed_norm[int(d)] = float(int(val))

    sum_fixed = float(sum(fixed_norm.values()))
    if sum_fixed > n_tot_score and sum_fixed > 0.0:
        scale = float(n_tot_score) / sum_fixed if sum_fixed > 0 else 0.0
        log_fn(
            f"[{label}] Sum of fixed targets ({int(sum_fixed)}) exceeds total objects "
            f"in range ({int(n_tot_score)}). Rescaling by factor {scale:.3f}.",
            always=True,
        )
        for d in list(fixed_norm.keys()):
            fixed_norm[d] *= scale
        sum_fixed = float(n_tot_score)

    for d, val in fixed_norm.items():
        idx = depths_sel.index(d)
        T[idx] = val

    N_rem = max(0.0, float(n_tot_score) - sum_fixed)
    if N_rem > 0.0:
        free_mask = np.ones_like(weights, dtype=bool)
        for d in fixed_norm:
            idx = depths_sel.index(d)
            free_mask[idx] = False

        W_free = float(weights[free_mask].sum())
        if W_free <= 0.0:
            n_free = int(free_mask.sum())
            if n_free > 0:
                T[free_mask] += N_rem / float(n_free)
        else:
            T[free_mask] += weights[free_mask] / W_free * N_rem

    T_cum = np.cumsum(T)
    Q = T_cum / float(n_tot_score) if n_tot_score > 0.0 else np.zeros_like(T_cum, dtype="float64")

    level_edges: np.ndarray = np.empty(len(depths_sel) + 1, dtype="float64")
    level_edges[0] = score_min
    for i, q in enumerate(Q, start=1):
        level_edges[i] = _quantile_from_histogram(cdf_hist, score_edges_hist, q)

    level_edges = np.maximum.accumulate(level_edges)

    # Enforce strictly increasing edges (except final clamp) to avoid zero-width slices.
    for i in range(1, len(level_edges)):
        if level_edges[i] <= level_edges[i - 1]:
            level_edges[i] = min(score_max, np.nextafter(level_edges[i - 1], np.inf))

    level_edges[0] = score_min
    level_edges[-1] = score_max
    return level_edges, T
