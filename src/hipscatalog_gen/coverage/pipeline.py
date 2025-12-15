from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import dask.dataframe as dd
import healpy as hp
import numpy as np
import pandas as pd

from ..io.output import build_header_line_from_keep
from ..pipeline.common import write_tiles_with_allsky
from ..utils import _HEALPIX_INDEX_RE, _fmt_dur, _get_meta_df, _log_depth_stats
from .utils import (
    _candidates_by_coverage_partition,
    _reduce_coverage_exact,
    _reduce_coverage_exact_dask,
    apply_fractional_k_per_cov,
    build_cov_thresholds,
    filter_remainder_by_coverage_partition,
)

__all__ = ["add_coverage_column", "run_coverage_selection"]


def add_coverage_column(
    ddf: Any,
    cfg: Any,
    is_hats: bool,
    ra_col: str,
    dec_col: str,
    log_fn,
) -> Any:
    """Attach __icov__ to the dataframe, honoring HATS-specific coverage when needed."""
    use_hats_cov = is_hats and bool(getattr(cfg.algorithm, "use_hats_as_coverage", False))

    base_meta = _get_meta_df(ddf)
    meta_with_icov = base_meta.copy()
    meta_with_icov["__icov__"] = pd.Series([], dtype="int64")

    if use_hats_cov:
        try:
            hp_pixels = ddf.get_healpix_pixels()
            n_hp_pixels = len(hp_pixels)
        except Exception:
            hp_pixels = None
            n_hp_pixels = None

        n_parts = ddf.npartitions

        msg = f"[coverage] Using HATS partitions as coverage cells (__icov__), " f"n_partitions={n_parts}"
        if n_hp_pixels is not None:
            msg += f", get_healpix_pixels() returned {n_hp_pixels} pixels"
        log_fn(msg, always=True)

        part_ids = dd.from_pandas(
            pd.Series(range(n_parts), dtype="int64"),
            npartitions=n_parts,
        )

        def _assign_icov(pdf: pd.DataFrame, part_series: pd.Series) -> pd.DataFrame:
            if pdf.empty:
                pdf["__icov__"] = pd.Series([], dtype="int64")
                return pdf
            cov_id = int(part_series.iloc[0])
            pdf = pdf.copy()
            pdf["__icov__"] = cov_id
            return pdf

        return ddf.map_partitions(
            _assign_icov,
            part_ids,
            meta=meta_with_icov,
        )

    Oc = int(cfg.algorithm.coverage_order)
    NSIDE_C = 1 << Oc

    def _add_icov(pdf: pd.DataFrame, ra_col_name: str, dec_col_name: str) -> pd.DataFrame:
        if len(pdf) == 0:
            pdf["__icov__"] = pd.Series([], dtype="int64")
            return pdf

        if is_hats:
            idx_name = getattr(pdf.index, "name", None)
            m = _HEALPIX_INDEX_RE.match(str(idx_name)) if idx_name else None

            if m is not None:
                base_order = int(m.group(1))
                if Oc <= base_order:
                    ipix_base = pdf.index.to_numpy()
                    shift = 2 * (base_order - Oc)
                    icov = (ipix_base >> shift).astype(np.int64)
                    pdf = pdf.copy()
                    pdf["__icov__"] = icov
                    return pdf

        theta = np.deg2rad(90.0 - pd.to_numeric(pdf[dec_col_name], errors="coerce").to_numpy())
        phi = np.deg2rad((pd.to_numeric(pdf[ra_col_name], errors="coerce").to_numpy()) % 360.0)
        icov = hp.ang2pix(NSIDE_C, theta, phi, nest=True).astype(np.int64)
        pdf = pdf.copy()
        pdf["__icov__"] = icov
        return pdf

    return ddf.map_partitions(
        _add_icov,
        ra_col,
        dec_col,
        meta=meta_with_icov,
    )


def _k_desired_from_total_profile(depth: int, algo: Any, n_cov: Optional[int]) -> float:
    mode = (getattr(algo, "density_mode", "constant") or "constant").lower()
    delta = max(0, depth - 1)

    tti = getattr(algo, "targets_total_initial", None)
    if tti is None:
        raise ValueError("targets_total_initial must be set when using total-profile density mode.")
    T0 = float(tti)

    if mode == "constant":
        T_desired = T0
    elif mode == "linear":
        T_desired = T0 * float(1 + delta)
    elif mode == "exp":
        base = float(getattr(algo, "density_exp_base", 2.0))
        if base <= 1.0:
            base = 2.0
        T_desired = T0 * (base ** float(delta))
    elif mode == "log":
        T_desired = T0 * math.log2(delta + 2.0)
    else:
        raise ValueError(f"Unknown density_mode: {algo.density_mode!r}")

    if n_cov is None or n_cov <= 0:
        return 0.0
    return T_desired / float(n_cov)


def _k_desired_from_density_profile(depth: int, algo: Any) -> float:
    k0 = float(algo.k_per_cov_initial)
    mode = (getattr(algo, "density_mode", "constant") or "constant").lower()
    delta = max(0, depth - 1)

    if mode == "constant":
        k_desired = k0
    elif mode == "linear":
        k_desired = k0 * float(1 + delta)
    elif mode == "exp":
        base = float(getattr(algo, "density_exp_base", 2.0))
        if base <= 1.0:
            base = 2.0
        k_desired = k0 * (base ** float(delta))
    elif mode == "log":
        k_desired = k0 * math.log2(delta + 2.0)
    else:
        raise ValueError(f"Unknown density_mode: {algo.density_mode!r}")

    return k_desired


def run_coverage_selection(
    remainder_ddf: Any,
    cfg: Any,
    densmaps: Dict[int, np.ndarray],
    keep_cols: list[str],
    ra_col: str,
    dec_col: str,
    out_dir,
    diag_ctx,
    log_fn,
    persist_ddfs: bool,
    avoid_computes: bool,
    is_hats: bool,
) -> None:
    """Execute the coverage-based selection loop."""
    score_col = getattr(cfg.algorithm, "coverage_score_column", None)
    if not score_col:
        raise ValueError("algorithm.coverage_score_column must be set for coverage selection.")

    if float(cfg.algorithm.k_per_cov_initial) <= 0.0 and not (cfg.algorithm.k_per_cov_per_level or {}):
        log_fn(
            "[selection] k_per_cov_initial <= 0 and no per-level overrides → "
            "nothing to select; finishing early.",
            always=True,
        )
        return

    Tmap = cfg.algorithm.targets_total_per_level or {}

    cov_order = int(cfg.algorithm.coverage_order)
    dens_cov: Dict[int, int] = {}
    if cov_order in densmaps:
        vec = densmaps[cov_order]
        dens_cov = {int(i): int(v) for i, v in enumerate(vec) if v > 0}

    from dask.distributed import wait

    for depth in range(1, cfg.algorithm.level_limit + 1):
        depth_t0 = time.time()

        with diag_ctx(f"dask_depth_{depth:02d}"):
            algo = cfg.algorithm
            mode = (getattr(algo, "density_mode", "constant") or "constant").lower()
            use_total_profile = getattr(algo, "targets_total_initial", None) is not None

            n_cov: Optional[int] = None
            if use_total_profile or depth in Tmap:
                try:
                    n_cov = int(remainder_ddf["__icov__"].dropna().nunique().compute())
                except Exception:
                    n_cov = 0

            if use_total_profile:
                k_desired = _k_desired_from_total_profile(depth, algo, n_cov)
                log_fn(
                    f"[DEPTH {depth}] start (density_mode={mode}, "
                    f"targets_total_initial={float(algo.targets_total_initial):.4f}, "
                    f"n_cov={n_cov}, k_desired_from_total={k_desired:.4f})",
                    always=True,
                )
            else:
                k_desired = _k_desired_from_density_profile(depth, algo)
                log_fn(
                    f"[DEPTH {depth}] start (density_mode={mode}, k_desired={k_desired:.4f})",
                    always=True,
                )

            if algo.k_per_cov_per_level and depth in algo.k_per_cov_per_level:
                k_desired = float(algo.k_per_cov_per_level[depth])

            k_desired = max(0.0, float(k_desired))
            _log_depth_stats(log_fn, depth, "start", counts=densmaps[depth])

            if depth in Tmap:
                T_L = float(Tmap[depth])
                if n_cov is None:
                    try:
                        n_cov = int(remainder_ddf["__icov__"].dropna().nunique().compute())
                    except Exception:
                        n_cov = 0

                if n_cov > 0 and T_L > 0.0:
                    cap_per_cov = T_L / float(n_cov)
                    if cap_per_cov < k_desired:
                        log_fn(
                            f"[DEPTH {depth}] applying total cap: T_L={int(T_L)}, "
                            f"N_cov={n_cov} → cap_per_cov={cap_per_cov:.4f} "
                            f"(before k_desired={k_desired:.4f})",
                            always=True,
                        )
                        k_desired = cap_per_cov
                else:
                    log_fn(
                        f"[DEPTH {depth}] cannot apply total cap: " f"T_L={T_L}, N_cov={n_cov}",
                        always=True,
                    )

            if k_desired <= 0.0:
                log_fn(
                    f"[DEPTH {depth}] k_desired <= 0 → skipping this depth",
                    always=True,
                )
                continue

            bias_mode = (getattr(algo, "density_bias_mode", "none") or "none").lower()
            use_bias = bias_mode in ("proportional", "inverse") and bool(dens_cov)

            k_per_cov_for_selection: Any
            k_per_cov_desired_map: Optional[Dict[int, float]] = None

            if use_bias:
                alpha = float(getattr(algo, "density_bias_exponent", 1.0))
                alpha = abs(alpha)
                eps = 1e-6

                w_raw: Dict[int, float] = {}
                for icov, cnt in dens_cov.items():
                    val = float(cnt) + eps
                    w = val**alpha if bias_mode == "proportional" else val ** (-alpha)
                    w_raw[int(icov)] = w

                if not w_raw:
                    use_bias = False
                    k_per_cov_for_selection = int(max(1, math.ceil(k_desired)))
                else:
                    w_vals = list(w_raw.values())
                    mean_w = float(np.mean(w_vals))
                    if mean_w <= 0.0 or not np.isfinite(mean_w):
                        use_bias = False
                        k_per_cov_for_selection = int(max(1, math.ceil(k_desired)))
                    else:
                        k_per_cov_desired: Dict[int, float] = {}
                        for icov, w in w_raw.items():
                            w_norm = w / mean_w
                            k_c = max(0.0, k_desired * w_norm)
                            k_per_cov_desired[int(icov)] = k_c

                        k_per_cov_int: Dict[int, int] = {}
                        for icov, k_c in k_per_cov_desired.items():
                            if k_c <= 0.0:
                                continue
                            k_per_cov_int[int(icov)] = max(1, int(math.ceil(k_c)))

                        if not k_per_cov_int:
                            use_bias = False
                            k_per_cov_for_selection = int(max(1, math.ceil(k_desired)))
                        else:
                            k_per_cov_for_selection = k_per_cov_int
                            k_per_cov_desired_map = k_per_cov_desired
                            log_fn(
                                f"[DEPTH {depth}] density bias active: "
                                f"mode={bias_mode}, exponent={alpha}, "
                                f"base_k_desired={k_desired:.4f}",
                                always=True,
                            )
            else:
                k_per_cov_for_selection = int(max(1, math.ceil(k_desired)))
                k_per_cov_desired_map = None

            needed_cols = list(remainder_ddf.columns)
            if score_col not in needed_cols:
                needed_cols.append(score_col)
            if "__icov__" not in needed_cols:
                needed_cols.append("__icov__")
            sel_ddf = remainder_ddf[needed_cols]

            meta_cand = _get_meta_df(sel_ddf)
            cand_ddf = sel_ddf.map_partitions(
                _candidates_by_coverage_partition,
                score_col=score_col,
                order_desc=cfg.algorithm.order_desc,
                k_per_cov=k_per_cov_for_selection,
                tie_buffer=int(cfg.algorithm.tie_buffer),
                ra_col=ra_col,
                dec_col=dec_col,
                meta=meta_cand,
            )

            target_parts = cfg.cluster.n_workers * cfg.cluster.threads_per_worker * 2

            if (not is_hats) and hasattr(cand_ddf, "shuffle"):
                cand_ddf = cand_ddf.shuffle(
                    "__icov__",
                    npartitions=max(target_parts, sel_ddf.npartitions),
                )
            elif not is_hats and hasattr(cand_ddf, "_ddf") and hasattr(cand_ddf._ddf, "shuffle"):
                base_ddf = cand_ddf._ddf  # type: ignore[attr-defined]
                cand_ddf = base_ddf.shuffle(
                    "__icov__",
                    npartitions=max(target_parts, sel_ddf.npartitions),
                )
            else:
                log_fn(
                    f"[DEPTH {depth}] HATS / LSDB path or no shuffle available "
                    f"→ keeping native partitioning for __icov__",
                    always=True,
                )

            if avoid_computes:
                selected_ddf = _reduce_coverage_exact_dask(
                    cand_ddf,
                    score_col=score_col,
                    order_desc=cfg.algorithm.order_desc,
                    k_per_cov=k_per_cov_for_selection,
                    ra_col=ra_col,
                    dec_col=dec_col,
                )

                selected_pdf = selected_ddf.compute()
                _log_depth_stats(
                    log_fn,
                    depth,
                    "selected_before_fractional",
                    selected_len=len(selected_pdf),
                )
            else:
                cand_pdf = cand_ddf.compute()
                _log_depth_stats(
                    log_fn,
                    depth,
                    "candidates",
                    candidates_len=len(cand_pdf),
                )

                selected_pdf = _reduce_coverage_exact(
                    cand_pdf,
                    score_col=score_col,
                    order_desc=cfg.algorithm.order_desc,
                    k_per_cov=k_per_cov_for_selection,
                    ra_col=ra_col,
                    dec_col=dec_col,
                )
                _log_depth_stats(
                    log_fn,
                    depth,
                    "selected_before_fractional",
                    selected_len=len(selected_pdf),
                )

            selected_pdf, _ = apply_fractional_k_per_cov(
                selected_pdf,
                k_desired=k_desired,
                score_col=score_col,
                order_desc=cfg.algorithm.order_desc,
                mode=getattr(cfg.algorithm, "fractional_mode", "random"),
                mode_logic=getattr(
                    cfg.algorithm,
                    "fractional_mode_logic",
                    "auto",
                ),
                ra_col=ra_col,
                dec_col=dec_col,
                k_per_cov_desired_map=k_per_cov_desired_map,
            )
            _log_depth_stats(
                log_fn,
                depth,
                "selected",
                selected_len=len(selected_pdf),
            )

            if len(selected_pdf) > 0:
                ra_vals = pd.to_numeric(
                    selected_pdf[ra_col],
                    errors="coerce",
                ).to_numpy()
                dec_vals = pd.to_numeric(
                    selected_pdf[dec_col],
                    errors="coerce",
                ).to_numpy()

                theta = np.deg2rad(90.0 - dec_vals)
                phi = np.deg2rad(ra_vals % 360.0)

                NSIDE_L = 1 << depth
                ipixL = hp.ang2pix(
                    NSIDE_L,
                    theta,
                    phi,
                    nest=True,
                ).astype(np.int64)
                selected_pdf["__ipix__"] = ipixL

            header_line = build_header_line_from_keep(keep_cols)
            counts = densmaps[depth]
            allsky_needed = depth in (1, 2)

            written_per_ipix, _ = write_tiles_with_allsky(
                out_dir=out_dir,
                depth=depth,
                header_line=header_line,
                ra_col=ra_col,
                dec_col=dec_col,
                counts=counts,
                selected=selected_pdf,
                order_desc=cfg.algorithm.order_desc,
                allsky_needed=allsky_needed,
                log_fn=log_fn,
            )
            _log_depth_stats(
                log_fn,
                depth,
                "written",
                counts=densmaps[depth],
                written=written_per_ipix,
            )

            thr_cov = build_cov_thresholds(
                selected_pdf,
                score_col=score_col,
                order_desc=cfg.algorithm.order_desc,
            )
            if len(thr_cov) == 0:
                log_fn(
                    f"[INFO] Depth {depth}: nothing selected; " "stopping selection loop.",
                    always=True,
                )
                break

            remainder_meta = _get_meta_df(remainder_ddf)

            remainder_ddf = remainder_ddf.map_partitions(
                filter_remainder_by_coverage_partition,
                score_expr=score_col,
                order_desc=cfg.algorithm.order_desc,
                thr_cov=thr_cov,
                ra_col=ra_col,
                dec_col=dec_col,
                meta=remainder_meta,
            )

            if persist_ddfs:
                remainder_ddf = remainder_ddf.persist()
                wait(remainder_ddf)

        log_fn(
            f"[DEPTH {depth}] done in {_fmt_dur(time.time() - depth_t0)}",
            always=True,
        )
