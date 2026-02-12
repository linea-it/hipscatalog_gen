"""Slice selections by value or score with HEALPix-aware ordering."""

from __future__ import annotations

import heapq
import os
import resource
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

import healpy as hp
import numpy as np
import pandas as pd

from ..io.output import build_header_line_from_keep, finalize_write_tiles
from ..pipeline.common import write_tiles_with_allsky
from ..utils import _fmt_dur, _get_meta_df, _log_depth_stats
from .common import add_ipix_column
from .levels import assign_level_edges
from .score import compute_score_histogram_ddf

try:  # pragma: no cover - optional parquet backend detail
    import pyarrow.parquet as _pq
except Exception:  # pragma: no cover - fallback path covered by runtime checks
    _pq = None

__all__ = ["select_by_value_slices", "select_by_score_slices"]


@dataclass
class _BucketWriteStats:
    """Aggregated stats for one bucket write task."""

    selected_len: int = 0
    tiles_written: int = 0
    rows_written: int = 0
    files_in: int = 0
    files_out: int = 0
    rounds: int = 0
    compacted: bool = False


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    """Read integer env var with fallback and lower bound enforcement."""
    raw = os.environ.get(name)
    if raw is None:
        return max(minimum, int(default))
    try:
        return max(minimum, int(raw))
    except Exception:
        return max(minimum, int(default))


def _detected_cpu_count() -> int:
    """Best-effort CPU count for the current process affinity."""
    try:
        affinity = os.sched_getaffinity(0)
        n_aff = int(len(affinity))
        if n_aff > 0:
            return n_aff
    except (AttributeError, OSError):
        pass
    n_cpu = os.cpu_count()
    if n_cpu is None or int(n_cpu) <= 0:
        return 1
    return int(n_cpu)


def _resolve_bucket_workers(n_buckets: int) -> tuple[int, int, bool]:
    """Resolve ThreadPool workers from env/defaults and current CPU availability."""
    n_bk = max(1, int(n_buckets))
    detected = _detected_cpu_count()
    raw_env = os.environ.get("HIPSCATALOG_STREAM_BUCKET_WORKERS")

    if raw_env is None:
        requested = max(1, min(8, detected))
        from_env = False
    else:
        try:
            requested = max(1, int(raw_env))
        except Exception:
            requested = max(1, min(8, detected))
        from_env = True

    workers = max(1, min(requested, n_bk))
    return workers, detected, from_env


def _resolve_local_scratch_base(out_dir: Path) -> Path:
    """Choose a local scratch base path for compaction intermediates."""
    candidates: list[Path] = []
    for env_name in ("HIPSCATALOG_STREAM_LOCAL_TMPDIR", "SLURM_TMPDIR", "TMPDIR"):
        raw = os.environ.get(env_name)
        if raw:
            candidates.append(Path(raw))
    candidates.append(Path(tempfile.gettempdir()))
    candidates.append(out_dir)

    for base in candidates:
        try:
            base.mkdir(parents=True, exist_ok=True)
            return base
        except OSError:
            pass
    return out_dir


def _should_compact_bucket(
    depth: int,
    files_in: int,
    *,
    mode: str,
    min_depth: int,
    min_files: int,
) -> bool:
    """Decide whether a bucket should run local compaction."""
    if files_in <= 1:
        return False
    if mode == "on":
        return True
    if mode == "off":
        return False
    # auto mode
    return (depth >= min_depth) or (files_in >= min_files)


def _merge_files_to_parquet(
    input_files: list[Path],
    output_file: Path,
    *,
    sort_cols: list[str],
    ascending: list[bool],
) -> None:
    """Merge many parquet fragments into one sorted parquet file."""
    merged = pd.concat((pd.read_parquet(fp) for fp in input_files), ignore_index=True)
    if len(merged) > 0:
        merged = merged.sort_values(
            ["__ipix__", *sort_cols],
            ascending=[True, *ascending],
            kind="mergesort",
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_file, index=False)


def _compact_bucket_fragments(
    frag_files: list[Path],
    *,
    local_bucket_dir: Path,
    sort_cols: list[str],
    ascending: list[bool],
    chunk_size: int,
    target_files: int,
) -> tuple[list[Path], int]:
    """Compact bucket fragments in rounds into local scratch."""
    current_files = list(frag_files)
    rounds = 0
    chunk = max(2, int(chunk_size))
    target = max(1, int(target_files))

    while len(current_files) > target:
        rounds += 1
        round_dir = local_bucket_dir / f"round_{rounds:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        next_files: list[Path] = []
        for start in range(0, len(current_files), chunk):
            batch = current_files[start : start + chunk]
            out_fp = round_dir / f"compact_{start // chunk:05d}.parquet"
            _merge_files_to_parquet(
                batch,
                out_fp,
                sort_cols=sort_cols,
                ascending=ascending,
            )
            next_files.append(out_fp)

        current_files = next_files

    return current_files, rounds


@dataclass(frozen=True)
class _ReverseSortValue:
    """Reverse-order wrapper for descending comparisons in heap keys."""

    value: Any

    def __lt__(self, other: _ReverseSortValue) -> bool:
        return bool(other.value < self.value)


def _sort_component(value: Any, *, ascending: bool) -> tuple[int, Any]:
    """Build a heap-sortable component matching pandas sort semantics."""
    if pd.isna(value):
        # Keep NaNs at the end regardless of ascending/descending.
        return (1, 0)
    if ascending:
        return (0, value)
    try:
        return (0, -value)
    except Exception:
        return (0, _ReverseSortValue(value))


def _row_sort_key(
    row: tuple[Any, ...], *, key_indices: list[int], key_ascending: list[bool]
) -> tuple[Any, ...]:
    """Compute global merge key for one streamed parquet row."""
    return tuple(
        _sort_component(row[idx], ascending=asc) for idx, asc in zip(key_indices, key_ascending, strict=True)
    )


def _iter_parquet_rows(file_path: Path, *, columns: list[str]) -> Iterator[tuple[Any, ...]]:
    """Yield row tuples from one parquet file in streaming batches."""
    if _pq is None:
        pdf = pd.read_parquet(file_path, columns=columns)
        yield from pdf.itertuples(index=False, name=None)
        return

    pf = _pq.ParquetFile(file_path)
    for batch in pf.iter_batches(columns=columns, batch_size=8192):
        pdf = batch.to_pandas()
        if len(pdf) == 0:
            continue
        yield from pdf.itertuples(index=False, name=None)


def _next_or_none(it: Iterator[tuple[Any, ...]]) -> tuple[Any, ...] | None:
    """Return next item from an iterator or None when exhausted."""
    try:
        return next(it)
    except StopIteration:
        return None


def _get_active_dask_client():
    """Return active distributed Client, or None when unavailable."""
    try:
        from dask.distributed import get_client

        return get_client()
    except Exception:
        return None


def _detect_worker_nthreads() -> int:
    """Best-effort detection of per-process worker task concurrency."""
    try:
        from dask.distributed import get_worker

        worker = get_worker()
        return max(1, int(getattr(worker, "nthreads", 1) or 1))
    except Exception:
        return 1


def _resolve_merge_max_open_files() -> int:
    """Resolve cap for simultaneous parquet files opened by one bucket task.

    Auto mode adapts to worker FD limits and task concurrency (nthreads).
    Manual override via HIPSCATALOG_STREAM_MERGE_MAX_OPEN_FILES is still honored
    but clipped to a safe upper bound derived from RLIMIT_NOFILE.
    """
    raw_override = os.environ.get("HIPSCATALOG_STREAM_MERGE_MAX_OPEN_FILES")
    override_val: int | None = None
    if raw_override is not None:
        try:
            override_val = max(2, int(raw_override))
        except Exception:
            override_val = None

    try:
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    except Exception:
        return override_val if override_val is not None else 32

    reserve_default = max(128, int(soft_limit // 4))
    reserve = _env_int("HIPSCATALOG_STREAM_MERGE_FD_RESERVE", reserve_default, minimum=8)
    available = max(8, int(soft_limit) - int(reserve))

    nthreads = _detect_worker_nthreads()

    # Per-task budget inside one worker process. Keep a conservative margin
    # because each task opens additional descriptors beyond parquet fragments.
    per_task_budget = max(8, available // nthreads)
    auto_cap = max(8, min(128, per_task_budget // 2))

    if override_val is not None:
        return int(max(2, min(override_val, per_task_budget)))
    return int(auto_cap)


def _process_bucket_dir(
    *,
    bucket_dir: Path,
    depth: int,
    out_dir: Path,
    header_line: str,
    counts: np.ndarray,
    sort_cols: list[str],
    ascending: list[bool],
    compaction_mode: str,
    compaction_min_depth: int,
    compaction_min_files: int,
    compaction_chunk_size: int,
    compaction_target_files: int,
) -> _BucketWriteStats:
    """Process one bucket (optional compaction + streaming k-way merge write)."""
    frag_files = sorted(bucket_dir.glob("part_*.parquet"))
    stats = _BucketWriteStats(files_in=len(frag_files), files_out=len(frag_files))
    if not frag_files:
        shutil.rmtree(bucket_dir, ignore_errors=True)
        return stats

    local_bucket_dir: Path | None = None

    try:
        merge_max_open_files = _resolve_merge_max_open_files()
        use_compaction = _should_compact_bucket(
            depth,
            stats.files_in,
            mode=compaction_mode,
            min_depth=compaction_min_depth,
            min_files=compaction_min_files,
        )
        working_files = frag_files
        force_fd_fan_in = len(working_files) > merge_max_open_files
        if use_compaction or force_fd_fan_in:
            local_scratch_base = _resolve_local_scratch_base(Path(out_dir))
            local_bucket_dir = local_scratch_base / (
                f"hipscatalog_stream_depth_{depth:02d}_{bucket_dir.name}_{uuid.uuid4().hex}"
            )
            local_bucket_dir.mkdir(parents=True, exist_ok=True)
            target_files = compaction_target_files
            chunk_size = compaction_chunk_size
            if force_fd_fan_in:
                # compaction_mode may be off, but we still need bounded fan-in
                # to avoid exhausting file descriptors on workers.
                target_files = int(max(1, merge_max_open_files))
                chunk_size = int(max(target_files, min(compaction_chunk_size, merge_max_open_files)))
            working_files, rounds = _compact_bucket_fragments(
                frag_files,
                local_bucket_dir=local_bucket_dir,
                sort_cols=sort_cols,
                ascending=ascending,
                chunk_size=chunk_size,
                target_files=target_files,
            )
            stats.rounds = int(rounds)
            stats.files_out = int(len(working_files))
            stats.compacted = stats.rounds > 0

        if not working_files:
            return stats

        if _pq is not None:
            schema_cols = [str(c) for c in _pq.ParquetFile(working_files[0]).schema_arrow.names]
        else:
            schema_cols = list(pd.read_parquet(working_files[0]).columns)

        required = ["__ipix__", *sort_cols]
        missing = [c for c in required if c not in schema_cols]
        if missing:
            raise KeyError(f"missing columns in bucket fragments: {missing!r}")

        header_cols = header_line.strip("\n").split("\t")
        internal = {"__ipix__", "__score__", "__icov__"}
        tile_cols = [c for c in header_cols if c not in internal and c in schema_cols]

        read_cols = list(dict.fromkeys(["__ipix__", *sort_cols, *tile_cols]))
        ipix_idx = read_cols.index("__ipix__")
        key_cols = ["__ipix__", *sort_cols]
        key_indices = [read_cols.index(c) for c in key_cols]
        key_ascending = [True, *ascending]
        tile_indices = [read_cols.index(c) for c in tile_cols]

        streams: list[Iterator[tuple[Any, ...]]] = []
        heap: list[tuple[tuple[Any, ...], int, tuple[Any, ...]]] = []
        for rank, fp in enumerate(working_files):
            row_it = iter(_iter_parquet_rows(fp, columns=read_cols))
            streams.append(row_it)
            first_row = _next_or_none(row_it)
            if first_row is None:
                continue
            heapq.heappush(
                heap,
                (
                    _row_sort_key(first_row, key_indices=key_indices, key_ascending=key_ascending),
                    rank,
                    first_row,
                ),
            )

        current_ipix: int | None = None
        current_rows: list[tuple[Any, ...]] = []

        def _flush_current_ipix() -> None:
            nonlocal current_rows, current_ipix
            if current_ipix is None or not current_rows:
                return
            ip = int(current_ipix)
            if ip < 0 or ip >= len(counts):
                current_rows = []
                return
            selected_ipix = pd.DataFrame(current_rows, columns=tile_cols)
            selected_ipix["__ipix__"] = np.int64(ip)
            written_per_ipix, _ = finalize_write_tiles(
                out_dir=out_dir,
                depth=depth,
                header_line=header_line,
                ra_col="",
                dec_col="",
                counts=counts,
                selected=selected_ipix,
                order_desc=False,
                allsky_collect=False,
            )
            if written_per_ipix:
                stats.tiles_written += int(len(written_per_ipix))
                stats.rows_written += int(sum(written_per_ipix.values()))
            current_rows = []

        while heap:
            _, rank, row = heapq.heappop(heap)
            ip = int(row[ipix_idx])

            if current_ipix is None:
                current_ipix = ip
            elif ip != current_ipix:
                _flush_current_ipix()
                current_ipix = ip

            stats.selected_len += 1
            if tile_indices:
                current_rows.append(tuple(row[idx] for idx in tile_indices))
            else:
                current_rows.append(tuple())

            nxt = _next_or_none(streams[rank])
            if nxt is not None:
                heapq.heappush(
                    heap,
                    (
                        _row_sort_key(nxt, key_indices=key_indices, key_ascending=key_ascending),
                        rank,
                        nxt,
                    ),
                )

        _flush_current_ipix()
        return stats
    finally:
        if local_bucket_dir is not None:
            shutil.rmtree(local_bucket_dir, ignore_errors=True)
        shutil.rmtree(bucket_dir, ignore_errors=True)


def _stream_write_depth_without_allsky(
    *,
    depth_ddf: Any,
    depth: int,
    value_col: str,
    order_desc: bool,
    tie_col: str | None,
    ra_col: str,
    dec_col: str,
    out_dir,
    header_line: str,
    counts: np.ndarray,
    log_fn,
) -> tuple[int, int, int]:
    """Write one depth without collecting all selected rows into driver memory.

    The function spills sorted partition fragments into bucketed temporary files
    and then merges per bucket before delegating final tile writing.
    """
    asc = not order_desc
    sort_cols = [value_col]
    ascending = [asc]
    if tie_col:
        sort_cols.append(tie_col)
        ascending.append(True)
    sort_cols.extend([ra_col, dec_col])
    ascending.extend([True, True])

    meta_depth = _get_meta_df(depth_ddf)
    if tie_col and tie_col not in meta_depth.columns:
        raise KeyError(f"tie_column '{tie_col}' not found in selected data.")

    meta_ipix = meta_depth.copy()
    meta_ipix["__ipix__"] = pd.Series([], dtype="int64")
    # Bucket by ipix to reduce small-file fan-out on distributed filesystems.
    # Keep bucket count bounded to control metadata overhead.
    n_buckets = max(16, min(128, int((len(counts) + 16383) // 16384)))
    meta_ipix["__bucket__"] = pd.Series([], dtype="int16")

    def _add_ipix_and_bucket(pdf: pd.DataFrame) -> pd.DataFrame:
        out = add_ipix_column(pdf, depth, ra_col, dec_col)
        if out.empty:
            out["__bucket__"] = pd.Series([], dtype="int16")
            return out
        out["__bucket__"] = np.mod(out["__ipix__"].to_numpy(dtype=np.int64), n_buckets).astype(np.int16)
        return out

    ddf_with_ipix = depth_ddf.map_partitions(_add_ipix_and_bucket, meta=meta_ipix)

    tmp_root = Path(out_dir) / f".tmp_stream_depth_{depth:02d}"
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    stats_meta = pd.DataFrame(
        {
            "rows": pd.Series([], dtype="int64"),
            "fragments": pd.Series([], dtype="int64"),
        }
    )

    def _spill_partition(pdf: pd.DataFrame, partition_info=None) -> pd.DataFrame:
        if pdf.empty:
            return pd.DataFrame({"rows": [0], "fragments": [0]}, dtype="int64")

        part_no = -1
        if isinstance(partition_info, dict) and "number" in partition_info:
            part_no = int(partition_info["number"])

        local_sort_cols = ["__bucket__", "__ipix__", *sort_cols]
        local_ascending = [True, True, *ascending]
        pdf_sorted = pdf.sort_values(local_sort_cols, ascending=local_ascending, kind="mergesort")

        n_frag = 0
        for bid, grp in pdf_sorted.groupby("__bucket__", sort=True):
            bucket = int(bid)
            bucket_dir = tmp_root / f"bucket_{bucket:04d}"
            bucket_dir.mkdir(parents=True, exist_ok=True)
            # LSDB may not provide partition_info.number; keep filenames unique to
            # avoid write collisions across concurrent tasks.
            frag_path = bucket_dir / f"part_{part_no:08d}_{uuid.uuid4().hex}.parquet"
            # LSDB partitions may be NestedFrame; cast to plain pandas before parquet IO
            # to keep a consistent writer signature across backends.
            pd.DataFrame(grp).reset_index(drop=True).to_parquet(frag_path, index=False)
            n_frag += 1

        return pd.DataFrame({"rows": [int(len(pdf_sorted))], "fragments": [int(n_frag)]}, dtype="int64")

    try:
        spill_stats = ddf_with_ipix.map_partitions(
            _spill_partition,
            partition_info=True,
            meta=stats_meta,
        ).compute()

        selected_len = int(spill_stats["rows"].sum()) if len(spill_stats) else 0
        if selected_len == 0:
            return 0, 0, 0
        bucket_dirs = sorted(
            [p for p in tmp_root.iterdir() if p.is_dir() and p.name.startswith("bucket_")],
            key=lambda p: int(p.name.split("_", 1)[1]),
        )
        if not bucket_dirs:
            return selected_len, 0, 0

        mode_env = str(os.environ.get("HIPSCATALOG_STREAM_COMPACTION_MODE", "off")).strip().lower()
        compaction_mode = "off"
        compaction_min_depth = _env_int("HIPSCATALOG_STREAM_COMPACTION_MIN_DEPTH", 8, minimum=0)
        compaction_min_files = _env_int("HIPSCATALOG_STREAM_COMPACTION_MIN_FILES", 4096, minimum=1)
        compaction_chunk_size = _env_int("HIPSCATALOG_STREAM_COMPACTION_CHUNK_SIZE", 128, minimum=2)
        compaction_target_files = _env_int("HIPSCATALOG_STREAM_COMPACTION_TARGET_FILES", 8, minimum=1)
        if compaction_chunk_size <= compaction_target_files:
            compaction_chunk_size = compaction_target_files + 1

        if mode_env != "off":
            log_fn(
                f"[stream] depth={depth} forcing compaction_mode='off' (env requested {mode_env!r}).",
                always=True,
                depth=depth,
            )

        stats_list: list[_BucketWriteStats] = []
        client = _get_active_dask_client()
        if client is None:
            raise RuntimeError(
                "No active dask.distributed Client found for streaming bucket merge. "
                "Start a Client (local cluster or SLURM-backed cluster) before running "
                f"streamed depth writes (depth={depth}, buckets={len(bucket_dirs)})."
            )

        worker_count = int(len(client.scheduler_info().get("workers", {})))
        log_fn(
            f"[stream] depth={depth} dask bucket submit: workers={worker_count} "
            f"buckets={len(bucket_dirs)} compaction_mode={compaction_mode}",
            always=True,
            depth=depth,
        )
        counts_payload: Any = counts
        try:
            counts_payload = client.scatter(counts, broadcast=True)
        except Exception:
            counts_payload = counts

        futures = [
            client.submit(
                _process_bucket_dir,
                bucket_dir=bucket_dir,
                depth=depth,
                out_dir=Path(out_dir),
                header_line=header_line,
                counts=counts_payload,
                sort_cols=sort_cols,
                ascending=ascending,
                compaction_mode=compaction_mode,
                compaction_min_depth=compaction_min_depth,
                compaction_min_files=compaction_min_files,
                compaction_chunk_size=compaction_chunk_size,
                compaction_target_files=compaction_target_files,
                pure=False,
            )
            for bucket_dir in bucket_dirs
        ]
        stats_list = list(client.gather(futures))

        tiles_written = int(sum(s.tiles_written for s in stats_list))
        rows_written = int(sum(s.rows_written for s in stats_list))
        files_in_total = int(sum(s.files_in for s in stats_list))
        files_out_total = int(sum(s.files_out for s in stats_list))
        compacted_buckets = int(sum(1 for s in stats_list if s.compacted))
        rounds_total = int(sum(s.rounds for s in stats_list))
        max_bucket_in = int(max((s.files_in for s in stats_list), default=0))

        if compacted_buckets > 0:
            reduction = 0.0
            if files_in_total > 0:
                reduction = 100.0 * (1.0 - (files_out_total / float(files_in_total)))
            log_fn(
                f"[stream-compaction] depth={depth} buckets={len(bucket_dirs)} compacted={compacted_buckets} "
                f"files_in={files_in_total} files_out={files_out_total} reduction={reduction:.1f}% "
                f"rounds_total={rounds_total} max_bucket_in={max_bucket_in}",
                always=True,
                depth=depth,
            )

        return selected_len, tiles_written, rows_written
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def select_by_value_slices(
    remainder_ddf: Any,
    densmaps: Dict[int, np.ndarray],
    depths_sel: Sequence[int],
    keep_cols: List[str],
    ra_col: str,
    dec_col: str,
    value_col: str,
    order_desc: bool,
    label: str,
    out_dir,
    diag_ctx,
    log_fn,
    *,
    level_edges: np.ndarray | None = None,
    tie_col: str | None = None,
    compute_hist_fn=None,
    value_min: float | None = None,
    value_max: float | None = None,
    hist_nbins: int | None = None,
    fixed_targets: Dict[int, float] | None = None,
    hist_diag_ctx_name: str | None = None,
    depth_diag_prefix: str | None = None,
) -> None:
    """Slice by per-depth value ranges and write tiles; expects value_col and densmaps populated."""
    if level_edges is None:
        if compute_hist_fn is None or hist_nbins is None or value_min is None or value_max is None:
            raise ValueError(
                f"{label}: missing histogram parameters (compute_hist_fn, hist_nbins, value_min, value_max)."
            )

        hist_ctx = hist_diag_ctx_name or f"dask_{label}_hist"
        with diag_ctx(hist_ctx):
            hist, edges_hist, n_tot = compute_hist_fn(
                remainder_ddf,
                value_col,
                value_min,
                value_max,
                hist_nbins,
            )

        if n_tot == 0:
            log_fn(
                f"[selection] {label}: no objects found in the range "
                f"[{value_min}, {value_max}] → nothing to select.",
                always=True,
            )
            return

        cdf_hist = hist.cumsum().astype("float64")
        if cdf_hist[-1] > 0:
            cdf_hist /= float(cdf_hist[-1])
        else:
            cdf_hist[:] = 0.0

        level_edges, _ = assign_level_edges(
            densmaps=densmaps,
            depths_sel=list(depths_sel),
            fixed_targets=fixed_targets or {},
            cdf_hist=cdf_hist,
            score_edges_hist=edges_hist,
            score_min=value_min,
            score_max=value_max,
            n_tot_score=float(n_tot),
            log_fn=log_fn,
            label=label,
        )
    else:
        level_edges = np.asarray(level_edges, dtype="float64")

    depths_list = list(depths_sel)
    log_fn(
        f"[selection] {label} mode: per-depth slices:\n"
        + "\n".join(
            f"  depth {d}: [{level_edges[i]:.6f}, {level_edges[i + 1]:.6f}"
            f"{')' if d != depths_list[-1] else ']'}"
            for i, d in enumerate(depths_list)
        ),
        always=True,
    )

    header_line = build_header_line_from_keep(keep_cols)
    depth_ctx = depth_diag_prefix or f"dask_{label}_depth"

    for i, depth in enumerate(depths_list):
        depth_t0 = time.time()
        v_lo = level_edges[i]
        v_hi = level_edges[i + 1]

        with diag_ctx(f"{depth_ctx}_{depth:02d}"):
            if depth != depths_list[-1]:
                val_mask = (remainder_ddf[value_col] >= v_lo) & (remainder_ddf[value_col] < v_hi)
            else:
                val_mask = (remainder_ddf[value_col] >= v_lo) & (remainder_ddf[value_col] <= v_hi)

            depth_ddf = remainder_ddf[val_mask]
            counts = densmaps[depth]
            allsky_needed = depth in (1, 2)
            if allsky_needed:
                selected_pdf = depth_ddf.compute()
                if tie_col and tie_col not in selected_pdf.columns:
                    raise KeyError(f"{label}: tie_column '{tie_col}' not found in selected data.")
                selected_len = int(len(selected_pdf))
                _log_depth_stats(
                    log_fn,
                    depth,
                    "selected",
                    counts=densmaps[depth],
                    selected_len=selected_len,
                )
                if selected_len == 0:
                    log_fn(
                        f"[DEPTH {depth}] {label}: no rows in slice [{v_lo:.6f}, {v_hi:.6f}] → skipping.",
                        always=True,
                        depth=depth,
                    )
                    log_fn(
                        f"[DEPTH {depth}] done in {_fmt_dur(time.time() - depth_t0)}",
                        always=True,
                        depth=depth,
                    )
                    continue

                ra_vals = pd.to_numeric(selected_pdf[ra_col], errors="coerce").to_numpy()
                dec_vals = pd.to_numeric(selected_pdf[dec_col], errors="coerce").to_numpy()

                theta = np.deg2rad(90.0 - dec_vals)
                phi = np.deg2rad(ra_vals % 360.0)

                nside_l = 1 << depth
                ipix_l = hp.ang2pix(nside_l, theta, phi, nest=True).astype(np.int64)
                selected_pdf["__ipix__"] = ipix_l

                sort_cols = [value_col]
                ascending = [not order_desc]
                if tie_col and tie_col in selected_pdf.columns:
                    sort_cols.append(tie_col)
                    ascending.append(True)
                if ra_col in selected_pdf.columns:
                    sort_cols.append(ra_col)
                    ascending.append(True)
                if dec_col in selected_pdf.columns:
                    sort_cols.append(dec_col)
                    ascending.append(True)
                selected_pdf = selected_pdf.sort_values(sort_cols, ascending=ascending, kind="mergesort")

                written_per_ipix, _ = write_tiles_with_allsky(
                    out_dir=out_dir,
                    depth=depth,
                    header_line=header_line,
                    ra_col=ra_col,
                    dec_col=dec_col,
                    counts=counts,
                    selected=selected_pdf,
                    order_desc=order_desc,
                    allsky_needed=allsky_needed,
                    log_fn=log_fn,
                )
                _log_depth_stats(log_fn, depth, "written", counts=densmaps[depth], written=written_per_ipix)
            else:
                selected_len, tiles_written, rows_written = _stream_write_depth_without_allsky(
                    depth_ddf=depth_ddf,
                    depth=depth,
                    value_col=value_col,
                    order_desc=order_desc,
                    tie_col=tie_col,
                    ra_col=ra_col,
                    dec_col=dec_col,
                    out_dir=out_dir,
                    header_line=header_line,
                    counts=counts,
                    log_fn=log_fn,
                )
                _log_depth_stats(
                    log_fn,
                    depth,
                    "selected",
                    counts=densmaps[depth],
                    selected_len=selected_len,
                )
                if selected_len == 0:
                    log_fn(
                        f"[DEPTH {depth}] {label}: no rows in slice [{v_lo:.6f}, {v_hi:.6f}] → skipping.",
                        always=True,
                        depth=depth,
                    )
                    log_fn(
                        f"[DEPTH {depth}] done in {_fmt_dur(time.time() - depth_t0)}",
                        always=True,
                        depth=depth,
                    )
                    continue
                _log_depth_stats(
                    log_fn,
                    depth,
                    "written",
                    counts=densmaps[depth],
                    tiles_written=tiles_written,
                    rows_written=rows_written,
                )

        log_fn(
            f"[DEPTH {depth}] done in {_fmt_dur(time.time() - depth_t0)}",
            always=True,
            depth=depth,
        )


def select_by_score_slices(
    remainder_ddf: Any,
    densmaps: Dict[int, np.ndarray],
    depths_sel: Sequence[int],
    keep_cols: List[str],
    ra_col: str,
    dec_col: str,
    score_col: str,
    score_min: float,
    score_max: float,
    hist_nbins: int,
    out_dir,
    diag_ctx,
    log_fn,
    label: str,
    order_desc: bool,
    fixed_targets: Dict[int, float] | None = None,
    hist_diag_ctx_name: str | None = None,
    depth_diag_prefix: str | None = None,
    tie_col: str | None = None,
) -> None:
    """Score-specialized wrapper around select_by_value_slices."""
    select_by_value_slices(
        remainder_ddf=remainder_ddf,
        densmaps=densmaps,
        depths_sel=depths_sel,
        keep_cols=keep_cols,
        ra_col=ra_col,
        dec_col=dec_col,
        value_col=score_col,
        order_desc=order_desc,
        label=label,
        out_dir=out_dir,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        compute_hist_fn=compute_score_histogram_ddf,
        value_min=score_min,
        value_max=score_max,
        hist_nbins=hist_nbins,
        fixed_targets=fixed_targets,
        hist_diag_ctx_name=hist_diag_ctx_name,
        depth_diag_prefix=depth_diag_prefix,
        tie_col=tie_col,
    )
