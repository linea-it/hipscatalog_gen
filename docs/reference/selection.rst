Selection
=========

Score/value slicing helpers and HEALPix binning used across modes.

Example: compute a histogram and resolve a range for scores::

   from hipscatalog_gen.selection import compute_score_histogram_ddf, resolve_value_range
   hist, edges, total = compute_score_histogram_ddf(ddf, "__score__", -5, 5, 128)
   lo, hi = resolve_value_range(ddf, "__score__", "complete", None, None, 128, compute_score_histogram_ddf, diag_ctx, log_fn, "scores")

.. autosummary::
   :toctree: generated/selection
   :nosignatures:

   hipscatalog_gen.selection.assign_level_edges
   hipscatalog_gen.selection.targets_per_tile
   hipscatalog_gen.selection.reduce_topk_by_group_dask
   hipscatalog_gen.selection.add_ipix_column
   hipscatalog_gen.selection.add_score_column
   hipscatalog_gen.selection.compute_histogram_ddf
   hipscatalog_gen.selection.compute_score_histogram_ddf
   hipscatalog_gen.selection.resolve_value_range
   hipscatalog_gen.selection.select_by_score_slices
   hipscatalog_gen.selection.select_by_value_slices
