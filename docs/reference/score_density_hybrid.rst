score_density_hybrid
====================

Hybrid density + score selection mode.

``density_up_to_depth`` defaults to 4 and must satisfy
``1 <= density_up_to_depth <= level_limit``. ``level_limit`` itself must be at
least 1, so configurations below level 4 must set ``density_up_to_depth``
explicitly to a compatible value.

When ``density_up_to_depth < level_limit``, deeper score slices consume the
rows left by the density stage. When both values are equal, the deepest density
level acts as a terminal spillover and consumes every remaining row instead of
dropping rows when individual pixel quotas cannot be filled.

With ``adaptive_range: complete`` and ``keep_invalid_values: true``, all rows
are retained when no explicit score bounds exclude finite values and the
coordinates and writes are valid. Invalid score values are mapped to the
terminal sentinel.

Usage snippet::

   from hipscatalog_gen.score_density_hybrid.pipeline import normalize_score_density_hybrid, run_score_density_hybrid_selection
   ddf_score, params = normalize_score_density_hybrid(ddf, cfg, diag_ctx, log_fn)
   run_score_density_hybrid_selection(ddf_score, densmaps, keep_cols, ra_col, dec_col, cfg, out_dir, diag_ctx, log_fn, params=params)

.. autosummary::
   :toctree: generated/score_density_hybrid
   :nosignatures:

   hipscatalog_gen.score_density_hybrid.pipeline.normalize_score_density_hybrid
   hipscatalog_gen.score_density_hybrid.pipeline.prepare_score_density_hybrid
   hipscatalog_gen.score_density_hybrid.pipeline.run_score_density_hybrid_selection
