score_global
============

Score-driven selection across all depths.

Usage snippet::

   from hipscatalog_gen.score_global.pipeline import normalize_score_global, run_score_global_selection
   ddf_score, params = normalize_score_global(ddf, cfg, diag_ctx, log_fn)
   run_score_global_selection(ddf_score, densmaps, keep_cols, ra_col, dec_col, cfg, out_dir, diag_ctx, log_fn, params=params)

.. autosummary::
   :toctree: generated/score_global
   :nosignatures:

   hipscatalog_gen.score_global.pipeline.normalize_score_global
   hipscatalog_gen.score_global.pipeline.prepare_score_global
   hipscatalog_gen.score_global.pipeline.run_score_global_selection
