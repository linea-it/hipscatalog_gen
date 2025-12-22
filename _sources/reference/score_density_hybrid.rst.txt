score_density_hybrid
====================

Hybrid density + score selection mode.

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
