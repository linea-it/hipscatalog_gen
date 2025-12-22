mag_global
==========

Magnitude-complete selection pipeline.

Usage snippet::

   from hipscatalog_gen.mag_global.pipeline import normalize_mag_global, run_mag_global_selection
   ddf_mag, params = normalize_mag_global(ddf, cfg, diag_ctx, log_fn)
   run_mag_global_selection(ddf_mag, densmaps, keep_cols, ra_col, dec_col, cfg, out_dir, diag_ctx, log_fn, params=params)

.. autosummary::
   :toctree: generated/mag_global
   :nosignatures:

   hipscatalog_gen.mag_global.pipeline.normalize_mag_global
   hipscatalog_gen.mag_global.pipeline.prepare_mag_global
   hipscatalog_gen.mag_global.pipeline.run_mag_global_selection
