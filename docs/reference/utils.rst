Utils
=====

Shared helpers exported from ``hipscatalog_gen.utils``. Most functions are
prefixed with underscores but are re-exported via ``__all__`` for pipeline use.

Example: resolve column names and validate RA/DEC::

   from hipscatalog_gen.utils import _resolve_col_name, _validate_and_normalize_radec
   ra = _resolve_col_name("RA", ddf, header=True)
   dec = _resolve_col_name("DEC", ddf, header=True)
   ddf = _validate_and_normalize_radec(ddf, ra, dec, log_fn)

.. autosummary::
   :toctree: generated/utils
   :nosignatures:

   hipscatalog_gen.utils._mkdirs
   hipscatalog_gen.utils._write_text
   hipscatalog_gen.utils._detect_hats_catalog_root
   hipscatalog_gen.utils._now_str
   hipscatalog_gen.utils._ts
   hipscatalog_gen.utils._fmt_dur
   hipscatalog_gen.utils._log_depth_stats
   hipscatalog_gen.utils._get_dask_base
   hipscatalog_gen.utils._score_deps
   hipscatalog_gen.utils._resolve_col_name
   hipscatalog_gen.utils._get_meta_df
   hipscatalog_gen.utils._validate_and_normalize_radec
