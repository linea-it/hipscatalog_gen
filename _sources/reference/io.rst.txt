I/O
===

Input loaders and HiPS writer utilities.

Typical flow::

   from hipscatalog_gen.io import _build_input_ddf, write_properties
   ddf, ra, dec, keep_cols = _build_input_ddf(paths, cfg)[:4]
   write_properties(out_dir, cfg.output, cfg.algorithm.level_limit, n_src=len(ddf))

.. autosummary::
   :toctree: generated/io
   :nosignatures:

   hipscatalog_gen.io.input._build_input_ddf
   hipscatalog_gen.io.input.compute_column_report_sample
   hipscatalog_gen.io.input.compute_column_report_global
   hipscatalog_gen.io.output.finalize_write_tiles
   hipscatalog_gen.io.output.build_header_line_from_keep
   hipscatalog_gen.io.output.write_properties
   hipscatalog_gen.io.output.write_arguments
   hipscatalog_gen.io.output.write_metadata_xml
   hipscatalog_gen.io.output.write_moc
   hipscatalog_gen.io.output.write_densmap_fits
