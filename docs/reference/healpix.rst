Healpix
=======

Helpers for HEALPix indexing and density maps.

Quick example::

   from hipscatalog_gen.healpix.densmap import densmap_for_depth
   dens = densmap_for_depth(ddf, ra_col="RA", dec_col="DEC", depth=4)

.. autosummary::
   :toctree: generated/healpix
   :nosignatures:

   hipscatalog_gen.healpix.densmap.ipix_for_depth
   hipscatalog_gen.healpix.densmap.densmap_for_depth_delayed
   hipscatalog_gen.healpix.densmap.densmap_for_depth
