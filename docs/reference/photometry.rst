Photometry
==========

The optional top-level ``photometry`` block derives magnitudes lazily from
physical flux columns. It can process multiple bands in one vectorized pass per
partition, optionally propagate flux errors, and optionally apply a row-wise
E(B-V) correction.

Equations
---------

For a finite, positive flux :math:`F`, the derived magnitude is

.. math::

   m = Z - 2.5 \log_{10}(F),

where :math:`Z` is ``photometry.mag_offset``. When a finite, non-negative flux
error :math:`\sigma_F` is configured, its propagated magnitude error is

.. math::

   \sigma_m = \frac{2.5}{\ln(10)} \frac{\sigma_F}{F}.

When dereddening is enabled, the corrected magnitude for band :math:`b` is

.. math::

   m_{b,\mathrm{corrected}} = m_b - A_b E(B-V).

The corrected magnitude error currently equals the flux-derived magnitude
error. Uncertainty in E(B-V) and in :math:`A_b` is not propagated.

Configuration
-------------

The top-level block described here is distinct from
``algorithm.mag_global.flux_column``. That mode-specific option is a
single-column convenience for the internal ``mag_global`` selection value. Use
top-level ``photometry`` when derived columns must be written, multiple bands
must be transformed, dereddening is required, or a score mode uses the result.

Use ``expand`` when input and output columns follow regular templates. The
following names and coefficients are illustrative only::

   photometry:
     mag_offset: 25.0
     invalid_value: nan
     replace_fluxes: true
     expand:
       bands: [g, r]
       flux_template: "{band}_flux"
       flux_error_template: "{band}_flux_error"
       mag_template: "{band}_mag"
       mag_error_template: "{band}_mag_error"
     dereddening:
       enabled: true
       ebv_column: extinction_ebv
       keep_observed_magnitudes: false
       output_suffix: "_corrected"
       coefficients:
         g: 3.0
         r: 2.0

Use ``measurements`` instead of ``expand`` for irregular schemas::

   photometry:
     mag_offset: 25.0
     measurements:
       - band: r
         flux_column: flux_r
         flux_error_column: flux_error_r
         mag_column: magnitude_r
         mag_error_column: magnitude_error_r

``measurements`` and ``expand`` are mutually exclusive. A flux error column and
its magnitude error output must be configured together. Dereddening requires an
E(B-V) column and one finite coefficient for every configured band.

Invalid values and row counts
-----------------------------

The transformation does not filter rows. Invalid numerical inputs produce
``NaN`` outputs:

- flux must be finite and greater than zero;
- flux error must be finite and non-negative;
- E(B-V) must be finite for corrected outputs.

``invalid_value`` currently accepts only ``nan``. To retain objects whose
derived selection value is invalid, configure the active selection block with
``adaptive_range: complete`` and ``keep_invalid_values: true``. The selection
maps invalid values to a sentinel in the last slice. At least one finite
selection value is required to resolve the range.

Output columns and projection
-----------------------------

With ``replace_fluxes: true``, source flux and flux-error columns remain
physical read dependencies but are omitted from the output schema. With
``replace_fluxes: false``, they may be retained according to ``columns.keep``.

When dereddening is enabled, ``keep_observed_magnitudes`` controls whether the
uncorrected magnitude columns are written alongside the corrected columns.
Every configured photometry output is retained when ``columns.keep`` is empty;
for a non-empty list, required coordinates, selection dependencies, and
photometry outputs are added automatically.

Only required physical Parquet columns are projected into the read graph. HATS
catalogs retain LSDB-native projection and partition operations whenever the
catalog supports them; conversion to a Dask DataFrame is used only as a
compatibility fallback for the transformation operation.

Derived columns in selection modes
----------------------------------

Photometry runs before selection normalization. A derived magnitude or error
column can therefore be used by ``mag_global`` or directly in score expressions
for ``score_global`` and ``score_density_hybrid``.

API
---

.. autosummary::
   :toctree: generated/photometry
   :nosignatures:

   hipscatalog_gen.config.PhotometryCfg
   hipscatalog_gen.config.PhotometryMeasurementCfg
   hipscatalog_gen.config.DereddeningCfg
   hipscatalog_gen.photometry.PhotometryPlan
   hipscatalog_gen.photometry.build_photometry_plan
   hipscatalog_gen.photometry.apply_photometry
   hipscatalog_gen.photometry.working_photometry_columns
