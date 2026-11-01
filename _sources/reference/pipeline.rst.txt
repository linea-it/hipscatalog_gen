Pipeline
========

Execution flow, mode registry, and shared pipeline utilities.

Use these helpers to wire configs, run the pipeline, or extend selection modes::

   from hipscatalog_gen.pipeline import get_selection_mode, run_pipeline
   mode = get_selection_mode("mag_global")
   run_pipeline(cfg)

.. autosummary::
   :toctree: generated/pipeline
   :nosignatures:

   hipscatalog_gen.pipeline.main.run_pipeline
   hipscatalog_gen.pipeline.modes.SelectionMode
   hipscatalog_gen.pipeline.modes.get_selection_mode
   hipscatalog_gen.pipeline.structure.PipelineStage
   hipscatalog_gen.pipeline.structure.PipelineContext
   hipscatalog_gen.pipeline.structure.run_stages
   hipscatalog_gen.pipeline.validation.validate_common_cfg
   hipscatalog_gen.pipeline.validation.validate_mag_global_cfg
   hipscatalog_gen.pipeline.validation.validate_score_global_cfg
   hipscatalog_gen.pipeline.validation.validate_score_density_hybrid_cfg
   hipscatalog_gen.pipeline.common.log_prologue
   hipscatalog_gen.pipeline.common.log_epilogue
   hipscatalog_gen.pipeline.common.build_and_prepare_input
   hipscatalog_gen.pipeline.common.compute_and_write_densmaps
   hipscatalog_gen.pipeline.common.write_tiles_with_allsky
   hipscatalog_gen.pipeline.common.write_counts_summaries
   hipscatalog_gen.pipeline.logging_utils.setup_structured_logger
