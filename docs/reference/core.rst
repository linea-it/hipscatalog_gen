Core
====

Entry points and configuration helpers exposed at the package root.

Typical usage::

   from hipscatalog_gen import load_config, run_pipeline
   cfg = load_config("config.yaml")
   run_pipeline(cfg)

.. autosummary::
   :toctree: generated/core
   :nosignatures:

   hipscatalog_gen.Config
   hipscatalog_gen.load_config
   hipscatalog_gen.run_pipeline
   hipscatalog_gen.config.display_available_configs
   hipscatalog_gen.config.load_config_from_dict
   hipscatalog_gen.cli.main
