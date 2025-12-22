Cluster
=======

Dask cluster orchestration and diagnostics contexts.

Quick usage::

   from hipscatalog_gen.cluster import setup_cluster, shutdown_cluster
   runtime, diag_ctx = setup_cluster(cfg.cluster, report_dir, log_fn)
   shutdown_cluster(runtime)

.. autosummary::
   :toctree: generated/cluster
   :nosignatures:

   hipscatalog_gen.cluster.runtime.ClusterRuntime
   hipscatalog_gen.cluster.runtime.setup_cluster
   hipscatalog_gen.cluster.runtime.shutdown_cluster
