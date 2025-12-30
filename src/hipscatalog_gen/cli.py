"""Command-line interface for running the HiPS catalog pipeline."""

from __future__ import annotations

import argparse
import sys
from typing import List

from .config import load_config

__all__ = ["main"]


def main(argv: List[str] | None = None) -> None:
    """Entry point for the command-line interface.

    Args:
        argv: Command-line arguments excluding the program name.
    """
    if argv is None:
        argv = sys.argv[1:]

    desc = (
        "HiPS Catalog Pipeline "
        "(Dask, Parquet, mag_global/score_global/score_density_hybrid selection). "
        "Use a YAML config file to control inputs, cluster, and algorithm options. "
        "Docs: https://linea-it.github.io/hipscatalog_gen/"
    )
    parser = argparse.ArgumentParser(description=desc)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        help="Path to the YAML configuration file.",
    )
    group.add_argument(
        "--list-modes",
        action="store_true",
        help="List available selection modes and exit.",
    )
    group.add_argument(
        "--check-config",
        metavar="CONFIG",
        help="Validate a YAML configuration file and exit without running the pipeline.",
    )
    group.add_argument(
        "--telemetry",
        metavar="FILE",
        help="Print a summary from an existing telemetry.json file and exit.",
    )

    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Also emit structured JSONL logs to process.jsonl (when running the pipeline).",
    )

    args = parser.parse_args(argv)

    if getattr(args, "list_modes", False):
        from .pipeline.modes import MODE_REGISTRY

        for name, entry in sorted(MODE_REGISTRY.items()):
            print(f"{name}: {entry.description}")
        return

    if getattr(args, "check_config", None):
        cfg = load_config(args.check_config)
        # Validation runs inside run_pipeline, but we surface success here.
        from .pipeline.validation import (
            validate_common_cfg,
            validate_mag_global_cfg,
            validate_score_density_hybrid_cfg,
            validate_score_global_cfg,
        )

        validate_common_cfg(cfg)
        mode = (getattr(cfg.algorithm, "selection_mode", "") or "").lower()
        if mode == "mag_global":
            validate_mag_global_cfg(cfg)
        elif mode == "score_global":
            validate_score_global_cfg(cfg)
        elif mode == "score_density_hybrid":
            validate_score_density_hybrid_cfg(cfg)
        else:
            raise ValueError(f"Unsupported selection_mode '{mode}' during config check.")
        print("Configuration is valid.")
        return

    if getattr(args, "telemetry", None):
        import json
        from pathlib import Path

        tfile = Path(args.telemetry)
        data = json.loads(tfile.read_text(encoding="utf-8"))
        stages = data.get("stages", {})
        top = sorted(
            ((name, info.get("duration_s", 0.0)) for name, info in stages.items()),
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        print(f"selection_mode: {data.get('selection_mode')}")
        print(f"input_rows: {data.get('input_rows')}")
        print(f"output_rows: {data.get('output_rows')}")
        print(f"total_duration_s: {data.get('total_duration_s')}")
        print("top_stages:")
        for name, dur in top:
            print(f"  - {name}: {dur}s")
        return

    # Import the pipeline lazily so that lightweight commands (list/check/telemetry)
    # do not pull heavier dependencies like dask.
    from .pipeline.main import run_pipeline

    cfg = load_config(args.config)
    run_pipeline(cfg, json_logs=bool(getattr(args, "json_logs", False)))


if __name__ == "__main__":
    main()
