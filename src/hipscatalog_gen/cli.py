from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .config import load_config
from .pipeline import run_pipeline

__all__ = ["main"]


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the command-line interface.

    Args:
        argv: Command-line arguments excluding the program name.
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description=(
            "HiPS Catalog Pipeline (Dask, Parquet, coverage/mag_global selection). "
            "Use a YAML config file to control inputs, cluster, and algorithm options."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )

    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
