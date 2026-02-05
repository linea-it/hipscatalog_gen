"""Command-line interface for running the HiPS catalog pipeline."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from contextlib import suppress
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List

from .config import load_config

__all__ = ["main"]


def _serve_output_dir(
    out_dir: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
) -> None:
    """Serve a generated HiPS output directory over HTTP for local preview."""
    out_path = Path(out_dir).expanduser().resolve()
    if not out_path.exists() or not out_path.is_dir():
        raise ValueError(f"--out must point to an existing directory: {out_path}")

    handler = partial(SimpleHTTPRequestHandler, directory=str(out_path))
    httpd = ThreadingHTTPServer((host, int(port)), handler)
    url = f"http://{host}:{int(port)}/index.html"
    print(f"Serving {out_path}")
    print(f"Open: {url}")
    print("Press Ctrl+C to stop.")

    if open_browser:
        with suppress(Exception):
            webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        httpd.server_close()


def main(argv: List[str] | None = None) -> None:
    """Entry point for the command-line interface.

    Args:
        argv: Command-line arguments excluding the program name.
    """
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "serve":
        parser = argparse.ArgumentParser(
            prog="hipscatalog-gen serve",
            description="Serve a generated HiPS output directory over local HTTP.",
        )
        parser.add_argument(
            "--out",
            required=True,
            help="Path to the generated output directory containing index.html.",
        )
        parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
        parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
        parser.add_argument(
            "--no-browser",
            action="store_true",
            help="Do not auto-open index.html in the default browser.",
        )
        args = parser.parse_args(argv[1:])
        _serve_output_dir(
            args.out,
            host=args.host,
            port=args.port,
            open_browser=not bool(args.no_browser),
        )
        return

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
