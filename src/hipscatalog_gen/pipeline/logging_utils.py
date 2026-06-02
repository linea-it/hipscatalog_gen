"""Structured logging utilities for the pipeline."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

LogFn = Callable[[str, bool], None]


class _MaxLevelFilter(logging.Filter):
    """Allow records below a maximum logging level."""

    def __init__(self, level: int) -> None:
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < self.level


@dataclass
class LogContext:
    """Mutable context for structured logging fields."""

    stage: str | None = None
    depth: int | None = None


def setup_structured_logger(
    out_dir: Path, selection_mode: str, *, json_logs: bool = False
) -> tuple[LogContext, LogFn]:
    """Configure a structured logger that writes to stdout and process.log; optionally JSON lines."""
    logger = logging.getLogger("hipscatalog_gen.pipeline")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fmt = (
        "%(asctime)s | %(levelname)s | mode=%(selection_mode)s stage=%(stage)s depth=%(depth)s | %(message)s"
    )
    formatter = logging.Formatter(fmt)

    fh = logging.FileHandler(out_dir / "process.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if json_logs:
        json_path = out_dir / "process.jsonl"
        json_handler = logging.FileHandler(json_path, encoding="utf-8")

        class _JsonFormatter(logging.Formatter):
            """Format log records as structured JSON lines."""

            def format(self, record: logging.LogRecord) -> str:
                """Render a log record to JSON with timestamp, level, and context."""
                payload = {
                    "ts": self.formatTime(record),
                    "level": record.levelname,
                    "selection_mode": getattr(record, "selection_mode", None),
                    "stage": getattr(record, "stage", None),
                    "depth": getattr(record, "depth", None),
                    "message": record.getMessage(),
                }
                return json.dumps(payload)

        json_handler.setFormatter(_JsonFormatter())
        logger.addHandler(json_handler)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(_MaxLevelFilter(logging.ERROR))
    logger.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    log_ctx = LogContext()

    def _log(msg: str, always: bool = False, *, stage: str | None = None, depth: int | None = None) -> None:
        """Emit a structured log line with optional stage/depth overrides."""
        level = logging.INFO
        extra = {
            "selection_mode": selection_mode,
            "stage": stage if stage is not None else log_ctx.stage,
            "depth": depth if depth is not None else log_ctx.depth,
        }
        logger.log(level, msg, extra=extra)

    return log_ctx, _log
