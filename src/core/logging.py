"""Logging setup: structlog, pretty to the console, JSON to ``logs/``.

Conventions (CLAUDE.md): no ``print()`` — structlog everywhere.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Any

import structlog

_configured = False

_SHARED_PROCESSORS: list[Any] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
]


def configure_logging(
    level: str = "INFO",
    json_file: bool = True,
    log_dir: str | Path | None = None,
) -> None:
    """Configure structlog + stdlib logging. Idempotent across a process."""
    global _configured
    if _configured:
        return

    structlog.configure(
        processors=[
            *_SHARED_PROCESSORS,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    root = logging.getLogger()
    root.setLevel(level.upper())

    console = logging.StreamHandler()
    console.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=True),
            foreign_pre_chain=_SHARED_PROCESSORS,
        )
    )
    root.addHandler(console)

    if json_file and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "music.jsonl",
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
                foreign_pre_chain=_SHARED_PROCESSORS,
            )
        )
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str | None = None) -> Any:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)
