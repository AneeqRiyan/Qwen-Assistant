"""
Structured logging for the Voice Personal Assistant.

Provides JSON-formatted log output with per-request latency metrics,
daily log rotation, and configurable log levels.
"""

from __future__ import annotations

import logging
import logging.handlers
import json
import time
import uuid
from pathlib import Path
from typing import Any

from app.config import get_config


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON lines for structured log analysis."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include extra fields if present (e.g., request metrics)
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "metrics"):
            log_entry["metrics"] = record.metrics

        # Include exception info if present
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging() -> logging.Logger:
    """
    Configure the application logger with console and file handlers.

    Returns:
        The root application logger instance.
    """
    config = get_config()

    # Create logs directory if needed
    log_path = Path(config.logging.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Root application logger
    logger = logging.getLogger("assistant")
    logger.setLevel(getattr(logging, config.logging.level.upper(), logging.INFO))

    # Avoid duplicate handlers on reload
    if logger.handlers:
        return logger

    # ── Console handler (human-readable) ──
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(name)-20s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # ── File handler (JSON, daily rotation) ──
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=str(log_path),
        when="midnight",
        interval=1,
        backupCount=config.logging.max_days,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    logger.info(
        "Logging initialized",
        extra={"metrics": {"log_level": config.logging.level, "log_file": str(log_path)}},
    )

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger for a specific module.

    Args:
        name: Module or component name (e.g., 'stt', 'llm', 'weather').

    Returns:
        A child logger under the 'assistant' namespace.
    """
    return logging.getLogger(f"assistant.{name}")


class RequestTimer:
    """
    Context manager for tracking per-request latency breakdown.

    Usage:
        timer = RequestTimer()
        with timer.measure("stt"):
            # ... STT processing ...
        with timer.measure("llm"):
            # ... LLM inference ...
        timer.log_metrics(logger)
    """

    def __init__(self) -> None:
        self.request_id: str = uuid.uuid4().hex[:12]
        self.metrics: dict[str, float] = {}
        self._start_time: float = time.perf_counter()

    class _MeasureContext:
        """Inner context manager for measuring a single phase."""

        def __init__(self, timer: "RequestTimer", phase: str) -> None:
            self.timer = timer
            self.phase = phase
            self.start: float = 0.0

        def __enter__(self) -> "_MeasureContext":
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args: Any) -> None:
            elapsed_ms = (time.perf_counter() - self.start) * 1000
            self.timer.metrics[f"{self.phase}_ms"] = round(elapsed_ms, 2)

    def measure(self, phase: str) -> _MeasureContext:
        """Create a context manager that measures the duration of a phase."""
        return self._MeasureContext(self, phase)

    @property
    def total_ms(self) -> float:
        """Total elapsed time since timer creation."""
        return round((time.perf_counter() - self._start_time) * 1000, 2)

    def log_metrics(self, logger: logging.Logger) -> None:
        """Log all collected metrics."""
        self.metrics["total_ms"] = self.total_ms
        logger.info(
            f"Request {self.request_id} completed",
            extra={"request_id": self.request_id, "metrics": self.metrics},
        )
