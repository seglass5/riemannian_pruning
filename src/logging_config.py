"""Shared logging configuration for the riemannian-pruning package."""

import logging
import sys


def setup_logging(level: str = "INFO", run_name: str | None = None) -> logging.Logger:
    """Configure root logger with a consistent format.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        run_name: Optional experiment name injected into the log prefix.

    Returns:
        The configured root logger.
    """
    fmt_parts = ["%(asctime)s", "%(levelname)-8s"]
    if run_name:
        fmt_parts.append(f"[{run_name}]")
    fmt_parts.append("%(name)s: %(message)s")
    fmt = " | ".join(fmt_parts)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Silence noisy third-party loggers unless debugging.
    for noisy in ("transformers", "datasets", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger()
