"""Logging configuration and utilities for taxonomise."""

import logging
import sys
from typing import TextIO


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the taxonomise prefix.

    Args:
        name: Logger name (will be prefixed with 'taxonomise.')

    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"taxonomise.{name}")


def configure_logging(
    level: int = logging.INFO,
    stream: TextIO | None = None,
    format_string: str | None = None,
) -> None:
    """Configure logging for the taxonomise package.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        stream: Output stream (defaults to sys.stderr)
        format_string: Custom format string (optional)
    """
    if stream is None:
        stream = sys.stderr

    if format_string is None:
        format_string = "%(levelname)-5s %(name)s: %(message)s"

    # Get the root taxonomise logger
    logger = logging.getLogger("taxonomise")
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create and configure handler
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(format_string))

    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False


def set_verbosity(verbose: bool = False, debug: bool = False, quiet: bool = False) -> None:
    """Set logging verbosity based on CLI flags.

    Args:
        verbose: Enable INFO level logging
        debug: Enable DEBUG level logging (takes precedence over verbose)
        quiet: Disable all logging except errors (takes precedence over all)
    """
    if quiet:
        configure_logging(level=logging.ERROR)
    elif debug:
        configure_logging(level=logging.DEBUG)
    elif verbose:
        configure_logging(level=logging.INFO)
    else:
        # Default: WARNING only (progress bars handle user feedback)
        configure_logging(level=logging.WARNING)
