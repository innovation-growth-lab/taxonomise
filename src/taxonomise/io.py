"""Multi-format input/output utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from taxonomise.logging import get_logger
from taxonomise.pipeline import ClassificationResults

logger = get_logger("io")


def write_results(
    results: ClassificationResults,
    path: Path | str,
    format: str | None = None,
) -> None:
    """Write classification results to a file.

    Args:
        results: Classification results to write
        path: Output file path
        format: Output format ('parquet', 'csv', 'json', 'jsonl'). Auto-detected if None.
    """
    path = Path(path)

    if format is None:
        suffix = path.suffix.lower().lstrip(".")
        if suffix == "jsonl":
            format = "jsonl"
        else:
            format = suffix

    logger.info(f"Writing {len(results)} results to {path} (format: {format})")

    records = results.to_records()

    if format == "parquet":
        _write_parquet(records, path)
    elif format == "csv":
        _write_csv(records, path)
    elif format == "json":
        _write_json(records, path)
    elif format == "jsonl":
        _write_jsonl(records, path)
    else:
        raise ValueError(f"Unknown output format: {format}. Supported: parquet, csv, json, jsonl")


def _write_parquet(records: list[dict[str, Any]], path: Path) -> None:
    """Write records to Parquet file."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for Parquet output")

    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    logger.info(f"Wrote {len(df)} rows to Parquet")


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    """Write records to CSV file."""
    try:
        import pandas as pd
    except ImportError:
        # Fallback to csv module
        import csv

        if not records:
            path.write_text("")
            return

        fieldnames = list(records[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        logger.info(f"Wrote {len(records)} rows to CSV")
        return

    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
    logger.info(f"Wrote {len(df)} rows to CSV")


def _write_json(records: list[dict[str, Any]], path: Path) -> None:
    """Write records to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)
    logger.info(f"Wrote {len(records)} records to JSON")


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records to JSONL file (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")
    logger.info(f"Wrote {len(records)} records to JSONL")
