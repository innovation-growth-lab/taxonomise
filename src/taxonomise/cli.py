"""Command-line interface for taxonomise."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from taxonomise import __version__
from taxonomise.config import PipelineConfig
from taxonomise.data import load_corpus, load_taxonomy
from taxonomise.io import write_results
from taxonomise.logging import set_verbosity
from taxonomise.pipeline import ClassificationPipeline

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="taxonomise")
def cli() -> None:
    """Taxonomise - Semantic taxonomy classification for document corpora."""
    pass


@cli.command()
@click.option(
    "--corpus",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to corpus file (CSV, Parquet, JSON, or JSONL)",
)
@click.option(
    "--taxonomy",
    "-t",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to taxonomy file (CSV or JSON)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path for results",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["parquet", "csv", "json", "jsonl"]),
    default=None,
    help="Output format (auto-detected from extension if not specified)",
)
# Corpus column configuration
@click.option(
    "--id-column",
    default="id",
    help="Column name for document IDs in corpus",
)
@click.option(
    "--text-columns",
    default="text",
    help="Comma-separated column names for text fields",
)
# Pipeline element flags
@click.option(
    "--enable-global/--disable-global",
    default=True,
    help="Enable/disable global document matching",
)
@click.option(
    "--enable-sentences/--disable-sentences",
    default=True,
    help="Enable/disable sentence-level matching",
)
@click.option(
    "--enable-keywords/--disable-keywords",
    default=True,
    help="Enable/disable keyword-based matching",
)
@click.option(
    "--enable-zeroshot/--disable-zeroshot",
    default=False,
    help="Enable/disable zero-shot NLI validation",
)
# Model configuration
@click.option(
    "--embedding-model",
    default="all-MiniLM-L6-v2",
    help="SentenceTransformer model for embeddings",
)
@click.option(
    "--zeroshot-model",
    default="tasksource/ModernBERT-large-nli",
    help="HuggingFace model for zero-shot validation",
)
# Performance
@click.option(
    "--batch-size",
    type=int,
    default=1000,
    help="Batch size for embedding computation",
)
@click.option(
    "--n-jobs",
    type=int,
    default=8,
    help="Number of parallel jobs",
)
@click.option(
    "--top-n",
    type=int,
    default=10,
    help="Number of top matches to keep per document/sentence",
)
# Weights
@click.option(
    "--sentence-weight",
    type=float,
    default=0.5,
    help="Weight for sentence-level scores (0-1)",
)
@click.option(
    "--global-weight",
    type=float,
    default=0.3,
    help="Weight for global scores (0-1, keyword = 1 - sentence - global)",
)
# Configuration file
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to YAML configuration file",
)
# Verbosity
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output (INFO level)",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug output (DEBUG level)",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress all output except errors",
)
def classify(
    corpus: Path,
    taxonomy: Path,
    output: Path,
    format: str | None,
    id_column: str,
    text_columns: str,
    enable_global: bool,
    enable_sentences: bool,
    enable_keywords: bool,
    enable_zeroshot: bool,
    embedding_model: str,
    zeroshot_model: str,
    batch_size: int,
    n_jobs: int,
    top_n: int,
    sentence_weight: float,
    global_weight: float,
    config: Path | None,
    verbose: bool,
    debug: bool,
    quiet: bool,
) -> None:
    """Classify documents against a taxonomy.

    Examples:

        taxonomise classify -c documents.csv -t taxonomy.json -o results.parquet

        taxonomise classify -c data.jsonl -t categories.csv -o out.json -f json
    """
    # Configure logging
    set_verbosity(verbose=verbose, debug=debug, quiet=quiet)

    # Load config from file if provided
    if config:
        pipeline_config = PipelineConfig.from_yaml(config)
    else:
        pipeline_config = PipelineConfig()

    # Apply CLI overrides
    cli_overrides = {
        "enable_global_matching": enable_global,
        "enable_sentence_matching": enable_sentences,
        "enable_keyword_matching": enable_keywords,
        "enable_zeroshot": enable_zeroshot,
        "embedding_model": embedding_model,
        "zeroshot_model": zeroshot_model,
        "batch_size": batch_size,
        "n_jobs": n_jobs,
        "top_n_matches": top_n,
        "sentence_weight": sentence_weight,
        "global_weight": global_weight,
    }
    pipeline_config = pipeline_config.merge(cli_overrides)

    # Parse text columns
    text_cols = [col.strip() for col in text_columns.split(",")]

    # Progress tracking
    progress_tasks: dict[str, int] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=quiet,
    ) as progress:
        main_task = progress.add_task("Classifying...", total=None)

        def update_progress(stage: str, current: int, total: int) -> None:
            """Callback to update progress display."""
            if stage not in progress_tasks:
                progress_tasks[stage] = progress.add_task(stage, total=total)
            progress.update(progress_tasks[stage], completed=current)

        try:
            # Load data
            progress.update(main_task, description="Loading corpus...")
            corpus_data = load_corpus(corpus, id_column=id_column, text_columns=text_cols)

            progress.update(main_task, description="Loading taxonomy...")
            taxonomy_data = load_taxonomy(taxonomy)

            if not quiet:
                console.print(
                    f"[green]Loaded {len(corpus_data)} documents and "
                    f"{len(taxonomy_data)} taxonomy labels[/green]"
                )

            # Run pipeline
            progress.update(main_task, description="Running classification...")
            pipeline = ClassificationPipeline(config=pipeline_config)
            results = pipeline.classify(
                corpus_data, taxonomy_data, progress_callback=update_progress
            )

            # Write output
            progress.update(main_task, description="Writing results...")
            write_results(results, output, format=format)

            progress.update(main_task, description="Done!", completed=100, total=100)

        except Exception as e:
            progress.stop()
            console.print(f"[red]Error: {e}[/red]")
            raise click.Abort()

    if not quiet:
        console.print(f"[green]Wrote {len(results)} results to {output}[/green]")

        # Print summary
        high_count = sum(1 for r in results if r.confidence_bin == "high")
        medium_count = sum(1 for r in results if r.confidence_bin == "medium")
        low_count = sum(1 for r in results if r.confidence_bin == "low")

        console.print(
            f"[dim]Confidence distribution: "
            f"high={high_count}, medium={medium_count}, low={low_count}[/dim]"
        )


if __name__ == "__main__":
    cli()
