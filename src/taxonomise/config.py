"""Configuration management for taxonomise."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PipelineConfig:
    """Configuration for the classification pipeline.

    Attributes:
        enable_global_matching: Enable document-level similarity matching
        enable_sentence_matching: Enable sentence-level similarity matching
        enable_keyword_matching: Enable keyword-based matching
        enable_zeroshot: Enable zero-shot NLI validation

        embedding_model: SentenceTransformer model name for embeddings
        zeroshot_model: HuggingFace model name for zero-shot classification
        zeroshot_hypothesis_template: Template for NLI hypothesis

        batch_size: Batch size for embedding computation
        top_n_matches: Number of top matches to retain per document/sentence
        n_jobs: Number of parallel jobs for processing

        sentence_weight: Weight for sentence-level scores in combination
        global_weight: Weight for global scores in combination
        (keyword_weight is computed as 1 - sentence_weight - global_weight)

        global_q2_threshold: Quantile threshold for medium confidence (global)
        global_q3_threshold: Quantile threshold for high confidence (global)
        local_q2_threshold: Quantile threshold for medium confidence (per-document)
        local_q3_threshold: Quantile threshold for high confidence (per-document)

        prune_sentence_threshold: Quantile threshold for pruning sentence matches
        prune_global_threshold: Quantile threshold for pruning global matches
        prune_keyword_threshold: Quantile threshold for pruning keyword matches
        use_quantile_pruning: Use quantile-based thresholds for pruning

        normalise_by_matches: Weight relevance by matching sentence ratio

        keyword_extractors: List of keyword extractors to use
        keyword_min_agreement: Minimum number of extractors that must agree
    """

    # Matching toggles
    enable_global_matching: bool = True
    enable_sentence_matching: bool = True
    enable_keyword_matching: bool = True
    enable_zeroshot: bool = False

    # Models
    embedding_model: str = "all-MiniLM-L6-v2"
    zeroshot_model: str = "tasksource/ModernBERT-large-nli"
    zeroshot_hypothesis_template: str = "This research project is about {}."

    # Processing
    batch_size: int = 1000
    top_n_matches: int = 10
    n_jobs: int = 8

    # Score weights (keyword_weight = 1 - sentence_weight - global_weight)
    sentence_weight: float = 0.5
    global_weight: float = 0.3

    # Global binning thresholds (quantiles)
    global_q2_threshold: float = 0.5
    global_q3_threshold: float = 0.75

    # Local binning thresholds (quantiles)
    local_q2_threshold: float = 0.5
    local_q3_threshold: float = 0.75

    # Pruning
    prune_sentence_threshold: float = 0.5
    prune_global_threshold: float = 0.5
    prune_keyword_threshold: float = 0.5
    use_quantile_pruning: bool = True

    # Score normalization
    normalise_by_matches: bool = False

    # Keywords
    keyword_extractors: list[str] = field(
        default_factory=lambda: ["rake", "yake", "keybert", "dbpedia"]
    )
    keyword_min_agreement: int = 2

    @property
    def keyword_weight(self) -> float:
        """Compute keyword weight from other weights."""
        return 1.0 - self.sentence_weight - self.global_weight

    @classmethod
    def from_yaml(cls, path: Path | str) -> "PipelineConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            PipelineConfig instance with values from file
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        """Create configuration from a dictionary.

        Args:
            data: Dictionary with configuration values

        Returns:
            PipelineConfig instance
        """
        # Flatten nested config structure if present
        flat_data: dict[str, Any] = {}

        for key, value in data.items():
            if isinstance(value, dict):
                # Handle nested sections like 'matching', 'embeddings', etc.
                for sub_key, sub_value in value.items():
                    flat_data[sub_key] = sub_value
            else:
                flat_data[key] = value

        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in flat_data.items() if k in valid_fields}

        return cls(**filtered)

    def merge(self, overrides: dict[str, Any]) -> "PipelineConfig":
        """Create a new config with overrides applied.

        Args:
            overrides: Dictionary of values to override

        Returns:
            New PipelineConfig with overrides applied
        """
        current = {k: getattr(self, k) for k in self.__dataclass_fields__}
        # Filter out None values from overrides
        filtered_overrides = {k: v for k, v in overrides.items() if v is not None}
        current.update(filtered_overrides)
        return PipelineConfig(**current)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If configuration is invalid
        """
        if not 0 <= self.sentence_weight <= 1:
            raise ValueError(f"sentence_weight must be in [0, 1], got {self.sentence_weight}")
        if not 0 <= self.global_weight <= 1:
            raise ValueError(f"global_weight must be in [0, 1], got {self.global_weight}")
        if self.sentence_weight + self.global_weight > 1:
            raise ValueError(
                f"sentence_weight + global_weight must be <= 1, "
                f"got {self.sentence_weight + self.global_weight}"
            )

        if not 0 <= self.global_q2_threshold <= 1:
            raise ValueError(f"global_q2_threshold must be in [0, 1]")
        if not 0 <= self.global_q3_threshold <= 1:
            raise ValueError(f"global_q3_threshold must be in [0, 1]")
        if self.global_q2_threshold >= self.global_q3_threshold:
            raise ValueError("global_q2_threshold must be < global_q3_threshold")

        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.top_n_matches < 1:
            raise ValueError(f"top_n_matches must be >= 1, got {self.top_n_matches}")
        if self.n_jobs < 1:
            raise ValueError(f"n_jobs must be >= 1, got {self.n_jobs}")

        if self.keyword_min_agreement < 1:
            raise ValueError(f"keyword_min_agreement must be >= 1")
