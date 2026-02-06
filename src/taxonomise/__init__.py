"""Taxonomise - Semantic taxonomy classification for document corpora."""

from taxonomise.config import PipelineConfig
from taxonomise.data import Corpus, Document, Taxonomy, TaxonomyNode
from taxonomise.pipeline import ClassificationPipeline, ClassificationResults

__version__ = "0.1.0"

__all__ = [
    "ClassificationPipeline",
    "ClassificationResults",
    "Corpus",
    "Document",
    "PipelineConfig",
    "Taxonomy",
    "TaxonomyNode",
    "__version__",
]
