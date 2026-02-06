"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path

from taxonomise.data import Corpus, Document, Taxonomy, TaxonomyNode
from taxonomise.config import PipelineConfig


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            id="doc1",
            text="Machine learning is a subset of artificial intelligence that enables "
            "computers to learn from data without being explicitly programmed.",
        ),
        Document(
            id="doc2",
            text="Quantum computing uses quantum mechanical phenomena like superposition "
            "and entanglement to perform computations.",
        ),
        Document(
            id="doc3",
            text="Climate change is causing significant impacts on global weather patterns "
            "and ecosystems around the world.",
        ),
    ]


@pytest.fixture
def sample_corpus(sample_documents: list[Document]) -> Corpus:
    """Create sample corpus for testing."""
    return Corpus(documents=sample_documents)


@pytest.fixture
def sample_taxonomy_nodes() -> list[TaxonomyNode]:
    """Create sample taxonomy nodes for testing."""
    return [
        TaxonomyNode(
            id="cs",
            label="Computer Science",
            full_path="Computer Science",
            level=0,
        ),
        TaxonomyNode(
            id="ai",
            label="Artificial Intelligence",
            full_path="Computer Science > Artificial Intelligence",
            parent_id="cs",
            level=1,
        ),
        TaxonomyNode(
            id="ml",
            label="Machine Learning",
            full_path="Computer Science > Artificial Intelligence > Machine Learning",
            parent_id="ai",
            level=2,
        ),
        TaxonomyNode(
            id="physics",
            label="Physics",
            full_path="Physics",
            level=0,
        ),
        TaxonomyNode(
            id="quantum",
            label="Quantum Physics",
            full_path="Physics > Quantum Physics",
            parent_id="physics",
            level=1,
        ),
        TaxonomyNode(
            id="env",
            label="Environmental Science",
            full_path="Environmental Science",
            level=0,
        ),
        TaxonomyNode(
            id="climate",
            label="Climate Science",
            full_path="Environmental Science > Climate Science",
            parent_id="env",
            level=1,
        ),
    ]


@pytest.fixture
def sample_taxonomy(sample_taxonomy_nodes: list[TaxonomyNode]) -> Taxonomy:
    """Create sample taxonomy for testing."""
    return Taxonomy(nodes=sample_taxonomy_nodes)


@pytest.fixture
def default_config() -> PipelineConfig:
    """Create default pipeline configuration."""
    return PipelineConfig()


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to fixtures directory."""
    return FIXTURES_DIR
