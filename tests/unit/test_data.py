"""Tests for data module."""

import pytest
from pathlib import Path

from taxonomise.data import (
    Corpus,
    Document,
    Taxonomy,
    TaxonomyNode,
    load_corpus,
    load_taxonomy,
    split_sentences,
)


class TestTaxonomyNode:
    """Tests for TaxonomyNode."""

    def test_create_basic(self):
        """Test creating a basic taxonomy node."""
        node = TaxonomyNode(id="test", label="Test Label")
        assert node.id == "test"
        assert node.label == "Test Label"
        assert node.full_path == "Test Label"
        assert node.level == 0
        assert node.parent_id is None

    def test_create_with_parent(self):
        """Test creating a node with parent."""
        node = TaxonomyNode(
            id="child",
            label="Child",
            parent_id="parent",
            level=1,
            full_path="Parent > Child",
        )
        assert node.parent_id == "parent"
        assert node.level == 1
        assert node.full_path == "Parent > Child"


class TestTaxonomy:
    """Tests for Taxonomy."""

    def test_create_empty(self):
        """Test creating an empty taxonomy."""
        tax = Taxonomy(nodes=[])
        assert len(tax) == 0

    def test_create_with_nodes(self, sample_taxonomy_nodes):
        """Test creating taxonomy with nodes."""
        tax = Taxonomy(nodes=sample_taxonomy_nodes)
        assert len(tax) == len(sample_taxonomy_nodes)

    def test_get_labels(self, sample_taxonomy):
        """Test getting all labels."""
        labels = sample_taxonomy.get_labels()
        assert len(labels) == len(sample_taxonomy)
        assert "Computer Science" in labels

    def test_get_leaf_nodes(self, sample_taxonomy):
        """Test getting leaf nodes."""
        leaves = sample_taxonomy.get_leaf_nodes()
        # ml, quantum, climate are leaves
        assert len(leaves) == 3
        leaf_ids = {node.id for node in leaves}
        assert leaf_ids == {"ml", "quantum", "climate"}

    def test_filter_leaves_only(self, sample_taxonomy):
        """Test filtering to leaves only."""
        leaves_only = sample_taxonomy.filter_leaves_only()
        assert len(leaves_only) == 3


class TestDocument:
    """Tests for Document."""

    def test_create_basic(self):
        """Test creating a basic document."""
        doc = Document(id="doc1", text="Test content")
        assert doc.id == "doc1"
        assert doc.text == "Test content"
        assert doc.metadata == {}

    def test_create_with_metadata(self):
        """Test creating document with metadata."""
        doc = Document(id="doc1", text="Test", metadata={"source": "test"})
        assert doc.metadata["source"] == "test"


class TestCorpus:
    """Tests for Corpus."""

    def test_create_empty(self):
        """Test creating an empty corpus."""
        corpus = Corpus(documents=[])
        assert len(corpus) == 0

    def test_get_texts(self, sample_corpus):
        """Test getting all texts."""
        texts = sample_corpus.get_texts()
        assert len(texts) == 3
        assert all(isinstance(t, str) for t in texts)


class TestLoadTaxonomy:
    """Tests for taxonomy loading."""

    def test_load_csv(self, fixtures_dir):
        """Test loading taxonomy from CSV."""
        path = fixtures_dir / "sample_taxonomy.csv"
        tax = load_taxonomy(path)
        assert len(tax) == 7

        # Check hierarchy
        ml_node = tax.get_node("ml")
        assert ml_node is not None
        assert ml_node.parent_id == "ai"
        assert "Machine Learning" in ml_node.full_path

    def test_load_json_tree(self, fixtures_dir):
        """Test loading taxonomy from JSON tree."""
        path = fixtures_dir / "sample_taxonomy.json"
        tax = load_taxonomy(path)

        # Should have root + children
        assert len(tax) >= 7

        # Check hierarchy was built correctly
        ml_node = tax.get_node("ml")
        assert ml_node is not None
        assert " > " in ml_node.full_path


class TestLoadCorpus:
    """Tests for corpus loading."""

    def test_load_csv(self, fixtures_dir):
        """Test loading corpus from CSV."""
        path = fixtures_dir / "sample_corpus.csv"
        corpus = load_corpus(path)
        assert len(corpus) == 3

    def test_load_jsonl(self, fixtures_dir):
        """Test loading corpus from JSONL."""
        path = fixtures_dir / "sample_corpus.jsonl"
        corpus = load_corpus(path)
        assert len(corpus) == 3


class TestSplitSentences:
    """Tests for sentence splitting."""

    def test_split_sentences(self, sample_corpus):
        """Test splitting documents into sentences."""
        corpus_with_sentences = split_sentences(sample_corpus)
        assert corpus_with_sentences.sentences is not None
        # Each test document is a single sentence, so we get 1 sentence per doc
        assert len(corpus_with_sentences.sentences) >= len(sample_corpus)

    def test_sentence_document_mapping(self, sample_corpus):
        """Test that sentences are mapped to correct documents."""
        corpus = split_sentences(sample_corpus)

        # All sentences should have valid document IDs
        doc_ids = {doc.id for doc in sample_corpus.documents}
        for sent in corpus.sentences:
            assert sent.document_id in doc_ids
