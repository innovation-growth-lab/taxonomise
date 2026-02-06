"""Data models and loaders for taxonomies and corpora."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from taxonomise.logging import get_logger

logger = get_logger("data")


# =============================================================================
# Taxonomy Models
# =============================================================================


@dataclass
class TaxonomyNode:
    """A single node in a taxonomy hierarchy.

    Attributes:
        id: Unique identifier for the node (required)
        label: Display label for the node
        description: Optional description text
        parent_id: ID of parent node (None for root nodes)
        level: Depth in hierarchy (0 = root)
        full_path: Full hierarchical label (e.g., "Science > Physics > Quantum")
    """

    id: str
    label: str
    description: str | None = None
    parent_id: str | None = None
    level: int = 0
    full_path: str = ""

    def __post_init__(self) -> None:
        if not self.full_path:
            self.full_path = self.label


@dataclass
class Taxonomy:
    """A complete taxonomy with hierarchical nodes.

    Attributes:
        nodes: List of taxonomy nodes
        embeddings: Optional pre-computed embeddings for labels
    """

    nodes: list[TaxonomyNode]
    embeddings: np.ndarray | None = None
    _id_to_node: dict[str, TaxonomyNode] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._id_to_node = {node.id: node for node in self.nodes}

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[TaxonomyNode]:
        return iter(self.nodes)

    def get_node(self, node_id: str) -> TaxonomyNode | None:
        """Get a node by its ID."""
        return self._id_to_node.get(node_id)

    def get_labels(self) -> list[str]:
        """Get all full_path labels for embedding."""
        return [node.full_path for node in self.nodes]

    def get_ids(self) -> list[str]:
        """Get all node IDs."""
        return [node.id for node in self.nodes]

    def get_leaf_nodes(self) -> list[TaxonomyNode]:
        """Get nodes that have no children (leaf/terminal nodes)."""
        parent_ids = {node.parent_id for node in self.nodes if node.parent_id}
        return [node for node in self.nodes if node.id not in parent_ids]

    def filter_leaves_only(self) -> Taxonomy:
        """Return a new Taxonomy containing only leaf nodes."""
        leaves = self.get_leaf_nodes()
        return Taxonomy(nodes=leaves)


# =============================================================================
# Corpus Models
# =============================================================================


@dataclass
class Document:
    """A single document in a corpus.

    Attributes:
        id: Unique identifier for the document
        text: Combined text content
        metadata: Optional additional metadata
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Sentence:
    """A sentence extracted from a document.

    Attributes:
        id: Unique identifier for the sentence
        document_id: ID of the parent document
        text: Sentence text
        index: Position in the document
    """

    id: str
    document_id: str
    text: str
    index: int


@dataclass
class Keyword:
    """A keyword extracted from documents.

    Attributes:
        id: Unique identifier
        keyword: The keyword text
        document_ids: IDs of documents containing this keyword
        num_extractors: Number of extractors that found this keyword
    """

    id: str
    keyword: str
    document_ids: list[str]
    num_extractors: int


@dataclass
class Corpus:
    """A collection of documents to classify.

    Attributes:
        documents: List of documents
        sentences: Optional extracted sentences
        keywords: Optional extracted keywords
        embeddings: Optional pre-computed document embeddings
        sentence_embeddings: Optional pre-computed sentence embeddings
    """

    documents: list[Document]
    sentences: list[Sentence] | None = None
    keywords: list[Keyword] | None = None
    embeddings: np.ndarray | None = None
    sentence_embeddings: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self) -> Iterator[Document]:
        return iter(self.documents)

    def get_texts(self) -> list[str]:
        """Get all document texts for embedding."""
        return [doc.text for doc in self.documents]

    def get_ids(self) -> list[str]:
        """Get all document IDs."""
        return [doc.id for doc in self.documents]

    def get_sentence_texts(self) -> list[str]:
        """Get all sentence texts for embedding."""
        if self.sentences is None:
            return []
        return [sent.text for sent in self.sentences]

    def get_document(self, doc_id: str) -> Document | None:
        """Get a document by its ID."""
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None


# =============================================================================
# Taxonomy Loaders
# =============================================================================


def load_taxonomy(path: Path | str, format: str | None = None) -> Taxonomy:
    """Load a taxonomy from a file.

    Args:
        path: Path to taxonomy file
        format: File format ('csv', 'json'). Auto-detected if None.

    Returns:
        Loaded Taxonomy

    Raises:
        ValueError: If required fields are missing or format is unknown
    """
    path = Path(path)

    if format is None:
        format = path.suffix.lower().lstrip(".")

    logger.info(f"Loading taxonomy from {path} (format: {format})")

    if format == "csv":
        return _load_taxonomy_csv(path)
    elif format == "json":
        return _load_taxonomy_json(path)
    else:
        raise ValueError(f"Unknown taxonomy format: {format}. Supported: csv, json")


def _load_taxonomy_csv(path: Path) -> Taxonomy:
    """Load taxonomy from CSV file.

    Expected columns:
    - id (required): Unique identifier
    - label (required): Node label
    - parent_id (optional): Parent node ID
    - description (optional): Description text
    """
    import csv

    nodes: list[TaxonomyNode] = []
    id_to_label: dict[str, str] = {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Validate required columns
        if reader.fieldnames is None:
            raise ValueError("CSV file is empty")

        fieldnames = set(reader.fieldnames)
        if "id" not in fieldnames:
            raise ValueError("CSV taxonomy must have 'id' column")
        if "label" not in fieldnames:
            raise ValueError("CSV taxonomy must have 'label' column")

        rows = list(reader)

    # First pass: collect all labels by ID
    for row in rows:
        id_to_label[row["id"]] = row["label"]

    # Second pass: build nodes with full paths
    for row in rows:
        node_id = row["id"]
        label = row["label"]
        parent_id = row.get("parent_id", "").strip() or None
        description = row.get("description", "").strip() or None

        # Build full path by traversing parents
        path_parts = [label]
        current_parent = parent_id
        level = 0

        while current_parent and current_parent in id_to_label:
            path_parts.insert(0, id_to_label[current_parent])
            # Find parent's parent
            for r in rows:
                if r["id"] == current_parent:
                    current_parent = r.get("parent_id", "").strip() or None
                    break
            else:
                current_parent = None
            level += 1

        full_path = " > ".join(path_parts)

        nodes.append(
            TaxonomyNode(
                id=node_id,
                label=label,
                description=description,
                parent_id=parent_id,
                level=level,
                full_path=full_path,
            )
        )

    logger.info(f"Loaded {len(nodes)} taxonomy nodes from CSV")
    return Taxonomy(nodes=nodes)


def _load_taxonomy_json(path: Path) -> Taxonomy:
    """Load taxonomy from JSON file.

    Supports two formats:
    1. Tree structure: {"id": "...", "label": "...", "children": [...]}
    2. Flat array: [{"id": "...", "label": "...", "parent_id": "..."}, ...]
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        # Flat array format
        return _parse_flat_json_taxonomy(data)
    elif isinstance(data, dict):
        # Tree format (single root) or wrapped array
        if "children" in data or "label" in data:
            return _parse_tree_json_taxonomy(data)
        elif "nodes" in data:
            return _parse_flat_json_taxonomy(data["nodes"])
        else:
            raise ValueError("JSON taxonomy must have 'children', 'label', or 'nodes' key")
    else:
        raise ValueError("JSON taxonomy must be an object or array")


def _parse_flat_json_taxonomy(data: list[dict[str, Any]]) -> Taxonomy:
    """Parse flat JSON array into Taxonomy."""
    nodes: list[TaxonomyNode] = []
    id_to_label: dict[str, str] = {}

    # First pass: collect labels
    for item in data:
        if "id" not in item:
            raise ValueError("Each taxonomy node must have an 'id' field")
        if "label" not in item:
            raise ValueError("Each taxonomy node must have a 'label' field")
        id_to_label[item["id"]] = item["label"]

    # Second pass: build nodes
    for item in data:
        node_id = item["id"]
        label = item["label"]
        parent_id = item.get("parent_id")
        description = item.get("description")

        # Build full path
        path_parts = [label]
        current_parent = parent_id
        level = 0

        while current_parent and current_parent in id_to_label:
            path_parts.insert(0, id_to_label[current_parent])
            # Find parent's parent
            for i in data:
                if i["id"] == current_parent:
                    current_parent = i.get("parent_id")
                    break
            else:
                current_parent = None
            level += 1

        full_path = " > ".join(path_parts)

        nodes.append(
            TaxonomyNode(
                id=node_id,
                label=label,
                description=description,
                parent_id=parent_id,
                level=level,
                full_path=full_path,
            )
        )

    logger.info(f"Loaded {len(nodes)} taxonomy nodes from flat JSON")
    return Taxonomy(nodes=nodes)


def _parse_tree_json_taxonomy(data: dict[str, Any]) -> Taxonomy:
    """Parse tree-structured JSON into Taxonomy."""
    nodes: list[TaxonomyNode] = []

    def traverse(
        node: dict[str, Any],
        parent_id: str | None,
        path_parts: list[str],
        level: int,
    ) -> None:
        if "id" not in node:
            raise ValueError("Each taxonomy node must have an 'id' field")
        if "label" not in node:
            raise ValueError("Each taxonomy node must have a 'label' field")

        node_id = node["id"]
        label = node["label"]
        description = node.get("description")

        current_path = path_parts + [label]
        full_path = " > ".join(current_path)

        nodes.append(
            TaxonomyNode(
                id=node_id,
                label=label,
                description=description,
                parent_id=parent_id,
                level=level,
                full_path=full_path,
            )
        )

        for child in node.get("children", []):
            traverse(child, node_id, current_path, level + 1)

    traverse(data, None, [], 0)

    logger.info(f"Loaded {len(nodes)} taxonomy nodes from tree JSON")
    return Taxonomy(nodes=nodes)


# =============================================================================
# Corpus Loaders
# =============================================================================


def load_corpus(
    path: Path | str,
    format: str | None = None,
    id_column: str = "id",
    text_columns: list[str] | str = "text",
) -> Corpus:
    """Load a corpus from a file.

    Args:
        path: Path to corpus file
        format: File format ('csv', 'parquet', 'json', 'jsonl'). Auto-detected if None.
        id_column: Column/field name for document IDs
        text_columns: Column/field name(s) for text content (combined if multiple)

    Returns:
        Loaded Corpus

    Raises:
        ValueError: If required fields are missing or format is unknown
    """
    path = Path(path)

    if format is None:
        suffix = path.suffix.lower().lstrip(".")
        if suffix == "jsonl":
            format = "jsonl"
        else:
            format = suffix

    if isinstance(text_columns, str):
        text_columns = [text_columns]

    logger.info(f"Loading corpus from {path} (format: {format})")

    if format == "csv":
        return _load_corpus_tabular(path, id_column, text_columns, "csv")
    elif format == "parquet":
        return _load_corpus_tabular(path, id_column, text_columns, "parquet")
    elif format == "json":
        return _load_corpus_json(path, id_column, text_columns)
    elif format == "jsonl":
        return _load_corpus_jsonl(path, id_column, text_columns)
    else:
        raise ValueError(f"Unknown corpus format: {format}. Supported: csv, parquet, json, jsonl")


def _load_corpus_tabular(
    path: Path,
    id_column: str,
    text_columns: list[str],
    format: str,
) -> Corpus:
    """Load corpus from CSV or Parquet file."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for loading tabular corpus files")

    if format == "csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    # Validate columns
    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found in {path}")

    missing_text = [col for col in text_columns if col not in df.columns]
    if missing_text:
        raise ValueError(f"Text column(s) not found in {path}: {missing_text}")

    documents: list[Document] = []

    for _, row in df.iterrows():
        doc_id = str(row[id_column])

        # Combine text columns
        text_parts = []
        for col in text_columns:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                text_parts.append(str(val).strip())
        text = ". ".join(text_parts)

        # Collect other columns as metadata
        metadata = {
            col: row[col]
            for col in df.columns
            if col != id_column and col not in text_columns
        }

        documents.append(Document(id=doc_id, text=text, metadata=metadata))

    logger.info(f"Loaded {len(documents)} documents from {format.upper()}")
    return Corpus(documents=documents)


def _load_corpus_json(
    path: Path,
    id_column: str,
    text_columns: list[str],
) -> Corpus:
    """Load corpus from JSON file (array of objects)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        if "documents" in data:
            data = data["documents"]
        else:
            raise ValueError("JSON corpus must be an array or have 'documents' key")

    return _parse_documents_from_dicts(data, id_column, text_columns)


def _load_corpus_jsonl(
    path: Path,
    id_column: str,
    text_columns: list[str],
) -> Corpus:
    """Load corpus from JSONL file (one object per line)."""
    data: list[dict[str, Any]] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return _parse_documents_from_dicts(data, id_column, text_columns)


def _parse_documents_from_dicts(
    data: list[dict[str, Any]],
    id_column: str,
    text_columns: list[str],
) -> Corpus:
    """Parse list of dicts into Corpus."""
    documents: list[Document] = []

    for item in data:
        if id_column not in item:
            raise ValueError(f"Document missing '{id_column}' field")

        doc_id = str(item[id_column])

        # Combine text fields
        text_parts = []
        for col in text_columns:
            val = item.get(col)
            if val and str(val).strip():
                text_parts.append(str(val).strip())

        if not text_parts:
            logger.warning(f"Document {doc_id} has no text content")
            continue

        text = ". ".join(text_parts)

        # Collect other fields as metadata
        metadata = {k: v for k, v in item.items() if k != id_column and k not in text_columns}

        documents.append(Document(id=doc_id, text=text, metadata=metadata))

    logger.info(f"Loaded {len(documents)} documents from JSON")
    return Corpus(documents=documents)


# =============================================================================
# Sentence Splitting
# =============================================================================


def split_sentences(corpus: Corpus) -> Corpus:
    """Split corpus documents into sentences using spaCy.

    Args:
        corpus: Corpus with documents

    Returns:
        Corpus with sentences populated
    """
    try:
        from spacy.lang.en import English
    except ImportError:
        raise ImportError("spaCy is required for sentence splitting")

    nlp = English()
    nlp.add_pipe("sentencizer")

    sentences: list[Sentence] = []

    for doc in corpus.documents:
        spacy_doc = nlp(doc.text)

        for idx, sent in enumerate(spacy_doc.sents):
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            # Generate sentence ID
            sent_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc.id}:{idx}:{sent_text[:50]}"))

            sentences.append(
                Sentence(
                    id=sent_id,
                    document_id=doc.id,
                    text=sent_text,
                    index=idx,
                )
            )

    logger.info(f"Split {len(corpus.documents)} documents into {len(sentences)} sentences")

    # Return new corpus with sentences
    return Corpus(
        documents=corpus.documents,
        sentences=sentences,
        keywords=corpus.keywords,
        embeddings=corpus.embeddings,
    )
