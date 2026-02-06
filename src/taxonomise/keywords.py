"""Keyword extraction and consensus aggregation."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

from taxonomise.data import Corpus, Document, Keyword
from taxonomise.logging import get_logger

logger = get_logger("keywords")


class KeywordExtractor(Protocol):
    """Protocol for keyword extractors."""

    @property
    def name(self) -> str:
        """Return extractor name."""
        ...

    def extract(self, text: str) -> list[str]:
        """Extract keywords from text.

        Args:
            text: Input text

        Returns:
            List of extracted keywords
        """
        ...


class BaseKeywordExtractor(ABC):
    """Base class for keyword extractors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return extractor name."""
        ...

    @abstractmethod
    def extract(self, text: str) -> list[str]:
        """Extract keywords from text."""
        ...

    def _preprocess(self, keyword: str) -> str:
        """Preprocess a keyword for consistency."""
        return keyword.lower().strip()

    def _postprocess(self, keywords: list[str]) -> list[str]:
        """Postprocess extracted keywords."""
        return [self._preprocess(kw) for kw in keywords if kw.strip()]


class RakeExtractor(BaseKeywordExtractor):
    """Keyword extractor using RAKE (Rapid Automatic Keyword Extraction)."""

    @property
    def name(self) -> str:
        return "rake"

    def extract(self, text: str) -> list[str]:
        """Extract keywords using RAKE.

        Args:
            text: Input text

        Returns:
            List of extracted keywords
        """
        try:
            from rake_nltk import Rake
        except ImportError:
            raise ImportError(
                "rake-nltk is required for RAKE extraction. "
                "Install with: pip install taxonomise[keywords]"
            )

        rake = Rake()
        rake.extract_keywords_from_text(text)
        keywords = rake.get_ranked_phrases()
        return self._postprocess(keywords)


class YakeExtractor(BaseKeywordExtractor):
    """Keyword extractor using YAKE."""

    def __init__(self, language: str = "en", n_gram: int = 3, top_n: int = 20) -> None:
        """Initialize YAKE extractor.

        Args:
            language: Language code
            n_gram: Maximum n-gram size
            top_n: Number of top keywords to return
        """
        self._language = language
        self._n_gram = n_gram
        self._top_n = top_n

    @property
    def name(self) -> str:
        return "yake"

    def extract(self, text: str) -> list[str]:
        """Extract keywords using YAKE.

        Args:
            text: Input text

        Returns:
            List of extracted keywords
        """
        try:
            import yake
        except ImportError:
            raise ImportError(
                "yake is required for YAKE extraction. "
                "Install with: pip install taxonomise[keywords]"
            )

        extractor = yake.KeywordExtractor(
            lan=self._language,
            n=self._n_gram,
            top=self._top_n,
        )
        keywords = extractor.extract_keywords(text)
        return self._postprocess([kw for kw, _ in keywords])


class KeyBertExtractor(BaseKeywordExtractor):
    """Keyword extractor using KeyBERT."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        ngram_range: tuple[int, int] = (1, 3),
        top_n: int = 10,
    ) -> None:
        """Initialize KeyBERT extractor.

        Args:
            model_name: SentenceTransformer model name
            ngram_range: Range for n-grams
            top_n: Number of top keywords to return
        """
        self._model_name = model_name
        self._ngram_range = ngram_range
        self._top_n = top_n
        self._model = None

    @property
    def name(self) -> str:
        return "keybert"

    def _get_model(self):
        """Lazy load KeyBERT model."""
        if self._model is None:
            try:
                from keybert import KeyBERT
            except ImportError:
                raise ImportError(
                    "keybert is required for KeyBERT extraction. "
                    "Install with: pip install taxonomise[keywords]"
                )
            self._model = KeyBERT(self._model_name)
        return self._model

    def extract(self, text: str) -> list[str]:
        """Extract keywords using KeyBERT.

        Args:
            text: Input text

        Returns:
            List of extracted keywords
        """
        model = self._get_model()
        keywords = model.extract_keywords(
            text,
            keyphrase_ngram_range=self._ngram_range,
            top_n=self._top_n,
        )
        return self._postprocess([kw for kw, _ in keywords])


class DBPediaExtractor(BaseKeywordExtractor):
    """Keyword extractor using DBpedia Spotlight entity linking."""

    def __init__(self, confidence: float = 0.5, support: int = 200) -> None:
        """Initialize DBpedia extractor.

        Args:
            confidence: Confidence threshold for annotations
            support: Support threshold for annotations
        """
        self._confidence = confidence
        self._support = support

    @property
    def name(self) -> str:
        return "dbpedia"

    def extract(self, text: str) -> list[str]:
        """Extract keywords using DBpedia Spotlight.

        Args:
            text: Input text

        Returns:
            List of extracted entity names
        """
        try:
            import requests
            from requests.adapters import HTTPAdapter, Retry
        except ImportError:
            raise ImportError(
                "requests is required for DBpedia extraction. "
                "Install with: pip install taxonomise[keywords]"
            )

        session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        session.mount("http://", HTTPAdapter(max_retries=retry))
        session.mount("https://", HTTPAdapter(max_retries=retry))

        base_url = "https://api.dbpedia-spotlight.org/en/annotate"
        headers = {"Accept": "application/json"}
        data = {
            "text": text,
            "confidence": self._confidence,
            "support": self._support,
            "policy": "whitelist",
        }

        try:
            response = session.post(base_url, data=data, headers=headers, timeout=30, verify=False)
            response.raise_for_status()
            response_json = response.json()
        except Exception as e:
            logger.warning(f"DBpedia request failed: {e}")
            return []

        resources = response_json.get("Resources", [])
        annotations = [
            resource.get("@URI", "")
            .replace("http://dbpedia.org/resource/", "")
            .replace("_", " ")
            for resource in resources
        ]

        return self._postprocess(list(set(annotations)))


def get_available_extractors() -> list[str]:
    """Get list of available extractor names based on installed packages."""
    available = []

    try:
        import rake_nltk  # noqa: F401

        available.append("rake")
    except ImportError:
        pass

    try:
        import yake  # noqa: F401

        available.append("yake")
    except ImportError:
        pass

    try:
        import keybert  # noqa: F401

        available.append("keybert")
    except ImportError:
        pass

    try:
        import requests  # noqa: F401

        available.append("dbpedia")
    except ImportError:
        pass

    return available


def create_extractor(name: str, **kwargs) -> KeywordExtractor:
    """Create a keyword extractor by name.

    Args:
        name: Extractor name ('rake', 'yake', 'keybert', 'dbpedia')
        **kwargs: Additional arguments for the extractor

    Returns:
        KeywordExtractor instance

    Raises:
        ValueError: If extractor name is unknown
    """
    extractors = {
        "rake": RakeExtractor,
        "yake": YakeExtractor,
        "keybert": KeyBertExtractor,
        "dbpedia": DBPediaExtractor,
    }

    if name not in extractors:
        raise ValueError(f"Unknown extractor: {name}. Available: {list(extractors.keys())}")

    return extractors[name](**kwargs)


@dataclass
class ExtractedKeywords:
    """Keywords extracted by a single extractor for a document.

    Attributes:
        document_id: Document ID
        extractor_name: Name of the extractor
        keywords: List of extracted keywords
    """

    document_id: str
    extractor_name: str
    keywords: list[str]


def extract_keywords_from_corpus(
    corpus: Corpus,
    extractor_names: list[str] | None = None,
    n_jobs: int = 1,
) -> list[ExtractedKeywords]:
    """Extract keywords from all documents using specified extractors.

    Args:
        corpus: Corpus with documents
        extractor_names: Names of extractors to use (default: all available)
        n_jobs: Number of parallel jobs

    Returns:
        List of ExtractedKeywords for each (document, extractor) pair
    """
    if extractor_names is None:
        extractor_names = get_available_extractors()

    if not extractor_names:
        logger.warning("No keyword extractors available")
        return []

    logger.info(f"Extracting keywords using: {extractor_names}")

    # Create extractors
    extractors = [create_extractor(name) for name in extractor_names]

    results: list[ExtractedKeywords] = []

    # Process each document
    for doc in corpus.documents:
        for extractor in extractors:
            try:
                keywords = extractor.extract(doc.text)
                results.append(
                    ExtractedKeywords(
                        document_id=doc.id,
                        extractor_name=extractor.name,
                        keywords=keywords,
                    )
                )
            except Exception as e:
                logger.warning(f"Extractor {extractor.name} failed on doc {doc.id}: {e}")
                results.append(
                    ExtractedKeywords(
                        document_id=doc.id,
                        extractor_name=extractor.name,
                        keywords=[],
                    )
                )

    logger.info(f"Extracted keywords from {len(corpus)} documents with {len(extractors)} extractors")

    return results


def aggregate_keywords(
    extracted: list[ExtractedKeywords],
    min_agreement: int = 2,
) -> list[Keyword]:
    """Aggregate keywords across extractors using consensus.

    Only keywords found by at least `min_agreement` extractors are kept.

    Args:
        extracted: List of ExtractedKeywords from multiple extractors
        min_agreement: Minimum number of extractors that must agree

    Returns:
        List of Keyword objects with consensus keywords
    """
    # Track which extractors found each keyword for each document
    # keyword -> set of extractor names that found it
    keyword_to_extractors: dict[str, set[str]] = defaultdict(set)
    # keyword -> set of document IDs where it appears
    keyword_to_docs: dict[str, set[str]] = defaultdict(set)

    for item in extracted:
        for keyword in item.keywords:
            keyword_to_extractors[keyword].add(item.extractor_name)
            keyword_to_docs[keyword].add(item.document_id)

    # Filter to keywords with enough agreement
    # Note: reference uses > (min_agreement - 1), which means >= min_agreement
    consensus_keywords: list[Keyword] = []

    for keyword, extractor_set in keyword_to_extractors.items():
        num_extractors = len(extractor_set)
        if num_extractors >= min_agreement:
            keyword_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, keyword))
            consensus_keywords.append(
                Keyword(
                    id=keyword_id,
                    keyword=keyword,
                    document_ids=list(keyword_to_docs[keyword]),
                    num_extractors=num_extractors,
                )
            )

    # Sort by number of extractors (descending), then alphabetically
    consensus_keywords.sort(key=lambda k: (-k.num_extractors, k.keyword))

    logger.info(
        f"Aggregated {len(keyword_to_extractors)} unique keywords -> "
        f"{len(consensus_keywords)} consensus keywords (min_agreement={min_agreement})"
    )

    return consensus_keywords


def extract_and_aggregate_keywords(
    corpus: Corpus,
    extractor_names: list[str] | None = None,
    min_agreement: int = 2,
    n_jobs: int = 1,
) -> Corpus:
    """Extract and aggregate keywords for a corpus.

    Args:
        corpus: Corpus with documents
        extractor_names: Names of extractors to use
        min_agreement: Minimum extractor agreement
        n_jobs: Number of parallel jobs

    Returns:
        Corpus with keywords populated
    """
    extracted = extract_keywords_from_corpus(
        corpus, extractor_names=extractor_names, n_jobs=n_jobs
    )
    keywords = aggregate_keywords(extracted, min_agreement=min_agreement)

    return Corpus(
        documents=corpus.documents,
        sentences=corpus.sentences,
        keywords=keywords,
        embeddings=corpus.embeddings,
        sentence_embeddings=corpus.sentence_embeddings,
    )
