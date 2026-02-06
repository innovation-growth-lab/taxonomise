"""Main classification pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from taxonomise.config import PipelineConfig
from taxonomise.confidence import (
    AggregatedScores,
    RefinedScores,
    aggregate_to_labels,
    enhance_with_zeroshot,
    refine_confidence_bins,
)
from taxonomise.data import Corpus, Taxonomy, split_sentences
from taxonomise.embeddings import EmbeddingProvider, create_embedding_provider
from taxonomise.keywords import extract_and_aggregate_keywords
from taxonomise.logging import get_logger
from taxonomise.matching import (
    CombinedScores,
    GlobalMatcher,
    KeywordMatcher,
    MatchResult,
    SentenceMatcher,
    combine_scores,
    prune_matches,
)

logger = get_logger("pipeline")


@dataclass
class ClassificationResult:
    """A single classification result for a document-taxonomy pair.

    Attributes:
        document_id: Document identifier
        taxonomy_label_id: Taxonomy label identifier
        taxonomy_label: Full taxonomy label path
        relevance_score: Combined relevance score
        confidence_bin: Final confidence bin (high/medium/low)
        similarity_score_global: Document-level similarity
        similarity_score_sent: Best sentence similarity
        similarity_score_key: Best keyword similarity
        global_bin: Global threshold bin
        local_bin: Per-document threshold bin
        num_matching_sentences: Number of sentences matching this label
        num_sentences: Total sentences in document
        zeroshot_score: Zero-shot classification score (if enabled)
        zeroshot_bin: Zero-shot confidence bin (if enabled)
    """

    document_id: str
    taxonomy_label_id: str
    taxonomy_label: str
    relevance_score: float
    confidence_bin: str
    similarity_score_global: float
    similarity_score_sent: float
    similarity_score_key: float
    global_bin: str
    local_bin: str
    num_matching_sentences: int
    num_sentences: int
    zeroshot_score: float | None = None
    zeroshot_bin: str | None = None
    max_confidence: str | None = None
    zeroshot_favouring_confidence: str | None = None
    sentence_favouring_confidence: str | None = None


@dataclass
class ClassificationResults:
    """Complete results from the classification pipeline.

    Attributes:
        results: List of individual classification results
        metadata: Pipeline configuration and statistics
    """

    results: list[ClassificationResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def filter_by_confidence(
        self, min_confidence: str = "medium"
    ) -> "ClassificationResults":
        """Filter results to keep only those at or above a confidence level.

        Args:
            min_confidence: Minimum confidence level ('high', 'medium', 'low')

        Returns:
            Filtered ClassificationResults
        """
        bin_order = {"high": 3, "medium": 2, "low": 1}
        min_val = bin_order.get(min_confidence, 0)

        filtered = [
            r for r in self.results if bin_order.get(r.confidence_bin, 0) >= min_val
        ]

        return ClassificationResults(results=filtered, metadata=self.metadata)

    def to_records(self) -> list[dict[str, Any]]:
        """Convert results to list of dictionaries.

        Returns:
            List of result dictionaries
        """
        records = []
        for r in self.results:
            record = {
                "document_id": r.document_id,
                "taxonomy_label_id": r.taxonomy_label_id,
                "taxonomy_label": r.taxonomy_label,
                "relevance_score": r.relevance_score,
                "confidence_bin": r.confidence_bin,
                "similarity_score_global": r.similarity_score_global,
                "similarity_score_sent": r.similarity_score_sent,
                "similarity_score_key": r.similarity_score_key,
                "global_bin": r.global_bin,
                "local_bin": r.local_bin,
                "num_matching_sentences": r.num_matching_sentences,
                "num_sentences": r.num_sentences,
            }
            if r.zeroshot_score is not None:
                record["zeroshot_score"] = r.zeroshot_score
                record["zeroshot_bin"] = r.zeroshot_bin
                record["sentence_bin"] = r.confidence_bin
                record["max_confidence"] = r.max_confidence
                record["zeroshot_favouring_confidence"] = r.zeroshot_favouring_confidence
                record["sentence_favouring_confidence"] = r.sentence_favouring_confidence
            records.append(record)
        return records


class ClassificationPipeline:
    """Main pipeline for taxonomy classification.

    Orchestrates the full classification workflow:
    1. Preprocess corpus (sentence splitting, keyword extraction)
    2. Generate embeddings
    3. Multi-level matching (global, sentence, keyword)
    4. Score combination and pruning
    5. Confidence binning
    6. Optional zero-shot validation
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the classification pipeline.

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        self._embedding_provider: EmbeddingProvider | None = None

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """Get or create the embedding provider."""
        if self._embedding_provider is None:
            self._embedding_provider = create_embedding_provider(
                self.config.embedding_model
            )
        return self._embedding_provider

    def classify(
        self,
        corpus: Corpus,
        taxonomy: Taxonomy,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> ClassificationResults:
        """Run the full classification pipeline.

        Args:
            corpus: Corpus with documents to classify
            taxonomy: Taxonomy to match against
            progress_callback: Optional callback(stage, current, total) for progress

        Returns:
            ClassificationResults with all matches and scores
        """
        self.config.validate()

        logger.info(
            f"Starting classification: {len(corpus)} documents, {len(taxonomy)} labels"
        )
        logger.info(
            f"Enabled: global={self.config.enable_global_matching}, "
            f"sentence={self.config.enable_sentence_matching}, "
            f"keyword={self.config.enable_keyword_matching}, "
            f"zeroshot={self.config.enable_zeroshot}"
        )

        # Build taxonomy label lookup
        taxonomy_labels = {node.id: node.full_path for node in taxonomy.nodes}

        # Step 1: Preprocess corpus
        if self.config.enable_sentence_matching and corpus.sentences is None:
            logger.info("Splitting documents into sentences")
            corpus = split_sentences(corpus)

        if self.config.enable_keyword_matching and corpus.keywords is None:
            logger.info("Extracting keywords")
            available_extractors = [
                e for e in self.config.keyword_extractors
                if self._extractor_available(e)
            ]
            if available_extractors:
                corpus = extract_and_aggregate_keywords(
                    corpus,
                    extractor_names=available_extractors,
                    min_agreement=self.config.keyword_min_agreement,
                )
            else:
                logger.warning("No keyword extractors available, skipping keyword matching")

        # Step 2: Run matching
        global_matches: MatchResult | None = None
        sentence_matches: MatchResult | None = None
        keyword_matches: MatchResult | None = None

        if self.config.enable_global_matching:
            logger.info("Running global matching")
            global_matcher = GlobalMatcher()
            global_matches = global_matcher.match(
                corpus, taxonomy, self.embedding_provider, self.config, progress_callback
            )
            if self.config.use_quantile_pruning:
                global_matches = prune_matches(
                    global_matches, self.config.prune_global_threshold, use_quantile=True
                )

        if self.config.enable_sentence_matching and corpus.sentences:
            logger.info("Running sentence matching")
            sentence_matcher = SentenceMatcher()
            sentence_matches = sentence_matcher.match(
                corpus, taxonomy, self.embedding_provider, self.config, progress_callback
            )
            if self.config.use_quantile_pruning:
                sentence_matches = prune_matches(
                    sentence_matches, self.config.prune_sentence_threshold, use_quantile=True
                )

        if self.config.enable_keyword_matching and corpus.keywords:
            logger.info("Running keyword matching")
            keyword_matcher = KeywordMatcher()
            keyword_matches = keyword_matcher.match(
                corpus, taxonomy, self.embedding_provider, self.config, progress_callback
            )
            if self.config.use_quantile_pruning:
                keyword_matches = prune_matches(
                    keyword_matches, self.config.prune_keyword_threshold, use_quantile=True
                )

        # Step 3: Combine scores
        # We need at least sentence matches as the base
        if sentence_matches is None:
            # Fall back to global matches as sentence-level if no sentence matching
            if global_matches is None:
                raise ValueError(
                    "At least one matching method must be enabled and produce results"
                )
            logger.info("Using global matches as base (no sentence matching)")
            sentence_matches = global_matches
            global_matches = None

        logger.info("Combining scores")
        combined = combine_scores(
            sentence_matches,
            global_matches,
            keyword_matches,
            sentence_weight=self.config.sentence_weight,
            global_weight=self.config.global_weight,
        )

        # Step 4: Aggregate to document-label level and assign confidence bins
        logger.info("Aggregating scores and assigning confidence bins")
        aggregated = aggregate_to_labels(combined, self.config)

        # Step 5: Optional zero-shot validation
        refined: RefinedScores | None = None
        if self.config.enable_zeroshot:
            logger.info("Running zero-shot validation")
            document_texts = {doc.id: doc.text for doc in corpus.documents}

            def zs_progress(current: int, total: int) -> None:
                if progress_callback:
                    progress_callback("Zero-shot validation", current, total)

            zeroshot = enhance_with_zeroshot(
                aggregated,
                document_texts,
                taxonomy_labels,
                self.config,
                progress_callback=zs_progress,
            )
            refined = refine_confidence_bins(aggregated, zeroshot)

        # Step 6: Build results
        results = self._build_results(aggregated, refined, taxonomy_labels)

        logger.info(f"Classification complete: {len(results)} results")

        return ClassificationResults(
            results=results,
            metadata={
                "config": self.config.to_dict(),
                "num_documents": len(corpus),
                "num_taxonomy_labels": len(taxonomy),
                "num_results": len(results),
            },
        )

    def _extractor_available(self, name: str) -> bool:
        """Check if a keyword extractor is available."""
        try:
            if name == "rake":
                import rake_nltk  # noqa: F401
            elif name == "yake":
                import yake  # noqa: F401
            elif name == "keybert":
                import keybert  # noqa: F401
            elif name == "dbpedia":
                import requests  # noqa: F401
            else:
                return False
            return True
        except ImportError:
            return False

    def _build_results(
        self,
        aggregated: AggregatedScores,
        refined: RefinedScores | None,
        taxonomy_labels: dict[str, str],
    ) -> list[ClassificationResult]:
        """Build ClassificationResult objects from aggregated scores."""
        results = []

        for i in range(len(aggregated)):
            doc_id = aggregated.document_ids[i]
            tax_id = aggregated.taxonomy_ids[i]

            result = ClassificationResult(
                document_id=doc_id,
                taxonomy_label_id=tax_id,
                taxonomy_label=taxonomy_labels.get(tax_id, tax_id),
                relevance_score=float(aggregated.relevance_score[i]),
                confidence_bin=str(aggregated.confidence_bin[i]),
                similarity_score_global=float(aggregated.similarity_score_global[i]),
                similarity_score_sent=float(aggregated.similarity_score_sent[i]),
                similarity_score_key=float(aggregated.similarity_score_key[i]),
                global_bin=str(aggregated.global_bin[i]),
                local_bin=str(aggregated.local_bin[i]),
                num_matching_sentences=int(aggregated.num_matching_sentences[i]),
                num_sentences=int(aggregated.num_sentences[i]),
            )

            if refined is not None:
                result.zeroshot_score = float(refined.zeroshot_score[i])
                result.zeroshot_bin = str(refined.zeroshot_bin[i])
                result.max_confidence = str(refined.max_confidence[i])
                result.zeroshot_favouring_confidence = str(
                    refined.zeroshot_favouring_confidence[i]
                )
                result.sentence_favouring_confidence = str(
                    refined.sentence_favouring_confidence[i]
                )

            results.append(result)

        return results
