"""Multi-level matching strategies for taxonomy classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from taxonomise.config import PipelineConfig
from taxonomise.data import Corpus, Taxonomy
from taxonomise.embeddings import EmbeddingProvider
from taxonomise.logging import get_logger
from taxonomise.similarity import BatchSimilarityResult, search_batch

logger = get_logger("matching")


@dataclass
class MatchResult:
    """Result from a matching operation.

    Attributes:
        document_ids: Document IDs
        taxonomy_ids: Matched taxonomy label IDs
        similarity_scores: Similarity scores
        sentence_ids: Optional sentence IDs (for sentence-level matching)
        keyword_ids: Optional keyword IDs (for keyword-level matching)
    """

    document_ids: np.ndarray
    taxonomy_ids: np.ndarray
    similarity_scores: np.ndarray
    sentence_ids: np.ndarray | None = None
    keyword_ids: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.document_ids)


class Matcher(Protocol):
    """Protocol for matching strategies."""

    def match(
        self,
        corpus: Corpus,
        taxonomy: Taxonomy,
        embedding_provider: EmbeddingProvider,
        config: PipelineConfig,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> MatchResult:
        """Perform matching.

        Args:
            corpus: Corpus with documents
            taxonomy: Taxonomy to match against
            embedding_provider: Provider for generating embeddings
            config: Pipeline configuration
            progress_callback: Optional callback(stage, current, total)

        Returns:
            MatchResult with matches
        """
        ...


class GlobalMatcher:
    """Document-level matching against taxonomy labels."""

    def match(
        self,
        corpus: Corpus,
        taxonomy: Taxonomy,
        embedding_provider: EmbeddingProvider,
        config: PipelineConfig,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> MatchResult:
        """Match full documents against taxonomy labels.

        Args:
            corpus: Corpus with documents
            taxonomy: Taxonomy to match against
            embedding_provider: Provider for generating embeddings
            config: Pipeline configuration
            progress_callback: Optional callback(stage, current, total)

        Returns:
            MatchResult with document-level matches
        """
        logger.info(f"Global matching: {len(corpus)} documents against {len(taxonomy)} labels")

        # Get or compute document embeddings
        if corpus.embeddings is None:
            texts = corpus.get_texts()

            def embed_progress(current: int, total: int) -> None:
                if progress_callback:
                    progress_callback("Embedding documents", current, total)

            corpus.embeddings = embedding_provider.encode(
                texts,
                batch_size=config.batch_size,
                normalize=True,
                progress_callback=embed_progress,
            )

        # Get or compute taxonomy embeddings
        if taxonomy.embeddings is None:
            labels = taxonomy.get_labels()

            def tax_progress(current: int, total: int) -> None:
                if progress_callback:
                    progress_callback("Embedding taxonomy", current, total)

            taxonomy.embeddings = embedding_provider.encode(
                labels,
                batch_size=config.batch_size,
                normalize=True,
                progress_callback=tax_progress,
            )

        # Search for matches
        def search_progress(current: int, total: int) -> None:
            if progress_callback:
                progress_callback("Global matching", current, total)

        result = search_batch(
            query_embeddings=corpus.embeddings,
            query_ids=corpus.get_ids(),
            target_embeddings=taxonomy.embeddings,
            target_ids=taxonomy.get_ids(),
            batch_size=config.batch_size,
            top_n=config.top_n_matches,
            normalized=True,
            progress_callback=search_progress,
        )

        logger.info(f"Global matching complete: {len(result)} matches")

        return MatchResult(
            document_ids=result.query_ids,
            taxonomy_ids=result.target_ids,
            similarity_scores=result.similarities,
        )


class SentenceMatcher:
    """Sentence-level matching against taxonomy labels."""

    def match(
        self,
        corpus: Corpus,
        taxonomy: Taxonomy,
        embedding_provider: EmbeddingProvider,
        config: PipelineConfig,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> MatchResult:
        """Match sentences against taxonomy labels.

        Args:
            corpus: Corpus with documents (must have sentences populated)
            taxonomy: Taxonomy to match against
            embedding_provider: Provider for generating embeddings
            config: Pipeline configuration
            progress_callback: Optional callback(stage, current, total)

        Returns:
            MatchResult with sentence-level matches
        """
        if corpus.sentences is None:
            raise ValueError("Corpus must have sentences populated for sentence matching")

        logger.info(
            f"Sentence matching: {len(corpus.sentences)} sentences "
            f"against {len(taxonomy)} labels"
        )

        # Get sentence texts and create ID mappings
        sentence_texts = corpus.get_sentence_texts()
        sentence_ids = [sent.id for sent in corpus.sentences]
        sentence_to_doc = {sent.id: sent.document_id for sent in corpus.sentences}

        # Compute sentence embeddings
        if corpus.sentence_embeddings is None:

            def embed_progress(current: int, total: int) -> None:
                if progress_callback:
                    progress_callback("Embedding sentences", current, total)

            corpus.sentence_embeddings = embedding_provider.encode(
                sentence_texts,
                batch_size=config.batch_size,
                normalize=True,
                progress_callback=embed_progress,
            )

        # Get or compute taxonomy embeddings
        if taxonomy.embeddings is None:
            labels = taxonomy.get_labels()

            def tax_progress(current: int, total: int) -> None:
                if progress_callback:
                    progress_callback("Embedding taxonomy", current, total)

            taxonomy.embeddings = embedding_provider.encode(
                labels,
                batch_size=config.batch_size,
                normalize=True,
                progress_callback=tax_progress,
            )

        # Search for matches
        def search_progress(current: int, total: int) -> None:
            if progress_callback:
                progress_callback("Sentence matching", current, total)

        result = search_batch(
            query_embeddings=corpus.sentence_embeddings,
            query_ids=sentence_ids,
            target_embeddings=taxonomy.embeddings,
            target_ids=taxonomy.get_ids(),
            batch_size=config.batch_size,
            top_n=config.top_n_matches,
            normalized=True,
            progress_callback=search_progress,
        )

        # Map sentence IDs back to document IDs
        document_ids = np.array([sentence_to_doc[sid] for sid in result.query_ids])

        logger.info(f"Sentence matching complete: {len(result)} matches")

        return MatchResult(
            document_ids=document_ids,
            taxonomy_ids=result.target_ids,
            similarity_scores=result.similarities,
            sentence_ids=result.query_ids,
        )


class KeywordMatcher:
    """Keyword-level matching against taxonomy labels."""

    def match(
        self,
        corpus: Corpus,
        taxonomy: Taxonomy,
        embedding_provider: EmbeddingProvider,
        config: PipelineConfig,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> MatchResult:
        """Match keywords against taxonomy labels.

        Args:
            corpus: Corpus with documents (must have keywords populated)
            taxonomy: Taxonomy to match against
            embedding_provider: Provider for generating embeddings
            config: Pipeline configuration
            progress_callback: Optional callback(stage, current, total)

        Returns:
            MatchResult with keyword-level matches (expanded to documents)
        """
        if corpus.keywords is None:
            raise ValueError("Corpus must have keywords populated for keyword matching")

        logger.info(
            f"Keyword matching: {len(corpus.keywords)} keywords "
            f"against {len(taxonomy)} labels"
        )

        # Get keyword texts
        keyword_texts = [kw.keyword for kw in corpus.keywords]
        keyword_ids = [kw.id for kw in corpus.keywords]
        keyword_to_docs = {kw.id: kw.document_ids for kw in corpus.keywords}

        # Compute keyword embeddings
        def embed_progress(current: int, total: int) -> None:
            if progress_callback:
                progress_callback("Embedding keywords", current, total)

        keyword_embeddings = embedding_provider.encode(
            keyword_texts,
            batch_size=config.batch_size,
            normalize=True,
            progress_callback=embed_progress,
        )

        # Get or compute taxonomy embeddings
        if taxonomy.embeddings is None:
            labels = taxonomy.get_labels()

            def tax_progress(current: int, total: int) -> None:
                if progress_callback:
                    progress_callback("Embedding taxonomy", current, total)

            taxonomy.embeddings = embedding_provider.encode(
                labels,
                batch_size=config.batch_size,
                normalize=True,
                progress_callback=tax_progress,
            )

        # Search for matches
        def search_progress(current: int, total: int) -> None:
            if progress_callback:
                progress_callback("Keyword matching", current, total)

        result = search_batch(
            query_embeddings=keyword_embeddings,
            query_ids=keyword_ids,
            target_embeddings=taxonomy.embeddings,
            target_ids=taxonomy.get_ids(),
            batch_size=config.batch_size,
            top_n=config.top_n_matches,
            normalized=True,
            progress_callback=search_progress,
        )

        # Expand keyword matches to document matches
        # Each keyword match applies to all documents containing that keyword
        expanded_doc_ids = []
        expanded_tax_ids = []
        expanded_scores = []
        expanded_kw_ids = []

        for kw_id, tax_id, score in zip(
            result.query_ids, result.target_ids, result.similarities
        ):
            doc_ids = keyword_to_docs[kw_id]
            for doc_id in doc_ids:
                expanded_doc_ids.append(doc_id)
                expanded_tax_ids.append(tax_id)
                expanded_scores.append(score)
                expanded_kw_ids.append(kw_id)

        logger.info(
            f"Keyword matching complete: {len(result)} keyword matches "
            f"-> {len(expanded_doc_ids)} document matches"
        )

        return MatchResult(
            document_ids=np.array(expanded_doc_ids),
            taxonomy_ids=np.array(expanded_tax_ids),
            similarity_scores=np.array(expanded_scores),
            keyword_ids=np.array(expanded_kw_ids),
        )


def prune_matches(
    matches: MatchResult,
    threshold: float,
    use_quantile: bool = True,
) -> MatchResult:
    """Prune low-scoring matches.

    Args:
        matches: Matches to prune
        threshold: Threshold value (quantile or absolute)
        use_quantile: If True, threshold is a quantile (0-1)

    Returns:
        Pruned MatchResult
    """
    if len(matches) == 0:
        return matches

    if use_quantile:
        cutoff = np.quantile(matches.similarity_scores, threshold)
    else:
        cutoff = threshold

    mask = matches.similarity_scores >= cutoff

    logger.debug(
        f"Pruning: {len(matches)} -> {mask.sum()} matches "
        f"(threshold={cutoff:.4f}, kept {100 * mask.mean():.1f}%)"
    )

    return MatchResult(
        document_ids=matches.document_ids[mask],
        taxonomy_ids=matches.taxonomy_ids[mask],
        similarity_scores=matches.similarity_scores[mask],
        sentence_ids=matches.sentence_ids[mask] if matches.sentence_ids is not None else None,
        keyword_ids=matches.keyword_ids[mask] if matches.keyword_ids is not None else None,
    )


@dataclass
class CombinedScores:
    """Combined scores from multiple matching levels.

    Attributes:
        document_ids: Document IDs
        sentence_ids: Sentence IDs (for sentence-level granularity)
        taxonomy_ids: Taxonomy label IDs
        similarity_score_sent: Sentence-level similarity
        similarity_score_global: Global (document) similarity
        similarity_score_key: Keyword similarity
        sentence_score: Combined weighted score
    """

    document_ids: np.ndarray
    sentence_ids: np.ndarray
    taxonomy_ids: np.ndarray
    similarity_score_sent: np.ndarray
    similarity_score_global: np.ndarray
    similarity_score_key: np.ndarray
    sentence_score: np.ndarray

    def __len__(self) -> int:
        return len(self.document_ids)


def combine_scores(
    sentence_matches: MatchResult,
    global_matches: MatchResult | None,
    keyword_matches: MatchResult | None,
    sentence_weight: float = 0.5,
    global_weight: float = 0.3,
) -> CombinedScores:
    """Combine scores from multiple matching levels.

    The combined sentence score is:
        sentence_weight * sent_score + global_weight * global_score + keyword_weight * key_score

    where keyword_weight = 1 - sentence_weight - global_weight

    Args:
        sentence_matches: Sentence-level matches (required)
        global_matches: Global (document-level) matches (optional)
        keyword_matches: Keyword-level matches (optional)
        sentence_weight: Weight for sentence scores
        global_weight: Weight for global scores

    Returns:
        CombinedScores with weighted combination
    """
    keyword_weight = 1.0 - sentence_weight - global_weight

    logger.info(
        f"Combining scores with weights: "
        f"sentence={sentence_weight:.2f}, global={global_weight:.2f}, keyword={keyword_weight:.2f}"
    )

    # Start with sentence matches as the base
    n_matches = len(sentence_matches)

    # Create arrays for combined scores
    similarity_global = np.zeros(n_matches)
    similarity_key = np.zeros(n_matches)

    # Build lookup for global scores: (doc_id, tax_id) -> score
    global_lookup: dict[tuple[str, str], float] = {}
    if global_matches is not None:
        for doc_id, tax_id, score in zip(
            global_matches.document_ids,
            global_matches.taxonomy_ids,
            global_matches.similarity_scores,
        ):
            key = (doc_id, tax_id)
            # Keep max score for each (doc, tax) pair
            if key not in global_lookup or score > global_lookup[key]:
                global_lookup[key] = score

    # Build lookup for keyword scores: (doc_id, tax_id) -> max_score
    keyword_lookup: dict[tuple[str, str], float] = {}
    if keyword_matches is not None:
        for doc_id, tax_id, score in zip(
            keyword_matches.document_ids,
            keyword_matches.taxonomy_ids,
            keyword_matches.similarity_scores,
        ):
            key = (doc_id, tax_id)
            # Keep max score for each (doc, tax) pair
            if key not in keyword_lookup or score > keyword_lookup[key]:
                keyword_lookup[key] = score

    # Fill in global and keyword scores for each sentence match
    for i, (doc_id, tax_id) in enumerate(
        zip(sentence_matches.document_ids, sentence_matches.taxonomy_ids)
    ):
        key = (doc_id, tax_id)
        if key in global_lookup:
            similarity_global[i] = global_lookup[key]
        if key in keyword_lookup:
            similarity_key[i] = keyword_lookup[key]

    # Compute combined sentence score
    sentence_score = (
        sentence_weight * sentence_matches.similarity_scores
        + global_weight * similarity_global
        + keyword_weight * similarity_key
    )

    logger.info(f"Combined {n_matches} matches")

    return CombinedScores(
        document_ids=sentence_matches.document_ids,
        sentence_ids=sentence_matches.sentence_ids,
        taxonomy_ids=sentence_matches.taxonomy_ids,
        similarity_score_sent=sentence_matches.similarity_scores,
        similarity_score_global=similarity_global,
        similarity_score_key=similarity_key,
        sentence_score=sentence_score,
    )
