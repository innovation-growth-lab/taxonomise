"""Confidence scoring, binning, and zero-shot validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from taxonomise.config import PipelineConfig
from taxonomise.logging import get_logger
from taxonomise.matching import CombinedScores

logger = get_logger("confidence")


# Bin ordering for comparisons
BIN_ORDER = {"very high": 4, "high": 3, "medium": 2, "low": 1, "very low": 0}
BIN_ORDER_3 = {"high": 3, "medium": 2, "low": 1}


@dataclass
class AggregatedScores:
    """Scores aggregated to document-label level.

    Attributes:
        document_ids: Document IDs
        taxonomy_ids: Taxonomy label IDs
        relevance_score: Max sentence score per doc-label pair
        similarity_score_global: Global similarity
        similarity_score_sent: Best sentence similarity
        similarity_score_key: Best keyword similarity
        num_matching_sentences: Count of matching sentences
        num_sentences: Total sentences per document
        global_bin: Confidence bin from global thresholds
        local_bin: Confidence bin from local thresholds
        confidence_bin: Combined confidence (min of global and local)
    """

    document_ids: np.ndarray
    taxonomy_ids: np.ndarray
    relevance_score: np.ndarray
    similarity_score_global: np.ndarray
    similarity_score_sent: np.ndarray
    similarity_score_key: np.ndarray
    num_matching_sentences: np.ndarray
    num_sentences: np.ndarray
    global_bin: np.ndarray
    local_bin: np.ndarray
    confidence_bin: np.ndarray

    def __len__(self) -> int:
        return len(self.document_ids)


def aggregate_to_labels(
    combined: CombinedScores,
    config: PipelineConfig,
) -> AggregatedScores:
    """Aggregate sentence-level scores to document-label pairs.

    Takes the MAX sentence score for each (document, label) pair.

    Args:
        combined: Combined scores from matching
        config: Pipeline configuration

    Returns:
        AggregatedScores with document-label level scores and bins
    """
    logger.info("Aggregating sentence scores to document-label pairs")

    # Build index: (doc_id, tax_id) -> list of indices
    pair_to_indices: dict[tuple[str, str], list[int]] = {}
    for i, (doc_id, tax_id) in enumerate(
        zip(combined.document_ids, combined.taxonomy_ids)
    ):
        key = (doc_id, tax_id)
        if key not in pair_to_indices:
            pair_to_indices[key] = []
        pair_to_indices[key].append(i)

    # Count total sentences per document
    doc_sentence_counts: dict[str, int] = {}
    seen_sentences: dict[str, set[str]] = {}
    if combined.sentence_ids is not None:
        for doc_id, sent_id in zip(combined.document_ids, combined.sentence_ids):
            if doc_id not in seen_sentences:
                seen_sentences[doc_id] = set()
            seen_sentences[doc_id].add(sent_id)
        doc_sentence_counts = {doc_id: len(sents) for doc_id, sents in seen_sentences.items()}

    # Aggregate scores
    n_pairs = len(pair_to_indices)
    doc_ids = np.empty(n_pairs, dtype=object)
    tax_ids = np.empty(n_pairs, dtype=object)
    relevance_scores = np.zeros(n_pairs)
    global_scores = np.zeros(n_pairs)
    sent_scores = np.zeros(n_pairs)
    key_scores = np.zeros(n_pairs)
    num_matching = np.zeros(n_pairs, dtype=int)
    num_sentences = np.zeros(n_pairs, dtype=int)

    for i, ((doc_id, tax_id), indices) in enumerate(pair_to_indices.items()):
        doc_ids[i] = doc_id
        tax_ids[i] = tax_id

        # Take max of sentence scores
        scores = combined.sentence_score[indices]
        max_idx = np.argmax(scores)
        relevance_scores[i] = scores[max_idx]

        # Take first global/key score (same for all sentences of this doc-label)
        global_scores[i] = combined.similarity_score_global[indices[0]]
        key_scores[i] = combined.similarity_score_key[indices[0]]
        sent_scores[i] = combined.similarity_score_sent[indices[max_idx]]

        # Count matching sentences
        if combined.sentence_ids is not None:
            unique_sents = set(combined.sentence_ids[indices])
            num_matching[i] = len(unique_sents)

        # Total sentences for this document
        num_sentences[i] = doc_sentence_counts.get(doc_id, len(indices))

    # Apply normalization if configured
    if config.normalise_by_matches:
        logger.info("Normalising scores by matching sentence ratio")
        ratio = num_matching / np.maximum(num_sentences, 1)
        relevance_scores = relevance_scores * ratio

    # Assign global bins
    global_bin = assign_global_bins(
        relevance_scores,
        q2_threshold=config.global_q2_threshold,
        q3_threshold=config.global_q3_threshold,
    )

    # Assign local bins (per-document)
    local_bin = assign_local_bins(
        doc_ids,
        relevance_scores,
        q2_threshold=config.local_q2_threshold,
        q3_threshold=config.local_q3_threshold,
    )

    # Combine bins (take minimum)
    confidence_bin = combine_bins(global_bin, local_bin)

    logger.info(f"Aggregated to {n_pairs} document-label pairs")

    return AggregatedScores(
        document_ids=doc_ids,
        taxonomy_ids=tax_ids,
        relevance_score=relevance_scores,
        similarity_score_global=global_scores,
        similarity_score_sent=sent_scores,
        similarity_score_key=key_scores,
        num_matching_sentences=num_matching,
        num_sentences=num_sentences,
        global_bin=global_bin,
        local_bin=local_bin,
        confidence_bin=confidence_bin,
    )


def assign_global_bins(
    scores: np.ndarray,
    q2_threshold: float = 0.5,
    q3_threshold: float = 0.75,
) -> np.ndarray:
    """Assign confidence bins based on global quantile thresholds.

    Args:
        scores: Relevance scores
        q2_threshold: Quantile threshold for medium confidence
        q3_threshold: Quantile threshold for high confidence

    Returns:
        Array of bin labels ("high", "medium", "low")
    """
    q2 = np.quantile(scores, q2_threshold)
    q3 = np.quantile(scores, q3_threshold)

    bins = np.where(
        scores > q3,
        "high",
        np.where(scores > q2, "medium", "low"),
    )

    logger.debug(f"Global bins: q2={q2:.4f}, q3={q3:.4f}")

    return bins


def assign_local_bins(
    document_ids: np.ndarray,
    scores: np.ndarray,
    q2_threshold: float = 0.5,
    q3_threshold: float = 0.75,
) -> np.ndarray:
    """Assign confidence bins based on per-document thresholds.

    For each document, bins are computed as the minimum of:
    1. Quantile-based bins (using local score distribution)
    2. Dropoff-based bins (using relative gaps between scores)

    Edge cases:
    - 1 label: Always "high"
    - 2 labels: First is "high", second is "medium"

    Args:
        document_ids: Document IDs
        scores: Relevance scores
        q2_threshold: Quantile threshold for medium confidence
        q3_threshold: Quantile threshold for high confidence

    Returns:
        Array of bin labels ("high", "medium", "low")
    """
    bins = np.empty(len(scores), dtype=object)

    # Group by document
    unique_docs = np.unique(document_ids)

    for doc_id in unique_docs:
        mask = document_ids == doc_id
        doc_scores = scores[mask]
        n_labels = len(doc_scores)

        # Edge cases
        if n_labels == 1:
            bins[mask] = "high"
            continue
        elif n_labels == 2:
            sorted_idx = np.argsort(-doc_scores)
            local_bins = np.array(["high", "medium"])
            # Map back to original order
            inverse_idx = np.argsort(sorted_idx)
            bins[mask] = local_bins[inverse_idx]
            continue

        # Sort scores descending
        sorted_idx = np.argsort(-doc_scores)
        sorted_scores = doc_scores[sorted_idx]

        # Quantile-based bins
        q2_local = np.quantile(sorted_scores, q2_threshold)
        q3_local = np.quantile(sorted_scores, q3_threshold)

        quantile_bins = np.where(
            sorted_scores > q3_local,
            "high",
            np.where(sorted_scores > q2_local, "medium", "low"),
        )

        # Dropoff-based bins (relative gap detection)
        dropoff_bins = _compute_dropoff_bins(sorted_scores)

        # Take minimum of both methods
        local_bins = _min_bins(quantile_bins, dropoff_bins)

        # Map back to original order
        inverse_idx = np.argsort(sorted_idx)
        bins[mask] = local_bins[inverse_idx]

    return bins


def _compute_dropoff_bins(sorted_scores: np.ndarray) -> np.ndarray:
    """Compute bins based on relative gaps between scores.

    1. Compute relative gaps: (score[i-1] - score[i]) / score[i-1]
    2. Find positions of two largest gaps (i*, j*)
    3. Assign bins: 0 to i* -> high, i*+1 to j* -> medium, j*+1 onwards -> low

    Args:
        sorted_scores: Scores sorted in descending order

    Returns:
        Array of bin labels
    """
    n = len(sorted_scores)
    bins = np.full(n, "low", dtype=object)

    if n <= 2:
        bins[:] = ["high", "medium"][:n]
        return bins

    # Compute relative gaps
    gaps = []
    for i in range(1, n):
        prev_score = sorted_scores[i - 1]
        curr_score = sorted_scores[i]
        if prev_score > 0:
            relative_gap = (prev_score - curr_score) / prev_score
        else:
            relative_gap = 0
        gaps.append((i - 1, relative_gap))  # Position of the score BEFORE the gap

    # Find two largest gaps
    gaps.sort(key=lambda x: x[1], reverse=True)

    if len(gaps) >= 2:
        gap_positions = sorted([gaps[0][0], gaps[1][0]])
        i_star, j_star = gap_positions[0], gap_positions[1]

        bins[: i_star + 1] = "high"
        bins[i_star + 1 : j_star + 1] = "medium"
        # j_star + 1 onwards stays "low"
    elif len(gaps) == 1:
        i_star = gaps[0][0]
        bins[: i_star + 1] = "high"
        bins[i_star + 1 :] = "medium"

    return bins


def _min_bins(bins1: np.ndarray, bins2: np.ndarray) -> np.ndarray:
    """Take element-wise minimum of two bin arrays."""
    result = np.empty(len(bins1), dtype=object)
    for i, (b1, b2) in enumerate(zip(bins1, bins2)):
        v1 = BIN_ORDER_3.get(b1, 0)
        v2 = BIN_ORDER_3.get(b2, 0)
        min_val = min(v1, v2)
        result[i] = {3: "high", 2: "medium", 1: "low"}[min_val]
    return result


def combine_bins(global_bin: np.ndarray, local_bin: np.ndarray) -> np.ndarray:
    """Combine global and local bins by taking the minimum.

    Args:
        global_bin: Global confidence bins
        local_bin: Local confidence bins

    Returns:
        Combined confidence bins
    """
    return _min_bins(global_bin, local_bin)


@dataclass
class ZeroshotScores:
    """Zero-shot validation scores.

    Attributes:
        document_ids: Document IDs
        taxonomy_ids: Taxonomy label IDs
        zeroshot_score: Classification confidence score
        zeroshot_bin: 5-level confidence bin
    """

    document_ids: np.ndarray
    taxonomy_ids: np.ndarray
    zeroshot_score: np.ndarray
    zeroshot_bin: np.ndarray


def enhance_with_zeroshot(
    aggregated: AggregatedScores,
    document_texts: dict[str, str],
    taxonomy_labels: dict[str, str],
    config: PipelineConfig,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ZeroshotScores:
    """Enhance scores using zero-shot NLI classification.

    Args:
        aggregated: Aggregated scores
        document_texts: Mapping of document ID to text
        taxonomy_labels: Mapping of taxonomy ID to label
        config: Pipeline configuration
        progress_callback: Optional callback(current, total)

    Returns:
        ZeroshotScores with NLI validation results
    """
    try:
        from transformers import pipeline as hf_pipeline
        import torch
    except ImportError:
        raise ImportError(
            "transformers and torch are required for zero-shot validation. "
            "Install with: pip install taxonomise[zeroshot]"
        )

    logger.info(f"Initializing zero-shot classifier: {config.zeroshot_model}")

    # Initialize classifier
    device = 0 if torch.cuda.is_available() else -1
    classifier = hf_pipeline(
        "zero-shot-classification",
        model=config.zeroshot_model,
        multi_label=True,
        batch_size=config.batch_size,
        truncation=True,
        device=device,
    )

    # Group labels by document
    doc_to_labels: dict[str, list[tuple[int, str]]] = {}
    for i, (doc_id, tax_id) in enumerate(
        zip(aggregated.document_ids, aggregated.taxonomy_ids)
    ):
        if doc_id not in doc_to_labels:
            doc_to_labels[doc_id] = []
        label = taxonomy_labels.get(tax_id, tax_id)
        doc_to_labels[doc_id].append((i, label))

    # Process each document
    n_docs = len(doc_to_labels)
    zeroshot_scores = np.zeros(len(aggregated))

    for doc_idx, (doc_id, label_info) in enumerate(doc_to_labels.items()):
        text = document_texts.get(doc_id, "")
        if not text:
            continue

        indices = [i for i, _ in label_info]
        labels = [label for _, label in label_info]

        try:
            result = classifier(
                text,
                candidate_labels=labels,
                hypothesis_template=config.zeroshot_hypothesis_template,
            )

            # Map scores back to indices
            label_to_score = dict(zip(result["labels"], result["scores"]))
            for idx, label in zip(indices, labels):
                zeroshot_scores[idx] = label_to_score.get(label, 0.0)

        except Exception as e:
            logger.warning(f"Zero-shot failed for doc {doc_id}: {e}")

        if progress_callback:
            progress_callback(doc_idx + 1, n_docs)

        # Clear GPU memory periodically
        if (doc_idx + 1) % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Assign 5-level bins
    zeroshot_bin = np.where(
        zeroshot_scores >= 0.9,
        "very high",
        np.where(
            zeroshot_scores >= 0.7,
            "high",
            np.where(
                zeroshot_scores >= 0.5,
                "medium",
                np.where(zeroshot_scores >= 0.25, "low", "very low"),
            ),
        ),
    )

    logger.info(f"Zero-shot validation complete for {n_docs} documents")

    return ZeroshotScores(
        document_ids=aggregated.document_ids,
        taxonomy_ids=aggregated.taxonomy_ids,
        zeroshot_score=zeroshot_scores,
        zeroshot_bin=zeroshot_bin,
    )


@dataclass
class RefinedScores:
    """Final refined scores with all confidence measures.

    Includes both sentence-based and zero-shot confidence bins,
    plus combined measures.
    """

    document_ids: np.ndarray
    taxonomy_ids: np.ndarray
    relevance_score: np.ndarray
    similarity_score_global: np.ndarray
    similarity_score_sent: np.ndarray
    similarity_score_key: np.ndarray
    num_matching_sentences: np.ndarray
    num_sentences: np.ndarray
    global_bin: np.ndarray
    local_bin: np.ndarray
    sentence_bin: np.ndarray  # Renamed from confidence_bin
    zeroshot_score: np.ndarray
    zeroshot_bin: np.ndarray
    max_confidence: np.ndarray
    zeroshot_favouring_confidence: np.ndarray
    sentence_favouring_confidence: np.ndarray

    def __len__(self) -> int:
        return len(self.document_ids)


def refine_confidence_bins(
    aggregated: AggregatedScores,
    zeroshot: ZeroshotScores,
) -> RefinedScores:
    """Refine confidence bins using both sentence and zero-shot scores.

    Args:
        aggregated: Aggregated scores with sentence-based bins
        zeroshot: Zero-shot validation scores

    Returns:
        RefinedScores with multiple confidence measures
    """
    logger.info("Refining confidence bins from sentence and zero-shot scores")

    n = len(aggregated)

    # Compute bin distances and combinations
    max_confidence = np.empty(n, dtype=object)
    zeroshot_favouring = np.empty(n, dtype=object)
    sentence_favouring = np.empty(n, dtype=object)

    for i in range(n):
        sent_bin = aggregated.confidence_bin[i]
        zero_bin = zeroshot.zeroshot_bin[i]

        sent_val = BIN_ORDER.get(sent_bin, 0)
        zero_val = BIN_ORDER.get(zero_bin, 0)

        # Max confidence
        if sent_val > zero_val:
            max_confidence[i] = sent_bin
        else:
            max_confidence[i] = zero_bin

        # Distance
        distance = abs(sent_val - zero_val)

        # Zeroshot favouring: use zeroshot when big disagreement
        if distance > 1:
            zeroshot_favouring[i] = zero_bin
        else:
            zeroshot_favouring[i] = max_confidence[i]

        # Sentence favouring: use sentence when big disagreement
        if distance > 1:
            sentence_favouring[i] = sent_bin
        else:
            sentence_favouring[i] = max_confidence[i]

    return RefinedScores(
        document_ids=aggregated.document_ids,
        taxonomy_ids=aggregated.taxonomy_ids,
        relevance_score=aggregated.relevance_score,
        similarity_score_global=aggregated.similarity_score_global,
        similarity_score_sent=aggregated.similarity_score_sent,
        similarity_score_key=aggregated.similarity_score_key,
        num_matching_sentences=aggregated.num_matching_sentences,
        num_sentences=aggregated.num_sentences,
        global_bin=aggregated.global_bin,
        local_bin=aggregated.local_bin,
        sentence_bin=aggregated.confidence_bin,
        zeroshot_score=zeroshot.zeroshot_score,
        zeroshot_bin=zeroshot.zeroshot_bin,
        max_confidence=max_confidence,
        zeroshot_favouring_confidence=zeroshot_favouring,
        sentence_favouring_confidence=sentence_favouring,
    )
