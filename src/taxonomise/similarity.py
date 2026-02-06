"""Vectorized similarity computation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from taxonomise.logging import get_logger

logger = get_logger("similarity")


@dataclass
class SimilarityMatch:
    """A single similarity match result.

    Attributes:
        query_id: ID of the query item
        target_id: ID of the matched target
        similarity: Cosine similarity score
    """

    query_id: str
    target_id: str
    similarity: float


@dataclass
class BatchSimilarityResult:
    """Results from batch similarity search.

    Attributes:
        query_ids: IDs of query items
        target_ids: IDs of matched targets
        similarities: Similarity scores
    """

    query_ids: np.ndarray
    target_ids: np.ndarray
    similarities: np.ndarray

    def __len__(self) -> int:
        return len(self.query_ids)


def compute_cosine_similarity(
    query_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    normalized: bool = True,
) -> np.ndarray:
    """Compute cosine similarity between query and target embeddings.

    Args:
        query_embeddings: Query embeddings of shape (n_queries, dim)
        target_embeddings: Target embeddings of shape (n_targets, dim)
        normalized: Whether embeddings are already L2-normalized

    Returns:
        Similarity matrix of shape (n_queries, n_targets)
    """
    if not normalized:
        # Normalize embeddings
        query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        target_norms = np.linalg.norm(target_embeddings, axis=1, keepdims=True)

        # Avoid division by zero
        query_norms = np.maximum(query_norms, 1e-10)
        target_norms = np.maximum(target_norms, 1e-10)

        query_embeddings = query_embeddings / query_norms
        target_embeddings = target_embeddings / target_norms

    # Compute similarity via matrix multiplication
    similarities = query_embeddings @ target_embeddings.T

    return similarities


def search_top_n(
    query_embeddings: np.ndarray,
    query_ids: list[str] | np.ndarray,
    target_embeddings: np.ndarray,
    target_ids: list[str] | np.ndarray,
    top_n: int = 10,
    normalized: bool = True,
) -> BatchSimilarityResult:
    """Find top-N most similar targets for each query.

    Args:
        query_embeddings: Query embeddings of shape (n_queries, dim)
        query_ids: IDs for each query
        target_embeddings: Target embeddings of shape (n_targets, dim)
        target_ids: IDs for each target
        top_n: Number of top matches to return per query
        normalized: Whether embeddings are already L2-normalized

    Returns:
        BatchSimilarityResult with matches
    """
    query_ids = np.asarray(query_ids)
    target_ids = np.asarray(target_ids)

    n_queries = query_embeddings.shape[0]
    n_targets = target_embeddings.shape[0]

    # Adjust top_n if larger than number of targets
    top_n = min(top_n, n_targets)

    # Compute similarity matrix
    similarities = compute_cosine_similarity(
        query_embeddings, target_embeddings, normalized=normalized
    )

    # Get top-N indices for each query using argpartition (faster than argsort)
    if top_n < n_targets:
        top_indices = np.argpartition(-similarities, top_n, axis=1)[:, :top_n]
    else:
        top_indices = np.tile(np.arange(n_targets), (n_queries, 1))

    # Gather results
    result_query_ids = []
    result_target_ids = []
    result_similarities = []

    for i, qid in enumerate(query_ids):
        idx = top_indices[i]
        scores = similarities[i, idx]

        # Sort by score within top-N
        sort_order = np.argsort(-scores)
        idx = idx[sort_order]
        scores = scores[sort_order]

        result_query_ids.extend([qid] * len(idx))
        result_target_ids.extend(target_ids[idx])
        result_similarities.extend(scores)

    return BatchSimilarityResult(
        query_ids=np.array(result_query_ids),
        target_ids=np.array(result_target_ids),
        similarities=np.array(result_similarities),
    )


def search_batch(
    query_embeddings: np.ndarray,
    query_ids: list[str] | np.ndarray,
    target_embeddings: np.ndarray,
    target_ids: list[str] | np.ndarray,
    batch_size: int = 1000,
    top_n: int = 10,
    normalized: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> BatchSimilarityResult:
    """Batch similarity search with memory-efficient processing.

    Args:
        query_embeddings: Query embeddings of shape (n_queries, dim)
        query_ids: IDs for each query
        target_embeddings: Target embeddings of shape (n_targets, dim)
        target_ids: IDs for each target
        batch_size: Number of queries to process per batch
        top_n: Number of top matches to return per query
        normalized: Whether embeddings are already L2-normalized
        progress_callback: Optional callback(current, total) for progress

    Returns:
        BatchSimilarityResult with all matches
    """
    query_ids = np.asarray(query_ids)
    target_ids = np.asarray(target_ids)

    n_queries = query_embeddings.shape[0]
    total_batches = (n_queries + batch_size - 1) // batch_size

    logger.debug(
        f"Batch search: {n_queries} queries, {len(target_ids)} targets, "
        f"{total_batches} batches of size {batch_size}"
    )

    all_query_ids = []
    all_target_ids = []
    all_similarities = []

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_queries)

        batch_result = search_top_n(
            query_embeddings[start:end],
            query_ids[start:end],
            target_embeddings,
            target_ids,
            top_n=top_n,
            normalized=normalized,
        )

        all_query_ids.append(batch_result.query_ids)
        all_target_ids.append(batch_result.target_ids)
        all_similarities.append(batch_result.similarities)

        if progress_callback:
            progress_callback(end, n_queries)

    return BatchSimilarityResult(
        query_ids=np.concatenate(all_query_ids),
        target_ids=np.concatenate(all_target_ids),
        similarities=np.concatenate(all_similarities),
    )
