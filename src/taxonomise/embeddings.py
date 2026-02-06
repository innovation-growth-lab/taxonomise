"""Embedding generation using SentenceTransformers."""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from taxonomise.logging import get_logger

logger = get_logger("embeddings")


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    @property
    def model_name(self) -> str:
        """Return the model name."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        ...

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Embeddings array of shape (len(texts), embedding_dim)
        """
        ...


class SentenceTransformerProvider:
    """Embedding provider using SentenceTransformers.

    Attributes:
        model_name: Name of the SentenceTransformer model
        embedding_dim: Dimension of the embeddings
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the embedding provider.

        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._embedding_dim = self._model.get_sentence_embedding_dimension()

        logger.info(f"Loaded embedding model: {model_name} (dim={self._embedding_dim})")

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self._embedding_dim

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Embeddings array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)

        logger.debug(f"Encoding {len(texts)} texts with batch_size={batch_size}")

        # Process in batches with progress tracking
        all_embeddings = []
        total = len(texts)

        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch_texts = texts[start_idx:end_idx]

            # Encode batch
            batch_embeddings = self._model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
            )

            all_embeddings.append(batch_embeddings)

            if progress_callback:
                progress_callback(end_idx, total)

            logger.debug(f"Encoded batch {start_idx}-{end_idx} of {total}")

        embeddings = np.vstack(all_embeddings)

        logger.info(f"Encoded {len(texts)} texts -> shape {embeddings.shape}")

        return embeddings


def create_embedding_provider(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingProvider:
    """Create an embedding provider.

    Args:
        model_name: Name of the embedding model

    Returns:
        EmbeddingProvider instance
    """
    return SentenceTransformerProvider(model_name)
