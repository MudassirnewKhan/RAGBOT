"""
Embedding service.
Loads the SentenceTransformer model once and provides vector encoding utilities.
"""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL


class EmbeddingService:
    """Singleton-style embedding service for dense vector representation."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or EMBEDDING_MODEL
        print(f"[INFO] Initializing Embedding Model: {self.model_name}...")
        try:
            # Try offline cached loading first to prevent network retry timeouts
            self.model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(self.model_name)
        try:
            self.dimension = self.model.get_embedding_dimension()
        except AttributeError:
            self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[INFO] Embedding Model active. Embedding dimension: {self.dimension}")

    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Encode a list of texts into dense vectors (float32).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string into a 1D vector (float32).
        """
        return self.encode([query])[0]
