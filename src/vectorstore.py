"""
FAISS Vector Store with Cosine Similarity and JSON Metadata.
Replaces insecure pickle storage with structured JSON serialization.
"""
import os
import json
from typing import List, Dict, Any
import faiss
import numpy as np
from src.config import FAISS_DIR


class VectorStore:
    """FAISS-backed vector store normalized for Cosine Similarity search."""

    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or FAISS_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index: faiss.IndexFlatIP = None
        self.metadata: List[Dict[str, Any]] = []

    def build(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """
        Build a FAISS IndexFlatIP index with L2-normalized vectors (cosine similarity).
        """
        if embeddings.shape[0] == 0:
            raise ValueError("Cannot build index from empty embeddings array.")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)

        # In-place L2 normalization transforms inner product into cosine similarity
        norm_embeddings = embeddings.copy()
        faiss.normalize_L2(norm_embeddings)
        self.index.add(norm_embeddings)
        self.metadata = metadata_list

        self.save()
        print(f"[INFO] Built VectorStore with {self.index.ntotal} vectors (dim={dim}).")

    def save(self):
        """Persist index binary and metadata JSON to disk."""
        if self.index is None:
            return

        index_file = os.path.join(self.persist_dir, "faiss.index")
        metadata_file = os.path.join(self.persist_dir, "metadata.json")

        faiss.write_index(self.index, index_file)
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2, default=str)

        print(f"[INFO] Saved vector store to {self.persist_dir}")

    def load(self):
        """Load index binary and metadata JSON from disk."""
        index_file = os.path.join(self.persist_dir, "faiss.index")
        metadata_file = os.path.join(self.persist_dir, "metadata.json")

        if not (os.path.exists(index_file) and os.path.exists(metadata_file)):
            raise FileNotFoundError(f"Vector store files missing in {self.persist_dir}")

        self.index = faiss.read_index(index_file)
        with open(metadata_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"[INFO] Loaded vector store: {self.index.ntotal} vectors indexed.")

    def exists(self) -> bool:
        """Check if both the index and metadata files exist on disk."""
        index_file = os.path.join(self.persist_dir, "faiss.index")
        metadata_file = os.path.join(self.persist_dir, "metadata.json")
        return os.path.isfile(index_file) and os.path.isfile(metadata_file)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search vector store with normalized query embedding for top_k matches.
        """
        if self.index is None:
            return []

        # Prepare 2D float32 normalized vector
        query_vec = query_embedding.reshape(1, -1).astype("float32").copy()
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)
        results: List[Dict[str, Any]] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            entry = self.metadata[idx]
            results.append({
                "score": float(score),
                "text": entry.get("text", ""),
                "metadata": {k: v for k, v in entry.items() if k != "text"},
            })

        return results
