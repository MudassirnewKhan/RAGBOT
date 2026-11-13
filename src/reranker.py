"""
Cross-Encoder Reranker.
Scores (query, passage) pairs jointly to refine hybrid retrieval candidates.
"""
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from src.config import RERANKER_MODEL, TOP_K_FINAL


class Reranker:
    """Cross-Encoder based neural reranker for second-stage ranking."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or RERANKER_MODEL
        print(f"[INFO] Initializing Cross-Encoder Reranker: {self.model_name}...")
        try:
            # Attempt loading from local cache first
            self.model = CrossEncoder(self.model_name, local_files_only=True)
        except Exception:
            self.model = CrossEncoder(self.model_name)
        print("[INFO] Cross-Encoder Reranker active.")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank a list of candidate chunk dicts using the cross-encoder model.

        Input candidates: [{ 'text': ..., 'score': ..., 'metadata': ... }, ...]
        Output: Top-k candidates re-sorted with 'rerank_score' attached.
        """
        k = top_k or TOP_K_FINAL

        if not candidates:
            return []

        # Prepare (query, passage_text) tuples for the cross-encoder
        pairs = [(query, c.get("text", "")) for c in candidates]

        # Compute cross-attention alignment scores
        scores = self.model.predict(pairs)

        # Attach scores to candidate dict copies
        reranked_candidates = []
        for candidate, score in zip(candidates, scores):
            entry = dict(candidate)
            entry["rerank_score"] = float(score)
            reranked_candidates.append(entry)

        # Sort descending by neural rerank score
        reranked_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked_candidates[:k]
