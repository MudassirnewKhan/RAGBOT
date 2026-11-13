"""
Hybrid Retriever combining FAISS vector search and BM25 lexical search
using Reciprocal Rank Fusion (RRF).
"""
from typing import List, Dict, Any
from src.vectorstore import VectorStore
from src.bm25_index import BM25Index
from src.embedding import EmbeddingService
from src.config import TOP_K_RETRIEVAL, BM25_WEIGHT, VECTOR_WEIGHT


class HybridRetriever:
    """Combines semantic vector search with keyword matching via RRF."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedding_service: EmbeddingService,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedding_service = embedding_service

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Execute hybrid retrieval:
        1. Retrieve top_k * 2 candidates from vector store.
        2. Retrieve top_k * 2 candidates from BM25 index.
        3. Fuse candidate rankings via Reciprocal Rank Fusion (RRF).
        4. Return top_k highest-ranking results.
        """
        k = top_k or TOP_K_RETRIEVAL
        candidate_pool = max(k * 2, 20)

        # 1. Vector Search
        query_emb = self.embedding_service.encode_query(query)
        vector_results = self.vector_store.search(query_emb, top_k=candidate_pool)

        # 2. BM25 Search
        bm25_results = self.bm25_index.search(query, top_k=candidate_pool)

        # 3. Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            weight_vector=VECTOR_WEIGHT,
            weight_bm25=BM25_WEIGHT,
        )

        return fused_results[:k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        weight_vector: float = 0.7,
        weight_bm25: float = 0.3,
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        RRF algorithm: score = sum( weight / (k + rank_i) )
        """
        scored_map: Dict[str, Dict[str, Any]] = {}

        # Add Vector results
        for rank, res in enumerate(vector_results):
            text_key = res["text"][:300]
            rrf_score = weight_vector * (1.0 / (k + rank + 1))
            if text_key not in scored_map:
                scored_map[text_key] = {
                    **res,
                    "rrf_score": rrf_score,
                    "retrieval_method": "vector",
                }
            else:
                scored_map[text_key]["rrf_score"] += rrf_score

        # Add BM25 results
        for rank, res in enumerate(bm25_results):
            text_key = res["text"][:300]
            rrf_score = weight_bm25 * (1.0 / (k + rank + 1))
            if text_key not in scored_map:
                scored_map[text_key] = {
                    **res,
                    "rrf_score": rrf_score,
                    "retrieval_method": "bm25",
                }
            else:
                scored_map[text_key]["rrf_score"] += rrf_score
                scored_map[text_key]["retrieval_method"] = "hybrid"

        # Sort descending by cumulative RRF score
        fused = sorted(scored_map.values(), key=lambda x: x["rrf_score"], reverse=True)
        return fused
