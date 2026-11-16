"""
RAG Engine — Production-grade end-to-end Retrieval-Augmented Generation pipeline.
Coordinates Document Ingestion, Smart Chunking, Embeddings, Hybrid Retrieval (FAISS + BM25),
Cross-Encoder Reranking, Strict Prompt Engineering, and Groq LLM Inference.
"""
import time
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    DATA_DIR,
    FAISS_DIR,
    TOP_K_RETRIEVAL,
    TOP_K_FINAL,
)
from src.text_parser import load_all_documents
from src.chunker import chunk_documents
from src.embedding import EmbeddingService
from src.vectorstore import VectorStore
from src.bm25_index import BM25Index
from src.retriever import HybridRetriever
from src.reranker import Reranker
from app.utils.rate_limiter import execute_with_retry


# --- Strict Production Prompts ---

SYSTEM_PROMPT = """You are a highly reliable and factual research assistant.
Your task is to answer user queries based STRICTLY on the retrieved context provided below.

Strict Directives:
1. Grounding: Rely ONLY on the provided context. Do NOT extrapolate or introduce external facts.
2. Insufficient Context: If the context does not contain sufficient details to answer, state:
   "I do not have sufficient information in the provided knowledge base to answer this query."
3. Citations: Every factual assertion must be attributed with in-line source citations, formatted as:
   [Source: <filename>, Page <number>]
4. Conciseness: Synthesize the context clearly, avoiding repetitive statements or conversational filler.
"""

USER_PROMPT_TEMPLATE = """Context Documents:
========================================
{context}
========================================

User Query: {query}

Provide a comprehensive, cited response:"""


class RAGEngine:
    """End-to-end RAG orchestrator with hybrid search, reranking, and citation tracking."""

    def __init__(self, rebuild_index: bool = False):
        print("=" * 60)
        print("🚀 Initializing Production RAG Engine...")
        print("=" * 60)

        # 1. Embedding Model (Singleton)
        self.embedding_service = EmbeddingService()

        # 2. Vector Store (FAISS Cosine Similarity)
        self.vector_store = VectorStore(FAISS_DIR)

        # 3. BM25 Lexical Index
        self.bm25_index = BM25Index()

        # 4. Check or Build Storage
        if self.vector_store.exists() and not rebuild_index:
            print("[INFO] Existing vector index found. Loading from disk...")
            self.vector_store.load()
            self._hydrate_bm25_from_metadata()
        else:
            print("[INFO] Building fresh vector and lexical indices from documents...")
            self.build_index()

        # 5. Hybrid Retriever (FAISS + BM25 with RRF)
        self.retriever = HybridRetriever(
            self.vector_store,
            self.bm25_index,
            self.embedding_service,
        )

        # 6. Neural Cross-Encoder Reranker
        self.reranker = Reranker()

        # 7. Groq LLM Inference Endpoint
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Please add it to your .env file or environment."
            )

        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=LLM_MODEL,
            temperature=0.1,  # Low temperature for deterministic, factual outputs
        )
        print(f"[INFO] Groq LLM initialized with model: {LLM_MODEL}")
        print("=" * 60)
        print("✅ Production RAG Engine fully initialized and ready.")
        print("=" * 60)

    def build_index(self):
        """Parse all documents from DATA_DIR, chunk them, embed, and construct indices."""
        documents = load_all_documents(DATA_DIR)
        if not documents:
            raise ValueError(f"No parseable documents found in target path: {DATA_DIR}")

        chunks = chunk_documents(documents)
        texts = [c.text for c in chunks]
        metadatas = [{**c.metadata, "text": c.text} for c in chunks]

        print(f"[INFO] Generating embeddings for {len(texts)} semantic chunks...")
        embeddings = self.embedding_service.encode(texts, show_progress=True)

        self.vector_store.build(embeddings, metadatas)
        self.bm25_index.build(texts, metadatas)
        print("[INFO] Index construction complete and synchronized.")

    def add_documents(self, documents: List[Any]) -> int:
        """
        Incrementally process and add new parsed documents to FAISS and BM25.
        Returns the number of new chunks added.
        """
        if not documents:
            return 0

        chunks = chunk_documents(documents)
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        metadatas = [{**c.metadata, "text": c.text} for c in chunks]

        print(f"[INFO] Incrementally encoding {len(texts)} new chunks...")
        embeddings = self.embedding_service.encode(texts, show_progress=False)

        self.vector_store.add(embeddings, metadatas)
        self._hydrate_bm25_from_metadata()
        print(f"[INFO] Incremental addition complete. {len(chunks)} chunks added.")
        return len(chunks)

    def _hydrate_bm25_from_metadata(self):
        """Populate the in-memory BM25 index using cached metadata JSON."""
        texts = [m.get("text", "") for m in self.vector_store.metadata]
        self.bm25_index.build(texts, self.vector_store.metadata)
        print(f"[INFO] BM25 index hydrated with {len(texts)} chunks.")

    def _prepare_query_context(self, question: str, top_k: int = TOP_K_FINAL) -> Dict[str, Any]:
        """Shared retrieval and reranking pipeline for both standard and streaming queries."""
        start_time = time.time()
        candidates = self.retriever.search(question, top_k=TOP_K_RETRIEVAL)
        if not candidates:
            return {
                "has_context": False,
                "context": "",
                "sources": [],
                "confidence": 0.0,
                "start_time": start_time,
            }

        reranked = self.reranker.rerank(question, candidates, top_k=top_k)
        context_blocks: List[str] = []
        sources: List[Dict[str, Any]] = []

        for item in reranked:
            meta = item.get("metadata", {})
            file_name = meta.get("source_file", "unknown")
            page_num = meta.get("page_number", "N/A")
            rerank_score = item.get("rerank_score", 0.0)

            source_tag = f"[Source: {file_name}, Page {page_num}]"
            context_blocks.append(f"{source_tag}\n{item.get('text', '')}")

            sources.append({
                "file": file_name,
                "page": page_num,
                "chunk_index": meta.get("chunk_index", 0),
                "relevance_score": round(rerank_score, 4),
                "retrieval_method": item.get("retrieval_method", "hybrid"),
            })

        consolidated_context = "\n\n--------------------\n\n".join(context_blocks)
        avg_score = (
            sum(s["relevance_score"] for s in sources) / len(sources) if sources else 0.0
        )
        confidence = round(1.0 / (1.0 + pow(2.71828, -avg_score)), 3)

        return {
            "has_context": True,
            "context": consolidated_context,
            "sources": sources,
            "confidence": confidence,
            "start_time": start_time,
        }

    def query(self, question: str, top_k: int = TOP_K_FINAL, min_confidence: float = 0.0) -> Dict[str, Any]:
        """
        Execute full RAG pipeline:
        1. Hybrid candidate search (top_k=TOP_K_RETRIEVAL).
        2. Neural cross-encoder reranking (top_k=top_k).
        3. Formulate structured context with metadata tags.
        4. Generate grounded LLM response with citation enforcement (with retry).
        5. Return rich telemetry: latency, confidence, sources.
        """
        ctx_data = self._prepare_query_context(question, top_k=top_k)
        start_time = ctx_data["start_time"]

        if not ctx_data["has_context"]:
            return {
                "answer": "No relevant documents found in the knowledge base matching your query.",
                "sources": [],
                "confidence": 0.0,
                "processing_time_ms": round((time.time() - start_time) * 1000, 1),
                "model": LLM_MODEL,
            }

        if ctx_data["confidence"] < min_confidence:
            return {
                "answer": f"Retrieved context confidence ({ctx_data['confidence']:.2f}) falls below required threshold ({min_confidence:.2f}).",
                "sources": ctx_data["sources"],
                "confidence": ctx_data["confidence"],
                "processing_time_ms": round((time.time() - start_time) * 1000, 1),
                "model": LLM_MODEL,
            }

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=USER_PROMPT_TEMPLATE.format(
                    context=ctx_data["context"],
                    query=question,
                )
            ),
        ]

        llm_response = execute_with_retry(lambda: self.llm.invoke(messages))
        elapsed_ms = round((time.time() - start_time) * 1000, 1)

        return {
            "answer": llm_response.content,
            "sources": ctx_data["sources"],
            "confidence": ctx_data["confidence"],
            "processing_time_ms": elapsed_ms,
            "model": LLM_MODEL,
        }

    def query_stream(self, question: str, top_k: int = TOP_K_FINAL, min_confidence: float = 0.0):
        """
        Stream LLM tokens while preserving grounding, sources, and telemetry.
        Yields chunk dictionaries:
        - {"type": "metadata", "sources": [...], "confidence": float}
        - {"type": "token", "content": "token_str"}
        - {"type": "done", "processing_time_ms": float, "model": str}
        """
        ctx_data = self._prepare_query_context(question, top_k=top_k)
        start_time = ctx_data["start_time"]

        if not ctx_data["has_context"]:
            yield {
                "type": "error",
                "message": "No relevant documents found in the knowledge base matching your query.",
            }
            return

        if ctx_data["confidence"] < min_confidence:
            yield {
                "type": "error",
                "message": f"Confidence ({ctx_data['confidence']:.2f}) below threshold ({min_confidence:.2f}).",
            }
            return

        yield {
            "type": "metadata",
            "sources": ctx_data["sources"],
            "confidence": ctx_data["confidence"],
        }

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=USER_PROMPT_TEMPLATE.format(
                    context=ctx_data["context"],
                    query=question,
                )
            ),
        ]

        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield {
                    "type": "token",
                    "content": chunk.content,
                }

        elapsed_ms = round((time.time() - start_time) * 1000, 1)
        yield {
            "type": "done",
            "processing_time_ms": elapsed_ms,
            "model": LLM_MODEL,
        }
