"""
Automated Phase 1 Verification Script.
Tests:
1. PyMuPDF & Text Parsing
2. Semantic Chunking
3. Vector Store (FAISS Cosine)
4. BM25 Lexical Index
5. Neural Cross-Encoder Reranker
6. End-to-end RAG Engine (with Groq LLM if API key provided)
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.text_parser import load_all_documents
from src.chunker import chunk_documents
from src.embedding import EmbeddingService
from src.vectorstore import VectorStore
from src.bm25_index import BM25Index
from src.retriever import HybridRetriever
from src.reranker import Reranker
from src.config import GROQ_API_KEY, DATA_DIR, FAISS_DIR


def run_checks():
    print("\n" + "=" * 65)
    print("🔍 RUNNING PHASE 1 PIPELINE VERIFICATION SUITE")
    print("=" * 65)

    # --- Check 1: Ingestion ---
    print("\n[TEST 1/5] Testing PyMuPDF & Text Parser...")
    docs = load_all_documents(DATA_DIR)
    assert len(docs) > 0, "No documents were parsed!"
    pdf_docs = [d for d in docs if d.metadata.get("file_type") == "pdf"]
    txt_docs = [d for d in docs if d.metadata.get("file_type") in ("txt", "md")]
    print(f"  ✅ Parsed {len(pdf_docs)} PDF pages and {len(txt_docs)} text files.")
    print(f"  ✅ First document source: '{docs[0].metadata.get('source_file')}'")
    print(f"  ✅ Headings detected: {docs[0].metadata.get('headings')[:2]}")

    # --- Check 2: Chunking ---
    print("\n[TEST 2/5] Testing Structure-Aware Chunking...")
    chunks = chunk_documents(docs)
    assert len(chunks) > 0, "No chunks created!"
    print(f"  ✅ Created {len(chunks)} chunks from {len(docs)} documents.")
    sample = chunks[0]
    assert "source_file" in sample.metadata, "Metadata missing source_file!"
    assert "page_number" in sample.metadata, "Metadata missing page_number!"
    assert "chunk_index" in sample.metadata, "Metadata missing chunk_index!"
    print(f"  ✅ Provenance metadata verified: {sample.metadata['source_file']} (p. {sample.metadata['page_number']}, chunk #{sample.metadata['chunk_index']})")

    # --- Check 3: Vector Store & BM25 ---
    print("\n[TEST 3/5] Testing VectorStore (FAISS) & BM25 Index...")
    vstore = VectorStore(FAISS_DIR)
    assert vstore.exists(), "Vector store files missing on disk! Run build_index() first."
    vstore.load()
    print(f"  ✅ Loaded FAISS index: {vstore.index.ntotal} vectors (Cosine FlatIP).")

    bm25 = BM25Index()
    texts = [m.get("text", "") for m in vstore.metadata]
    bm25.build(texts, vstore.metadata)
    print(f"  ✅ Built BM25 lexical index: {len(texts)} document chunks.")

    # --- Check 4: Hybrid Retrieval & Reranker ---
    print("\n[TEST 4/5] Testing Hybrid Retrieval & Cross-Encoder Reranker...")
    emb_service = EmbeddingService()
    retriever = HybridRetriever(vstore, bm25, emb_service)
    test_query = "What is the attention mechanism in transformers?"
    candidates = retriever.search(test_query, top_k=10)
    assert len(candidates) > 0, "Hybrid search returned 0 results!"
    print(f"  ✅ Hybrid search retrieved {len(candidates)} candidate chunks.")

    reranker = Reranker()
    reranked = reranker.rerank(test_query, candidates, top_k=3)
    assert len(reranked) == 3, "Reranker failed to return top 3!"
    top_doc = reranked[0]
    print(f"  ✅ Neural reranker active. Top score: {top_doc['rerank_score']:+.4f}")
    print(f"  ✅ Top snippet source: {top_doc['metadata'].get('source_file')}, page {top_doc['metadata'].get('page_number')}")
    print(f"  ✅ Snippet preview: {top_doc['text'][:120].replace(chr(10), ' ')}...")

    # --- Check 5: Groq LLM Inference ---
    print("\n[TEST 5/5] Testing Groq LLM End-to-End Generation...")
    if not GROQ_API_KEY or GROQ_API_KEY.startswith("your_"):
        print("  ⚠️ SKIPPED: GROQ_API_KEY not configured in .env.")
        print("  👉 Create a .env file with your Groq API key to test live LLM answers:")
        print("     GROQ_API_KEY=gsk_your_key_here")
    else:
        from src.rag_engine import RAGEngine
        engine = RAGEngine()
        response = engine.query(test_query)
        print("  ✅ Groq LLM inference succeeded!")
        print(f"  💡 Answer: {response['answer'][:180]}...")
        print(f"  📚 Citations: {len(response['sources'])} sources")
        print(f"  ⏱️ Latency: {response['processing_time_ms']} ms | Confidence: {response['confidence'] * 100:.1f}%")

    print("\n" + "=" * 65)
    print("🎉 ALL CORE PHASE 1 PIPELINE TESTS PASSED!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_checks()
