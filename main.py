"""
RAGBOT — Production Knowledge Base Interface
Run: python main.py
"""
import sys
import os
from pathlib import Path

# Ensure project root is available on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_engine import RAGEngine


def format_header():
    print()
    print("=" * 65)
    print("🤖  RAGBOT: Production-Grade Knowledge Base Assistant")
    print("=" * 65)
    print("  • Architecture : PyMuPDF + Hybrid (FAISS Cosine + Okapi BM25)")
    print("  • Neural Rank  : Cross-Encoder (ms-marco-MiniLM-L-6-v2)")
    print("  • LLM Engine   : Groq Llama 3.3 70B (Versatile)")
    print("=" * 65)
    print("Commands:")
    print("  Type your question and press [Enter]")
    print("  Type 'rebuild' to force document re-indexing")
    print("  Type 'exit', 'quit', or 'q' to terminate session")
    print("=" * 65)
    print()


def run_interactive_session():
    format_header()

    try:
        engine = RAGEngine()
    except Exception as e:
        print(f"\n❌ Initialization Failure: {e}")
        print("Please check your .env file and ensure GROQ_API_KEY is properly set.")
        return

    print("\n✅ System active! Ask anything from your knowledge base.\n")

    while True:
        try:
            print("-" * 65)
            query = input("❓ Question: ").strip()

            if not query:
                continue

            if query.lower() in ("exit", "quit", "q"):
                print("\n👋 Disconnecting RAG session. Happy learning!\n")
                break

            if query.lower() == "rebuild":
                print("\n🔄 Rebuilding knowledge base from data/ directory...")
                engine.build_index()
                print("✅ Rebuild complete!\n")
                continue

            print("\n🔍 Traversing hybrid vector space & reranking candidates...")
            result = engine.query(query)

            # Display AI Answer
            print("\n💡 AI Response:")
            print(result["answer"])

            # Display Attributed Citations
            sources = result.get("sources", [])
            if sources:
                print(f"\n📚 Citations & Provenance ({len(sources)} sources consulted):")
                for s in sources:
                    score_str = f"score: {s['relevance_score']:+.3f}"
                    method_str = f"[{s.get('retrieval_method', 'hybrid')}]"
                    print(
                        f"   • {s['file']} (Page {s['page']}) "
                        f"{method_str:10} | {score_str}"
                    )

            # Display Performance Telemetry
            latency = result.get("processing_time_ms", 0)
            confidence = result.get("confidence", 0.0)
            model = result.get("model", "N/A")

            print(
                f"\n⏱️  Latency: {latency} ms | "
                f"Confidence: {confidence * 100:.1f}% | "
                f"Model: {model}"
            )
            print("-" * 65 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted gracefully. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n⚠️ Runtime Exception encountered: {e}\n")


if __name__ == "__main__":
    run_interactive_session()
