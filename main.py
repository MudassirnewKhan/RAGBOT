import os
import sys
from dotenv import load_dotenv

# Absolute path configuration to resolve internal 'src' cross-imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Load environment configurations from your root .env file
load_dotenv(os.path.join(project_root, ".env"))

try:
    from src.search import RAGSearch
except ModuleNotFoundError as e:
    print(f"\n❌ Import Error: {e}")
    print("👉 Solution: Ensure you have prefixed internal imports inside your src/ files with 'src.'")
    print("Example: Inside src/search.py, change 'from vectorstore...' to 'from src.vectorstore...'")
    sys.exit(1)

def main():
    print("=" * 60)
    print("🤖 KNOWLEDGE BASE RAG PIPELINE ACTIVE")
    print("=" * 60)
    
    # Using Llama 3.3 to bypass the decommissioned Gemma2 model endpoint error
    try:
        rag = RAGSearch(
            persist_dir="faiss_store", 
            embedding_model="all-MiniLM-L6-v2", 
            llm_model="llama-3.3-70b-versatile"
        )
        print("\n✅ System initialized successfully! Type 'exit' or 'quit' to terminate.\n")
    except Exception as e:
        print(f"\n❌ Initialization Failure: {e}")
        print("Please check that GROQ_API_KEY is properly set in your .env file.")
        return

    # Continuous interactive QA session loop
    while True:
        try:
            print("-" * 60)
            query = input("❓ Ask a question: ").strip()
            
            if not query:
                continue
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Disconnecting RAG session. Happy learning!")
                break
                
            print("\n🔍 Traversing vector space and synthesizing response...")
            summary = rag.search_and_summarize(query, top_k=3)
            
            print("\n💡 AI Engine Response:")
            print(summary)
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Process execution interrupted gracefully. Exiting...")
            break
        except Exception as e:
            print(f"\n⚠️ Runtime Exception encountered: {e}\n")

if __name__ == "__main__":
    main()