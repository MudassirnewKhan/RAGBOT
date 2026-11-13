"""
Central configuration for the RAG pipeline.
All constants, model names, and pipeline hyper-parameters are defined here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Model Settings ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Chunking Settings ---
CHUNK_SIZE = 800          # Characters per chunk (tuned for tighter context)
CHUNK_OVERLAP = 150       # Character overlap between consecutive chunks
MIN_CHUNK_LENGTH = 50     # Discard fragments shorter than this threshold

# --- Retrieval Settings ---
TOP_K_RETRIEVAL = 20      # Initial candidate pool retrieved by hybrid search
TOP_K_FINAL = 5           # Final candidate count passed to LLM after reranking
BM25_WEIGHT = 0.3         # Weight for BM25 in Reciprocal Rank Fusion
VECTOR_WEIGHT = 0.7       # Weight for dense vector search in RRF

# --- Storage Paths ---
DATA_DIR = str(PROJECT_ROOT / "data")
FAISS_DIR = str(PROJECT_ROOT / "faiss_store")
