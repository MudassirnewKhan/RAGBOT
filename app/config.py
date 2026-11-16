"""
Application configuration and API constants.
"""
from typing import List


class APISettings:
    """REST API Configuration Settings."""
    PROJECT_NAME: str = "RAGBOT API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = (
        "Production-grade Retrieval-Augmented Generation (RAG) REST API "
        "featuring Hybrid Retrieval (FAISS + BM25), Neural Cross-Encoder Reranking, "
        "Strict Source Citations, and Groq LLM Inference."
    )
    API_V1_STR: str = "/api/v1"
    CORS_ORIGINS: List[str] = ["*"]
    HOST: str = "0.0.0.0"
    PORT: int = 8000


settings = APISettings()
