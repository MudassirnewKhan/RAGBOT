"""
FastAPI dependencies for dependency injection across API endpoints.
Provides the RAGEngine singleton with graceful fallback.
"""
from typing import Optional
from fastapi import Request, HTTPException, status
from src.rag_engine import RAGEngine

_singleton_engine: Optional[RAGEngine] = None


def set_rag_engine(engine: RAGEngine) -> None:
    """Store the global singleton instance."""
    global _singleton_engine
    _singleton_engine = engine


def get_rag_engine(request: Request = None) -> RAGEngine:
    """
    Retrieve the initialized RAGEngine singleton from application state or global cache.
    Raises HTTP 503 if engine is unavailable.
    """
    global _singleton_engine

    if request and hasattr(request, "app") and getattr(request.app.state, "rag_engine", None) is not None:
        return request.app.state.rag_engine

    if _singleton_engine is not None:
        return _singleton_engine

    try:
        _singleton_engine = RAGEngine()
        return _singleton_engine
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"RAG Engine service is not initialized or currently unavailable: {exc}",
        ) from exc
