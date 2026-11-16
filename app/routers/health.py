"""
Health and readiness probe endpoints.
"""
from fastapi import APIRouter, Depends
from app.schemas.health import HealthResponse, ReadinessResponse
from app.dependencies import get_rag_engine
from src.rag_engine import RAGEngine
from src.config import LLM_MODEL, RERANKER_MODEL

router = APIRouter(tags=["Health & Monitoring"])


@router.get("/health", response_model=HealthResponse, summary="Liveness Probe")
def health_check() -> HealthResponse:
    """Returns basic service health heartbeat for load balancers and orchestrators."""
    return HealthResponse(status="ok")


@router.get("/readiness", response_model=ReadinessResponse, summary="Readiness Probe")
def readiness_check(engine: RAGEngine = Depends(get_rag_engine)) -> ReadinessResponse:
    """Validates that the FAISS index and embedding models are in memory and ready."""
    vector_count = engine.vector_store.index.ntotal if engine.vector_store.index else 0
    dim = engine.embedding_service.dimension if engine.embedding_service else 0

    return ReadinessResponse(
        ready=vector_count > 0,
        vector_count=vector_count,
        embedding_dimension=dim,
        llm_model=LLM_MODEL,
        reranker_model=RERANKER_MODEL,
    )
