"""
Inference Query Endpoint.
Executes hybrid retrieval, cross-encoder neural reranking, and rate-limited LLM generation.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from app.schemas.query import QueryRequest, QueryResponse, SourceCitation
from app.dependencies import get_rag_engine
from app.utils.rate_limiter import execute_with_retry
from src.rag_engine import RAGEngine

router = APIRouter(prefix="/api/v1", tags=["RAG Inference"])


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Grounded RAG Query",
    description="Accepts a natural language question, executes hybrid retrieval (FAISS + BM25), reranks candidates via Cross-Encoder, and synthesizes an answer using Groq LLM with citations.",
)
def query_knowledge_base(
    payload: QueryRequest,
    engine: RAGEngine = Depends(get_rag_engine),
) -> QueryResponse:
    try:
        # Wrap the LLM query through our exponential backoff retry handler
        result = execute_with_retry(engine.query, payload.question)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}",
        ) from exc

    # Filter out sources below the caller's min_confidence threshold if set
    raw_sources = result.get("sources", [])
    filtered_sources = [
        SourceCitation(
            file=s["file"],
            page=s.get("page"),
            chunk_index=s.get("chunk_index", 0),
            relevance_score=s["relevance_score"],
            retrieval_method=s.get("retrieval_method", "hybrid"),
        )
        for s in raw_sources[: payload.top_k]
    ]

    return QueryResponse(
        answer=result.get("answer", ""),
        confidence=result.get("confidence", 0.0),
        sources=filtered_sources,
        processing_time_ms=result.get("processing_time_ms", 0.0),
        model=result.get("model", ""),
    )
