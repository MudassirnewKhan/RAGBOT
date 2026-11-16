"""
Query request and response schemas with strict Pydantic v2 validation.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class QueryRequest(BaseModel):
    """Incoming user query validation schema."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "question": "What is the attention mechanism in transformers?",
                "top_k": 5,
                "min_confidence": 0.0,
            }
        },
    )

    question: str = Field(
        ...,
        min_length=3,
        max_length=1500,
        description="User query string to retrieve and answer from knowledge base",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of final context snippets to use after reranking",
    )
    min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold required for the response",
    )


class SourceCitation(BaseModel):
    """Attributed source reference for grounded assertions."""
    file: str = Field(..., description="Source document filename")
    page: Optional[int | str] = Field(default=None, description="Page number where snippet originated")
    chunk_index: int = Field(default=0, description="Sequential chunk index within document")
    relevance_score: float = Field(..., description="Neural cross-encoder relevance score")
    retrieval_method: str = Field(default="hybrid", description="Retrieval method (vector, bm25, or hybrid)")


class QueryResponse(BaseModel):
    """Structured response payload returned to client."""
    answer: str = Field(..., description="Grounded LLM-synthesized response with in-text citations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Normalized answer confidence score")
    sources: List[SourceCitation] = Field(default_factory=list, description="List of consulted source passages")
    processing_time_ms: float = Field(..., description="Total pipeline latency in milliseconds")
    model: str = Field(..., description="LLM model used for synthesis")
