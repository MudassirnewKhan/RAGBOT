"""
Health check and readiness probe schemas.
"""
from datetime import datetime
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness probe response model."""
    status: str = Field(default="ok", description="Service liveness state")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Current UTC timestamp")
    service: str = Field(default="RAGBOT REST API", description="Service name")


class ReadinessResponse(BaseModel):
    """Readiness probe response model inspecting index and model availability."""
    ready: bool = Field(..., description="Whether service is ready to accept inference requests")
    vector_count: int = Field(..., description="Total vectors currently indexed in FAISS")
    embedding_dimension: int = Field(..., description="Embedding vector dimensionality")
    llm_model: str = Field(..., description="Configured Groq LLM model name")
    reranker_model: str = Field(..., description="Configured Cross-Encoder model name")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Probe execution timestamp")
