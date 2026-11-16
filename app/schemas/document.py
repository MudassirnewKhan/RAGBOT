"""
Document ingestion and metadata schemas.
"""
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    """Metadata of an ingested document."""
    filename: str = Field(..., description="Document file name")
    file_type: str = Field(..., description="File extension / type (pdf, txt, md)")
    total_pages: int = Field(default=1, description="Number of extracted pages")
    chunk_count: int = Field(default=0, description="Total semantic chunks derived from document")


class DocumentUploadResponse(BaseModel):
    """Response returned upon uploading documents."""
    message: str = Field(..., description="Ingestion status summary")
    uploaded_files: List[str] = Field(..., description="Names of uploaded files")
    chunks_created: int = Field(..., description="Total new chunks derived and indexed")
    total_vectors_in_store: int = Field(..., description="Updated total vector count in FAISS index")


class RebuildIndexResponse(BaseModel):
    """Response returned when manually triggering full index rebuild."""
    message: str = Field(..., description="Rebuild status message")
    documents_scanned: int = Field(..., description="Total documents processed from data directory")
    chunks_indexed: int = Field(..., description="Total chunks embedded and added to index")
    total_vectors: int = Field(..., description="Final vector count in FAISS index")
