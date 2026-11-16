"""
Document management and ingestion endpoints.
Handles PDF / TXT file uploads, metadata inspections, and dynamic index rebuilding.
"""
import shutil
from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from app.schemas.document import DocumentUploadResponse, DocumentInfo, RebuildIndexResponse
from app.dependencies import get_rag_engine
from src.rag_engine import RAGEngine
from src.config import DATA_DIR

router = APIRouter(prefix="/api/v1/documents", tags=["Document Management"])


@router.get(
    "",
    response_model=List[DocumentInfo],
    summary="List Ingested Documents",
    description="Returns metadata for all documents currently indexed in the knowledge base.",
)
def list_documents(engine: RAGEngine = Depends(get_rag_engine)) -> List[DocumentInfo]:
    docs_map = {}
    for entry in engine.vector_store.metadata:
        filename = entry.get("source_file", "unknown")
        if filename not in docs_map:
            docs_map[filename] = {
                "filename": filename,
                "file_type": entry.get("file_type", "unknown"),
                "total_pages": entry.get("total_pages", 1),
                "chunk_count": 0,
            }
        docs_map[filename]["chunk_count"] += 1

    return [DocumentInfo(**info) for info in docs_map.values()]


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload & Ingest Documents",
    description="Uploads one or more PDF, TXT, or Markdown documents, saves them to storage, and rebuilds the indices.",
)
async def upload_documents(
    files: List[UploadFile] = File(..., description="PDF or text files to ingest"),
    engine: RAGEngine = Depends(get_rag_engine),
) -> DocumentUploadResponse:
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded.")

    saved_filenames = []
    data_path = Path(DATA_DIR)
    pdf_dir = data_path / "pdf"
    txt_dir = data_path / "text_files"

    pdf_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        filename = file.filename
        ext = Path(filename).suffix.lower()

        if ext == ".pdf":
            dest = pdf_dir / filename
        elif ext in (".txt", ".md"):
            dest = txt_dir / filename
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format '{ext}'. Only .pdf, .txt, and .md are supported.",
            )

        with open(dest, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        saved_filenames.append(filename)

    # Re-index dynamically with the newly uploaded files
    engine.build_index()

    vector_count = engine.vector_store.index.ntotal if engine.vector_store.index else 0
    return DocumentUploadResponse(
        message=f"Successfully ingested {len(saved_filenames)} document(s).",
        uploaded_files=saved_filenames,
        chunks_created=len(engine.vector_store.metadata),
        total_vectors_in_store=vector_count,
    )


@router.post(
    "/rebuild",
    response_model=RebuildIndexResponse,
    summary="Rebuild Knowledge Base Indices",
    description="Manually re-scans all files in the data directory and rebuilds FAISS & BM25 indices.",
)
def rebuild_knowledge_base(engine: RAGEngine = Depends(get_rag_engine)) -> RebuildIndexResponse:
    engine.build_index()
    vector_count = engine.vector_store.index.ntotal if engine.vector_store.index else 0

    unique_docs = len({m.get("source_file") for m in engine.vector_store.metadata})

    return RebuildIndexResponse(
        message="Knowledge base indices successfully rebuilt from source documents.",
        documents_scanned=unique_docs,
        chunks_indexed=len(engine.vector_store.metadata),
        total_vectors=vector_count,
    )
