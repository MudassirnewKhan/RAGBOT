"""
Structure-aware text chunker.
Splits text respecting paragraph and line boundaries, ensuring chunks retain semantic cohesion
and full document metadata with overlap.
"""
from typing import List
from dataclasses import dataclass, field
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH
from src.pdf_parser import ParsedDocument


@dataclass
class Chunk:
    """A semantic chunk of text ready for embedding and indexing."""
    text: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def chunk_document(
    text: str,
    metadata: dict,
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[Chunk]:
    """
    Split a single document's text into chunks while preserving paragraph boundaries.

    Strategy:
    1. Split on double newlines (paragraphs).
    2. Sub-split paragraphs longer than chunk_size on single newlines.
    3. Merge small consecutive segments up to chunk_size.
    4. Apply chunk_overlap across consecutive chunks for context continuity.
    """
    target_size = chunk_size or CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP

    # Step 1: Split into paragraphs
    raw_paragraphs = text.split("\n\n")
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    # Step 2: Ensure individual segments don't exceed target_size
    segments: List[str] = []
    for para in paragraphs:
        if len(para) <= target_size:
            segments.append(para)
        else:
            # Split over-sized paragraphs on single newlines
            lines = para.split("\n")
            current = ""
            for line in lines:
                if len(current) + len(line) + 1 > target_size and current:
                    segments.append(current.strip())
                    current = line
                else:
                    current = f"{current}\n{line}" if current else line
            if current.strip():
                segments.append(current.strip())

    # Step 3: Merge segments up to target_size
    merged_texts: List[str] = []
    current_text = ""

    for segment in segments:
        candidate = f"{current_text}\n\n{segment}" if current_text else segment
        if len(candidate) <= target_size:
            current_text = candidate
        else:
            if current_text and len(current_text) >= MIN_CHUNK_LENGTH:
                merged_texts.append(current_text)
            current_text = segment

    if current_text and len(current_text) >= MIN_CHUNK_LENGTH:
        merged_texts.append(current_text)

    # Step 4: Add overlap between consecutive chunks
    final_chunks: List[Chunk] = []
    for i, chunk_text in enumerate(merged_texts):
        if i > 0 and overlap > 0:
            prev_tail = merged_texts[i - 1][-overlap:]
            chunk_text = f"{prev_tail}\n{chunk_text}"

        final_chunks.append(
            Chunk(
                text=chunk_text,
                chunk_index=i,
                metadata={
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(merged_texts),
                }
            )
        )

    # Update total_chunks in case length changed
    total = len(final_chunks)
    for c in final_chunks:
        c.metadata["total_chunks"] = total

    return final_chunks


def chunk_documents(documents: List[ParsedDocument]) -> List[Chunk]:
    """
    Chunk a collection of ParsedDocument objects into a unified list of Chunks.
    """
    all_chunks: List[Chunk] = []

    for doc in documents:
        doc_chunks = chunk_document(doc.text, doc.metadata)
        all_chunks.extend(doc_chunks)

    print(f"[INFO] Chunking complete: {len(documents)} documents -> {len(all_chunks)} chunks")
    return all_chunks
