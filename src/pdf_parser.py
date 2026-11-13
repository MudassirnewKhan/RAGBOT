"""
PDF parser using PyMuPDF.
Extracts text with structural metadata: page numbers, headings, and source tracking.
"""
import pymupdf
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class ParsedDocument:
    """A single parsed document page or segment with structural metadata."""
    text: str
    metadata: dict = field(default_factory=dict)


def parse_pdf(file_path: str) -> List[ParsedDocument]:
    """
    Parse a PDF file and return a list of ParsedDocument objects,
    one per page, with structural metadata and heading detection.
    """
    path = Path(file_path)
    documents = []

    try:
        doc = pymupdf.open(str(path))
    except Exception as e:
        print(f"[ERROR] Cannot open PDF {path.name}: {e}")
        return documents

    total_pages = len(doc)

    for page_num in range(total_pages):
        page = doc[page_num]

        # Extract text blocks with layout positioning
        text_page = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)
        blocks = text_page.get("blocks", [])

        page_text_parts = []
        page_headings = []

        for block in blocks:
            # type 0 represents text blocks (skip type 1 image blocks)
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                line_text = ""
                max_font_size = 0.0

                for span in line.get("spans", []):
                    line_text += span.get("text", "")
                    span_size = span.get("size", 0.0)
                    if span_size > max_font_size:
                        max_font_size = span_size

                line_text = line_text.strip()
                if not line_text:
                    continue

                # Heuristic: font size > 12pt and reasonable length indicates a section heading
                if max_font_size > 12.0 and len(line_text) < 200:
                    page_headings.append(line_text)

                page_text_parts.append(line_text)

        page_text = "\n".join(page_text_parts).strip()
        if not page_text:
            continue

        documents.append(
            ParsedDocument(
                text=page_text,
                metadata={
                    "source_file": path.name,
                    "file_path": str(path.resolve()),
                    "page_number": page_num + 1,
                    "total_pages": total_pages,
                    "headings": page_headings,
                    "file_type": "pdf",
                }
            )
        )

    doc.close()
    print(f"[INFO] Parsed PDF: {path.name} ({len(documents)} pages extracted)")
    return documents
