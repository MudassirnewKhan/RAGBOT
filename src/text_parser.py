"""
Text and Markdown parser with unified document ingestion.
Ingests TXT, MD, and PDF documents uniformly into ParsedDocument objects.
"""
from pathlib import Path
from typing import List
from src.pdf_parser import ParsedDocument, parse_pdf


def parse_text_file(file_path: str) -> List[ParsedDocument]:
    """Parse a plain text (.txt) or markdown (.md) file into ParsedDocument objects."""
    path = Path(file_path)

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="latin-1")
        except Exception as e:
            print(f"[ERROR] Cannot decode {path.name}: {e}")
            return []
    except Exception as e:
        print(f"[ERROR] Cannot read {path.name}: {e}")
        return []

    stripped_text = text.strip()
    if not stripped_text:
        return []

    # Detect markdown headers (# Heading) or ALL-CAPS section titles
    headings = []
    for line in stripped_text.splitlines():
        line_clean = line.strip()
        if line_clean.startswith("#"):
            headings.append(line_clean.lstrip("# ").strip())
        elif line_clean.isupper() and 5 < len(line_clean) < 150:
            headings.append(line_clean)

    file_extension = path.suffix.lower().lstrip(".")
    print(f"[INFO] Parsed {file_extension.upper()}: {path.name} ({len(stripped_text)} chars)")

    return [
        ParsedDocument(
            text=stripped_text,
            metadata={
                "source_file": path.name,
                "file_path": str(path.resolve()),
                "page_number": 1,
                "total_pages": 1,
                "headings": headings,
                "file_type": file_extension or "txt",
            }
        )
    ]


def load_all_documents(data_dir: str) -> List[ParsedDocument]:
    """
    Scans data_dir recursively for PDFs, TXT, and Markdown files,
    returning a consolidated list of ParsedDocument objects.
    """
    data_path = Path(data_dir).resolve()
    documents: List[ParsedDocument] = []

    if not data_path.exists():
        print(f"[WARNING] Data directory does not exist: {data_path}")
        return documents

    # Ingest all PDF documents
    pdf_files = sorted(data_path.glob("**/*.pdf"))
    for pdf_file in pdf_files:
        documents.extend(parse_pdf(str(pdf_file)))

    # Ingest all TXT documents
    txt_files = sorted(data_path.glob("**/*.txt"))
    for txt_file in txt_files:
        documents.extend(parse_text_file(str(txt_file)))

    # Ingest all Markdown documents
    md_files = sorted(data_path.glob("**/*.md"))
    for md_file in md_files:
        documents.extend(parse_text_file(str(md_file)))

    print(f"[INFO] Ingestion complete. Total document segments: {len(documents)}")
    return documents

