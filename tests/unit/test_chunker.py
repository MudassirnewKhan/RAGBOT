from src.chunker import chunk_document

def test_chunk_document_basic():
    sample_text = (
        "Paragraph 1 contains enough text to form a valid paragraph. "
        "It talks about retrieval-augmented generation and search.\n\n"
        "Paragraph 2 continues the discussion on vector databases and embeddings. "
        "It provides additional context for chunking."
    )
    metadata = {"source": "test.txt", "file_type": "txt"}
    chunks = chunk_document(sample_text, metadata=metadata, chunk_size=150, chunk_overlap=30)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk.text.strip()) > 0
        assert chunk.metadata["source"] == "test.txt"
        assert "chunk_index" in chunk.metadata

def test_chunk_document_empty_input():
    chunks = chunk_document("", metadata={"source": "empty.txt"}, chunk_size=500, chunk_overlap=50)
    assert chunks == []
