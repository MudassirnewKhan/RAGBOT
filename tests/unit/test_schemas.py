import pytest
from pydantic import ValidationError
from app.schemas.query import QueryRequest

def test_query_request_valid():
    req = QueryRequest(question="What is RAG?", top_k=5)
    assert req.question == "What is RAG?"
    assert req.top_k == 5

def test_query_request_empty_query():
    with pytest.raises(ValidationError):
        QueryRequest(question="   ", top_k=5)

def test_query_request_invalid_top_k():
    with pytest.raises(ValidationError):
        QueryRequest(question="Valid question", top_k=0)
