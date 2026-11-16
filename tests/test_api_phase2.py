"""
Automated Test Suite for Phase 2: FastAPI REST API.
Verifies endpoints, Pydantic v2 validation gates, tracing headers, and error envelopes.
"""
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure project root is available on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app
from src.config import GROQ_API_KEY


def run_api_tests():
    print("\n" + "=" * 65)
    print("🧪 RUNNING PHASE 2 API VERIFICATION SUITE")
    print("=" * 65)

    with TestClient(app) as client:
        # --- TEST 1: Health Liveness Probe ---
        print("\n[TEST 1/5] Testing GET /health (Liveness)...")
        resp = client.get("/health")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert data.get("status") == "ok", f"Expected status 'ok', got {data}"
        req_id = resp.headers.get("x-request-id") or resp.headers.get("X-Request-ID")
        proc_time = resp.headers.get("x-process-time") or resp.headers.get("X-Process-Time")
        assert req_id is not None, f"Missing X-Request-ID header in: {dict(resp.headers)}"
        assert proc_time is not None, f"Missing X-Process-Time header in: {dict(resp.headers)}"
        print(f"  ✅ Health check passed: {data}")
        print(f"  ✅ Tracing Headers verified: X-Request-ID={req_id[:8]}... | X-Process-Time={proc_time}")

        # --- TEST 2: Readiness Probe ---
        print("\n[TEST 2/5] Testing GET /readiness (Index & Models)...")
        resp = client.get("/readiness")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert data.get("ready") is True, f"Expected ready=True, got {data}"
        assert data.get("vector_count", 0) > 0, "No vectors reported in readiness!"
        print(f"  ✅ Readiness probe passed: {data.get('vector_count')} vectors indexed across {data.get('embedding_dimension')} dimensions.")

        # --- TEST 3: Schema Validation Gate (HTTP 422) ---
        print("\n[TEST 3/5] Testing Strict Pydantic Schema Validation (HTTP 422 Rejection)...")
        # Test case: question too short (< 3 chars)
        invalid_payload = {"question": "hi", "top_k": 5}
        resp = client.post("/api/v1/query", json=invalid_payload)
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"
        err_data = resp.json()
        assert err_data.get("error_type") == "VALIDATION_ERROR", f"Unexpected error envelope: {err_data}"
        print(f"  ✅ Strict rejection verified: Rejected '{invalid_payload['question']}' with HTTP 422.")
        print(f"  ✅ Structured RFC 7807 Error Envelope: {err_data.get('message')}")

        # --- TEST 4: List Documents Endpoint ---
        print("\n[TEST 4/5] Testing GET /api/v1/documents...")
        resp = client.get("/api/v1/documents")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        docs = resp.json()
        assert isinstance(docs, list) and len(docs) > 0, "No documents listed!"
        print(f"  ✅ Documents endpoint passed: Found {len(docs)} registered files.")
        for d in docs[:2]:
            print(f"     • {d.get('filename')} ({d.get('file_type')}, {d.get('chunk_count')} chunks)")

        # --- TEST 5: Live RAG Query Endpoint ---
        print("\n[TEST 5/5] Testing POST /api/v1/query...")
        if not GROQ_API_KEY or GROQ_API_KEY.startswith("your_"):
            print("  ⚠️ SKIPPED: GROQ_API_KEY not configured in .env.")
        else:
            query_payload = {
                "question": "What is the attention mechanism?",
                "top_k": 3,
                "min_confidence": 0.0,
            }
            resp = client.post("/api/v1/query", json=query_payload)
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
            res_data = resp.json()
            assert "answer" in res_data, "Missing answer in response!"
            assert "sources" in res_data, "Missing sources in response!"
            assert len(res_data["sources"]) > 0, "Expected at least 1 source citation!"
            print(f"  ✅ Query Endpoint passed! Latency: {res_data.get('processing_time_ms')}ms")
            print(f"  💡 Answer Preview: {res_data['answer'][:150]}...")
            print(f"  📚 Attributed Sources: {len(res_data['sources'])} citations returned:")
            for s in res_data["sources"]:
                print(f"     • {s.get('file')} (Page {s.get('page')}) | score: {s.get('relevance_score'):+.4f}")

    print("\n" + "=" * 65)
    print("🎉 ALL PHASE 2 REST API TESTS PASSED SUCCESSFULLY!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_api_tests()
