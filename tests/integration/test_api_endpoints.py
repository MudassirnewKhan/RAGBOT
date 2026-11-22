def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    # Verify structured tracing header is present
    assert "x-request-id" in response.headers or "X-Request-ID" in response.headers

def test_readiness_endpoint(client):
    response = client.get("/readiness")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data

def test_query_validation_error(client):
    # Empty question should trigger 422 Unprocessable Entity
    response = client.post("/api/v1/query", json={"question": ""})
    assert response.status_code == 422
    data = response.json()
    assert data["error_type"] == "VALIDATION_ERROR"
    assert "details" in data or "message" in data
