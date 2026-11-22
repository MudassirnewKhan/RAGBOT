import sys
import os
from pathlib import Path

# Add project root to sys.path so app and src can be imported cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure test environment variables
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "mock_test_key"

import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client
