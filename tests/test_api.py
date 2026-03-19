import pytest
from fastapi.testclient import TestClient
from src.api.main import app
client = TestClient(app)

class TestAPI:
    def test_health(self): assert client.get("/health").status_code == 200
    def test_generate(self):
        resp = client.post("/generate", json={"prompt": "cat", "steps": 50})
        assert resp.status_code == 200
        assert resp.json()["status"] == "generated"
