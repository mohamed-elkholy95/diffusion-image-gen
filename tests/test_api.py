"""Tests for the FastAPI image generation service."""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_status_code(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_fields(self):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "version" in data


class TestModelInfoEndpoint:
    """Test the model configuration endpoint."""

    def test_model_info_status(self):
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_fields(self):
        data = client.get("/model/info").json()
        assert data["image_size"] == 64
        assert data["diffusion_steps"] > 0
        assert data["noise_schedule"] in ("linear", "cosine")
        assert "architecture" in data

    def test_model_info_hyperparams(self):
        """Config values should be positive and reasonable."""
        data = client.get("/model/info").json()
        assert data["learning_rate"] > 0
        assert data["batch_size"] > 0
        assert data["base_channels"] > 0


class TestGenerateEndpoint:
    """Test the image generation endpoint."""

    def test_generate_default(self):
        response = client.post("/generate", json={"prompt": "cat", "steps": 50})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "generated"
        assert "64x64" in data["image_size"]

    def test_generate_with_seed(self):
        """Seed should be echoed back in the response."""
        data = client.post(
            "/generate", json={"prompt": "dog", "steps": 25, "seed": 42}
        ).json()
        assert data["seed"] == 42

    def test_generate_with_cosine_schedule(self):
        data = client.post(
            "/generate",
            json={"prompt": "sunset", "steps": 100, "schedule": "cosine"},
        ).json()
        assert data["schedule"] == "cosine"

    def test_generate_invalid_schedule(self):
        """Invalid schedule should return 400."""
        response = client.post(
            "/generate",
            json={"prompt": "test", "steps": 50, "schedule": "invalid"},
        )
        assert response.status_code == 400

    def test_generate_includes_timing(self):
        """Response should include elapsed_ms for latency tracking."""
        data = client.post("/generate", json={"prompt": "test", "steps": 10}).json()
        assert "elapsed_ms" in data
        assert isinstance(data["elapsed_ms"], float)

    def test_generate_empty_prompt_rejected(self):
        """Empty prompt should fail validation."""
        response = client.post("/generate", json={"prompt": "", "steps": 50})
        assert response.status_code == 422  # Pydantic validation error

    def test_generate_excessive_steps_rejected(self):
        """Steps > 1000 should fail validation."""
        response = client.post("/generate", json={"prompt": "test", "steps": 5000})
        assert response.status_code == 422


class TestSchedulesEndpoint:
    """Test the noise schedule comparison endpoint."""

    def test_schedules_status(self):
        response = client.get("/schedules")
        assert response.status_code == 200

    def test_schedules_both_present(self):
        data = client.get("/schedules").json()
        assert "linear" in data["schedules"]
        assert "cosine" in data["schedules"]

    def test_schedules_have_stats(self):
        data = client.get("/schedules").json()
        for sched_name in ("linear", "cosine"):
            stats = data["schedules"][sched_name]
            assert "beta_range" in stats
            assert "final_alpha_bar" in stats
            assert "num_steps" in stats
