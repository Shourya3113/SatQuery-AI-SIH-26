"""
SatQuery AI - FastAPI REST Gateway Integration Test Suite
Tests health check, operational stats, settings, and query telemetry.
"""

import io
import pytest
from fastapi.testclient import TestClient
from api.main import app, stats_store


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["isro_problem_id"] == "SIH26167"


def test_stats_endpoint(client):
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_processed" in data
    assert "success_rate" in data
    assert "avg_time" in data
    assert "recent_queries" in data


def test_settings_endpoints(client):
    # Test GET settings
    get_res = client.get("/api/settings")
    assert get_res.status_code == 200
    data = get_res.json()
    assert "google_api_key_masked" in data
    assert "has_key" in data
    assert "cesium_ion_token_masked" in data
    assert "has_cesium_key" in data

    # Test POST settings with Google Key and Cesium Ion Token
    post_res = client.post("/api/settings", json={
        "google_api_key": "AIzaSyDummyKeyForTesting12345",
        "cesium_ion_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dummyCesiumToken12345"
    })
    assert post_res.status_code == 200
    assert post_res.json()["message"] == "Settings updated successfully"

    # Clean up test token so dev environment is not polluted
    client.post("/api/settings", json={"cesium_ion_token": ""})


def test_query_endpoint_with_telemetry(client, tmp_path):
    from PIL import Image
    # Create valid dummy PNG image bytes
    dummy_image = io.BytesIO()
    img = Image.new("RGB", (64, 64), color=(60, 120, 40))
    img.save(dummy_image, format="PNG")
    dummy_image.seek(0)
    dummy_image.name = "test_scene.png"

    initial_processed = stats_store["total_processed"]

    response = client.post(
        "/api/query",
        data={"query": "Identify flood inundation zones in this scene"},
        files={"files": ("test_scene.png", dummy_image, "image/png")}
    )

    assert response.status_code == 200
    data = response.json()

    # Verify unified fields
    assert "trace_id" in data
    assert "query" in data
    assert data["query"] == "Identify flood inundation zones in this scene"
    assert "execution_trace" in data
    trace = data["execution_trace"]
    assert "selected_tool" in trace
    assert "reasoning" in trace
    assert "inputs" in trace
    assert "test_scene.png" in trace["inputs"]
    assert "validation" in data

    # Verify operational stats updated
    assert stats_store["total_processed"] == initial_processed + 1
    assert len(stats_store["recent_queries"]) > 0
    assert stats_store["recent_queries"][0] == "Identify flood inundation zones in this scene"


def test_benchmark_evaluation_endpoint(client):
    response = client.get("/api/benchmarks/evaluate")
    assert response.status_code == 200
    data = response.json()
    assert data["problem_statement"] == "SIH26167 (ISRO / SAC)"
    assert "benchmarks" in data
    assert "BigEarthNet-MM" in data["benchmarks"]
    assert "VRSBench" in data["benchmarks"]
    assert "RSVQA" in data["benchmarks"]
    assert "CDVQA" in data["benchmarks"]
    assert data["summary"]["passed_benchmarks"] == 4
    assert data["summary"]["normalized_composite_score"] >= 80.0

