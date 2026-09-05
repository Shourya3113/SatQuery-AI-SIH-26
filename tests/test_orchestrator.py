"""
SatQuery AI - Central Orchestrator & Integration Test Suite
Owner: Peter (Team Leader, Chief Architect) & Pradipti (QA Lead)
Tests all 5 task categories, parameter bounding guardrails, Affine GeoJSON generation,
FastAPI endpoints, and ReportLab PDF dossier synthesis.
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from core.schemas import TaskCategory, ModalityType, QueryResponse
from services.orchestrator import AgenticTaskRouter
from api.main import app, execution_traces, stored_responses
from core.config import UPLOADS_DIR, OUTPUTS_DIR


@pytest.fixture(scope="session")
def setup_test_samples():
    """Ensure test sample rasters exist in UPLOADS_DIR."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    optical_path = UPLOADS_DIR / "test_opt_eval.png"
    sar_path = UPLOADS_DIR / "test_sar_eval.png"

    # Create dummy optical 3-channel image
    opt_img = Image.new("RGB", (256, 256), color=(60, 120, 40))
    opt_img.save(optical_path)

    # Create dummy SAR single-channel image
    sar_img = Image.new("L", (256, 256), color=100)
    sar_img.save(sar_path)

    return optical_path, sar_path


def test_orchestrator_task_classification():
    router = AgenticTaskRouter()

    # 1. Captioning
    cat1 = router.classify_task(
        "Describe the land cover and dominant terrain features",
        1,
        [ModalityType.OPTICAL_RGB]
    )
    assert cat1 == TaskCategory.SINGLE_IMAGE_CAPTIONING

    # 2. VQA
    cat2 = router.classify_task(
        "What is the spatial resolution of this imagery?",
        1,
        [ModalityType.OPTICAL_RGB]
    )
    assert cat2 == TaskCategory.SINGLE_IMAGE_VQA

    # 3. Grounding
    cat3 = router.classify_task(
        "Highlight and delineate the boundary of all water bodies",
        1,
        [ModalityType.OPTICAL_RGB]
    )
    assert cat3 == TaskCategory.TEXT_GUIDED_GROUNDING

    # 4. Bi-Temporal Change
    cat4 = router.classify_task(
        "Detect the change between these two dates and quantify expansion",
        2,
        [ModalityType.OPTICAL_RGB, ModalityType.OPTICAL_RGB]
    )
    assert cat4 == TaskCategory.BI_TEMPORAL_CHANGE_DETECTION

    # 5. Cross-Modal Joint Analysis
    cat5 = router.classify_task(
        "Use the optical and SAR images together to identify built-up structures",
        2,
        [ModalityType.OPTICAL_RGB, ModalityType.SAR_C_BAND]
    )
    assert cat5 == TaskCategory.CROSS_MODAL_JOINT_ANALYSIS


def test_parameter_bounding_guardrails():
    router = AgenticTaskRouter()

    raw_params = {
        "confidence_threshold": 2.5,   # max allowed 0.99
        "change_threshold": -0.8,      # min allowed 0.1
        "speckle_filter_kernel": 9,    # not in allowed [3, 5, 7]
        "max_tokens": 1000             # max allowed 512
    }

    bounded = router.validate_and_bound_parameters(raw_params)
    assert bounded["confidence_threshold"] == 0.99
    assert bounded["change_threshold"] == 0.1
    assert bounded["speckle_filter_kernel"] == 5  # reverted to default
    assert bounded["max_tokens"] == 512


def test_single_image_captioning_pipeline(setup_test_samples):
    opt_path, _ = setup_test_samples
    router = AgenticTaskRouter()

    res = router.process_query(
        query="Describe the terrain and dominant land use visible in this image",
        file_paths=[opt_path]
    )

    assert isinstance(res, QueryResponse)
    assert res.task_category == TaskCategory.SINGLE_IMAGE_CAPTIONING
    assert len(res.text_response) > 20
    assert res.confidence_score > 0.70
    assert len(res.execution_trace.orchestration["pipeline_steps"]) >= 2
    assert res.execution_trace.input_audit["count"] == 1


def test_spatial_grounding_pipeline(setup_test_samples):
    opt_path, _ = setup_test_samples
    router = AgenticTaskRouter()

    res = router.process_query(
        query="Highlight and segment the water body in this image",
        file_paths=[opt_path]
    )

    assert res.task_category == TaskCategory.TEXT_GUIDED_GROUNDING
    assert len(res.vector_layers) >= 1
    vl = res.vector_layers[0]
    assert vl.geojson["type"] == "FeatureCollection"
    assert len(vl.geojson["features"]) >= 1
    assert vl.metrics["total_area_hectares"] > 0
    assert "SAM-2" in res.text_response or "delineated" in res.text_response


def test_bitemporal_change_pipeline(setup_test_samples):
    opt_path, _ = setup_test_samples
    router = AgenticTaskRouter()

    res = router.process_query(
        query="What changed between these two temporal acquisitions?",
        file_paths=[opt_path, opt_path],
        raw_params={"change_threshold": 0.5}
    )

    assert res.task_category == TaskCategory.BI_TEMPORAL_CHANGE_DETECTION
    assert len(res.vector_layers) >= 1
    assert "change" in res.text_response.lower()
    assert res.execution_trace.input_audit["count"] == 2


def test_cross_modal_fusion_pipeline(setup_test_samples):
    opt_path, sar_path = setup_test_samples
    router = AgenticTaskRouter()

    res = router.process_query(
        query="Use optical and SAR together to detect built-up and water covered regions",
        file_paths=[opt_path, sar_path],
        raw_params={"speckle_filter_kernel": 5}
    )

    assert res.task_category == TaskCategory.CROSS_MODAL_JOINT_ANALYSIS
    assert len(res.vector_layers) >= 1
    assert res.confidence_score >= 0.85
    assert any("built" in vl.layer_name or "water" in vl.layer_name for vl in res.vector_layers)


def test_api_full_roundtrip_and_pdf_export(setup_test_samples):
    client = TestClient(app)
    opt_path, _ = setup_test_samples

    # 1. Health check
    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "HEALTHY"

    # 2. Process query via REST API
    with open(opt_path, "rb") as f:
        query_resp = client.post(
            "/api/query",
            data={
                "query": "Delineate and highlight all vegetation parcels",
                "confidence_threshold": 0.80
            },
            files={"image": ("test_opt_eval.png", f, "image/png")}
        )

    assert query_resp.status_code == 200
    data = query_resp.json()
    trace_id = data["trace_id"]
    assert data["task_category"] == "TEXT_GUIDED_GROUNDING"
    assert len(data["vector_layers"]) >= 1

    # 3. Retrieve auditable trace
    trace_resp = client.get(f"/api/trace/{trace_id}")
    assert trace_resp.status_code == 200
    trace_data = trace_resp.json()
    assert trace_data["trace_id"] == trace_id
    assert trace_data["user_query"] == "Delineate and highlight all vegetation parcels"

    # 4. Generate intelligence dossier PDF
    pdf_resp = client.post(f"/api/export-report/{trace_id}")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    assert len(pdf_resp.content) > 1000
