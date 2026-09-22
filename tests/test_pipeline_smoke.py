"""
SatQuery AI - End-to-End Smoke Test Suite
Owner: Pradipti (Research, Benchmarks, QA & Pitch Lead)
"""

import pytest
from core.schemas import ModalityType, TaskCategory, InputImageMetadata
from services.orchestrator import AgenticTaskRouter


def test_agentic_task_router_classification():
    router = AgenticTaskRouter()

    # Test single-image captioning
    task1 = router.classify_task("Describe the land cover and dominant objects visible in this scene", 1, [ModalityType.OPTICAL_RGB])
    assert task1 == TaskCategory.SINGLE_IMAGE_CAPTIONING

    # Test spatial grounding
    task2 = router.classify_task("Highlight the water body referred to in the query", 1, [ModalityType.OPTICAL_RGB])
    assert task2 == TaskCategory.TEXT_GUIDED_GROUNDING

    # Test bi-temporal change
    task3 = router.classify_task("What changed between these two dates and where did change occur?", 2, [ModalityType.OPTICAL_RGB, ModalityType.OPTICAL_RGB])
    assert task3 == TaskCategory.BI_TEMPORAL_CHANGE_DETECTION

    # Test optical-SAR fusion
    task4 = router.classify_task("Use the optical and SAR images together to identify built-up and water-covered regions", 2, [ModalityType.OPTICAL_RGB, ModalityType.SAR_C_BAND])
    assert task4 == TaskCategory.CROSS_MODAL_JOINT_ANALYSIS

    metadata = InputImageMetadata(
        filename="test.tif",
        format="GeoTIFF",
        modality=ModalityType.OPTICAL_MULTISPECTRAL,
        width=512,
        height=512,
        bands=4
    )

    band_result = router.validate_band_count(metadata)

    assert band_result["valid"] is True
    assert band_result["bands"] == 4


def test_parameter_bounding_guardrails():
    router = AgenticTaskRouter()
    raw_params = {
        "confidence_threshold": 1.5,  # Out of bounds
        "change_threshold": -0.2,     # Out of bounds
        "speckle_filter_kernel": 9    # Not in allowed [3, 5, 7]
    }
    bounded = router.validate_and_bound_parameters(raw_params)
    assert bounded["confidence_threshold"] <= 0.99
    assert bounded["change_threshold"] >= 0.1
    assert bounded["speckle_filter_kernel"] == 5  # Reverts to default
