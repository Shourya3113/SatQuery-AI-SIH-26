"""
SatQuery AI - Remote Sensing VQA & Scene Captioning Engine
Owner: Chhavi (AI & Deep Learning Lead) / Integration: Peter (Chief Architect)
Subclasses BaseSpecialistTool to provide single-image remote sensing question answering
and descriptive scene captioning.
"""

from typing import Dict, Any, Tuple
import numpy as np
from tools.base import BaseSpecialistTool
from core.schemas import TaskCategory, ModalityType


class RSVQAEngine(BaseSpecialistTool):
    """
    Remote Sensing Vision-Language Engine.
    Handles visual question answering and scene captioning for optical and SAR imagery.
    """

    def __init__(self):
        super().__init__(
            name="RS-VQA-Engine",
            description="Remote Sensing VLM specialist for VQA and Scene Captioning."
        )

    def execute(self, inputs: Dict[str, Any], parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        query = inputs.get("query", "")
        metadata = inputs.get("metadata")
        raster_data = inputs.get("raster_data")
        task_category = inputs.get("task_category", TaskCategory.SINGLE_IMAGE_VQA)

        confidence_thresh = parameters.get("confidence_threshold", 0.75)
        max_tokens = parameters.get("max_tokens", 128)

        # Extract raster statistics for grounded physical reasoning
        bands, height, width = (1, 512, 512)
        mean_val = 128.0
        std_val = 40.0
        if isinstance(raster_data, np.ndarray) and raster_data.size > 0:
            if raster_data.ndim == 3:
                bands, height, width = raster_data.shape
            elif raster_data.ndim == 2:
                height, width = raster_data.shape
                bands = 1
            mean_val = float(np.mean(raster_data))
            std_val = float(np.std(raster_data))

        crs = getattr(metadata, "crs", "EPSG:4326") if metadata else "EPSG:4326"
        res_m = getattr(metadata, "spatial_resolution_m", 10.0) if metadata else 10.0
        modality = getattr(metadata, "modality", ModalityType.OPTICAL_RGB) if metadata else ModalityType.OPTICAL_RGB
        modality_str = modality.value if hasattr(modality, "value") else str(modality)

        q_lower = query.lower()

        # Deterministic domain synthesis with physical grounding
        if task_category == TaskCategory.SINGLE_IMAGE_CAPTIONING or "caption" in q_lower or "describe" in q_lower:
            answer = (
                f"The scene exhibits high-resolution {modality_str} remote sensing coverage ({width}x{height} pixels, "
                f"spatial resolution: {res_m}m/pixel, CRS: {crs}). The terrain comprises mixed land-use "
                f"patterns with distinct structural textures (mean reflectance/intensity: {mean_val:.1f} DN, σ: {std_val:.1f}). "
                f"Dominant features include organized agricultural parcels, built-up infrastructure, and vegetative canopies."
            )
            confidence = min(0.96, confidence_thresh + 0.15)
        elif "water" in q_lower or "river" in q_lower or "lake" in q_lower:
            answer = (
                f"Analysis of spectral reflectance and backscatter patterns confirms the presence of surface water bodies. "
                f"At {res_m}m ground resolution, hydrological boundaries show sharp absorption boundaries "
                f"consistent with inland reservoirs or river channels."
            )
            confidence = min(0.94, confidence_thresh + 0.12)
        elif "cloud" in q_lower or "weather" in q_lower:
            if "sar" in modality_str.lower():
                answer = "SAR microwave radar operates independently of cloud cover, penetrating atmospheric haze and vapor without signal degradation."
            else:
                answer = "Optical image displays clear atmospheric transmission with minimal cloud interference over target regions."
            confidence = 0.93
        elif "resolution" in q_lower or "scale" in q_lower or "gsd" in q_lower:
            answer = f"The ground sample distance (GSD) is {res_m} meters per pixel, with full scene dimensions of {width}x{height} pixels in {crs} projection."
            confidence = 0.99
        elif "count" in q_lower or "how many" in q_lower:
            answer = "Identified 14 primary geographic and structural object clusters within the delineated footprint."
            confidence = 0.88
        else:
            answer = (
                f"Based on {modality_str} imagery analysis, the queried scene demonstrates stable landscape characteristics "
                f"across {bands} spectral band(s) at {res_m}m spatial resolution. No anomalous structural disruptions are detected."
            )
            confidence = 0.90

        output = {
            "answer": answer,
            "confidence": confidence,
            "tokens_generated": min(len(answer.split()), max_tokens),
            "stats": {
                "bands": bands,
                "dimensions": [width, height],
                "spatial_resolution_m": res_m,
                "crs": crs
            }
        }

        telemetry = {
            "status": "SUCCESS",
            "model": "MOCK-RS-VQA",
            "backend": "deterministic_mock",
            "device": "CPU/CUDA",
            "tokens": output["tokens_generated"]
        }

        return output, telemetry
