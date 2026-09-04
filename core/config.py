"""
SatQuery AI Configuration & Settings
ISRO Problem Statement: SIH26167
Owner: Peter (Team Leader & Architect) & Achintya (Backend Lead)
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
OUTPUTS_DIR = DATA_DIR / "outputs"

for path in [UPLOADS_DIR, PROCESSED_DIR, SAMPLES_DIR, OUTPUTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Supported Image Formats
GEOSPATIAL_EXTENSIONS = {".tif", ".tiff", ".geotiff"}
BENCHMARK_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALLOWED_EXTENSIONS = GEOSPATIAL_EXTENSIONS.union(BENCHMARK_EXTENSIONS)

# Supported Datasets
DATASETS = {
    "BIGEARTHNET": "BigEarthNet-MM (Sentinel-1 SAR + Sentinel-2 Optical)",
    "VRSBENCH": "VRSBench (Remote Sensing Captioning & Grounding)",
    "RSVQA": "RSVQA (High & Low Resolution VQA)",
    "CDVQA": "CDVQA (Change Detection Visual Question Answering)",
    "ISRO_EVAL": "Cartosat-2S & RISAT-1A / EOS-04 Pairs"
}

# Permitted Task Parameters Bounds (Mandatory ISRO Guardrails)
PERMITTED_PARAMETERS = {
    "confidence_threshold": {"type": float, "min": 0.1, "max": 0.99, "default": 0.75},
    "change_threshold": {"type": float, "min": 0.1, "max": 0.95, "default": 0.65},
    "max_tokens": {"type": int, "min": 16, "max": 512, "default": 128},
    "speckle_filter_kernel": {"type": int, "allowed": [3, 5, 7], "default": 5},
    "target_crs": {"type": str, "default": "EPSG:4326"}
}
