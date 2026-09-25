"""
SatQuery AI - Model Weights Downloader & Offline Cache Verifier
Owner: Achintya (Backend Lead) & Peter (AI/ML Lead)

Pre-downloads all HuggingFace weights to the local cache so the platform
can operate 100% offline in air-gapped defense or disaster management environments.

Supported Models:
  1. VQA / Captioning: Qwen/Qwen2-VL-2B-Instruct
  2. Text Grounding: IDEA-Research/grounding-dino-tiny
  3. Segmentation: facebook/sam-vit-base
  4. Domain Adapter Backbone: openai/clip-vit-base-patch16
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("satquery.download_models")

MODELS_REGISTRY = {
    "vqa": {
        "id": "Qwen/Qwen2-VL-2B-Instruct",
        "description": "Vision-Language Foundation Model for RS VQA & Captioning",
        "type": "vlm",
    },
    "grounding": {
        "id": "IDEA-Research/grounding-dino-tiny",
        "description": "Zero-shot Object Detector for Region Grounding",
        "type": "detection",
    },
    "sam": {
        "id": "facebook/sam-vit-base",
        "description": "Segment Anything Model (SAM) for Pixel Mask Extraction",
        "type": "segmentation",
    },
    "clip": {
        "id": "openai/clip-vit-base-patch16",
        "description": "Base Visual Encoder for BigEarthNet Domain Adaptation",
        "type": "backbone",
    },
}


def download_model(key: str, info: Dict[str, Any]) -> bool:
    model_id = info["id"]
    model_type = info["type"]
    logger.info("=" * 60)
    logger.info("Downloading [%s]: %s", key.upper(), model_id)
    logger.info("Description: %s", info["description"])
    logger.info("=" * 60)

    try:
        from transformers import AutoProcessor, AutoTokenizer

        if model_type == "vlm":
            from transformers import Qwen2VLForConditionalGeneration
            logger.info("Fetching processor...")
            AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            logger.info("Fetching model weights...")
            Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

        elif model_type == "detection":
            from transformers import AutoModelForZeroShotObjectDetection
            logger.info("Fetching processor...")
            AutoProcessor.from_pretrained(model_id)
            logger.info("Fetching model weights...")
            AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

        elif model_type == "segmentation":
            from transformers import SamModel, SamProcessor
            logger.info("Fetching processor...")
            SamProcessor.from_pretrained(model_id)
            logger.info("Fetching model weights...")
            SamModel.from_pretrained(model_id)

        elif model_type == "backbone":
            from transformers import CLIPVisionModelWithProjection, AutoProcessor
            logger.info("Fetching processor...")
            AutoProcessor.from_pretrained(model_id)
            logger.info("Fetching model weights...")
            CLIPVisionModelWithProjection.from_pretrained(model_id)

        logger.info("[SUCCESS] %s cached locally.", model_id)
        return True

    except Exception as e:
        logger.error("[FAILED] Could not download %s: %s", model_id, e)
        return False


def verify_cache() -> Dict[str, bool]:
    """Check which models are currently available in the HuggingFace cache."""
    from huggingface_hub import scan_cache_dir

    logger.info("Scanning local HuggingFace cache...")
    status = {}
    try:
        cache_info = scan_cache_dir()
        cached_repos = {repo.repo_id for repo in cache_info.repos}
        for k, v in MODELS_REGISTRY.items():
            is_cached = v["id"] in cached_repos
            status[k] = is_cached
            state_str = "CACHED" if is_cached else "NOT CACHED"
            logger.info("  [%-10s] %-36s : %s", k, v["id"], state_str)
    except Exception as e:
        logger.warning("Could not scan HuggingFace cache: %s", e)
        for k in MODELS_REGISTRY:
            status[k] = False
    return status


def main():
    parser = argparse.ArgumentParser(description="Pre-download SatQuery AI model weights for offline execution")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "vqa", "grounding", "sam", "clip"],
        help="Specify which model to download (default: all)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only inspect local cache status without downloading",
    )
    args = parser.parse_args()

    if args.check:
        verify_cache()
        return

    targets = list(MODELS_REGISTRY.keys()) if args.model == "all" else [args.model]

    logger.info("Target models to download: %s", targets)
    successes = 0
    for target in targets:
        ok = download_model(target, MODELS_REGISTRY[target])
        if ok:
            successes += 1

    logger.info("=" * 60)
    logger.info("Download Summary: %d / %d models successfully cached", successes, len(targets))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
