"""
SatQuery AI - Centralized Model Manager
Owner: Peter (Team Leader & AI/ML Lead)

Provides lazy-loading, device auto-detection, optional 4-bit quantization,
and memory tracking for all specialist AI models:
  1. Qwen2-VL-2B-Instruct  (VQA / Captioning)
  2. Grounding DINO Tiny    (Text-guided object detection)
  3. SAM ViT-Base           (Segment Anything - mask generation)
  4. CLIP ViT-B/16 + LoRA   (BigEarthNet domain adapter)

Set env var SATQUERY_MOCK_ONLY=1 to skip all model loading (for tests/CI).
"""

import gc
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch

logger = logging.getLogger("satquery.model_manager")

# ---------------------------------------------------------------------------
# Device Detection
# ---------------------------------------------------------------------------

def detect_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_info() -> Dict[str, Any]:
    """Return human-readable device information."""
    device = detect_device()
    info: Dict[str, Any] = {"device": str(device)}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["vram_total_gb"] = round(props.total_memory / 1e9, 2)
        info["vram_free_gb"] = round(torch.cuda.mem_get_info(0)[0] / 1e9, 2)
        info["cuda_version"] = torch.version.cuda or "unknown"
    elif device.type == "mps":
        info["gpu_name"] = "Apple Silicon (MPS)"
        info["vram_total_gb"] = "shared"
    else:
        info["gpu_name"] = "CPU only"
    return info


# ---------------------------------------------------------------------------
# Quantization Config Builder
# ---------------------------------------------------------------------------

def _build_bnb_config() -> Optional[Any]:
    """
    Build a BitsAndBytesConfig for 4-bit quantization.
    Returns None if bitsandbytes is not installed or device is not CUDA.
    """
    if detect_device().type != "cuda":
        return None
    try:
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    except ImportError:
        logger.warning("bitsandbytes not installed — loading models without 4-bit quantization")
        return None
    except Exception as e:
        logger.warning("Failed to create BitsAndBytesConfig: %s", e)
        return None


# ---------------------------------------------------------------------------
# Singleton Model Manager
# ---------------------------------------------------------------------------

class ModelManager:
    """
    Centralized singleton that lazily loads and caches AI models.
    Call .get_<model>() to obtain the (model, processor) pair on first use.
    Call .unload(<name>) or .unload_all() to free VRAM.
    """

    _instance: Optional["ModelManager"] = None

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.mock_only = os.environ.get("SATQUERY_MOCK_ONLY", "0") == "1"
        self.device = detect_device()
        self._cache: Dict[str, Any] = {}
        if self.mock_only:
            logger.info("ModelManager initialized in MOCK-ONLY mode (no models will be loaded)")
        else:
            logger.info("ModelManager initialized — device=%s", self.device)

    def _guard(self):
        """Raise if mock-only mode is active."""
        if self.mock_only:
            raise RuntimeError("SATQUERY_MOCK_ONLY=1 — model loading disabled")

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def vram_used_gb(self) -> float:
        if self.device.type == "cuda":
            return round(torch.cuda.memory_allocated(0) / 1e9, 3)
        return 0.0

    def _log_vram(self, tag: str):
        if self.device.type == "cuda":
            alloc = torch.cuda.memory_allocated(0) / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info("[%s] VRAM: %.2f / %.2f GB", tag, alloc, total)

    # ------------------------------------------------------------------
    # 1. Qwen2-VL-2B-Instruct  (VQA / Captioning)
    # ------------------------------------------------------------------

    def get_vqa_model(self) -> Tuple[Any, Any]:
        """Returns (model, processor) for Qwen2-VL-2B-Instruct."""
        self._guard()
        key = "qwen2_vl"
        if key in self._cache:
            return self._cache[key]

        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        model_id = "Qwen/Qwen2-VL-2B-Instruct"
        logger.info("Loading VQA model: %s", model_id)

        bnb_config = _build_bnb_config()

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
        }
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["device_map"] = "auto"
        elif self.device.type == "cuda":
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
        elif self.device.type == "mps":
            load_kwargs["torch_dtype"] = torch.float16
        # CPU: default float32

        model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)

        if self.device.type == "mps":
            model = model.to(self.device)

        model.eval()

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        self._cache[key] = (model, processor)
        self._log_vram("VQA loaded")
        return model, processor

    # ------------------------------------------------------------------
    # 2. Grounding DINO Tiny  (Zero-shot object detection)
    # ------------------------------------------------------------------

    def get_grounding_dino(self) -> Tuple[Any, Any]:
        """Returns (model, processor) for Grounding DINO Tiny."""
        self._guard()
        key = "grounding_dino"
        if key in self._cache:
            return self._cache[key]

        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        model_id = "IDEA-Research/grounding-dino-tiny"
        logger.info("Loading Grounding DINO: %s", model_id)

        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, torch_dtype=torch.float32
        ).to(self.device)
        model.eval()

        processor = AutoProcessor.from_pretrained(model_id)

        self._cache[key] = (model, processor)
        self._log_vram("Grounding DINO loaded")
        return model, processor

    # ------------------------------------------------------------------
    # 3. SAM ViT-Base  (Segment Anything)
    # ------------------------------------------------------------------

    def get_sam(self) -> Tuple[Any, Any]:
        """Returns (model, processor) for SAM ViT-Base."""
        self._guard()
        key = "sam"
        if key in self._cache:
            return self._cache[key]

        from transformers import SamModel, SamProcessor

        model_id = "facebook/sam-vit-base"
        logger.info("Loading SAM: %s", model_id)

        model = SamModel.from_pretrained(model_id, torch_dtype=torch.float32).to(self.device)
        model.eval()

        processor = SamProcessor.from_pretrained(model_id)

        self._cache[key] = (model, processor)
        self._log_vram("SAM loaded")
        return model, processor

    # ------------------------------------------------------------------
    # 4. BigEarthNet-Adapted Visual Backbone (PEFT LoRA CLIP)
    # ------------------------------------------------------------------

    def get_bigearth_adapter(self) -> Any:
        """Returns the domain-adapted RemoteSensingCLIPAdapter model."""
        self._guard()
        key = "bigearth_adapter"
        if key in self._cache:
            return self._cache[key]

        from mlops.train_adapter import RemoteSensingCLIPAdapter, NUM_CLASSES
        logger.info("Loading BigEarthNet domain-adapted vision adapter")

        adapter = RemoteSensingCLIPAdapter(num_classes=NUM_CLASSES, use_lora=True)
        ckpt_path = Path("models") / "bigearth_adapter.pth"
        if ckpt_path.exists():
            try:
                state_dict = torch.load(str(ckpt_path), map_location=self.device)
                adapter.load_state_dict(state_dict, strict=False)
                logger.info("Loaded weights from %s", ckpt_path)
            except Exception as e:
                logger.warning("Could not load adapter state dict (%s), using base weights", e)

        adapter = adapter.to(self.device)
        adapter.eval()
        self._cache[key] = adapter
        self._log_vram("BigEarthNet adapter loaded")
        return adapter

    # ------------------------------------------------------------------
    # Unloading
    # ------------------------------------------------------------------

    def unload(self, key: str):
        """Unload a specific model from cache to free VRAM."""
        if key in self._cache:
            del self._cache[key]
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Unloaded model: %s", key)

    def unload_all(self):
        """Unload all cached models."""
        keys = list(self._cache.keys())
        self._cache.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Unloaded all models: %s", keys)

    def loaded_models(self) -> list:
        """Return names of currently loaded models."""
        return list(self._cache.keys())

    def status(self) -> Dict[str, Any]:
        """Return full status dict for API exposure."""
        return {
            **get_device_info(),
            "vram_used_gb": self.vram_used_gb(),
            "loaded_models": self.loaded_models(),
        }
