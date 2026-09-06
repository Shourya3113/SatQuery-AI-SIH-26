"""
SatQuery AI - GPU Memory & Inference Resource Manager
Owner: Chhavi (AI & Deep Learning Lead + MLOps)

Provides CUDA detection, device selection, memory monitoring,
and CUDA cache cleanup to reduce GPU out-of-memory failures.
"""

from typing import Dict, Any

import torch


class GPUManager:
    """
    Lightweight GPU resource manager for SatQuery AI inference.
    """

    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")

    def get_device(self) -> torch.device:
        """Return the best available inference device."""
        return self.device

    def get_device_name(self) -> str:
        """Return the CUDA GPU name or CPU."""
        if self.cuda_available:
            return torch.cuda.get_device_name(0)
        return "CPU"

    def get_memory_info(self) -> Dict[str, Any]:
        """
        Return current GPU memory statistics.

        Values are reported in MB. On CPU-only systems,
        zero values are returned.
        """
        if not self.cuda_available:
            return {
                "device": "CPU",
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "free_mb": 0.0,
                "total_mb": 0.0,
            }

        allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        free = max(0.0, total - reserved)

        return {
            "device": self.get_device_name(),
            "allocated_mb": round(allocated, 2),
            "reserved_mb": round(reserved, 2),
            "free_mb": round(free, 2),
            "total_mb": round(total, 2),
        }

    def clear_cache(self) -> None:
        """Release unused CUDA cached memory."""
        if self.cuda_available:
            torch.cuda.empty_cache()

    def cleanup(self) -> None:
        """
        Perform lightweight GPU cleanup after inference.
        """
        self.clear_cache()

    def status(self) -> Dict[str, Any]:
        """Return a complete GPU manager status report."""
        return {
            "cuda_available": self.cuda_available,
            "device": str(self.device),
            "device_name": self.get_device_name(),
            "memory": self.get_memory_info(),
        }


gpu_manager = GPUManager()