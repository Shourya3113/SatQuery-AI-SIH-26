"""
SatQuery AI - Remote Sensing Preprocessing
Owner: Chhavi (AI & Deep Learning Lead)

Converts raster inputs into model-ready PyTorch tensors while
preserving the information required for downstream geospatial
processing.

This module currently focuses on deterministic preprocessing.
The actual BIT model is integrated separately.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class RemoteSensingPreprocessor:
    """
    Preprocessing utilities for bi-temporal remote-sensing imagery.

    Expected internal raster layout:
        (bands, height, width)

    Expected model tensor layout:
        (batch, channels, height, width)
    """

    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_height = target_size[0]
        self.target_width = target_size[1]

    @staticmethod
    def validate_raster(raster: np.ndarray) -> None:
        """
        Validate that a raster is a non-empty NumPy array with
        band-first layout.
        """
        if not isinstance(raster, np.ndarray):
            raise TypeError("Raster must be a NumPy array.")

        if raster.size == 0:
            raise ValueError("Raster is empty.")

        if raster.ndim != 3:
            raise ValueError(
                "Raster must have shape (bands, height, width)."
            )

        bands, height, width = raster.shape

        if bands < 1:
            raise ValueError("Raster must contain at least one band.")

        if height < 1 or width < 1:
            raise ValueError("Raster dimensions must be greater than zero.")

    @staticmethod
    def _replace_invalid_values(raster: np.ndarray) -> np.ndarray:
        """
        Replace NaN and infinite values with finite values.

        NaN/Inf values can occur in remote-sensing products because of
        nodata pixels or invalid measurements.
        """
        raster = raster.astype(np.float32, copy=True)

        finite_mask = np.isfinite(raster)

        if not finite_mask.any():
            raise ValueError("Raster contains no finite pixel values.")

        finite_values = raster[finite_mask]

        replacement = float(np.median(finite_values))

        raster[~finite_mask] = replacement

        return raster

    @staticmethod
    def _select_channels(raster: np.ndarray) -> np.ndarray:
        """
        Convert different raster modalities into a consistent
        three-channel representation.

        Rules:
        - 1 band  -> replicate the band into 3 channels
        - 2 bands -> use both bands and repeat the second channel
        - 3 bands -> keep all 3
        - 4+ bands -> use the first 3 bands

        This is an interface-level conversion for a BIT model that
        expects three-channel image inputs. It does not claim that
        arbitrary multispectral bands are spectrally equivalent to RGB.
        """
        bands = raster.shape[0]

        if bands == 1:
            return np.repeat(raster, 3, axis=0)

        if bands == 2:
            return np.concatenate(
                [raster, raster[1:2]],
                axis=0
            )

        if bands == 3:
            return raster

        return raster[:3]

    @staticmethod
    def _normalize_per_channel(raster: np.ndarray) -> np.ndarray:
        """
        Normalize each channel independently to [0, 1].

        Per-channel normalization avoids one high-dynamic-range band
        dominating the other channels.
        """
        raster = raster.astype(np.float32, copy=False)

        normalized = np.empty_like(raster)

        for channel in range(raster.shape[0]):
            band = raster[channel]

            band_min = float(np.min(band))
            band_max = float(np.max(band))

            if band_max - band_min < 1e-6:
                normalized[channel] = 0.0
            else:
                normalized[channel] = (
                    (band - band_min) /
                    (band_max - band_min)
                )

        return normalized

    def to_tensor(
        self,
        raster: np.ndarray,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Convert a raster to a BIT-compatible tensor.

        Returns:
            Tensor with shape:
            (1, 3, target_height, target_width)
        """
        self.validate_raster(raster)

        raster = self._replace_invalid_values(raster)
        raster = self._select_channels(raster)

        if normalize:
            raster = self._normalize_per_channel(raster)

        tensor = torch.from_numpy(
            np.ascontiguousarray(raster)
        ).float()

        # Add batch dimension:
        # (C, H, W) -> (1, C, H, W)
        tensor = tensor.unsqueeze(0)

        # Resize spatial dimensions while preserving channels.
        if (
            tensor.shape[-2] != self.target_height
            or tensor.shape[-1] != self.target_width
        ):
            tensor = F.interpolate(
                tensor,
                size=(self.target_height, self.target_width),
                mode="bilinear",
                align_corners=False
            )

        return tensor

    def prepare_bitemporal_pair(
        self,
        raster_t1: np.ndarray,
        raster_t2: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare two temporal acquisitions for a change-detection model.

        Both images receive the same spatial target size and channel
        convention.
        """
        tensor_t1 = self.to_tensor(raster_t1)
        tensor_t2 = self.to_tensor(raster_t2)

        return tensor_t1, tensor_t2

    def describe_tensor(
        self,
        tensor: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Return basic diagnostic information for debugging and telemetry.
        """
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "min": round(float(tensor.min().item()), 6),
            "max": round(float(tensor.max().item()), 6),
            "mean": round(float(tensor.mean().item()), 6),
        }