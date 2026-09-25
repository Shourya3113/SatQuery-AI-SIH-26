"""
"""
SatQuery AI - Bi-Temporal Change Detection & CDVQA Engine
Owner: Chhavi (AI & Deep Learning Lead) / Guided Integration: Peter

Provides bi-temporal remote-sensing change detection using:
    1. Pretrained BiT model for learned change detection.
    2. Deterministic multi-channel fallback with morphology.

The fallback is explicitly reported and must not be presented as
pretrained BiT inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

from services.bit_inference import BiTInferenceAdapter
from services.geospatial import GeospatialEngine
from services.preprocessing import RemoteSensingPreprocessor
from tools.base import BaseSpecialistTool


logger = logging.getLogger("satquery.change_engine")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIT_CHECKPOINT = PROJECT_ROOT / "models" / "best_ckpt.pt"


class BiTemporalChangeEngine(BaseSpecialistTool):
    """
    Bi-temporal change detection specialist.

    Expected inputs:
        raster_t1:
            NumPy array with shape (bands, height, width).

        raster_t2:
            NumPy array with shape (bands, height, width).

        metadata_t1 / metadata_t2:
            InputImageMetadata-like objects.

        affine_transform:
            Rasterio affine transform when available.

    Parameters:
        backend:
            "auto", "bit", or "fallback".

        change_threshold:
            Used by the deterministic fallback.

        checkpoint:
            Optional pretrained BiT checkpoint path.

        query:
            Optional natural-language CDVQA query.

        include_xai:
            Whether to generate an XAI explanation.
    """

    def __init__(self):
        super().__init__(
            name="BiTemporalChange-Engine",
            description=(
                "Bi-temporal remote-sensing change detection using "
                "pretrained BiT with deterministic fallback, with "
                "spatial boundary delineation and CDVQA reasoning."
            ),
        )

        self.preprocessor = RemoteSensingPreprocessor(
            target_size=(256, 256)
        )

        self._bit_model: Optional[BiTInferenceAdapter] = None
        self._bit_load_error: Optional[str] = None

    # ==================================================================
    # BiT backend
    # ==================================================================

    def _get_bit_model(
        self,
        checkpoint: Optional[str] = None,
    ) -> BiTInferenceAdapter:
        """Lazily load the pretrained BiT model."""
        checkpoint_path = Path(
            checkpoint
            if checkpoint
            else DEFAULT_BIT_CHECKPOINT
        )

        if self._bit_model is not None:
            return self._bit_model

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"BiT checkpoint not found: {checkpoint_path}"
            )

        try:
            self._bit_model = BiTInferenceAdapter(
                checkpoint_path=checkpoint_path
            )
            self._bit_load_error = None
            return self._bit_model
        except Exception as exc:
            self._bit_load_error = str(exc)
            raise RuntimeError(
                f"Unable to load pretrained BiT checkpoint: {exc}"
            ) from exc

    @staticmethod
    def _resize_mask_to_raster(
        mask: np.ndarray,
        target_height: int,
        target_width: int,
    ) -> np.ndarray:
        """Resize the BiT mask to the source raster grid."""
        if mask.ndim != 2:
            raise ValueError("BiT mask must be a 2D array.")

        if (
            mask.shape[0] == target_height
            and mask.shape[1] == target_width
        ):
            return (mask > 0).astype(np.uint8)

        mask_image = Image.fromarray(
            (mask > 0).astype(np.uint8) * 255,
            mode="L",
        )

        resized = mask_image.resize(
            (target_width, target_height),
            resample=Image.Resampling.NEAREST,
        )

        return (np.asarray(resized) > 0).astype(np.uint8)

    def _run_bit(
        self,
        raster_t1: np.ndarray,
        raster_t2: np.ndarray,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the verified pretrained BiT model."""
        checkpoint = parameters.get("checkpoint")
        model = self._get_bit_model(checkpoint=checkpoint)

        native_mask, native_change_percentage = model.predict(
            raster_t1=raster_t1,
            raster_t2=raster_t2,
        )

        target_height = int(raster_t1.shape[-2])
        target_width = int(raster_t1.shape[-1])

        native_mask = native_mask.astype(np.uint8)
        output_mask = self._resize_mask_to_raster(
            mask=native_mask,
            target_height=target_height,
            target_width=target_width,
        )

        changed_pixels = int(output_mask.sum())
        change_percentage = (
            changed_pixels / max(1, output_mask.size)
        ) * 100.0

        return {
            "binary_mask": output_mask,
            "native_mask": native_mask,
            "change_percentage": round(change_percentage, 2),
            "native_change_percentage": round(
                float(native_change_percentage), 4
            ),
            "changed_pixels": changed_pixels,
            "checkpoint": str(
                checkpoint
                if checkpoint
                else DEFAULT_BIT_CHECKPOINT
            ),
        }

    # ==================================================================
    # Fallback preparation
    # ==================================================================

    @staticmethod
    def _prepare_channels(raster: Any) -> np.ndarray:
        """Convert raster to float32 CHW representation."""
        if not isinstance(raster, np.ndarray):
            raise TypeError(
                "Raster input must be a NumPy array."
            )

        if raster.size == 0:
            raise ValueError("Raster input is empty.")

        if raster.ndim == 2:
            return raster[np.newaxis, :, :].astype(np.float32)

        if raster.ndim != 3:
            raise ValueError(
                "Raster must have shape (bands, height, width)."
            )

        return raster.astype(np.float32, copy=True)

    @staticmethod
    def _align_fallback_pair(
        t1: np.ndarray,
        t2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align arrays to their common overlapping grid."""
        height = min(t1.shape[1], t2.shape[1])
        width = min(t1.shape[2], t2.shape[2])
        bands = min(t1.shape[0], t2.shape[0])

        if bands <= 0 or height <= 0 or width <= 0:
            raise ValueError(
                "T1 and T2 do not contain a valid overlapping grid."
            )

        return (
            t1[:bands, :height, :width],
            t2[:bands, :height, :width],
        )

    @staticmethod
    def _normalize_multichannel_pair(
        t1: np.ndarray,
        t2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize corresponding channels with shared percentiles."""
        norm1 = np.zeros_like(t1, dtype=np.float32)
        norm2 = np.zeros_like(t2, dtype=np.float32)

        for index in range(t1.shape[0]):
            a = t1[index]
            b = t2[index]

            finite = np.concatenate(
                [a[np.isfinite(a)], b[np.isfinite(b)]]
            )

            if finite.size == 0:
                continue

            lower = float(np.percentile(finite, 1))
            upper = float(np.percentile(finite, 99))

            if upper - lower < 1e-6:
                continue

            norm1[index] = np.clip(
                (a - lower) / (upper - lower),
                0.0,
                1.0,
            )
            norm2[index] = np.clip(
                (b - lower) / (upper - lower),
                0.0,
                1.0,
            )

            norm1[index][~np.isfinite(a)] = 0.0
            norm2[index][~np.isfinite(b)] = 0.0

        return norm1, norm2

    # ==================================================================
    # Deterministic fallback
    # ==================================================================

    def _run_fallback(
        self,
        raster_t1: np.ndarray,
        raster_t2: np.ndarray,
        parameters: Dict[str, Any],
        metadata_t1: Optional[Any],
        affine_transform: Optional[Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run deterministic multi-channel change detection."""
        change_threshold = float(
            parameters.get("change_threshold", 0.65)
        )
        change_threshold = min(
            max(change_threshold, 0.0),
            1.0,
        )

        query = str(parameters.get("query", "")).lower()

        t1 = self._prepare_channels(raster_t1)
        t2 = self._prepare_channels(raster_t2)
        t1, t2 = self._align_fallback_pair(t1, t2)

        norm1, norm2 = self._normalize_multichannel_pair(
            t1,
            t2,
        )

        channel_diffs = np.abs(norm2 - norm1)
        raw_diff = np.mean(channel_diffs, axis=0)

        effective_threshold = max(
            0.05,
            change_threshold * 0.40,
        )

        raw_mask = (
            raw_diff > effective_threshold
        ).astype(np.uint8)

        structure = ndimage.generate_binary_structure(2, 2)

        cleaned_mask = ndimage.binary_opening(
            raw_mask,
            structure=structure,
            iterations=1,
        )
        cleaned_mask = ndimage.binary_closing(
            cleaned_mask,
            structure=structure,
            iterations=1,
        ).astype(np.uint8)

        total_pixels = cleaned_mask.size
        changed_pixels = int(cleaned_mask.sum())
        change_percentage = round(
            (changed_pixels / max(1, total_pixels)) * 100.0,
            2,
        )

        if changed_pixels > 0:
            changed_area = cleaned_mask == 1
            t1_mean = float(np.mean(norm1[:, changed_area]))
            t2_mean = float(np.mean(norm2[:, changed_area]))
            mean_delta = t2_mean - t1_mean

            if mean_delta > 0.08:
                change_type = (
                    "built-up expansion / surface hardening "
                    "(reflectance increase)"
                )
                built_status = "increased"
            elif mean_delta < -0.08:
                change_type = (
                    "water inundation / vegetation clearance "
                    "(reflectance decrease)"
                )
                built_status = "decreased"
            else:
                change_type = "mixed land-use modification"
                built_status = "altered"
        else:
            mean_delta = 0.0
            change_type = (
                "stable surface conditions "
                "(no detectable change)"
            )
            built_status = "remained unchanged"

        pixel_size_m = (
            getattr(
                metadata_t1,
                "spatial_resolution_m",
                10.0,
            )
            if metadata_t1 is not None
            else 10.0
        )

        crs = (
            getattr(metadata_t1, "crs", None)
            if metadata_t1 is not None
            else None
        )

        vector_result = (
            GeospatialEngine.raster_mask_to_geojson(
                binary_mask=cleaned_mask,
                affine_transform=affine_transform,
                layer_name="bitemporal_change_delineation",
                pixel_size_m=pixel_size_m,
                crs=crs,
            )
        )

        total_area_hectares = float(
            vector_result.get(
                "metrics",
                {},
            ).get(
                "total_area_hectares",
                0.0,
            )
        )

        if (
            "increased" in query
            or "decreased" in query
            or "unchanged" in query
        ):
            answer = (
                "Based on bi-temporal change analysis, "
                f"the target area has {built_status}. "
                f"Detected {total_area_hectares:.2f} hectares "
                f"({change_percentage}% of the footprint) "
                f"undergoing alteration, characterized "
                f"primarily by {change_type}."
            )
        elif changed_pixels == 0:
            answer = (
                "Bi-temporal change detection verified that "
                "the target landscape has remained unchanged "
                "between the two observation dates."
            )
        else:
            answer = (
                "Bi-temporal change analysis detected "
                f"{total_area_hectares:.2f} hectares "
                f"({change_percentage}% of the footprint) "
                "undergoing significant change between T1 "
                "and T2 acquisitions. The primary mode of "
                f"alteration is {change_type}."
            )

        if changed_pixels == 0:
            computed_confidence = 0.95
        else:
            contrast = float(
                np.mean(raw_diff[cleaned_mask == 1])
            )
            computed_confidence = min(
                0.95,
                max(
                    0.60,
                    round(0.50 + contrast * 0.80, 3),
                ),
            )

        output = {
            "answer": answer,
            "confidence": computed_confidence,
            "change_percentage": change_percentage,
            "changed_area_hectares": total_area_hectares,
            "change_type": change_type,
            "built_status": built_status,
            "binary_mask": cleaned_mask,
            "vector_layer": vector_result,
        }

        if parameters.get("include_xai", False):
            try:
                from mlops.xai_engine import RSAIXEngine

                xai_engine = RSAIXEngine()
                output["xai_explanation"] = (
                    xai_engine.explain_change_detection(
                        raster_t1=raster_t1,
                        raster_t2=raster_t2,
                        change_mask=cleaned_mask,
                        change_pct=change_percentage,
                        confidence=computed_confidence,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Change detection XAI generation failed: %s",
                    exc,
                )

        telemetry = {
            "status": "SUCCESS",
            "model": "Deterministic Bi-Temporal Change Engine",
            "backend": (
                "bitemporal_multi_channel_difference_and_morphology"
            ),
            "changed_pixels": changed_pixels,
            "change_percentage": change_percentage,
            "threshold_applied": round(
                effective_threshold,
                4,
            ),
            "area_hectares": total_area_hectares,
            "mean_signed_delta": round(mean_delta, 6),
        }

        return output, telemetry

    # ==================================================================
    # Main execution
    # ==================================================================

    def execute(
        self,
        inputs: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Execute bi-temporal change detection."""
        raster_t1 = inputs.get("raster_t1")
        raster_t2 = inputs.get("raster_t2")
        metadata_t1 = inputs.get("metadata_t1")
        affine_transform = inputs.get("affine_transform")

        if raster_t1 is None or raster_t2 is None:
            raise ValueError(
                "Bi-temporal change detection requires both T1 and T2."
            )

        if not isinstance(raster_t1, np.ndarray):
            raise TypeError(
                "raster_t1 must be a NumPy array."
            )

        if not isinstance(raster_t2, np.ndarray):
            raise TypeError(
                "raster_t2 must be a NumPy array."
            )

        if raster_t1.ndim != 3 or raster_t2.ndim != 3:
            raise ValueError(
                "Bi-temporal rasters must have shape "
                "(bands, height, width)."
            )

        backend = str(
            parameters.get("backend", "auto")
        ).lower()

        # --------------------------------------------------------------
        # Pretrained BiT
        # --------------------------------------------------------------
        if backend in {"auto", "bit"}:
            try:
                bit_result = self._run_bit(
                    raster_t1=raster_t1,
                    raster_t2=raster_t2,
                    parameters=parameters,
                )

                binary_mask = bit_result["binary_mask"]
                changed_pixels = bit_result["changed_pixels"]
                change_percentage = bit_result["change_percentage"]

                pixel_size_m = (
                    getattr(
                        metadata_t1,
                        "spatial_resolution_m",
                        10.0,
                    )
                    if metadata_t1 is not None
                    else 10.0
                )

                crs = (
                    getattr(
                        metadata_t1,
                        "crs",
                        None,
                    )
                    if metadata_t1 is not None
                    else None
                )

                changed_area_hectares = round(
                    changed_pixels
                    * (pixel_size_m ** 2)
                    / 10000.0,
                    4,
                ) if pixel_size_m > 0 else 0.0

                vector_layer = (
                    GeospatialEngine.raster_mask_to_geojson(
                        binary_mask=binary_mask,
                        affine_transform=affine_transform,
                        layer_name="bitemporal_change_mask",
                        pixel_size_m=pixel_size_m,
                        crs=crs,
                    )
                )

                output = {
                    "answer": (
                        "Pretrained BiT detected "
                        f"{change_percentage}% pixel-level change "
                        "across the analyzed footprint."
                    ),
                    "confidence": None,
                    "change_percentage": change_percentage,
                    "changed_area_hectares": (
                        changed_area_hectares
                    ),
                    "change_type": (
                        "model-detected pixel-level "
                        "surface alteration"
                    ),
                    "built_status": None,
                    "binary_mask": binary_mask,
                    "vector_layer": vector_layer,
                }

                if parameters.get("include_xai", False):
                    try:
                        from mlops.xai_engine import RSAIXEngine

                        xai_engine = RSAIXEngine()
                        output["xai_explanation"] = (
                            xai_engine.explain_change_detection(
                                raster_t1=raster_t1,
                                raster_t2=raster_t2,
                                change_mask=binary_mask,
                                change_pct=change_percentage,
                                confidence=None,
                            )
                        )
                    except Exception as exc:
                        logger.warning(
                            "BiT XAI generation failed: %s",
                            exc,
                        )

                telemetry = {
                    "status": "SUCCESS",
                    "model": "BiT",
                    "backend": "pretrained_bit",
                    "device": "cpu",
                    "checkpoint": bit_result["checkpoint"],
                    "changed_pixels": changed_pixels,
                    "change_percentage": change_percentage,
                    "native_change_percentage": (
                        bit_result["native_change_percentage"]
                    ),
                    "native_mask_shape": list(
                        bit_result["native_mask"].shape
                    ),
                    "output_mask_shape": list(
                        binary_mask.shape
                    ),
                }

                return output, telemetry

            except Exception as exc:
                self._bit_load_error = str(exc)

                if backend == "bit":
                    raise

        # --------------------------------------------------------------
        # Deterministic fallback
        # --------------------------------------------------------------
        return self._run_fallback(
            raster_t1=raster_t1,
            raster_t2=raster_t2,
            parameters=parameters,
            metadata_t1=metadata_t1,
            affine_transform=affine_transform,
        )
