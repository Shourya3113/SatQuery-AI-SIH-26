"""
SatQuery AI - Bi-Temporal Change Detection Engine
Owner: Chhavi (AI & Deep Learning Lead)

This engine provides a stable SatQuery interface for bi-temporal
change detection.

Model backends:
    - Original pretrained BiT implementation from BIT_CD.
    - Deterministic pixel-difference fallback.

The BiT backend uses the verified pretrained LEVIR-CD checkpoint.
The fallback is explicitly reported in telemetry and must not be
presented as pretrained BiT inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from services.bit_inference import BiTInferenceAdapter
from services.geospatial import GeospatialEngine
from tools.base import BaseSpecialistTool


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
            Threshold used only by the deterministic fallback.

        checkpoint:
            Optional path to the pretrained BiT checkpoint.
    """

    def __init__(self):
        super().__init__(
            name="BiTemporalChange-Engine",
            description=(
                "Bi-temporal remote-sensing change detection using "
                "pretrained BiT with deterministic fallback."
            ),
        )

        self._bit_model: Optional[BiTInferenceAdapter] = None
        self._bit_load_error: Optional[str] = None

    # ------------------------------------------------------------------
    # BiT backend
    # ------------------------------------------------------------------

    def _get_bit_model(
        self,
        checkpoint: Optional[str] = None,
    ) -> BiTInferenceAdapter:
        """
        Lazily load the pretrained BiT model.

        Lazy loading keeps SatQuery usable in environments where the
        optional BiT dependency/checkpoint is unavailable.
        """
        checkpoint_path = Path(
            checkpoint if checkpoint else DEFAULT_BIT_CHECKPOINT
        )

        if self._bit_model is not None:
            return self._bit_model

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
        """
        Resize the 256x256 BiT mask back to the source raster grid.

        Nearest-neighbour interpolation preserves binary labels.
        """
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

        resized_array = np.asarray(resized)

        return (resized_array > 0).astype(np.uint8)

    def _run_bit(
        self,
        raster_t1: np.ndarray,
        raster_t2: np.ndarray,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run the verified pretrained BiT implementation.
        """
        checkpoint = parameters.get("checkpoint")

        model = self._get_bit_model(checkpoint=checkpoint)

        model_mask, model_change_percentage = model.predict(
            raster_t1=raster_t1,
            raster_t2=raster_t2,
        )

        target_height = int(raster_t1.shape[-2])
        target_width = int(raster_t1.shape[-1])

        native_mask = model_mask.astype(np.uint8)

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
                float(model_change_percentage),
                4,
            ),
            "changed_pixels": changed_pixels,
            "checkpoint": str(
                checkpoint if checkpoint else DEFAULT_BIT_CHECKPOINT
            ),
        }

    # ------------------------------------------------------------------
    # Deterministic fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_2d(raster: Any) -> np.ndarray:
        """
        Extract a representative channel for the deterministic fallback.
        """
        if not isinstance(raster, np.ndarray) or raster.size == 0:
            raise ValueError("Raster input is missing or empty.")

        if raster.ndim == 2:
            return raster.astype(np.float32)

        if raster.ndim != 3:
            raise ValueError(
                "Raster must have shape (bands, height, width)."
            )

        return raster[0].astype(np.float32)

    @staticmethod
    def _normalize_pair(
        arr1: np.ndarray,
        arr2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize both images using one shared intensity range.
        """
        combined = np.concatenate(
            [arr1.reshape(-1), arr2.reshape(-1)]
        )

        finite = combined[np.isfinite(combined)]

        if finite.size == 0:
            raise ValueError(
                "Bi-temporal pair contains no finite pixel values."
            )

        lower = float(np.percentile(finite, 1))
        upper = float(np.percentile(finite, 99))

        if upper - lower < 1e-6:
            return (
                np.zeros_like(arr1, dtype=np.float32),
                np.zeros_like(arr2, dtype=np.float32),
            )

        norm1 = np.clip(
            (arr1 - lower) / (upper - lower),
            0.0,
            1.0,
        ).astype(np.float32)

        norm2 = np.clip(
            (arr2 - lower) / (upper - lower),
            0.0,
            1.0,
        ).astype(np.float32)

        return norm1, norm2

    @staticmethod
    def _resize_to_common_grid(
        arr1: np.ndarray,
        arr2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Put fallback arrays onto a common overlapping grid.
        """
        height = min(arr1.shape[0], arr2.shape[0])
        width = min(arr1.shape[1], arr2.shape[1])

        return (
            arr1[:height, :width],
            arr2[:height, :width],
        )

    @staticmethod
    def _fallback_change_mask(
        raster_t1: np.ndarray,
        raster_t2: np.ndarray,
        threshold: float,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Deterministic pixel-difference fallback.

        Returns:
            binary_mask,
            change_percentage,
            mean_signed_delta
        """
        arr1 = BiTemporalChangeEngine._extract_2d(raster_t1)
        arr2 = BiTemporalChangeEngine._extract_2d(raster_t2)

        arr1, arr2 = BiTemporalChangeEngine._resize_to_common_grid(
            arr1,
            arr2,
        )

        norm1, norm2 = BiTemporalChangeEngine._normalize_pair(
            arr1,
            arr2,
        )

        difference = np.abs(norm2 - norm1)

        binary_mask = (
            difference >= float(threshold)
        ).astype(np.uint8)

        total_pixels = binary_mask.size
        changed_pixels = int(binary_mask.sum())

        change_percentage = (
            changed_pixels / max(1, total_pixels)
        ) * 100.0

        if changed_pixels:
            signed_delta = (
                norm2[binary_mask == 1]
                - norm1[binary_mask == 1]
            )
            mean_signed_delta = float(np.mean(signed_delta))
        else:
            mean_signed_delta = 0.0

        return (
            binary_mask,
            round(change_percentage, 2),
            mean_signed_delta,
        )

    @staticmethod
    def _describe_change(
        mean_signed_delta: float,
        changed_pixels: int,
    ) -> str:
        """
        Describe intensity direction only.

        This does not claim semantic land-cover categories.
        """
        if changed_pixels == 0:
            return "no significant pixel-level change"

        if mean_signed_delta > 0.05:
            return "positive surface-intensity change"

        if mean_signed_delta < -0.05:
            return "negative surface-intensity change"

        return "mixed surface-intensity change"

    # ------------------------------------------------------------------
    # Geospatial output
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_area(
        changed_pixels: int,
        metadata: Optional[Any],
    ) -> float:
        """
        Calculate changed area in hectares from pixel resolution.
        """
        if metadata is None:
            return 0.0

        pixel_size_m = getattr(
            metadata,
            "spatial_resolution_m",
            None,
        )

        if pixel_size_m is None or pixel_size_m <= 0:
            return 0.0

        area_m2 = changed_pixels * (pixel_size_m ** 2)

        return round(area_m2 / 10000.0, 4)

    @staticmethod
    def _build_vector(
        binary_mask: np.ndarray,
        metadata: Optional[Any],
        affine_transform: Optional[Any],
    ) -> Dict[str, Any]:
        """
        Convert a binary change mask into GeoJSON.
        """
        pixel_size_m = (
            getattr(metadata, "spatial_resolution_m", 10.0)
            if metadata
            else 10.0
        )

        return GeospatialEngine.raster_mask_to_geojson(
            binary_mask=binary_mask,
            affine_transform=affine_transform,
            layer_name="bitemporal_change_mask",
            pixel_size_m=pixel_size_m,
        )

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def execute(
        self,
        inputs: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        raster_t1 = inputs.get("raster_t1")
        raster_t2 = inputs.get("raster_t2")

        metadata_t1 = inputs.get("metadata_t1")
        affine_transform = inputs.get("affine_transform")

        if raster_t1 is None or raster_t2 is None:
            raise ValueError(
                "Bi-temporal change detection requires both T1 and T2."
            )

        if not isinstance(raster_t1, np.ndarray):
            raise TypeError("raster_t1 must be a NumPy array.")

        if not isinstance(raster_t2, np.ndarray):
            raise TypeError("raster_t2 must be a NumPy array.")

        if raster_t1.ndim != 3 or raster_t2.ndim != 3:
            raise ValueError(
                "Bi-temporal rasters must have shape "
                "(bands, height, width)."
            )

        backend = str(
            parameters.get("backend", "auto")
        ).lower()

        threshold = float(
            parameters.get("change_threshold", 0.20)
        )

        threshold = min(max(threshold, 0.0), 1.0)

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

                changed_area_hectares = self._calculate_area(
                    changed_pixels=changed_pixels,
                    metadata=metadata_t1,
                )

                vector_layer = self._build_vector(
                    binary_mask=binary_mask,
                    metadata=metadata_t1,
                    affine_transform=affine_transform,
                )

                answer = (
                    f"Pretrained BiT detected "
                    f"{change_percentage}% pixel-level change "
                    f"across the analyzed footprint."
                )

                output = {
                    "answer": answer,
                    "confidence": None,
                    "change_percentage": change_percentage,
                    "changed_area_hectares": changed_area_hectares,
                    "binary_mask": binary_mask,
                    "vector_layer": vector_layer,
                }

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

                if backend == "bit":
                    raise

                bit_error = str(exc)

        else:
            bit_error = None

        # --------------------------------------------------------------
        # Deterministic fallback
        # --------------------------------------------------------------
        if backend != "fallback":
            # If auto mode reaches here, BiT was unavailable/failed.
            # Continue with deterministic fallback.
            pass

        binary_mask, change_percentage, mean_signed_delta = (
            self._fallback_change_mask(
                raster_t1=raster_t1,
                raster_t2=raster_t2,
                threshold=threshold,
            )
        )

        changed_pixels = int(binary_mask.sum())

        change_description = self._describe_change(
            mean_signed_delta=mean_signed_delta,
            changed_pixels=changed_pixels,
        )

        changed_area_hectares = self._calculate_area(
            changed_pixels=changed_pixels,
            metadata=metadata_t1,
        )

        vector_layer = self._build_vector(
            binary_mask=binary_mask,
            metadata=metadata_t1,
            affine_transform=affine_transform,
        )

        answer = (
            f"Deterministic bi-temporal pixel analysis detected "
            f"{change_percentage}% pixel-level change "
            f"({changed_area_hectares:.4f} hectares) "
            f"with {change_description}."
        )

        output = {
            "answer": answer,
            "confidence": None,
            "change_percentage": change_percentage,
            "changed_area_hectares": changed_area_hectares,
            "binary_mask": binary_mask,
            "vector_layer": vector_layer,
        }

        telemetry = {
            "status": "SUCCESS",
            "model": "DETERMINISTIC-FALLBACK",
            "backend": "deterministic_pixel_difference",
            "changed_pixels": changed_pixels,
            "change_percentage": change_percentage,
            "threshold_applied": threshold,
            "mean_signed_delta": round(
                mean_signed_delta,
                6,
            ),
            "bit_load_error": locals().get(
                "bit_error",
                None,
            ),
        }

        return output, telemetry