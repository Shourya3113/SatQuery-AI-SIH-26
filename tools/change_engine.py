"""
SatQuery AI - Bi-Temporal Change Detection & CDVQA Engine
Owner: Peter (Team Leader & AI/ML Lead) / Guided Integration: Chhavi
Subclasses BaseSpecialistTool to perform bi-temporal change detection,
morphological noise filtering, spatial boundary delineation, and change visual question answering.
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy import ndimage
from tools.base import BaseSpecialistTool
from services.geospatial import GeospatialEngine

logger = logging.getLogger("satquery.change_engine")


class BiTemporalChangeEngine(BaseSpecialistTool):
    """
    Bi-Temporal Change Detection & CDVQA Engine.
    Performs multi-channel chromatic and intensity difference analysis,
    morphological jitter filtering, directional alteration classification,
    and natural language change visual question answering.
    """

    def __init__(self):
        super().__init__(
            name="BiTemporalChange-Engine",
            description="Bi-temporal change detection, spatial boundary delineation, and CDVQA reasoning."
        )

    def execute(self, inputs: Dict[str, Any], parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        query = inputs.get("query", "")
        meta_t1 = inputs.get("metadata_t1") or inputs.get("metadata")
        meta_t2 = inputs.get("metadata_t2") or inputs.get("metadata")
        raster_t1 = inputs.get("raster_t1")
        raster_t2 = inputs.get("raster_t2")
        affine_transform = inputs.get("affine_transform")

        change_thresh = float(parameters.get("change_threshold", 0.65))
        pixel_size_m = getattr(meta_t1, "spatial_resolution_m", 10.0) if meta_t1 else 10.0
        bounds = getattr(meta_t1, "bounding_box", None) if meta_t1 else None

        # -------------------------------------------------------------
        # 1. Prepare Arrays and Align Dimensions
        # -------------------------------------------------------------
        def prepare_channels(r) -> np.ndarray:
            if not isinstance(r, np.ndarray) or r.size == 0:
                return np.zeros((1, 256, 256), dtype=np.float32)
            if r.ndim == 2:
                return r[np.newaxis, :, :].astype(np.float32)
            return r.astype(np.float32)

        t1_bands = prepare_channels(raster_t1)
        t2_bands = prepare_channels(raster_t2)

        # Match dimensions
        h = min(t1_bands.shape[1], t2_bands.shape[1])
        w = min(t1_bands.shape[2], t2_bands.shape[2])
        c = min(t1_bands.shape[0], t2_bands.shape[0])

        t1 = t1_bands[:c, :h, :w]
        t2 = t2_bands[:c, :h, :w]

        # Normalize per-channel
        norm1 = np.zeros_like(t1)
        norm2 = np.zeros_like(t2)
        for i in range(c):
            b1, b2 = t1[i], t2[i]
            ptp1 = np.ptp(b1)
            ptp2 = np.ptp(b2)
            norm1[i] = (b1 - np.min(b1)) / (ptp1 + 1e-6) if ptp1 > 1e-6 else b1
            norm2[i] = (b2 - np.min(b2)) / (ptp2 + 1e-6) if ptp2 > 1e-6 else b2

        # -------------------------------------------------------------
        # 2. Multi-Channel Difference & Morphological Filtering
        # -------------------------------------------------------------
        channel_diffs = np.abs(norm2 - norm1)
        raw_diff = np.mean(channel_diffs, axis=0)  # (H, W)

        # Threshold based on change_threshold parameter [0.1, 0.95]
        # Scaled threshold ensures sensitivity across high and low contrast changes
        effective_thresh = max(0.05, change_thresh * 0.40)
        raw_mask = (raw_diff > effective_thresh).astype(np.uint8)

        # Morphological noise filtering: remove single-pixel registration jitter
        # Opening removes isolated false-positive noise pixels; closing bridges contiguous polygons
        struct = ndimage.generate_binary_structure(2, 2)
        cleaned_mask = ndimage.binary_opening(raw_mask, structure=struct, iterations=1)
        cleaned_mask = ndimage.binary_closing(cleaned_mask, structure=struct, iterations=1).astype(np.uint8)

        total_pixels = h * w
        changed_pixels = int(np.sum(cleaned_mask))
        change_pct = round((changed_pixels / max(1, total_pixels)) * 100.0, 2)

        # -------------------------------------------------------------
        # 3. Directional & Semantic Alteration Reasoning
        # -------------------------------------------------------------
        q_lower = query.lower()

        if changed_pixels > 0:
            # Mean intensity delta in changed regions
            t1_mean = float(np.mean(norm1[:, cleaned_mask == 1]))
            t2_mean = float(np.mean(norm2[:, cleaned_mask == 1]))
            delta = t2_mean - t1_mean

            if delta > 0.08:
                change_type = "built-up expansion / surface hardening (reflectance increase)"
                built_status = "increased"
            elif delta < -0.08:
                change_type = "water inundation / vegetation clearance (reflectance decrease)"
                built_status = "decreased"
            else:
                change_type = "mixed land-use modification"
                built_status = "altered"
        else:
            delta = 0.0
            change_type = "stable surface conditions (no detectable change)"
            built_status = "remained unchanged"

        # -------------------------------------------------------------
        # 4. Affine Projection & GeoJSON Vector Layer
        # -------------------------------------------------------------
        layer_name = "bitemporal_change_delineation"

        vector_result = GeospatialEngine.raster_mask_to_geojson(
            binary_mask=cleaned_mask,
            affine_transform=affine_transform,
            layer_name=layer_name,
            pixel_size_m=pixel_size_m,
            crs=getattr(meta_t1, "crs", None) if meta_t1 else None
        )

        total_area_ha = vector_result["metrics"].get("total_area_hectares", 0.0)

        # -------------------------------------------------------------
        # 5. Natural Language CDVQA Synthesis
        # -------------------------------------------------------------
        # Directly answers queries like "Has the built-up area increased, decreased, or remained unchanged?"
        if "increased" in q_lower or "decreased" in q_lower or "unchanged" in q_lower:
            answer = (
                f"Based on bi-temporal change analysis, the target area has {built_status}. "
                f"Detected {total_area_ha:.2f} hectares ({change_pct}% of footprint) undergoing alteration, "
                f"characterized primarily by {change_type}."
            )
        elif changed_pixels == 0:
            answer = (
                f"Bi-temporal change detection verified that the target landscape has remained unchanged "
                f"between the two observation dates (0.00% significant transformation detected)."
            )
        else:
            answer = (
                f"Bi-temporal change analysis detected {total_area_ha:.2f} hectares ({change_pct}% of footprint) "
                f"undergoing significant change between T1 and T2 acquisitions. "
                f"The primary mode of alteration is {change_type}."
            )

        if changed_pixels == 0:
            computed_conf = 0.95
        else:
            contrast = float(np.mean(raw_diff[cleaned_mask == 1])) if changed_pixels > 0 else 0.0
            computed_conf = min(0.95, max(0.60, round(0.50 + contrast * 0.8, 3)))

        output = {
            "answer": answer,
            "confidence": computed_conf,
            "change_percentage": change_pct,
            "changed_area_hectares": total_area_ha,
            "change_type": change_type,
            "built_status": built_status,
            "binary_mask": cleaned_mask,
            "vector_layer": vector_result
        }

        # RS-XAI Integration
        if parameters.get("include_xai", False):
            try:
                from mlops.xai_engine import RSAIXEngine
                xai_engine = RSAIXEngine()
                output["xai_explanation"] = xai_engine.explain_change_detection(
                    raster_t1=b1,
                    raster_t2=b2,
                    change_mask=cleaned_mask,
                    change_pct=change_pct,
                    confidence=computed_conf
                )
            except Exception as e:
                logger.warning("Change detection XAI generation failed: %s", e)

        telemetry = {
            "status": "SUCCESS",
            "model": "Deterministic Bi-Temporal Change Engine",
            "backend": "bitemporal_multi_channel_difference_and_morphology",
            "changed_pixels": changed_pixels,
            "change_percentage": change_pct,
            "threshold_applied": round(effective_thresh, 4),
            "area_hectares": total_area_ha
        }

        return output, telemetry
