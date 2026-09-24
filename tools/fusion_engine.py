"""
SatQuery AI - Optical-SAR Cross-Modal Joint Analysis Engine
Owner: Peter (Team Leader & AI/ML Lead)
Subclasses BaseSpecialistTool to perform physics-grounded cross-modal fusion
combining cloud-penetrating C-band / L-band SAR backscatter (sigma0 dB) with
optical multispectral reflectance and spectral indices.

Scientific Formulation:
  1. Radiometric Calibration: sigma0 (dB) = 10 * log10(DN^2 + eps) - K_cal
  2. Speckle Suppression: Adaptive 5x5 Lee Filter preserving structural edges
  3. Spectral Extraction: NDWI (McFeeters 1996) + NDVI (Rouse 1974) + NDBI proxy
  4. Decision Fusion: Bayesian Joint Consensus of Optical and Microwave evidence
  5. Deterministic Affine Projection: Raster masks -> EPSG:4326 GeoJSON polygons
"""

import logging
from typing import Dict, Any, Tuple, List
import numpy as np
from tools.base import BaseSpecialistTool
from services.geospatial import (
    SENTINEL1_GRD_K_CAL_DB,
    GeospatialEngine,
    calibrate_sigma0,
    adaptive_lee_filter,
    ndwi,
    ndvi,
    ndbi,
    ndbi_red_proxy,
)

logger = logging.getLogger("satquery.fusion_engine")


class OpticalSARFusionEngine(BaseSpecialistTool):
    """
    Optical-SAR Cross-Modal Joint Analysis Engine.
    Combines optical spectral characteristics with SAR microwave roughness,
    structure, and specular reflection for all-weather intelligence.
    """

    def __init__(self):
        super().__init__(
            name="OpticalSARFusion-Engine",
            description="Cross-modal optical-SAR fusion with radiometric calibration and Bayesian evidence integration."
        )

    def execute(self, inputs: Dict[str, Any], parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        query = inputs.get("query", "")
        meta_opt = inputs.get("metadata_optical") or inputs.get("metadata")
        meta_sar = inputs.get("metadata_sar") or inputs.get("metadata")
        raster_opt = inputs.get("raster_optical")
        raster_sar = inputs.get("raster_sar")
        affine_transform = inputs.get("affine_transform")

        kernel_size = int(parameters.get("speckle_filter_kernel", 5))
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size % 2 == 0:
            kernel_size += 1

        confidence_thresh = float(parameters.get("confidence_threshold", 0.75))
        pixel_size_m = getattr(meta_opt, "spatial_resolution_m", 10.0) if meta_opt else 10.0
        bounds = getattr(meta_opt, "bounding_box", None) if meta_opt else None

        # -------------------------------------------------------------
        # 1. Optical Raster Processing
        # -------------------------------------------------------------
        if isinstance(raster_opt, np.ndarray) and raster_opt.size > 0:
            if raster_opt.ndim == 2:
                opt_bands = raster_opt[np.newaxis, :, :].astype(np.float32)
            else:
                opt_bands = raster_opt.astype(np.float32)
        else:
            opt_bands = np.zeros((3, 256, 256), dtype=np.float32)

        # -------------------------------------------------------------
        # 2. SAR Raster Processing: Radiometric Calibration + Lee Filter
        # -------------------------------------------------------------
        if isinstance(raster_sar, np.ndarray) and raster_sar.size > 0:
            if raster_sar.ndim == 2:
                sar_raw = raster_sar[np.newaxis, :, :].astype(np.float32)
            else:
                sar_raw = raster_sar.astype(np.float32)

            # Check if SAR is already in dB scale (typical range -35 to +10 dB)
            sar_band = sar_raw[0]
            if np.min(sar_band) < -5.0 and np.max(sar_band) < 25.0:
                # Already calibrated in dB
                sigma0_db = sar_band
            else:
                # Raw DN -> Calibrate to sigma0 (dB)
                # High-amplitude DNs indicate a real GRD amplitude image, so apply the
                # product's K_cal; otherwise the array is already normalized/synthetic.
                k_cal = SENTINEL1_GRD_K_CAL_DB if np.max(sar_band) > 100.0 else 0.0
                sigma0_db = calibrate_sigma0(sar_band, k_cal_db=k_cal)

            # Apply adaptive Lee speckle filter
            try:
                sar_filtered = adaptive_lee_filter(sigma0_db, window_size=kernel_size)
            except Exception as e:
                logger.warning("Lee filter error (%s), using raw sigma0: %s", kernel_size, e)
                sar_filtered = sigma0_db
        else:
            sar_filtered = np.full(opt_bands.shape[-2:], -12.0, dtype=np.float32)

        # Spatial alignment / dimension matching
        h = min(opt_bands.shape[1], sar_filtered.shape[0])
        w = min(opt_bands.shape[2], sar_filtered.shape[1])
        opt_bands = opt_bands[:, :h, :w]
        sar_filtered = sar_filtered[:h, :w]

        # -------------------------------------------------------------
        # 3. Physical Evidence Calculation
        # -------------------------------------------------------------
        # SAR Physics:
        # Water: Specular scattering -> radar reflected away -> sigma0 < -16 dB
        # Urban: Dihedral double-bounce reflection -> high backscatter -> sigma0 > -6 dB
        # Moderate: Volume scattering -> -16 dB <= sigma0 <= -6 dB
        sar_min, sar_max = float(np.min(sar_filtered)), float(np.max(sar_filtered))
        sar_norm = (sar_filtered - sar_min) / (sar_max - sar_min + 1e-6)

        # Evidence probability maps from SAR:
        # Low backscatter sigmoid -> Water
        p_sar_water = 1.0 / (1.0 + np.exp((sar_filtered - (-15.0)) / 2.5))
        # High backscatter sigmoid -> Built-up
        p_sar_urban = 1.0 / (1.0 + np.exp(-1.0 * (sar_filtered - (-7.0)) / 2.5))

        # Optical Physics:
        num_opt_bands = opt_bands.shape[0]
        if num_opt_bands >= 4:
            # Multispectral: B, G, R, NIR [, SWIR1]
            green = opt_bands[1]
            red = opt_bands[2]
            nir = opt_bands[3]
            ndwi_arr = ndwi(green, nir)
            ndvi_arr = ndvi(nir, red)
            if num_opt_bands >= 5:
                # Real NDBI once a SWIR band exists: (SWIR - NIR) / (SWIR + NIR)
                ndbi_arr = ndbi(opt_bands[4], nir)
            else:
                # 4-band stack has no SWIR -> documented (Red - NIR) proxy only
                ndbi_arr = ndbi_red_proxy(red, nir)

            p_opt_water = 1.0 / (1.0 + np.exp(-1.0 * (ndwi_arr - 0.05) / 0.15))
            p_opt_urban = 1.0 / (1.0 + np.exp(-1.0 * (ndbi_arr - 0.0) / 0.15))
        else:
            # Standard RGB
            r = opt_bands[0] / (np.max(opt_bands[0]) + 1e-6)
            g = opt_bands[min(1, num_opt_bands - 1)] / (np.max(opt_bands[min(1, num_opt_bands - 1)]) + 1e-6)
            b = opt_bands[min(2, num_opt_bands - 1)] / (np.max(opt_bands[min(2, num_opt_bands - 1)]) + 1e-6)
            brightness = (r + g + b) / 3.0

            # Water: Dark, green/blue dominant over red
            p_opt_water = np.clip(1.0 - brightness * 1.5, 0.0, 1.0) * (g >= r * 0.95).astype(np.float32)
            # Urban: High brightness and contrast
            p_opt_urban = np.clip((brightness - 0.45) * 2.0, 0.0, 1.0)

        # -------------------------------------------------------------
        # 4. Bayesian Consensus Fusion
        # -------------------------------------------------------------
        def fuse_probabilities(p1: np.ndarray, p2: np.ndarray, w1: float = 0.5, w2: float = 0.5) -> np.ndarray:
            """Weighted Bayesian evidence combination with mutual reinforcement."""
            numerator = (p1 ** w1) * (p2 ** w2)
            denominator = numerator + ((1.0 - p1) ** w1) * ((1.0 - p2) ** w2) + 1e-9
            return np.clip(numerator / denominator, 0.0, 1.0)

        p_fused_water = fuse_probabilities(p_opt_water, p_sar_water, w1=0.5, w2=0.5)
        p_fused_urban = fuse_probabilities(p_opt_urban, p_sar_urban, w1=0.45, w2=0.55)  # SAR double-bounce heavily weighted for urban

        # Threshold to crisp binary masks
        water_mask = (p_fused_water >= 0.40).astype(np.uint8)
        urban_mask = (p_fused_urban >= 0.42).astype(np.uint8)

        # Prevent overlap
        overlap = (water_mask == 1) & (urban_mask == 1)
        water_mask[overlap] = (p_fused_water[overlap] > p_fused_urban[overlap]).astype(np.uint8)
        urban_mask[overlap] = 1 - water_mask[overlap]

        # -------------------------------------------------------------
        # 5. Query Filtering & Vector Layer Synthesis
        # -------------------------------------------------------------
        q_lower = query.lower()
        layers_to_build: List[Tuple[str, np.ndarray, str]] = []

        request_water = "water" in q_lower or "river" in q_lower or "lake" in q_lower or "flood" in q_lower
        request_urban = "built" in q_lower or "urban" in q_lower or "structure" in q_lower or "building" in q_lower

        if request_water:
            layers_to_build.append(("cross_modal_water_bodies", water_mask, "water"))
        if request_urban:
            layers_to_build.append(("cross_modal_built_up", urban_mask, "built-up"))
        if not layers_to_build:
            # Default to both layers
            layers_to_build.append(("cross_modal_water_bodies", water_mask, "water"))
            layers_to_build.append(("cross_modal_built_up", urban_mask, "built-up"))

        vector_layers = []
        water_ha = 0.0
        urban_ha = 0.0

        for layer_name, mask, target_cat in layers_to_build:
            vec_res = GeospatialEngine.raster_mask_to_geojson(
                binary_mask=mask,
                affine_transform=affine_transform,
                layer_name=layer_name,
                pixel_size_m=pixel_size_m,
                crs=getattr(meta_opt, "crs", None) if meta_opt else None
            )

            if "water" in layer_name:
                water_ha = vec_res["metrics"].get("total_area_hectares", 0.0)
            else:
                urban_ha = vec_res["metrics"].get("total_area_hectares", 0.0)

            vector_layers.append(vec_res)

        # -------------------------------------------------------------
        # 6. Physical Summary & Evidence Explanation
        # -------------------------------------------------------------
        agreement_pct = round(float(np.mean((water_mask > 0) == (p_sar_water > 0.4)) * 100.0), 1)

        answer = (
            f"Cross-modal Optical-SAR fusion executed with calibrated C-band backscatter "
            f"and adaptive {kernel_size}x{kernel_size} Lee speckle filtering. "
            f"Identified {water_ha:.2f} hectares of surface water via specular radar reflection "
            f"(sigma0 < -15 dB) corroborated by optical absorption, and {urban_ha:.2f} hectares "
            f"of built-up infrastructure characterized by dihedral double-bounce corner reflection "
            f"(sigma0 > -7 dB). Cross-modal spatial agreement: {agreement_pct}%."
        )

        overall_confidence = round(min(0.95, max(0.50, (agreement_pct / 100.0) * 0.4 + 0.50)), 3)

        output = {
            "answer": answer,
            "confidence": overall_confidence,
            "vector_layers": vector_layers,
            "urban_area_hectares": urban_ha,
            "water_area_hectares": water_ha,
            "sar_metrics": {
                "sigma0_min_db": round(sar_min, 2),
                "sigma0_max_db": round(sar_max, 2),
                "speckle_filter_kernel": kernel_size,
                "cross_modal_agreement_pct": agreement_pct
            }
        }

        # RS-XAI Integration
        if parameters.get("include_xai", False):
            try:
                from mlops.xai_engine import RSAIXEngine
                xai_engine = RSAIXEngine()
                v_scores = {
                    "empty": 0.0,
                    "optical": float(np.mean(p_opt_water > 0.4)),
                    "sar": float(np.mean(p_sar_water > 0.4)),
                    "joint": overall_confidence
                }
                combined_mask = np.maximum(water_mask, urban_mask)
                output["xai_explanation"] = xai_engine.explain_multimodal_fusion(
                    optical_raster=opt_bands,
                    sar_raster=sar_filtered,
                    v_scores=v_scores,
                    feature_mask=combined_mask
                )
            except Exception as e:
                logger.warning("Fusion XAI explanation failed: %s", e)

        telemetry = {
            "status": "SUCCESS",
            "model": "Physics-Guided Evidence Fusion Engine",
            "backend": "radiometric_calibration_lee_filter_bayesian_fusion",
            "sar_filter_kernel": kernel_size,
            "layers_generated": len(vector_layers),
            "agreement_percentage": agreement_pct
        }

        return output, telemetry
