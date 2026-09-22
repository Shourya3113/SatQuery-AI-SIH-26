"""
SatQuery AI - Remote Sensing Explainable AI (RS-XAI) Test Suite
Owner: Peter (Chief Architect)

Validates mathematical correctness, efficiency properties, zero-autograd memory safety,
and physical consistency for RS-XAI components:
- PatchEnergyVisualizer (L2 token norms, torch.no_grad)
- AnalyticalShapleyDecomposer (n=2 exact coalitions, efficiency, dummy player, symmetry)
- PhysicsScatteringDecomposer (Woodhouse empirical SAR classification, analytical NDWI/NDVI derivatives)
- PolygonMaskedOverlayBuilder (base64 RGBA generation, alpha masking)
- RSAIXEngine (Pydantic schema conformity, telemetry)
"""

import base64
import numpy as np
import pytest
import torch

from core.schemas import XAIExplanation
from mlops.xai_engine import (
    PatchEnergyVisualizer,
    AnalyticalShapleyDecomposer,
    PhysicsScatteringDecomposer,
    PolygonMaskedOverlayBuilder,
    RSAIXEngine,
)


# ==============================================================================
# 1. Patch Energy Visualizer Tests
# ==============================================================================

class TestPatchEnergyVisualizer:

    def test_token_energy_3d_tensor(self):
        # Shape (B=1, N=64, D=384) representing 8x8 patch grid
        tokens = torch.randn(1, 64, 384, requires_grad=True)

        heatmap = PatchEnergyVisualizer.compute_token_energy(
            tokens=tokens,
            output_hw=(128, 128)
        )

        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (128, 128)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0
        # Verify no autograd was tracked in outputs
        assert not tokens.grad  # grad should be None since executed in no_grad

    def test_token_energy_with_explicit_grid(self):
        # Shape (N=32, D=128) representing 4x8 grid
        tokens = np.random.rand(32, 128).astype(np.float32)

        heatmap = PatchEnergyVisualizer.compute_token_energy(
            tokens=tokens,
            output_hw=(64, 128),
            grid_hw=(4, 8)
        )

        assert heatmap.shape == (64, 128)
        assert np.all((heatmap >= 0.0) & (heatmap <= 1.0))

    def test_gradient_free_saliency(self):
        # Create image with a bright spot in the center
        img = np.zeros((3, 64, 64), dtype=np.uint8)
        img[:, 24:40, 24:40] = 255

        saliency = PatchEnergyVisualizer.compute_gradient_free_saliency(img, patch_size=8)

        assert saliency.shape == (64, 64)
        assert np.all((saliency >= 0.0) & (saliency <= 1.0))
        # Center region should have high saliency compared to corners
        assert saliency[28:36, 28:36].mean() > saliency[0:8, 0:8].mean()


# ==============================================================================
# 2. Analytical Shapley Decomposer Tests
# ==============================================================================

class TestAnalyticalShapleyDecomposer:

    def test_efficiency_property(self):
        """Efficiency property: phi_O + phi_S == v(joint) - v(empty)"""
        v_empty = 0.1
        v_opt = 0.70
        v_sar = 0.60
        v_joint = 0.95

        res = AnalyticalShapleyDecomposer.decompose_two_players(
            v_empty=v_empty,
            v_optical=v_opt,
            v_sar=v_sar,
            v_joint=v_joint
        )

        phi_o = res["raw_shapley_values"]["phi_optical"]
        phi_s = res["raw_shapley_values"]["phi_sar"]
        total_gain = v_joint - v_empty

        assert pytest.approx(phi_o + phi_s, abs=1e-5) == total_gain
        assert res["efficiency_satisfied"] is True
        assert res["efficiency_error"] < 1e-5

    def test_normalized_attribution_sum_to_one(self):
        res = AnalyticalShapleyDecomposer.decompose_two_players(
            v_empty=0.0,
            v_optical=0.65,
            v_sar=0.55,
            v_joint=0.90
        )
        norm_o = res["attribution"]["optical"]
        norm_s = res["attribution"]["sar"]

        assert pytest.approx(norm_o + norm_s, abs=1e-4) == 1.0
        assert norm_o > norm_s  # Optical had higher individual score

    def test_symmetry_property(self):
        """If v(O) == v(S), then phi_O == phi_S"""
        res = AnalyticalShapleyDecomposer.decompose_two_players(
            v_empty=0.0,
            v_optical=0.60,
            v_sar=0.60,
            v_joint=0.85
        )
        phi_o = res["raw_shapley_values"]["phi_optical"]
        phi_s = res["raw_shapley_values"]["phi_sar"]

        assert pytest.approx(phi_o, abs=1e-5) == phi_s
        assert res["attribution"]["optical"] == 0.5
        assert res["attribution"]["sar"] == 0.5

    def test_dummy_player_property(self):
        """If SAR adds zero value in both coalitions, phi_SAR == 0"""
        res = AnalyticalShapleyDecomposer.decompose_two_players(
            v_empty=0.2,
            v_optical=0.8,
            v_sar=0.2,      # adds 0 to empty
            v_joint=0.8     # adds 0 to optical
        )
        phi_s = res["raw_shapley_values"]["phi_sar"]
        assert pytest.approx(phi_s, abs=1e-5) == 0.0
        assert res["attribution"]["optical"] == 1.0
        assert res["attribution"]["sar"] == 0.0


# ==============================================================================
# 3. Physics & Scattering Decomposer Tests
# ==============================================================================

class TestPhysicsScatteringDecomposer:

    def test_sar_backscatter_empirical_classification(self):
        # Create synthetic SAR dB image with 3 distinct patches
        sar_db = np.zeros((30, 30), dtype=np.float32)
        sar_db[0:10, :] = -20.0   # Surface scattering (< -16 dB)
        sar_db[10:20, :] = -10.0  # Volume scattering ([-16, -6] dB)
        sar_db[20:30, :] = -2.0   # Double bounce (> -6 dB)

        res = PhysicsScatteringDecomposer.classify_sar_backscatter_empirical(
            sar_array=sar_db,
            is_db=True
        )

        dist = res["scattering_distribution_pct"]
        assert pytest.approx(dist["surface_specular_pct"], abs=1.0) == 33.33
        assert pytest.approx(dist["volume_canopy_pct"], abs=1.0) == 33.33
        assert pytest.approx(dist["double_bounce_urban_pct"], abs=1.0) == 33.33
        assert "Woodhouse" in res["citation"]

    def test_analytical_ndwi_sensitivity(self):
        # 4-band: [Blue, Green, Red, NIR]
        # Pure water signature: High Green, Low NIR
        data = np.zeros((4, 20, 20), dtype=np.uint8)
        data[1, :, :] = 180  # Green (reflection)
        data[3, :, :] = 20   # NIR (absorption)

        sens = PhysicsScatteringDecomposer.compute_analytical_spectral_sensitivity(
            raster_data=data,
            index_type="NDWI"
        )

        d_green = sens["mean_partial_derivatives"]["d_green"]
        d_nir = sens["mean_partial_derivatives"]["d_nir"]

        # Analytical proof: d_NDWI/d_Green must be strictly positive, d_NDWI/d_NIR strictly negative
        assert d_green > 0.0
        assert d_nir < 0.0

        attrib = sens["band_sensitivity_attribution"]
        assert pytest.approx(attrib["green_band"] + attrib["nir_band"], abs=1e-3) == 1.0

    def test_analytical_ndvi_sensitivity(self):
        # 4-band: [Blue, Green, Red, NIR]
        # Dense vegetation: High NIR, Low Red
        data = np.zeros((4, 20, 20), dtype=np.uint8)
        data[2, :, :] = 30   # Red (chlorophyll absorption)
        data[3, :, :] = 200  # NIR (canopy reflection)

        sens = PhysicsScatteringDecomposer.compute_analytical_spectral_sensitivity(
            raster_data=data,
            index_type="NDVI"
        )

        d_nir = sens["mean_partial_derivatives"]["d_nir"]
        d_red = sens["mean_partial_derivatives"]["d_red"]

        # Analytical proof: d_NDVI/d_NIR must be positive, d_NDVI/d_Red negative
        assert d_nir > 0.0
        assert d_red < 0.0

        attrib = sens["band_sensitivity_attribution"]
        assert pytest.approx(attrib["nir_band"] + attrib["red_band"], abs=1e-3) == 1.0


# ==============================================================================
# 4. Polygon-Masked Overlay Builder Tests
# ==============================================================================

class TestPolygonMaskedOverlayBuilder:

    def test_overlay_generation(self):
        heatmap = np.random.rand(64, 64).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 1

        b64 = PolygonMaskedOverlayBuilder.build_overlay_base64(
            heatmap_2d=heatmap,
            binary_mask=mask,
            colormap="turbo"
        )

        assert b64.startswith("data:image/png;base64,")
        # Ensure valid base64 payload
        raw_b64 = b64.split(",", 1)[1]
        decoded = base64.b64decode(raw_b64)
        assert len(decoded) > 100  # valid PNG header and payload


# ==============================================================================
# 5. RSAIXEngine Unified Pipeline Tests
# ==============================================================================

class TestRSAIXEngine:

    def test_explain_multimodal_fusion(self):
        engine = RSAIXEngine()

        opt = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
        sar = np.random.randint(0, 255, (64, 64), dtype=np.uint8)

        explanation = engine.explain_multimodal_fusion(
            optical_raster=opt,
            sar_raster=sar,
            v_scores={"empty": 0.0, "optical": 0.70, "sar": 0.60, "joint": 0.92}
        )

        assert isinstance(explanation, XAIExplanation)
        assert explanation.method == "Hybrid Shapley-Physics Cross-Modal Attribution"
        assert "optical" in explanation.modality_attribution
        assert "sar" in explanation.modality_attribution
        assert pytest.approx(
            explanation.modality_attribution["optical"] + explanation.modality_attribution["sar"],
            abs=1e-3
        ) == 1.0
        assert explanation.heatmap_overlay_base64.startswith("data:image/png;base64,")
        assert explanation.runtime_ms > 0.0
        assert len(explanation.limitations) >= 2
        assert explanation.faithfulness_metrics["shapley_efficiency_error"] < 1e-4

    def test_explain_spatial_grounding(self):
        engine = RSAIXEngine()

        opt = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 1

        explanation = engine.explain_spatial_grounding(
            raster_data=opt,
            binary_mask=mask,
            target_class="water_bodies",
            confidence=0.88
        )

        assert isinstance(explanation, XAIExplanation)
        assert explanation.modality_attribution["optical"] == 1.0
        assert explanation.confidence == 0.88
        assert "water_bodies" in explanation.summary

    def test_explain_change_detection(self):
        engine = RSAIXEngine()
        t1 = np.zeros((3, 64, 64), dtype=np.uint8)
        t2 = np.ones((3, 64, 64), dtype=np.uint8) * 100
        mask = np.ones((64, 64), dtype=np.uint8)

        explanation = engine.explain_change_detection(
            raster_t1=t1,
            raster_t2=t2,
            change_mask=mask,
            change_pct=15.5,
            confidence=0.91
        )

        assert isinstance(explanation, XAIExplanation)
        assert explanation.method == "Differential Radiometric Energy Decomposition"
        assert explanation.physics_rationale["surface_change_percentage"] == 15.5
        assert explanation.heatmap_overlay_base64.startswith("data:image/png;base64,")

    def test_explain_vqa(self):
        engine = RSAIXEngine()
        img = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)

        explanation = engine.explain_vqa(
            raster_data=img,
            query="What is the dominant land cover?",
            answer="Agricultural crop fields",
            confidence=0.85
        )

        assert isinstance(explanation, XAIExplanation)
        assert explanation.method == "Visual Token Energy Saliency"
        assert explanation.confidence == 0.85


# ==============================================================================
# 6. Orchestrator End-to-End XAI Integration Tests
# ==============================================================================

class TestOrchestratorXAIIntegration:

    def test_orchestrator_fusion_with_xai(self, tmp_path):
        from PIL import Image
        from services.orchestrator import AgenticTaskRouter

        opt_file = tmp_path / "opt_xai.png"
        sar_file = tmp_path / "sar_xai.png"

        Image.new("RGB", (128, 128), color=(50, 150, 50)).save(opt_file)
        Image.new("L", (128, 128), color=80).save(sar_file)

        router = AgenticTaskRouter()
        res = router.process_query(
            query="Use optical and SAR together to detect built-up and water covered regions",
            file_paths=[opt_file, sar_file],
            raw_params={"include_xai": True}
        )

        assert res.xai_explanation is not None
        assert isinstance(res.xai_explanation, XAIExplanation)
        assert "optical" in res.xai_explanation.modality_attribution
        assert "sar" in res.xai_explanation.modality_attribution
        assert res.xai_explanation.heatmap_overlay_base64.startswith("data:image/png;base64,")

    def test_orchestrator_grounding_with_xai(self, tmp_path):
        from PIL import Image
        from services.orchestrator import AgenticTaskRouter

        opt_file = tmp_path / "opt_ground_xai.png"
        # Draw green background with blue water region
        img = Image.new("RGB", (128, 128), color=(40, 120, 40))
        img.save(opt_file)

        router = AgenticTaskRouter()
        res = router.process_query(
            query="Highlight and segment the water body in this image",
            file_paths=[opt_file],
            raw_params={"include_xai": True}
        )

        assert res.xai_explanation is not None
        assert isinstance(res.xai_explanation, XAIExplanation)
        assert res.xai_explanation.heatmap_overlay_base64.startswith("data:image/png;base64,")
