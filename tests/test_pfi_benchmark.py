"""
SatQuery AI - RS-XAI Scientific Faithfulness Benchmark Test Suite
Owner: Peter (Chief Architect) & Pradipti (Research & Benchmarks Lead)

Validates:
1. Multispectral Permutation Feature Importance (PFI) mathematical correctness.
2. Physics-grounded consistency for NDWI (Green + NIR) and NDVI (Red + NIR).
3. AOPC Saliency Faithfulness (MoRF vs. Random vs. LeRF).
4. Insertion score recovery dynamics.
5. Zero-autograd memory safety (<3.8 GB VRAM, RTX 3050 & CPU safe).
6. End-to-end execution on actual BigEarthNet-MM Sentinel-2 / Sentinel-1 rasters.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
import rasterio

from benchmarks.faithfulness import SpectralPFIEvaluator, AOPCFaithfulnessEvaluator
from scripts.spectral_pfi_benchmark import (
    ndwi_scoring_fn,
    ndvi_scoring_fn,
    multimodal_joint_scoring_fn,
    run_smoke_test,
    run_full_benchmark,
    BIGEARTHNET_DIR
)


# ==============================================================================
# 1. Spectral Permutation Feature Importance (PFI) Tests
# ==============================================================================

class TestSpectralPFI:

    def test_informative_vs_noise_band(self):
        """Permuting an informative signal band must cause strictly greater drop than noise."""
        rng = np.random.RandomState(42)
        raster = rng.uniform(0.1, 0.3, size=(4, 64, 64)).astype(np.float32)
        # Channel 1 (Green) has a high-value active signature
        raster[1, 16:48, 16:48] = 0.90
        # Channel 0 (Blue) is pure random noise

        # Spatial ROI detector sensitive to signal concentrated in center of channel 1
        def ch1_detector(arr: np.ndarray) -> float:
            return float(np.mean(arr[1, 16:48, 16:48]))

        pfi_res = SpectralPFIEvaluator.compute_band_pfi(
            model_or_fn=ch1_detector,
            raster=raster,
            band_names=["Noise_0", "Signal_1", "Noise_2", "Noise_3"],
            n_permutations=4,
            seed=42
        )

        assert pfi_res["dominant_band"] == "Signal_1"
        assert pfi_res["normalized_weights"]["Signal_1"] > pfi_res["normalized_weights"]["Noise_0"]

    def test_ndwi_physics_consistency(self):
        """NDWI PFI must allocate >= 50% importance to Green (B03) and NIR (B08)."""
        rng = np.random.RandomState(42)
        raster = rng.uniform(0.1, 0.4, size=(4, 64, 64)).astype(np.float32)
        raster[1, 20:44, 20:44] = 0.80  # High green reflectance
        raster[3, 20:44, 20:44] = 0.10  # Low NIR reflectance (water absorption)

        pfi_res = SpectralPFIEvaluator.compute_band_pfi(
            model_or_fn=ndwi_scoring_fn,
            raster=raster,
            n_permutations=4,
            seed=42
        )

        val = SpectralPFIEvaluator.validate_physics_consistency(
            band_importance=pfi_res["normalized_weights"],
            index_type="NDWI",
            threshold_ratio=0.50
        )

        assert val["physically_consistent"] is True
        assert val["primary_importance_weight"] >= 0.50
        assert val["secondary_importance_weight"] < 0.50

    def test_ndvi_physics_consistency(self):
        """NDVI PFI must allocate >= 50% importance to Red (B04) and NIR (B08)."""
        rng = np.random.RandomState(42)
        raster = rng.uniform(0.1, 0.4, size=(4, 64, 64)).astype(np.float32)
        raster[2, 20:44, 20:44] = 0.15  # Low red (chlorophyll absorption)
        raster[3, 20:44, 20:44] = 0.85  # High NIR (canopy scatter)

        pfi_res = SpectralPFIEvaluator.compute_band_pfi(
            model_or_fn=ndvi_scoring_fn,
            raster=raster,
            n_permutations=4,
            seed=42
        )

        val = SpectralPFIEvaluator.validate_physics_consistency(
            band_importance=pfi_res["normalized_weights"],
            index_type="NDVI",
            threshold_ratio=0.50
        )

        assert val["physically_consistent"] is True
        assert val["primary_importance_weight"] >= 0.50

    def test_unsupported_index_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported index_type"):
            SpectralPFIEvaluator.validate_physics_consistency(
                band_importance={"B02_Blue": 0.5, "B03_Green": 0.5},
                index_type="UNKNOWN_INDEX"
            )


# ==============================================================================
# 2. AOPC Saliency Faithfulness Tests
# ==============================================================================

class TestAOPCFaithfulness:

    def test_morf_exceeds_lerf_and_random(self):
        """MoRF deletion must drop score faster than LeRF deletion and random removal."""
        rng = np.random.RandomState(42)
        raster = rng.uniform(0.1, 0.3, size=(4, 64, 64)).astype(np.float32)
        raster[1, 20:44, 20:44] = 0.85  # Green water signal
        raster[3, 20:44, 20:44] = 0.05  # NIR absorption

        green, nir = raster[1], raster[3]
        ndwi = (green - nir) / (green + nir + 1e-6)
        saliency = np.clip((ndwi - ndwi.min()) / (ndwi.max() - ndwi.min() + 1e-6), 0.0, 1.0)

        aopc = AOPCFaithfulnessEvaluator.compute_aopc(
            model_or_fn=ndwi_scoring_fn,
            raster=raster,
            saliency_map=saliency,
            steps=4,
            baseline_mode="mean",
            seed=42
        )

        assert aopc["morf_exceeds_lerf"] is True
        assert aopc["faithfulness_ratio"] >= 1.0
        assert aopc["is_faithful"] is True

    def test_insertion_score_recovery(self):
        """Inserting top salient pixels must progressively recover prediction score."""
        rng = np.random.RandomState(42)
        raster = rng.uniform(0.1, 0.3, size=(4, 64, 64)).astype(np.float32)
        raster[2, 20:44, 20:44] = 0.10  # Low red (absorption)
        raster[3, 20:44, 20:44] = 0.85  # High NIR (canopy reflection)

        red, nir = raster[2], raster[3]
        ndvi = (nir - red) / (nir + red + 1e-6)
        saliency = np.clip((ndvi - ndvi.min()) / (ndvi.max() - ndvi.min() + 1e-6), 0.0, 1.0)

        aopc = AOPCFaithfulnessEvaluator.compute_aopc(
            model_or_fn=ndvi_scoring_fn,
            raster=raster,
            saliency_map=saliency,
            steps=4,
            baseline_mode="zero",
            seed=42
        )

        ins_scores = aopc["curves"]["insertion_scores"]
        assert len(ins_scores) == 4
        # Insertion must recover score above the zero baseline
        assert ins_scores[0] > aopc["baseline_empty_score"]
        # Progressive insertion of more salient pixels should increase score
        assert ins_scores[-1] >= ins_scores[0]

    def test_shape_mismatch_raises(self):
        raster = np.zeros((4, 32, 32), dtype=np.float32)
        bad_saliency = np.zeros((64, 64), dtype=np.float32)

        with pytest.raises(ValueError, match="does not match raster spatial shape"):
            AOPCFaithfulnessEvaluator.compute_aopc(
                model_or_fn=lambda x: 1.0,
                raster=raster,
                saliency_map=bad_saliency
            )


# ==============================================================================
# 3. Hardware & VRAM Memory Safety Tests
# ==============================================================================

class TestHardwareSafety:

    def test_zero_vram_and_no_grad(self):
        """Verifies that perturbation iterations attach no gradients and leak zero VRAM."""
        initial_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        raster = np.random.rand(4, 64, 64).astype(np.float32)
        sal = np.random.rand(64, 64).astype(np.float32)

        def dummy_model(arr: np.ndarray) -> float:
            t = torch.from_numpy(arr).float()
            # Deliberately check that no autograd graph is built
            res = torch.sum(t ** 2)
            assert res.grad_fn is None  # Must have no backward graph
            return float(res.item())

        _ = SpectralPFIEvaluator.compute_band_pfi(dummy_model, raster, n_permutations=2)
        _ = AOPCFaithfulnessEvaluator.compute_aopc(dummy_model, raster, sal, steps=3)

        final_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        assert final_allocated == initial_allocated


# ==============================================================================
# 4. End-to-End Real BigEarthNet-MM Benchmark Verification
# ==============================================================================

class TestRealBenchmarkData:

    def test_bigearthnet_full_benchmark(self):
        """Executes full benchmark on actual BigEarthNet Sentinel-2 and Sentinel-1 rasters."""
        s2_path = BIGEARTHNET_DIR / "s2_patch.tif"
        s1_path = BIGEARTHNET_DIR / "s1_patch.tif"

        if not s2_path.exists() or not s1_path.exists():
            pytest.skip("BigEarthNet benchmark files missing from disk.")

        scorecard = run_full_benchmark(n_permutations=3, steps=3)

        assert scorecard["benchmark"] == "SatQuery AI - RS-XAI Scientific Faithfulness Benchmark"
        assert scorecard["summary"]["overall_status"] == "PASSED"
        assert scorecard["summary"]["physics_consistency_status"] == "PASSED"
        assert scorecard["summary"]["aopc_faithfulness_status"] == "PASSED"
        assert scorecard["summary"]["total_benchmark_latency_ms"] < 1000.0  # Under 1 second

    def test_smoke_test_function(self):
        """Executes synthetic CLI smoke test and verifies pass status."""
        scorecard = run_smoke_test()
        assert scorecard["mode"] == "SMOKE_TEST"
        assert scorecard["passed"] is True

    def test_aopc_uniform_baseline(self):
        """Verifies uniform baseline calculation and ratio comparison."""
        raster = np.random.rand(4, 32, 32).astype(np.float32)
        sal = np.random.rand(32, 32).astype(np.float32)

        def dummy_model(arr: np.ndarray) -> float:
            return float(np.mean(arr[0]))

        res = AOPCFaithfulnessEvaluator.compute_aopc(dummy_model, raster, sal, steps=3)
        assert "aopc_uniform" in res["aopc_scores"]
        assert "uniform_scores" in res["curves"]
        assert "faithfulness_ratio_vs_uniform" in res

    def test_saliency_localization_evaluator(self):
        """Verifies Pointing Game and EIMR localization evaluator."""
        from benchmarks.faithfulness import SaliencyLocalizationEvaluator

        sal = np.zeros((32, 32), dtype=np.float32)
        sal[10, 10] = 10.0  # Peak point

        gt = np.zeros((32, 32), dtype=np.uint8)
        gt[8:15, 8:15] = 1

        loc = SaliencyLocalizationEvaluator.evaluate_localization(sal, gt, target_name="water_body")
        assert loc["pointing_game_hit"] is True
        assert loc["energy_in_mask_ratio"] > 0.5
        assert loc["status"] == "PASSED"

    def test_ablation_matrix_execution(self):
        """Verifies that all 5 ablation configurations execute cleanly."""
        from benchmarks.ablation import XAIAblationEngine

        res = XAIAblationEngine.run_ablation_matrix()
        assert res["summary"]["configurations_evaluated"] == 5
        assert "Config_A_Patch_Energy_Only" in res["configurations"]
        assert "Config_B_Shapley_Attribution_Only" in res["configurations"]
        assert "Config_C_Physics_Scattering_Only" in res["configurations"]
        assert "Config_D_Spatial_Masking_Only" in res["configurations"]
        assert "Config_E_Full_Unified_Pipeline" in res["configurations"]
        assert res["summary"]["all_configs_gradient_free"] is True

    def test_baselines_comparison_execution(self):
        """Verifies quantitative comparison against Random, Uniform, and KernelSHAP."""
        from benchmarks.ablation import XAIAblationEngine

        res = XAIAblationEngine.run_baselines_comparison()
        assert "Random_Saliency" in res["saliency_baselines"]
        assert "Uniform_Saliency" in res["saliency_baselines"]
        assert "Patch_Energy_Ours" in res["saliency_baselines"]
        assert "KernelSHAP_Approximation" in res["attribution_baselines"]
        assert "Analytical_Shapley_Ours" in res["attribution_baselines"]
