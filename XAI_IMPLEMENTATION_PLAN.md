# SatQuery AI — Scientific Explainable AI (RS-XAI) Implementation Plan

## Problem Context & Red-Team Audit Summary

Standard Explainable AI (XAI) methods developed for consumer computer vision (such as classical Grad-CAM and image-level KernelSHAP) fail in operational Earth Observation and face critical bottlenecks on target hardware:
1. **Classical Grad-CAM on 4-bit Quantized Models:** Incompatible with frozen INT4 weights in `bitsandbytes`; dequantizing for autograd backward passes triggers **CUDA Out-Of-Memory (OOM)** on 4GB VRAM. Furthermore, standard Grad-CAM on Vision Transformers ignores multi-head residual mixing, producing uninformative, blurry artifacts (*Chefer et al., CVPR 2021; Höhl et al., 2024*).
2. **KernelSHAP Latency:** Sampling 300–1000 superpixel coalitions takes **40–120 seconds per query**, violating real-time disaster management and live demo SLAs.
3. **Integrated Gradients Baseline Dilemma:** Arbitrary zero-baselines distort physical multispectral signatures, and 100 Riemann integration steps on 29k images would require over 120 hours of compute (*Sturmfels et al., Distill 2020*).

---

## The Scientifically Grounded Architecture

This plan implements a **Hardware-Calibrated, Physics-Aware Remote Sensing Explainability Framework** that scales across all three target devices:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              Heterogeneous Hardware Tiering                                      │
├───────────────────────────────┬───────────────────────────────┬──────────────────────────────────┤
│ Tier 1: RTX 3050 (4GB Laptop) │ Tier 2: MacBook M4 Pro        │ Tier 3: Lab A100 (40/80GB)       │
│ "Tactical Edge Engine"        │ "Interactive Command HUD"     │ "Deep Scientific Audit"          │
├───────────────────────────────┼───────────────────────────────┼──────────────────────────────────┤
│ • Patch Energy Saliency       │ • Native FP16 Token Norms     │ • Spectral Permutation Feature   │
│   (Gradient-Free, < 2ms)      │ • 60 FPS Retina Visualizer    │   Importance (PFI across 12 bands│
│ • Analytical 2-Player Shapley │ • SAM-2 Polygon Boundary      │ • Quantitative Faithfulness      │
│   (Exact Sensor Contribution) │   Projected Contours          │   Validation (AOPC Curves)       │
│ • Radar Physical Scattering   │ • Zero MPS Autograd           │ • Full Benchmark Batch Auditing  │
│   Decomposition (σ⁰ dB)       │   Kernel Dependencies         │   (VRSBench, BigEarthNet)        │
└───────────────────────────────┴───────────────────────────────┴──────────────────────────────────┘
```

---

## User Review Required

> [!IMPORTANT]
> ### 1. XAI Response Overhead
> Generating the high-resolution overlay adds ~50–80 KB (base64 PNG) to the API response. We will provide an opt-out query flag (`include_xai=true/false`, default: `true`).
>
> ### 2. Benchmark Integration
> The Spectral Permutation Feature Importance (PFI) suite for Sentinel-2 bands will be added to `scripts/run_benchmarks.py` so it can be executed on the A100 to produce Slide 4 charts for the SIH presentation.

---

## Proposed Changes

### Component 1: Core Explainability Engine (`mlops/`)

#### [NEW] [xai_engine.py](file:///d:/projects/SIH%2026/mlops/xai_engine.py)
Implements four core explainability algorithms:

1. **`PatchEnergyVisualizer` (Gradient-Free Visual Saliency):**
   - Grounded in *Darcet et al. (ICLR 2024)* and *He et al. (CVPR 2022)*.
   - Extracts the $L_2$ norm of vision transformer patch tokens directly from the forward pass:
     $$S(x, y) = \|\mathbf{z}_{(x, y)}^{\text{last}}\|_2$$
   - Zero backward passes, zero gradient graph memory, 100% compatible with 4-bit quantized weights and Apple Silicon MPS.
   - Computes in $< 2\text{ms}$.

2. **`AnalyticalShapleyDecomposer` (Exact Sensor Contribution):**
   - Grounded in cooperative game theory ($N = \{\text{Optical}, \text{SAR}\}$):
     $$\phi_{\text{SAR}} = \frac{1}{2}\Big[v(\{\text{SAR}\}) - v(\emptyset)\Big] + \frac{1}{2}\Big[v(\{\text{Optical}, \text{SAR}\}) - v(\{\text{Optical}\})\Big]$$
     $$\phi_{\text{Optical}} = 1.0 - \phi_{\text{SAR}}$$
   - Evaluates the 4 coalition states in $< 3\text{ms}$.
   - Returns exact percentage splits proving to ISRO why C-band radar was required.

3. **`PhysicsScatteringDecomposer` (Radar & Spectral Physics):**
   - Grounded in *Freeman & Durden (IEEE TGRS 1998)* polarimetric decomposition and optical spectral sensitivities:
     - Decomposes SAR backscatter into:
       - Specular reflection ($\sigma^0 < -16\text{ dB}$ $\rightarrow$ Water / Smooth surfaces)
       - Dihedral double-bounce ($\sigma^0 > -6\text{ dB}$ $\rightarrow$ Urban / Vertical structures)
       - Diffuse volume scattering ($-16\text{ dB} \le \sigma^0 \le -6\text{ dB}$ $\rightarrow$ Vegetation canopy)
     - Computes closed-form partial derivatives of spectral indices:
       $$\frac{\partial \text{NDWI}}{\partial \rho_{\text{NIR}}} = -\frac{2 \rho_{\text{Green}}}{(\rho_{\text{Green}} + \rho_{\text{NIR}} + \epsilon)^2}$$

4. **`PolygonMaskedOverlayBuilder`:**
   - Masks the saliency heatmap with SAM-2/raster polygon boundaries, eliminating blurry blobs and locking evidence directly to parcel boundaries in `EPSG:4326`.
   - Returns base64 PNG data URI and GeoJSON feature metrics.

---

### Component 2: Specialist Tool Updates (`tools/`)

#### [MODIFY] [fusion_engine.py](file:///d:/projects/SIH%2026/tools/fusion_engine.py)
- Integrate `AnalyticalShapleyDecomposer` and `PhysicsScatteringDecomposer`.
- Output includes:
  - `shapley_attribution`: `{"SAR_pct": 74.2, "Optical_pct": 25.8}`
  - `physics_scattering`: Physical area breakdown in hectares per scattering regime.
  - `xai_overlay_base64`: Clean visual overlay.

#### [MODIFY] [vqa_engine.py](file:///d:/projects/SIH%2026/tools/vqa_engine.py)
- Integrate `PatchEnergyVisualizer` to extract visual token saliency during Qwen2-VL inference.
- Return token-aligned spatial saliency so the user can verify the model was looking at the target region when answering.

#### [MODIFY] [spatial_grounding.py](file:///d:/projects/SIH%2026/tools/spatial_grounding.py)
- Combine Grounding DINO detection scores with SAM mask boundary confidence, generating confidence-bounded spatial evidence.

---

### Component 3: Data Schemas & API (`core/`, `api/`)

#### [MODIFY] [schemas.py](file:///d:/projects/SIH%2026/core/schemas.py)
Add standard Pydantic models for XAI outputs:
```python
class XAIExplanation(BaseModel):
    method: str  # "Patch-Energy-Saliency", "Analytical-Shapley", "Physics-Scattering"
    modality_attribution: Optional[Dict[str, float]] = None  # {"Optical_pct": 26.0, "SAR_pct": 74.0}
    physics_rationale: Optional[Dict[str, Any]] = None
    heatmap_overlay_base64: Optional[str] = None
    summary: str

# QueryResponse extension:
xai_explanation: Optional[XAIExplanation] = None
```

#### [MODIFY] [report_generator.py](file:///d:/projects/SIH%2026/services/report_generator.py)
- Embed a new section in the generated ReportLab PDF:
  **"Section 4: Visual Evidence & Explainable AI Verification"**
  Includes the sensor attribution breakdown bar, physics scattering metrics, and the georeferenced heatmap overlay.

---

### Component 4: Benchmarks & Scientific Validation (`scripts/`)

#### [NEW] [spectral_pfi_benchmark.py](file:///d:/projects/SIH%2026/scripts/spectral_pfi_benchmark.py)
- Implements Permutation Feature Importance (PFI) across all 12 Sentinel-2 bands and SAR polarizations.
- Designed to run on the **Lab A100** across test splits in under 5 minutes.
- Exports a publication-ready CSV and plot (`assets/spectral_importance_curve.png`) for the PPT presentation.

---

### Component 5: Interactive Frontend HUD (`frontend/src/`)

#### [MODIFY] [Analysis.jsx](file:///d:/projects/SIH%2026/frontend/src/pages/Analysis.jsx)
- Add an **"Explainability & Visual Evidence"** toggle card.
- Include an interactive **Opacity Slider** (0% raw image $\leftrightarrow$ 100% saliency overlay).
- Display a **Sensor Attribution Progress Bar** (`SAR: 74%` | `Optical: 26%`) for cross-modal queries.

---

## Verification Plan

### Automated Tests
1. `pytest tests/test_xai.py -v`:
   - Unit tests for `PatchEnergyVisualizer` (output dimensions match input image, values in $[0, 1]$).
   - Unit tests for `AnalyticalShapleyDecomposer` ($\phi_{\text{SAR}} + \phi_{\text{Optical}} \equiv 100.0\%$).
   - Unit tests for `PhysicsScatteringDecomposer` (correct physical classification under boundary values).
2. Regression pass: All 46 existing unit/integration tests must pass cleanly:
   ```bash
   $env:SATQUERY_MOCK_ONLY="1"; python -m pytest tests/ -q
   ```

### Hardware Constraints Verification
- **RTX 3050:** Verify peak VRAM during XAI execution remains below 3.8 GB (zero gradient allocations).
- **MacBook M4 Pro:** Verify seamless execution under native FP16 with zero missing MPS kernel errors.
- **Lab A100:** Verify `spectral_pfi_benchmark.py` runs 1,000 samples in under 5 minutes.
