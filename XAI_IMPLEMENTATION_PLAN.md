# SatQuery AI — Scientific Explainable AI (RS-XAI) Implementation Plan (Revised)

## 1. Problem Context & Red-Team Audit Summary

Several conventional XAI methods developed for natural-image computer vision require modification or careful adaptation for operational Earth Observation workloads, particularly under quantized inference, multispectral inputs, multimodal fusion, and constrained edge hardware:

1. **Quantized Inference & Activation Memory Constraints:**
   Classical Grad-CAM can become impractical for frozen INT4 inference pipelines because gradient-based attribution may require additional autograd/activation memory and, depending on the quantization/runtime configuration, may trigger dequantization or unsupported backward operations on 4GB VRAM.
2. **Structural Mismatch on Vision Transformers:**
   Naive Grad-CAM adaptations for Vision Transformers can produce spatially diffuse or poorly localized explanations because transformer token interactions, residual connections, and attention mixing are not explicitly accounted for (*Chefer et al., CVPR 2021; Höhl et al., 2024*).
3. **Perturbation Complexity in Operational Workflows:**
   KernelSHAP can introduce substantial query latency when hundreds of superpixel coalitions are evaluated per sample; the exact runtime depends on model inference latency, number of coalitions, masking strategy, hardware, and batching.
4. **Multispectral Baseline Ambiguity:**
   Integrated Gradients requires a meaningful baseline for multispectral inputs; an all-zero baseline may not represent a physically meaningful observation and can therefore produce difficult-to-interpret attribution paths (*Sturmfels et al., Distill 2020*).

---

## 2. Hardware-Calibrated Architecture

The explainability engine is calibrated across three operational compute tiers:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              Heterogeneous Hardware Tiering                                      │
├───────────────────────────────┬───────────────────────────────┬──────────────────────────────────┤
│ Tier 1: RTX 3050 (4GB Laptop) │ Tier 2: MacBook M4 Pro        │ Tier 3: Lab A100 (40/80GB)       │
│ "Tactical Edge Engine"        │ "Interactive Command HUD"     │ "Deep Scientific Audit"          │
├───────────────────────────────┼───────────────────────────────┼──────────────────────────────────┤
│ • Gradient-Free Patch-Token   │ • Native FP16 Token Norms     │ • Spectral Permutation Feature   │
│   Saliency (Target: < 2 ms)   │ • Target: 60 FPS interactive  │   Importance (PFI on benchmark   │
│ • Two-Modality Shapley Model  │   HUD, subject to rendering   │   Sentinel-2 and SAR channels)   │
│   Attribution (n=2 exact)     │ • Polygon-Constrained Spatial │ • Quantitative Faithfulness      │
│ • Empirical SAR Backscatter   │   Saliency Overlays           │   Validation (AOPC / Insertion)  │
│   Classification (σ⁰ dB)      │ • Zero autograd dependency    │ • Cross-Model Baseline Auditing  │
└───────────────────────────────┴───────────────────────────────┴──────────────────────────────────┘
```

---

## 3. User Review & Configuration Decisions

> [!IMPORTANT]
> ### 1. Real-Time Latency Safeguard: `include_xai` Defaults to `false`
> To prevent serialization overhead and maintain sub-second query performance during live operational routing, the API defaults to:
> ```python
> include_xai: bool = False
> ```
> The frontend HUD or analyst explicitly sets `include_xai=True` when opening the Explainability inspector.
>
> ### 2. Scientific Framing: Model Attribution vs. Physical Causality
> Saliency maps and Shapley splits represent **model-internal feature and modality attribution** under the defined coalition setup, preprocessing, and value function. They are never presented as direct causal physical proof without independent physical validation.

---

## 4. Proposed Changes

### Component 1: Core Explainability Engine (`mlops/`)

#### [NEW] [xai_engine.py](file:///d:/projects/SIH%2026/mlops/xai_engine.py)
Implements four core explainability modules:

1. **`PatchEnergyVisualizer` (Gradient-Free Patch-Token Saliency):**
   - Motivated by prior work on vision-transformer representations and token-level visual information (*Darcet et al., ICLR 2024; He et al., CVPR 2022*).
   - Designed to avoid gradient/autograd memory overhead and support INT4 inference runtimes and Apple Silicon MPS, subject to model/runtime hidden-state access.
   - Strictly wrapped in `torch.no_grad()` to ensure zero gradient graph memory is allocated.
   - Extracts patch-token $L_2$ norms:
     $$S(x, y) = \|\mathbf{z}_{(x, y)}^{\text{last}}\|_2$$
   - Target latency: $< 2\text{ ms}$ on the RTX 3050, to be validated through benchmarking.
   - *Interpretation note:* Patch-token $L_2$ norm is treated as a model-internal saliency proxy rather than a causal explanation. Faithfulness must therefore be evaluated independently using deletion/insertion or perturbation-based metrics.

2. **`AnalyticalShapleyDecomposer` (Two-Modality Model Attribution):**
   - Formulated for the two-modality cooperative game $N = \{\text{Optical}, \text{SAR}\}$.
   - *Note on exactness:* $n=2$ case — exact by construction, no combinatorial sampling needed.
     $$\phi_{\text{SAR}} = \frac{1}{2}\Big[v(\{\text{SAR}\}) - v(\emptyset)\Big] + \frac{1}{2}\Big[v(\{\text{Optical}, \text{SAR}\}) - v(\{\text{Optical}\})\Big]$$
     $$\phi_{\text{Optical}} = 1.0 - \phi_{\text{SAR}}$$
   - *Value-function definition:* Explicitly defines $v(S)$ where $S$ evaluates model output confidence under modality masking/substitution with zero/neutral priors.
   - *Interpretation constraint:* The resulting SAR/Optical percentages represent model attribution under the selected model, coalition definitions, preprocessing pipeline, and value function. They should not be interpreted as causal physical contribution percentages.
   - Returns normalized Shapley attribution values quantifying the model's reliance on SAR relative to optical inputs for the specified query.

3. **`PhysicsScatteringDecomposer` (Empirical SAR Regimes & Analytical Sensitivities):**
   - Motivated by established SAR scattering decomposition literature, including Freeman-Durden. The implementation distinguishes between the original polarimetric covariance/coherency matrix eigen-decomposition assumptions and the simplified threshold-based interpretation used here:
     - Low backscatter regime: $\sigma^0 < -16\text{ dB}$ $\rightarrow$ commonly associated with smooth/open-water-like surfaces under the target acquisition conditions.
     - High backscatter regime: $\sigma^0 > -6\text{ dB}$ $\rightarrow$ commonly associated with strong double-bounce or other high-return structures.
     - Intermediate regime: $-16\text{ dB} \le \sigma^0 \le -6\text{ dB}$ $\rightarrow$ potentially associated with volume/diffuse scattering.
     - *Important:* These thresholds are operational heuristics and must be validated against the sensor, polarization, incidence angle, preprocessing/calibration, and target dataset. They are not presented as universal physical laws.
   - Computes analytical sensitivities of selected spectral indices with respect to input reflectance bands:
     $$\frac{\partial \text{NDWI}}{\partial \rho_{\text{NIR}}} = -\frac{2 \rho_{\text{Green}}}{(\rho_{\text{Green}} + \rho_{\text{NIR}} + \epsilon)^2}$$
     $$\frac{\partial \text{NDVI}}{\partial \rho_{\text{Red}}} = -\frac{2 \rho_{\text{NIR}}}{(\rho_{\text{NIR}} + \rho_{\text{Red}} + \epsilon)^2}$$
     *Interpretation note:* These derivatives quantify local mathematical sensitivity of the index to an input band; they should not be interpreted as causal evidence that the corresponding physical phenomenon is driven by that band.

4. **`PolygonMaskedOverlayBuilder` (Boundary-Aligned Visualization):**
   - Constrains the saliency visualization to detected/segmented spatial regions to improve spatial interpretability.
   - Aligns the visualization with available spatial boundaries or segmentation masks.
   - *Coordinate-system validation:* Verifies CRS transformations, raster-to-vector alignment, pixel-to-geographic coordinate conversion, and polygon clipping accuracy before presenting the overlay as georeferenced evidence in `EPSG:4326`.

---

### Component 2: Specialist Tool Updates (`tools/`)

#### [MODIFY] [fusion_engine.py](file:///d:/projects/SIH%2026/tools/fusion_engine.py)
- Incorporates `AnalyticalShapleyDecomposer` and `PhysicsScatteringDecomposer` when `include_xai=True`.
- Returns structured attribution traceable to: (1) model output explained, (2) coalition/masking strategy, (3) preprocessing, (4) normalization, and (5) inference configuration:
  ```python
  "shapley_attribution": {
      "SAR_pct": 74.2,
      "Optical_pct": 25.8,
      "value_function": "fused_class_consensus_prob",
      "interpretation": "model attribution under the defined coalition setup"
  }
  ```

#### [MODIFY] [vqa_engine.py](file:///d:/projects/SIH%2026/tools/vqa_engine.py)
- Returns token-aligned spatial saliency that allows the user to inspect whether model-internal activation patterns overlap with the queried target region.
- Does not claim that saliency overlap proves the model relied causally on the target region.

#### [MODIFY] [spatial_grounding.py](file:///d:/projects/SIH%2026/tools/spatial_grounding.py)
- Combines Grounding DINO detection confidence and SAM-derived segmentation confidence to generate spatial evidence with explicitly reported confidence components.
- Documents how the two confidence measures are normalized and combined; avoids presenting the resulting value as a calibrated probability unless calibration is performed.

---

### Component 3: Data Schemas & API (`core/`, `api/`)

#### [MODIFY] [schemas.py](file:///d:/projects/SIH%2026/core/schemas.py)
Add strongly typed Pydantic models for XAI outputs:
```python
from typing import Literal, Optional, Dict, List, Any
from pydantic import BaseModel, Field

class XAIExplanation(BaseModel):
    method: Literal[
        "Patch-Energy-Saliency",
        "Analytical-Shapley",
        "Physics-Scattering"
    ]
    modality_attribution: Optional[Dict[str, float]] = None
    spectral_sensitivity: Optional[Dict[str, float]] = None  # Typed analytical sensitivities
    physics_rationale: Optional[Dict[str, Any]] = None
    heatmap_overlay_base64: Optional[str] = None
    confidence: Optional[float] = None
    faithfulness_metrics: Optional[Dict[str, float]] = None
    runtime_ms: Optional[float] = None
    hardware_tier: Optional[str] = None
    limitations: Optional[List[str]] = Field(default_factory=list)
    summary: str

# In QueryResponse:
xai_explanation: Optional[XAIExplanation] = None
```

#### [MODIFY] [report_generator.py](file:///d:/projects/SIH%2026/services/report_generator.py)
- Embeds a new auditable section in the generated ReportLab PDF:
  **"Section 4: Visual Evidence & Explainability Verification"**
  Includes:
  - Explanation method & version
  - Input modalities evaluated
  - Attribution values under defined value function
  - Spectral sensitivities
  - Runtime and hardware tier
  - Faithfulness metrics where available
  - Known limitations and interpretation boundaries

---

### Component 4: Benchmarks & Scientific Validation (`scripts/`)

#### [NEW] [spectral_pfi_benchmark.py](file:///d:/projects/SIH%2026/scripts/spectral_pfi_benchmark.py)
- Implements Permutation Feature Importance (PFI) across the selected Sentinel-2 bands and available SAR channels/polarizations in the benchmark dataset.
- Target runtime: $< 5\text{ minutes}$ on the specified A100 configuration; benchmark runtime must be measured and reported alongside sample count, batch size, model configuration, and number of permutations.
- Reports uncertainty or variability across repeated permutations where computationally feasible.

---

### Component 5: Interactive Frontend HUD (`frontend/src/`)

#### [MODIFY] [Analysis.jsx](file:///d:/projects/SIH%2026/frontend/src/pages/Analysis.jsx)
- **Explainability HUD Toggle:** Defaults to inactive (`include_xai=false`) to preserve query latency.
- **Sensor Attribution Display:**
  `SAR: 74% | Optical: 26% (Model attribution under defined coalition/value function)`
  Includes a tooltip explaining what the percentage represents and what it does not represent.
- **Adjustable Opacity Slider:** Ensures the raw satellite image remains visible at all times through an adjustable overlay ($0\%\leftrightarrow 100\%$), with a clear legend for the saliency scale.

---

## 5. Methodological Rigor & Validation Modules

### XAI Faithfulness Validation
The explainability system must distinguish visual plausibility from explanation faithfulness:
1. **Deletion / AOPC Test:** Progressively remove the most salient regions and measure prediction degradation rate.
2. **Insertion Test:** Progressively add salient regions into a neutral canvas and measure prediction recovery.
3. **Random Baseline:** Compare attribution against randomly selected spatial regions.
4. **Uniform Baseline:** Compare attribution against uniform saliency.
5. **Runtime Benchmark:** Measure explanation generation latency.
6. **Memory Benchmark:** Measure peak VRAM and host RAM usage.
7. **Localization Evaluation:** Where ground-truth regions are available, compare saliency overlap using IoU or pointing-game metrics.

### XAI Baselines Comparison
To demonstrate that the proposed methods provide meaningful explanations, compare against appropriate baselines where feasible:
- Random saliency
- Uniform saliency
- Gradient-based attribution where hardware permits
- Standard/naive Grad-CAM where applicable
- Integrated Gradients on a documented baseline
- KernelSHAP on a small benchmark subset

### Reproducibility Requirements
Every benchmark run must log:
- Model name and version
- Quantization configuration (e.g. INT4 NF4 vs FP16)
- Dataset split and version
- Input resolution and CRS
- Preprocessing and normalization configuration
- Hardware specification and driver/CUDA versions
- Batch size and permutation count
- Random seed
- Execution latency and peak memory

### Ablation Study
Evaluate the contribution of each explainability component independently:
- **A.** Patch-energy saliency only
- **B.** Shapley modality attribution only
- **C.** Physics interpretation only
- **D.** Spatial masking only
- **E.** Combined XAI pipeline

### XAI Failure & Fallback Handling
If an explanation method cannot be executed because of hardware, model architecture, missing hidden states, unsupported operators, or unavailable modality inputs, the API must return a structured explanation status rather than silently failing:
```json
{
    "status": "partial",
    "available_methods": ["Physics-Scattering"],
    "unavailable_methods": ["Patch-Energy-Saliency"],
    "reason": "Required hidden states unavailable in current inference backend"
}
```

### Decoupling Prediction Confidence from Explanation
Prediction confidence and explanation confidence must remain strictly separate concepts:
- A high model confidence score does not imply that the explanation is faithful.
- A visually compelling explanation does not imply high prediction confidence.

---

## 6. Verification Plan & Automated Tests

### Automated Unit & Performance Tests
1. `tests/test_xai.py`:
   - **VRAM Safety & No-Grad Enforcement:** Assert that `PatchEnergyVisualizer` executes within `torch.no_grad()`, verifies `requires_grad=False` across all intermediate tensors, and asserts that memory allocation does not exceed the predefined safety budget (target $< 3.8\text{ GB}$ on the 4GB RTX 3050).
   - **Two-Modality Shapley Tests:** Assert $\phi_{\text{SAR}} + \phi_{\text{Optical}} \equiv 1.0 \pm 10^{-6}$ across synthetic cases with known modality dependence.
   - **Spectral Sensitivity Tests:** Assert analytical derivatives match finite-difference approximations for NDWI/NDVI within numerical tolerance.
   - **Schema & Typed Field Validation:** Assert `spectral_sensitivity` and schema fields serialize correctly.
2. Full regression pass:
   ```bash
   $env:SATQUERY_MOCK_ONLY="1"; python -m pytest tests/ -q
   ```
   All 46 existing tests must continue to pass.

### Hardware-Specific Acceptance Tests
- **RTX 3050:** Verify peak VRAM remains below the safety threshold under representative workloads with zero backward passes.
- **MacBook M4 Pro:** Verify successful execution under the selected MPS/FP16 configuration with no unsupported-operation failures for the tested model and workload.
- **Lab A100:** Target 1,000 benchmark samples in $< 5\text{ minutes}$ on the specified configuration, recording actual runtime and configuration.

---

## 7. Acceptance Criteria

The implementation is considered complete when:
1. All existing 46 unit and integration tests pass.
2. XAI unit tests in `tests/test_xai.py` pass.
3. RTX 3050 execution stays within the defined VRAM budget.
4. M4 Pro execution succeeds using the supported inference path.
5. A100 benchmark completes within the target runtime.
6. Explanation outputs conform to the typed API schema (including `spectral_sensitivity`).
7. Faithfulness metrics are generated for supported methods.
8. Random/uniform baselines are included.
9. Explanation limitations are surfaced to the user.
10. PDF report and frontend representations match backend attribution values.
11. No explanation is presented as causal physical proof unless separately validated.
