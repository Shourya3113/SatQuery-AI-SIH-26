# SatQuery AI — 1-Week Sprint Plan (Final)

## Hardware & Model Strategy

| Machine | Specs | Role |
|---------|-------|------|
| **Uni Lab A100** | 40/80 GB VRAM | Training (BigEarthNet LoRA), full benchmark runs, heavy inference |
| **MacBook Pro M4 Pro** | 18/36 GB unified memory | Demo machine (MLX inference), frontend dev, testing |
| **RTX 3050** | 4 GB VRAM | Lightweight inference fallback, backend dev |
| **Google Colab** | T4 16GB (free) / A100 (Pro) | Backup training, notebook prototyping |

### Model Selections (Calibrated to Hardware)

| Component | Model | Size | VRAM (4-bit) | HuggingFace ID | Runs on RTX 3050? |
|-----------|-------|------|-------------|----------------|-------------------|
| **VQA / Captioning** | Qwen2-VL-2B-Instruct | 2B params | ~3.2 GB | `Qwen/Qwen2-VL-2B-Instruct` | ✅ Yes (tight) |
| **Grounding (Stage 1)** | Grounding DINO Tiny | 172M params | ~0.8 GB | `IDEA-Research/grounding-dino-tiny` | ✅ Yes |
| **Grounding (Stage 2)** | SAM ViT-Base | 91M params | ~0.5 GB | `facebook/sam-vit-base` | ✅ Yes |
| **Change Detection** | BIT (Binary change Transformer) | ~3M params | ~0.2 GB | `open-cd` model zoo | ✅ Yes |
| **BigEarthNet Adapter** | CLIP ViT-B/16 + LoRA | 86M + 2M LoRA | ~0.5 GB | `openai/clip-vit-base-patch16` | ✅ Yes |
| **Fusion** | Physics-based (no DL model) | N/A | 0 GB | N/A | ✅ Yes |

> [!TIP]
> Total peak VRAM for all models loaded simultaneously: **~5.2 GB** — fits on RTX 3050 with margin. On M4 Pro, runs entirely in unified memory. On A100, trivial.

---

## Team Assignments (7-Day Sprint)

### Day 1–2: Real AI Engines (CRITICAL PATH)

#### Peter (Team Leader, AI/ML Lead) — on A100 / MacBook M4 Pro
- [ ] **Build Model Manager** (`mlops/model_manager.py`): Centralized lazy-loading, 4-bit quantization via `bitsandbytes`, GPU/CPU/MPS auto-detection, memory tracking
- [ ] **Rewrite VQA engine** (`tools/vqa_engine.py`): Load Qwen2-VL-2B-Instruct, implement real `model.generate()` inference, GeoTIFF→PIL conversion, RS system prompt engineering, keep mock as CPU fallback
- [ ] **Rewrite grounding engine** (`tools/spatial_grounding.py`): Grounding DINO Tiny → bounding boxes, SAM ViT-Base → pixel masks, pipe to existing affine projection
- [ ] Test both engines on sample GeoTIFF images

#### Achintya (Backend Lead) — on RTX 3050
- [ ] **Fix orchestrator** (`services/orchestrator.py`): Remove hardcoded telemetry durations, fix `co_registered` hardcoding, add bounding box overlap check
- [ ] **Update requirements.txt**: Add `bitsandbytes`, `peft`, `datasets`, `groundingdino-py`, `segment-anything`, `open-clip-torch`
- [ ] **Write model download script** (`scripts/download_models.py`): Pre-download all HuggingFace weights to `models/` directory for offline demo
- [ ] **Update API endpoints** (`api/main.py`): Add model status endpoint (`GET /api/models/status`), add GPU info endpoint
- [ ] **Add startup model preloading**: Load models on server start (not per-request)
- [ ] **Update input validation**: Real band-count checks, spatial overlap verification for pairs

#### Vinayak (Frontend Lead)
- [ ] **Add model loading indicator**: Show loading spinner while models initialize on backend
- [ ] **Display real confidence scores**: Update Analysis.jsx to show per-model confidence
- [ ] **Add change map visualization**: Render change detection binary mask as colored overlay on Cesium globe

---

### Day 3–4: Multi-Image + Domain Adaptation

#### Peter (AI/ML Lead) — on A100
- [ ] **BigEarthNet LoRA training** (`mlops/train_adapter.py`):
  - Download BigEarthNet-MM subset (10K S1+S2 patches)
  - Fine-tune CLIP ViT-B/16 with LoRA rank=8 on multi-label classification
  - ~2-3 hours on A100, export to `models/bigearth_adapter.pth`
- [ ] **Integrate adapter**: Use BigEarthNet-adapted CLIP as the visual encoder for VQA context embeddings
- [ ] **Rewrite fusion engine** (`tools/fusion_engine.py`): Replace Boolean rules with physics-grounded σ⁰ dB classification + optical spectral indices + Dempster-Shafer evidence fusion
- [ ] **Write change VQA logic**: After change mask generation, feed mask stats + query to VQA model for natural language change description
- [ ] **Integration testing**: Run all 5 representative queries from the problem statement end-to-end

#### Chhavi (AI Support & Data Pipeline)
- [ ] **Rewrite change detection engine** (`tools/change_engine.py`): Load pre-trained BIT model from `open-cd`, implement real bi-temporal inference (Peter provides architecture guidance)
- [ ] **Data preprocessing pipeline**: Build GeoTIFF→tensor pipeline for all engines, handle band normalization, multi-resolution alignment
- [ ] Test change engine on sample bi-temporal image pairs

#### Misha (Database/GIS Lead)
- [ ] **Enhance geospatial engine**: Add proper co-registration verification (bounding box intersection over union)
- [ ] **SAR calibration refinement**: Verify Lee filter params, add proper σ⁰ dB calibration with correct K_cal per sensor
- [ ] **NDBI index**: Add Normalized Difference Built-up Index for optical urban detection in fusion

#### Pradipti (Research/QA & Pitch Lead)
- [ ] **Download benchmark datasets** (`scripts/download_benchmarks.py`):
  - VRSBench test split from HuggingFace
  - RSVQA-LR test split
  - CDVQA test split
  - BigEarthNet-MM test split (subset)
- [ ] **Prepare sample Cartosat-2S + RISAT GeoTIFF pairs** for demo (from NRSC Bhoonidhi or ISRO open data)
- [ ] **Update master plan & PPT** with honest benchmark numbers once available

---

### Day 5–6: Benchmarks + Integration + Polish

#### Peter (AI/ML Lead)
- [ ] **Run real benchmarks** on A100:
  - `python scripts/run_benchmarks.py --real --dataset vrsbench`
  - `python scripts/run_benchmarks.py --real --dataset cdvqa`
  - `python scripts/run_benchmarks.py --real --dataset rsvqa`
  - `python scripts/run_benchmarks.py --real --dataset bigearth`
- [ ] Record honest scores, update `README.md` and master plan
- [ ] **Profile VRAM usage**: Verify total < 5.8 GB for the "edge deployment" claim
- [ ] **Profile latency**: Measure real end-to-end pipeline time (target < 5s per query)
- [ ] **Rewrite `run_benchmarks.py`**: Replace rigged predictions with real pipeline calls
- [ ] **Update `eval_harness.py`**: Add proper dataset loaders for each benchmark
- [ ] **Model optimization**: Quantization tuning, batch sizing, inference speed

#### Chhavi (AI Support)
- [ ] **Assist Peter** on benchmark runs
- [ ] Test edge cases: single-band SAR, 4-band multispectral, corrupted inputs, zero-mask fallbacks

#### Achintya (Backend Lead)
- [ ] **Optimize model loading**: Implement model caching, lazy unloading of unused models
- [ ] **Error handling**: Graceful degradation when GPU OOM, timeout handling for slow inference
- [ ] **Update PDF report**: Include model names, real confidence scores, and benchmark citations

#### Vinayak (Frontend Lead)
- [ ] **Polish UI**: Loading states, error handling, responsive layout for demo
- [ ] **Add benchmark dashboard**: Display real scores on Benchmarks page
- [ ] **Cesium 3D integration**: Overlay change detection polygons on 3D globe with before/after toggle

#### Misha (Database/GIS Lead)
- [ ] **End-to-end GeoTIFF pipeline testing**: Verify Cartosat-2S + RISAT pair flows through entire pipeline
- [ ] **Prepare 5 demo scenarios** with real GeoTIFF samples:
  1. Single optical image → VQA + Caption
  2. Single optical image → Grounding ("highlight water body")
  3. Bi-temporal pair → Change detection + Change VQA
  4. Optical + SAR pair → Cross-modal fusion
  5. Complex query → Agentic routing demonstration

#### Pradipti (Research/QA & Pitch Lead)
- [ ] **Prepare jury defense**: Update battle card with real benchmark numbers
- [ ] **Record demo video**: Screen capture of full pipeline working
- [ ] **Update PPT slides 2, 4, 6** with real architecture and real scores

---

### Day 7: Demo Day Rehearsal

#### All Team Members
- [ ] **Full demo run-through** on MacBook M4 Pro (or A100 via SSH tunnel)
- [ ] **Test all 5 representative queries** from the problem statement
- [ ] **Test with a novel GeoTIFF** (not in training data) to prove generalization
- [ ] **Rehearse jury Q&A** using the battle card
- [ ] **Git tag release**: `v1.0.0-sih-final`
- [ ] **Verify offline capability**: All models cached locally, no internet dependency

---

## File Change Summary

| File | Action | Owner | Priority |
|------|--------|-------|----------|
| `mlops/model_manager.py` | **NEW** | Peter | Day 1 |
| `tools/vqa_engine.py` | **REWRITE** | Peter | Day 1 |
| `tools/spatial_grounding.py` | **REWRITE** | Peter | Day 2 |
| `tools/change_engine.py` | **REWRITE** | Chhavi (guided by Peter) | Day 3 |
| `tools/fusion_engine.py` | **REWRITE** | Peter | Day 3 |
| `mlops/train_adapter.py` | **NEW** | Peter | Day 3 |
| `scripts/download_models.py` | **NEW** | Achintya | Day 1 |
| `scripts/download_benchmarks.py` | **NEW** | Pradipti | Day 4 |
| `scripts/run_benchmarks.py` | **REWRITE** | Peter | Day 5 |
| `services/orchestrator.py` | **MODIFY** | Achintya | Day 1 |
| `services/geospatial.py` | **MODIFY** | Misha | Day 4 |
| `api/main.py` | **MODIFY** | Achintya | Day 2 |
| `requirements.txt` | **MODIFY** | Achintya | Day 1 |
| `frontend/src/pages/Analysis.jsx` | **MODIFY** | Vinayak | Day 2 |
| `benchmarks/eval_harness.py` | **MODIFY** | Peter | Day 5 |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| A100 lab not available some days | Colab T4 as backup for training; M4 Pro for inference testing |
| Qwen2-VL-2B doesn't fit in 4GB RTX 3050 | Use CPU offloading or switch to smaller Florence-2-base (0.23B) |
| BigEarthNet download too slow (66GB) | Use curated 10K-sample subset (~2GB) |
| Real benchmark scores much lower than claimed | Update all documentation with honest numbers; real 60% > fake 100% at jury evaluation |
| Demo machine crashes during presentation | Pre-record backup video of full pipeline working |

> [!IMPORTANT]
> **The #1 priority is getting VQA + Grounding working with real models by end of Day 2.** Everything else builds on top. If we hit Day 3 with real inference running, we're on track to win.
