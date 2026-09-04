# SatQuery AI 🛰️
### Agentic Multimodal Remote Sensing Intelligence Platform

[![Smart India Hackathon 2026](https://img.shields.io/badge/SIH-2026-blue.svg)](https://sih.gov.in/)
[![Problem Statement](https://img.shields.io/badge/ISRO-SIH26167-orange.svg)](https://www.isro.gov.in/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![MapLibre GL](https://img.shields.io/badge/MapLibre_GL-WebGL-blueviolet.svg)](https://maplibre.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Executive Summary

**SatQuery AI** is a software-based agentic vision-language assistant built for the **Indian Space Research Organisation (ISRO) / Space Applications Centre (SAC)** under **Smart India Hackathon 2026 (Problem Statement: SIH26167)**.

Traditional remote sensing tools are isolated applications requiring deep GIS expertise, manual band selection, and complex parameter tuning. Furthermore, generic VLMs (e.g., standard GPT-4V, LLaVA) fail because they only process 3-channel RGB images, hallucinate arbitrary geographic coordinates, and cannot interpret Synthetic Aperture Radar (SAR) backscatter or multi-band GeoTIFFs.

**SatQuery AI resolves this through an Agentic, Query-Driven Framework:**
* **Multimodal Ingestion:** Ingests single optical/SAR images, co-registered Optical–SAR pairs, and multitemporal ($T_1, T_2$) pairs in native **GeoTIFF** format.
* **Zero-Hallucination Affine Projection:** Mathematically projects pixel-level segmentation boundaries to world Earth coordinates $(\text{lon}, \text{lat})$ via GeoTIFF Affine Matrices.
* **Cloud-Penetrating Vision:** Combines passive optical reflectance (Cartosat-2S / Sentinel-2) with active microwave C-band SAR backscatter (RISAT-1A / Sentinel-1) to detect surface features through 100% cloud cover.
* **Observable Execution Telemetry:** Outputs a verifiable, auditable JSON execution trace proving which models, parameters, and CRS transformations were invoked.

---

## 🏛️ System Architecture

```
              ┌──────────────────────────────────────────────────┐
              │         User Query & Multi-Modal Upload          │
              │ (Single / Cross-Modal Pair / Bi-Temporal Pair)   │
              └────────────────────────┬─────────────────────────┘
                                       │
                                       ▼
              ┌──────────────────────────────────────────────────┐
              │       Input Validation & Ingestion Engine        │
              │ • GDAL / Rasterio Header & CRS Projection Check  │
              │ • Format Audit (GeoTIFF vs Benchmark PNG/JPEG)   │
              │ • Spatial Footprint & Resolution Verification    │
              └────────────────────────┬─────────────────────────┘
                                       │
                                       ▼
              ┌──────────────────────────────────────────────────┐
              │            Agentic Task Orchestrator             │
              │ • Query Intent Classifier & Dynamic Tool Router  │
              │ • Permitted Parameter Guardrails & Bounds Check  │
              └────────────────────────┬─────────────────────────┘
                                       │
       ┌───────────────────┬───────────┴───────────┬───────────────────┐
       ▼                   ▼                       ▼                   ▼
┌──────────────┐    ┌──────────────┐        ┌──────────────┐    ┌──────────────┐
│Tool 1: RS-VLM│    │Tool 2: SAM-2 │        │Tool 3: CDVQA │    │Tool 4: SAR + │
│Single VQA &  │    │Grounding DINO│        │Bi-Temporal   │    │Optical Fusion│
│Captioning    │    │(Vector Mask) │        │ChangeFormer  │    │Cross-Attn    │
└──────┬───────┘    └──────┬───────┘        └──────┬───────┘    └──────┬───────┘
       │                   │                       │                   │
       └───────────────────┴───────────┬───────────┴───────────────────┘
                                       │
                                       ▼
              ┌──────────────────────────────────────────────────┐
              │         Geospatial Coordinate Projector          │
              │ • Affine Transformation Matrix: Pixel -> CRS     │
              │ • GeoJSON Vector Layer & Surface Area Compute    │
              └────────────────────────┬─────────────────────────┘
                                       │
                                       ▼
              ┌──────────────────────────────────────────────────┐
              │       Synthesizer & Auditable Trace Engine       │
              │ • Grounded Natural Language Text Response        │
              │ • Interactive GeoJSON Vector Layer Overlays      │
              │ • Observable JSON Execution Trace Telemetry      │
              │ • Downloadable Government PDF Intelligence Report│
              └──────────────────────────────────────────────────┘
```

---

## 🔬 The 4 Core AI Specialist Engines

1. **Tool 1 (`RS-VQA-Engine`):** Remote-sensing vision-language model adapted on **BigEarthNet-MM** with LoRA adapters for scene captioning, attribute reasoning, and counting.
2. **Tool 2 (`SpatialGrounding-Engine`):** Open-vocabulary text-guided grounding combining **Grounding DINO** with **SAM-2 (Segment Anything Model 2)** to output pixel-level organic polygon masks.
3. **Tool 3 (`BiTemporalChange-Engine`):** Multitemporal change detection engine using **ChangeFormer-V2** cross-attention difference heads, evaluating on **CDVQA**.
4. **Tool 4 (`OpticalSARFusion-Engine`):** Dual-stream Siamese cross-modal encoder merging optical spectral bands with calibrated radar backscatter ($\sigma^0\text{ dB}$) for all-weather vision.

---

## 👥 Team Roster & Engineering Track Ownership

Our team operates on dedicated, non-overlapping engineering tracks:

```
┌────────────────────────────────────────────────────────────────────────┐
│             SatQuery AI — 6-Person Core Team Roster                    │
├──────────────────────────────────┬─────────────────────────────────────┤
│ Peter (Team Leader & Architect)  │ Pradipti (Research, QA & Pitch)     │
│ • Central Agentic Orchestrator   │ • ISRO Sensor Physics & Datasets    │
│ • Git Repo & Integration Lead    │ • Metric Benchmarks (mIoU/BLEU/F1)  │
│ • Cross-Module Unblocker & Guide │ • Official 8-Slide PPT & Jury Pitch │
├──────────────────────────────────┼─────────────────────────────────────┤
│ Chhavi (AI & Deep Learning Lead) │ Vinayak (Frontend & Secondary BE)   │
│ • RS-VLM & SAM-2 Specialist      │ • React + MapLibre GL Canvas        │
│ • ChangeFormer & Optical-SAR     │ • Dual-Pane Swipe Comparison Slider │
│ • MLOps: Quantization (ONNX)     │ • Dynamic Vector Layer Overlays     │
│ • GPU VRAM Optimization & Latency│ • Secondary Backend REST Routes     │
├──────────────────────────────────┼─────────────────────────────────────┤
│ Achintya (Backend Lead & System) │ Misha (Database & Geospatial Lead)  │
│ • FastAPI Gateway & REST Schemas │ • PostGIS Spatial Database Lake     │
│ • System Integration & Security  │ • Rasterio GeoTIFF Parser & COGs    │
│ • Auditable JSON Trace Logger    │ • Affine Matrix Coordinate Mapping  │
│ • PDF Dossier Exporter Service   │ • Band Math (NDVI/NDWI/SAR dB)      │
└──────────────────────────────────┴─────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
SatQuery-AI-SIH-26/
├── api/                             # [Achintya + Vinayak] FastAPI REST Gateway
│   ├── main.py                      # Core API server & routing
│   └── routes_frontend.py           # Upload streaming & UI helper endpoints
├── benchmarks/                      # [Pradipti] Evaluation harness & test splits
│   ├── eval_harness.py              # mIoU, BLEU-4, F1 metric evaluators
│   └── test_splits/                 # BigEarthNet, RSVQA, CDVQA test subsets
├── core/                            # [Peter + Achintya] Shared core contracts
│   ├── config.py                    # Environment settings, bounds, paths
│   └── schemas.py                   # Pydantic v2 schemas (Inputs, Trace, Outputs)
├── db/                              # [Misha] PostGIS database & spatial indices
│   ├── models.py                    # Spatial tables & metadata schemas
│   └── connection.py                # Database pool connection
├── mlops/                           # [Chhavi] Model acceleration & GPU pooling
│   ├── quantize.py                  # ONNX Runtime FP16/INT8 export pipeline
│   └── gpu_manager.py               # CUDA memory pooler & OOM safeguards
├── services/                        
│   ├── orchestrator.py              # [Peter] Central Agentic Task Router
│   ├── geospatial.py                # [Misha] Rasterio GeoTIFF parser & Affine transform
│   └── report_generator.py          # [Achintya] ReportLab PDF Intelligence Dossier export
├── tests/                           # [Pradipti] Automated integration test suites
│   ├── test_pipeline_smoke.py       # End-to-end pipeline smoke tests
│   └── test_geospatial_crs.py       # CRS & Affine projection unit tests
├── tools/                           # [Chhavi] 4 Core Specialist AI Engines
│   ├── base.py                      # Specialist Tool Base Class with telemetry
│   ├── vqa_engine.py                # Tool 1: Single VQA & Scene Captioning
│   ├── spatial_grounding.py         # Tool 2: Grounding DINO + SAM-2
│   ├── change_engine.py             # Tool 3: ChangeFormer-V2 CDVQA
│   └── fusion_engine.py             # Tool 4: Optical-SAR Cross-Modal Fusion
├── satquery-frontend/               # [Vinayak] Web-GIS User Interface
│   ├── src/
│   │   ├── components/map/          # MapLibre GL interactive map canvas
│   │   ├── components/swipe/        # Dual-pane synchronized split swipe slider
│   │   ├── components/chat/         # Conversational drawer with preset prompts
│   │   └── components/trace/        # Live Auditable JSON Trace inspector
│   ├── package.json
│   └── tailwind.config.js
├── SIH_2026_Problem_Statements_Master_Plan.pdf
├── SIH_2026_SatQuery_Team_Roles_and_Responsibilities.pdf
├── SIH_2026_SatQuery_AI_Builders_Guidebook.pdf
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart & Development Setup

### 1. Prerequisites
* Python 3.11+
* Node.js 18+ and npm
* Git

### 2. Backend Setup
```bash
# Clone the repository
git clone https://github.com/Shourya3113/SatQuery-AI-SIH-26.git
cd SatQuery-AI-SIH-26

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running Backend Server
```bash
uvicorn api.main:app --reload --port 8000
```
Interactive Swagger API docs available at: `http://localhost:8000/docs`

### 4. Running Frontend
```bash
cd satquery-frontend
npm install
npm run dev
```
Web-GIS interface available at: `http://localhost:5173`

---

## 📚 Master Documentation & Team Handbooks

All master operational documents and PDFs are located directly in the repository root:
1. **[Master Technical Plan (PDF)](SIH_2026_Problem_Statements_Master_Plan.pdf) | [Markdown](SIH_2026_Problem_Statements_Master_Plan.md):** Complete problem deconstruction, deep-tech dual-use startup strategy, system architecture, and official 8-slide PPT blueprint.
2. **[Team Roles & Responsibilities Handbook (PDF)](SIH_2026_SatQuery_Team_Roles_and_Responsibilities.pdf) | [Markdown](SIH_2026_SatQuery_Team_Roles_and_Responsibilities.md):** 6-person work breakdown, Git branching strategy, Day-1 mock contracts, and 36-hour hackathon schedule.
3. **[Conceptual Masterclass & Builder's Guidebook (PDF)](SIH_2026_SatQuery_AI_Builders_Guidebook.pdf) | [Markdown](SIH_2026_SatQuery_AI_Builders_Guidebook.md):** Deep scientific principles, sensor physics, affine math, radar backscatter theory, and Peter's grilling checklists.

---

## 📄 License
This project is developed for the **Smart India Hackathon 2026** under the **Indian Space Research Organisation (ISRO)** problem statement. Licensed under the [MIT License](LICENSE).
