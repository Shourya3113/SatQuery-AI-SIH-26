# SatQuery AI (ISRO — SIH26167)
## Master Architecture Plan, Deep-Tech Dual-Use Strategy & 6-Person Team Execution Roadmap

**Prepared For:** Peter & Team (6 Members)  
**Hackathon:** Smart India Hackathon (SIH) 2026  
**Problem Statement ID:** SIH26167  
**Ministry / Organization:** Indian Space Research Organisation (ISRO) / Space Applications Centre (SAC)  
**Category:** Software Track (High-Velocity, Multimodal Remote Sensing & Agentic AI)  

---

## Table of Contents
1. [Official Problem Statement & Requirements Breakdown](#1-official-problem-statement--requirements-breakdown)
2. [Strategic Positioning: The "Deep-Tech Dual-Use" Startup Model](#2-strategic-positioning-the-deep-tech-dual-use-startup-model)
3. [End-to-End System Architecture](#3-end-to-end-system-architecture)
4. [The 4 Core Specialist AI Engines](#4-the-4-core-specialist-ai-engines)
5. [Geospatial Ingestion, GeoTIFFs & Affine Coordinate Mapping](#5-geospatial-ingestion-geotiffs--affine-coordinate-mapping)
6. [Agentic Task Orchestrator & Auditable JSON Execution Trace](#6-agentic-task-orchestrator--auditable-json-execution-trace)
7. [Benchmark Datasets & Domain Adaptation Strategy](#7-benchmark-datasets--domain-adaptation-strategy)
8. [Interactive Web-GIS Interface & UX Specifications](#8-interactive-web-gis-interface--ux-specifications)
9. [6-Person Work Breakdown Structure (AI, Research, Backend, Frontend, DB, MLOps)](#9-6-person-work-breakdown-structure-ai-research-backend-frontend-db-mlops)
10. [Inter-Role Handshake Contracts & Data Exchange Matrix](#10-inter-role-handshake-contracts--data-exchange-matrix)
11. [36-Hour Hackathon Grand Finale Execution Schedule](#11-36-hour-hackathon-grand-finale-execution-schedule)
12. [Official 8-Slide SIH Idea Submission PPT Blueprint](#12-official-8-slide-sih-idea-submission-ppt-blueprint)
13. [ISRO / SAC Jury Defense & Q&A Battle Card](#13-isro--sac-jury-defense--qa-battle-card)

---

# 1. Official Problem Statement & Requirements Breakdown

### Background & Context
Remote-sensing imagery is widely used for agricultural monitoring, disaster management, urban planning, forest monitoring, water-resource assessment, infrastructure mapping, and environmental analysis. However, most existing remote-sensing AI solutions are developed as isolated applications for a single predefined task, such as land-cover classification, object detection, visual question answering, or change detection. These systems often require users to understand satellite-data characteristics, GIS workflows, model selection, and task-specific parameters. Consequently, non-expert users find it difficult to obtain meaningful information from satellite imagery through simple natural-language queries.

Many operational remote-sensing questions cannot be answered reliably using a single optical image. Relevant information is often distributed across paired or multiple observations acquired at different times or by different sensors:
* **Optical & Multispectral:** Provides spectral and contextual information (RGB, NIR, SWIR).
* **Synthetic Aperture Radar (SAR):** Provides complementary structural/dielectric information and supports all-weather, day-and-night acquisition through thick cloud cover and smoke.
* **Multitemporal Image Pairs:** Required to identify and interpret changes over time.
* **Co-registered Optical–SAR Pairs:** Deliver far more complete and reliable intelligence than either modality alone.

### The Core Mandate
Generic VLMs (e.g., standard GPT-4V, LLaVA) fail because they lack domain adaptation to remote-sensing imagery, sensor backscatter characteristics, and GIS coordinate systems. **SatQuery AI** resolves this via an **agentic, query-driven framework** that validates inputs, selects specialist models from a predefined registry, executes constrained workflows, and produces evidence-grounded responses accompanied by an **observable, auditable execution trace**.

### 1:1 Requirement Compliance Matrix

| Requirement Dimension | Mandatory Scope Details | SatQuery AI Architectural Solution |
| :--- | :--- | :--- |
| **Input Modality 1: Single Image** | One optical/multispectral or SAR image for VQA, captioning, and text-guided region grounding. | Native `Rasterio` ingestion extracting CRS, GSD, and band layouts; routes to RS-VLM or Grounding SAM-2. |
| **Input Modality 2: Cross-Modal Pair** | Co-registered optical/multispectral and SAR images of the same geographic area. | Dual-stream Siamese encoder combining optical spectral bands (RGB/NIR) with SAR calibrated backscatter ($\sigma^0\text{ dB}$). |
| **Input Modality 3: Bi-Temporal Pair** | Two spatially corresponding images ($T_1, T_2$) for change detection, change description, and CDVQA. | `ChangeFormer-V2` / BIT cross-attention difference head generating both pixel change masks and natural-language summaries. |
| **Supported File Formats** | **GeoTIFF / TIFF** mandatory for operational geospatial data. PNG/JPEG accepted only for prescribed public benchmark datasets. | Full GeoTIFF affine matrix parser and multi-band slicer; fallback loader for public benchmark images. |
| **Domain Adaptation** | At least one visual or vision-language component fine-tuned on `BigEarthNet.txt` or open EO data. | Vision-language projection head fine-tuned with LoRA on BigEarthNet-MM (Sentinel-1 SAR + Sentinel-2 optical pairs). |
| **Mandatory Single-Image Tasks** | VQA mandatory + either Captioning or Text-Guided Region Grounding. | **Implements both:** Descriptive scene captioner and open-vocabulary polygon grounding via Grounding DINO + SAM-2. |
| **Multi-Image Change Analysis** | Change description or change-based VQA (CDVQA) mandatory; spatial change map generated when reference masks exist. | Quantified change reporting (area in hectares, % growth/decay) with interactive GeoJSON polygon overlays. |
| **Cross-Modal Pair Analysis** | Extract complementary information from co-registered optical and SAR images. | Cloud-penetrating all-weather feature extraction (calibrated water specular returns and urban double-bounce). |
| **Agentic Tool Orchestration** | Automatic task classification, input validation, tool selection, parameter bounding, output synthesis. | Deterministic `AgenticTaskRouter` + Tool Registry with strict guardrails and no hallucinated tool invocations. |
| **Auditable Execution Trace** | Verifiable JSON trace containing selected task, models/tools, parameters, confidence, and latency. | Standardized JSON telemetry schema rendered in real-time UI Inspector and exportable in PDF reports. |
| **ISRO/SAC Evaluation Readiness** | Evaluation on undisclosed Cartosat-2S (optical) and RISAT-1A (SAR) test pairs. | Pre-configured sensor ingestion profiles for Cartosat-2S (0.65m GSD pan-sharpened) and RISAT-1A / EOS-04 (C-band SAR). |

---

# 2. Strategic Positioning: The "Deep-Tech Dual-Use" Startup Model

### Should it sound like a startup idea?
**Yes, but strictly as a "Deep-Tech Dual-Use Platform," NOT a generic consumer SaaS.**

In SIH, the jury consists of senior scientists from ISRO (Space Applications Centre - SAC, National Remote Sensing Centre - NRSC) and ministry directors.

```
             ┌────────────────────────────────────────────────────┐
             │             SatQuery AI Core Platform              │
             │ (Agentic Multimodal Remote Sensing AI Engine)      │
             └─────────────────────────┬──────────────────────────┘
                                       │
           ┌───────────────────────────┴───────────────────────────┐
           ▼                                                       ▼
┌────────────────────────────┐               ┌────────────────────────────┐
│  Anchor 1: Institutional   │               │ Anchor 2: Dual-Use Startup │
│   Public Mission (ISRO)    │               │    Commercial Spin-Off     │
├────────────────────────────┤               ├────────────────────────────┤
│• Native Bhuvan & MOSDAC    │               │• Parametric Crop Insurance │
│• NDMA Disaster Management  │               │  (PMFBY claim validation)  │
│• All-Weather Flood Mapping │               │• Infrastructure Auditing   │
│• Border & Defense Terrain  │               │  (NHAI highway progress)   │
│• Forest & Wetland Ecology  │               │• ESG & Carbon MRV Audits   │
└────────────────────────────┘               └────────────────────────────┘
```

#### 1. The Institutional Anchor (What Wins the Hackathon):
* Position SatQuery AI as a sovereign intelligence layer designed to integrate natively into **Bhuvan (ISRO's Geoportal)**, **MOSDAC**, and the **National Disaster Management Authority (NDMA)**.
* Highlight public-good missions: Rapid flood delineation under monsoon clouds (when optical satellites are blinded), landslide damage assessment, and municipal urban encroachment monitoring.

#### 2. The Commercial Dual-Use Spin-Off (What Proves Startup Viability):
* **Parametric Agriculture & Crop Insurance:** Automating Pradhan Mantri Fasal Bima Yojana (PMFBY) claim verifications using optical-SAR fusion to detect crop lodging, drought, or waterlogging at village cluster levels.
* **National Infrastructure Monitoring:** Tracking National Highways Authority of India (NHAI) road expansion and railway corridor construction timelines.
* **Environmental & ESG Compliance:** Monitoring real-time forest cover depletion and coastal wetland conservation.

This dual-use framing demonstrates that your team respects national defense and public sector realities while building a commercially viable, self-sustaining technological enterprise.

---

# 3. End-to-End System Architecture

SatQuery AI decouples high-level natural language reasoning from spatial compute using an **Agentic Query-Driven Orchestrator**:

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

# 4. The 4 Core Specialist AI Engines

### Engine 1: Remote-Sensing Single-Image VQA & Captioning (`RS-VQA-Engine`)
* **Underlying Architecture:** Remote-sensing adapted vision-language backbone (e.g., GeoChat / RemoteCLIP / LLaVA-Earth with Low-Rank Adaptation - LoRA).
* **Task Capabilities:**
  - **Descriptive Captioning:** Generates comprehensive scene summaries (*"High-resolution optical scene showing urban settlement, transport corridors, and bounded agricultural plots"*).
  - **Quantitative Counting:** Counts discrete targets (*"Identified 7 aircraft on the tarmac"*).
  - **Attribute Reasoning:** Analyzes surface textures, roof types, and water turbidity.
* **Evaluation Benchmark:** RSVQA (High/Low Resolution) and VRSBench.

### Engine 2: Open-Vocabulary Text-Guided Region Grounding (`SpatialGrounding-Engine`)
* **Underlying Architecture:** Grounding DINO feature backbone paired with Segment Anything Model 2 (SAM-2).
* **Workflow:**
  1. User specifies a text query (*"Highlight the active sediment plume near the river mouth"*).
  2. Grounding DINO detects candidate bounding boxes based on language tokens.
  3. SAM-2 predicts zero-shot pixel-precise segmentation boundaries.
  4. The Affine Coordinate Projector converts the binary raster mask into georeferenced GeoJSON polygons.
* **Output:** Interactive vector layer with surface area computed in hectares ($1\text{ ha} = 10,000\text{ m}^2$) and square kilometers.

### Engine 3: Bi-Temporal Change Detection & CDVQA (`BiTemporalChange-Engine`)
* **Underlying Architecture:** `ChangeFormer-V2` / Bitemporal Image Transformer (BIT) coupled with a lightweight CDVQA cross-attention language head.
* **Workflow:**
  1. Ingests spatially aligned observations $T_1$ (Date 1) and $T_2$ (Date 2).
  2. Computes multi-scale difference feature maps across temporal channels.
  3. Classifies nature of change: Urban expansion, water recession/flooding, or vegetation loss.
  4. Generates both qualitative text (*"Built-up area increased by 18.4% between March 2023 and March 2024"*) and a pixel-accurate change mask.
* **Evaluation Benchmark:** CDVQA dataset.

### Engine 4: Cross-Modal Optical–SAR Fusion (`OpticalSARFusion-Engine`)
* **Underlying Architecture:** Dual-stream Siamese encoder with cross-attention feature fusion.
* **The Physical Principle:**
  - **Optical Sensor (Cartosat / Sentinel-2):** High spectral detail (RGB/NIR/SWIR), but easily blocked by monsoon clouds, smoke, and darkness.
  - **SAR Sensor (RISAT-1A / Sentinel-1):** C-band microwave radar ($\lambda \approx 5.6\text{ cm}$) penetrates clouds, haze, and rain. Radar backscatter ($\sigma^0\text{ in dB}$) reveals physical roughness: calm water creates specular reflection (dark/very low return), while man-made orthogonal walls create dihedral double-bounce (bright/very high return).
* **Fusion Logic:** Merges co-registered optical bands and dual-pol (VV/VH) radar backscatter. Enables reliable water and built-up segmentation even under 100% cloud cover.

---

# 5. Geospatial Ingestion, GeoTIFFs & Affine Coordinate Mapping

A key differentiator for SatQuery AI is its strict adherence to professional GIS principles rather than treating satellite imagery as generic web images:

### Supported Formats & Native Parsing:
* **GeoTIFF / Cloud-Optimized GeoTIFF (COG):** Ingested via `Rasterio` and `GDAL`.
* **Benchmark Formats:** PNG/JPEG supported exclusively for evaluation on public benchmark splits.

### Coordinate Reference Systems (CRS) & Projection:
* Automatically parses embedded projection systems (e.g., `EPSG:4326` WGS84 or `EPSG:32643` UTM Zone 43N).
* Verifies spatial alignment and Ground Sample Distance (GSD) across input pairs.

### The Affine Transformation Matrix (Eliminating Lat/Long Hallucinations):
SatQuery AI ensures the language model **never invents geographic coordinates**. All spatial coordinates are mathematically projected from pixel space $(x_{\text{pixel}}, y_{\text{pixel}})$ into world coordinates $(\text{longitude}, \text{latitude})$ using the affine transform matrix:

$$\begin{bmatrix} \text{longitude} \\ \text{latitude} \\ 1 \end{bmatrix} = \begin{bmatrix} a & b & c \\ d & e & f \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x_{\text{pixel}} \\ y_{\text{pixel}} \\ 1 \end{bmatrix}$$

Where:
* $a$ = pixel width (resolution in longitude degrees/meters)
* $e$ = pixel height (negative value, scanning north to south)
* $c, f$ = coordinates of the upper-left corner $(x_{\text{origin}}, y_{\text{origin}})$
* $b, d$ = rotation/shear coefficients (typically zero for north-up rasters)

### Multispectral Band Math & SAR Preprocessing:
* **NDVI (Vegetation Index):** $\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$
* **NDWI (Water Index):** $\text{NDWI} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}$
* **SAR Radiometric Calibration:** $\sigma^0\text{ (dB)} = 10 \cdot \log_{10}(\text{DN}^2 + \epsilon)$
* **Speckle Filtering:** Lee/Frost filter ($5\times 5$ kernel) to suppress radar multiplicative noise.

---

# 6. Agentic Task Orchestrator & Auditable JSON Execution Trace

### Orchestration Workflow
1. **Query Intent Classification:** Analyzes prompt semantics to identify the required task category (`SINGLE_IMAGE_VQA`, `TEXT_GUIDED_GROUNDING`, `BI_TEMPORAL_CHANGE_DETECTION`, or `CROSS_MODAL_JOINT_ANALYSIS`).
2. **Input Verification:** Audits file count, formats, band count, and spatial co-registration.
3. **Dynamic Tool Assembly:** Sequences necessary preprocessing filters, neural backbones, and post-processing vectorizers.
4. **Parameter Bounding:** Restricts execution parameters to safe operational bounds (confidence threshold $\in [0.1, 0.99]$, change threshold $\in [0.1, 0.95]$, speckle kernel $\in \{3, 5, 7\}$).

### Standardized Auditable Execution Trace (JSON Schema)
As explicitly evaluated by the ISRO jury, every query produces a transparent, observable telemetry log:

```json
{
  "trace_id": "satquery-exec-2026-9481",
  "timestamp": "2026-09-04T16:15:00Z",
  "user_query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
  "input_audit": {
    "count": 2,
    "modalities": ["OPTICAL_MULTISPECTRAL", "SAR_C_BAND"],
    "format": "GeoTIFF",
    "spatial_alignment": {
      "co_registered": true,
      "crs": "EPSG:32643",
      "spatial_resolution_meters": 10.0,
      "bounding_box": [72.82, 18.95, 72.95, 19.08]
    }
  },
  "orchestration": {
    "selected_task": "CROSS_MODAL_JOINT_ANALYSIS",
    "pipeline_steps": [
      {
        "step_number": 1,
        "tool_name": "GeospatialPreprocessor",
        "parameters": {
          "sar_filter": "lee_speckle_5x5",
          "optical_normalization": "min_max_percentile_2_98"
        },
        "status": "SUCCESS",
        "duration_ms": 42.5
      },
      {
        "step_number": 2,
        "tool_name": "OpticalSARFusion-Engine",
        "parameters": {
          "speckle_filter_kernel": 5,
          "confidence_threshold": 0.85
        },
        "status": "SUCCESS",
        "duration_ms": 310.2
      },
      {
        "step_number": 3,
        "tool_name": "AffineCoordinateProjector",
        "parameters": {
          "simplification_tolerance": 0.0001,
          "target_crs": "EPSG:4326"
        },
        "status": "SUCCESS",
        "duration_ms": 18.4
      }
    ]
  },
  "results": {
    "overall_confidence": 0.95,
    "vector_layer_count": 2,
    "textual_summary": "Cross-modal Optical–SAR fusion successfully executed. SAR backscatter penetrated atmospheric interference to delineate 1,230.5 hectares of surface water and 4,820.2 hectares of high-density built-up structures."
  }
}
```

---

# 7. Benchmark Datasets & Domain Adaptation Strategy

SatQuery AI establishes strict compliance with all datasets designated in the problem statement:

```
                ┌──────────────────────────────────────────────┐
                │        Training Datasets & Benchmarks        │
                └──────────────────────┬───────────────────────┘
                                       │
         ┌─────────────────────────────┴─────────────────────────────┐
         ▼                                                           ▼
┌──────────────────────────────┐               ┌──────────────────────────────┐
│   BigEarthNet-MM (Domain)    │               │   CDVQA (Bi-Temporal QA)     │
├──────────────────────────────┤               ├──────────────────────────────┤
│ • 590K Sentinel-1 & 2 pairs  │               │ • Multitemporal change VQA   │
│ • S1 SAR backscatter (VV/VH) │               │ • Growth/decay reasoning     │
│ • S2 12-band multispectral   │               │ • Pixel-level change masks   │
└──────────────────────────────┘               └──────────────────────────────┘
         │                                                           │
         ▼                                                           ▼
┌──────────────────────────────┐               ┌──────────────────────────────┐
│  RSVQA & VRSBench (Single)   │               │ ISRO / SAC Evaluation Set    │
├──────────────────────────────┤               ├──────────────────────────────┤
│ • High & Low Res Visual QA   │               │ • Cartosat-2S (Optical Pan)  │
│ • Scene captioning baselines │               │ • RISAT-1A / EOS-04 (SAR)    │
│ • Open-vocabulary grounding  │               │ • Blind test verification    │
└──────────────────────────────┘               └──────────────────────────────┘
```

1. **Domain Adaptation & Fine-Tuning:**
   - **`BigEarthNet.txt` (BigEarthNet-MM):** 590,326 Sentinel-1 SAR and Sentinel-2 multispectral patch pairs used to adapt the vision encoder for cross-modal remote sensing representations.
2. **Single-Image Benchmarks:**
   - **`RSVQA` (High & Low Resolution):** Evaluates visual question answering accuracy on aerial and satellite imagery.
   - **`VRSBench`:** Evaluates high-resolution remote sensing captioning and region grounding.
3. **Multi-Temporal Benchmark:**
   - **`CDVQA`:** Evaluates change detection question answering and temporal feature difference reasoning.
4. **ISRO / SAC Final Evaluation Readiness:**
   - Dedicated data adapters for **Cartosat-2S** (0.65m optical) and **RISAT-1A / EOS-04** (C-band SAR in Ground Range Detected - GRD format) ensuring zero pipeline friction during the blind jury evaluation.

---

# 8. Interactive Web-GIS Interface & UX Specifications

The frontend is an operational, geospatial command console built with **React** and **MapLibre GL**:

* **Dual-Pane Interactive Map Canvas:** 
  - Dynamic raster layer rendering with GPU-accelerated WebGL.
  - Interactive vector polygon overlays rendered from GeoJSON with color-coded classification tags (e.g., Cyan for Water, Amber for Built-Up, Red for Change).
* **Split-Screen Swipe Comparison Slider:**
  - Swipe between $T_1$ and $T_2$ bi-temporal captures to reveal dynamic urban sprawl or flood inundation.
  - Swipe between Optical RGB and SAR backscatter to demonstrate cloud-penetration capabilities.
* **Conversational AI Drawer:**
  - Query input bar with operational prompt recommendations (*"Detect flood inundation"*, *"Highlight runway boundaries"*, *"Quantify built-up expansion"*).
* **Auditable Trace Inspector:**
  - Real-time side drawer displaying the live JSON execution log, tool invocations, parameters, and confidence scores.
* **One-Click Intelligence Dossier Exporter:**
  - Exports a professional, government-ready PDF report including query metadata, satellite parameters, masked preview imagery, and calculated area tables.

---

# 9. 6-Person Work Breakdown Structure (AI, Research, Backend, Frontend, DB, MLOps)

To guarantee high velocity, zero duplicate effort, and flawless parallel development, responsibilities are mapped to 6 distinct, specialized engineering tracks:

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

### Detailed Track Profiles & Deliverables:

#### Track 1: AI & Deep Learning Lead (Core Models & Specialist Engines)
* **Domain Focus:** Neural Architectures, Vision-Language Adaptation, Segmentation, Temporal Attention.
* **Core Responsibilities:**
  - **Tool 1 (`RS-VQA-Engine`):** Adapts Remote-Sensing VLM (GeoChat / RemoteCLIP / LLaVA-Earth) with LoRA weights. Implements descriptive captioning and counting heads.
  - **Tool 2 (`SpatialGrounding-Engine`):** Connects Grounding DINO open-vocabulary prompt tokens with SAM-2 (Segment Anything Model 2) zero-shot polygon mask generation.
  - **Tool 3 (`BiTemporalChange-Engine`):** Implements `ChangeFormer-V2` / BIT cross-attention difference heads; writes directional change reasoning (growth/decay).
  - **Tool 4 (`OpticalSARFusion-Engine`):** Constructs dual-stream Siamese cross-modal feature fusion for all-weather vision.
* **Code Ownership:** `satquery/tools/` (`vqa_engine.py`, `spatial_grounding.py`, `change_engine.py`, `fusion_engine.py`).
* **Primary Tech Stack:** PyTorch, Hugging Face Transformers, PEFT/LoRA, SAM-2, Grounding DINO.

#### Track 2: Research & Benchmark Lead (Domain Adaptation & ISRO Alignment)
* **Domain Focus:** Remote Sensing Physics, Benchmark Evaluation Splits, Scientific Rigor & Defense.
* **Core Responsibilities:**
  - **Dataset Preparation:** Curates official evaluation subsets from `BigEarthNet.txt` (BigEarthNet-MM), `RSVQA`, `VRSBench`, and `CDVQA`.
  - **ISRO Sensor Specs:** Studies and formalizes sensor models for **Cartosat-2S** (0.65m GSD pan-sharpened optical) and **RISAT-1A / EOS-04** (C-band SAR backscatter calibration, incidence angles, VV/VH polarizations).
  - **Benchmark Validation:** Writes evaluation harness to compute official performance metrics: mIoU for grounding, BLEU/CIDEr for VQA/captions, F1-score for change detection.
  - **Scientific Documentation:** Drafts the technical methodology documentation and heads the ISRO jury defense Q&A preparation.
* **Deliverable Ownership:** `satquery/benchmarks/`, validation scripts, and Jury Q&A Battle Card.
* **Primary Tech Stack:** Python, Scikit-Learn, Torchmetrics, Jupyter, LaTeX.

#### Track 3: Backend & Agentic Orchestrator Lead (FastAPI, Routing & Telemetry)
* **Domain Focus:** Distributed Systems, Agentic Routing, Input Schema Verification, Observability.
* **Core Responsibilities:**
  - **Central Orchestrator:** Implements `AgenticTaskRouter` classifying queries into single VQA, grounding, change, or cross-modal tasks.
  - **Input Compatibility Auditor:** Verifies file counts, formats, band configurations, and co-registration alignment.
  - **Parameter Bounding Engine:** Enforces strict mathematical guardrails on thresholds, filter kernels, and token limits.
  - **Auditable Execution Trace:** Builds standardized JSON telemetry generator recording models invoked, latency, and confidence scores.
  - **REST API Endpoints:** Exposes `/api/upload`, `/api/query`, `/api/trace/{id}`, and `/api/export-report`.
* **Code Ownership:** `satquery/services/orchestrator.py` & `satquery/api/main.py`.
* **Primary Tech Stack:** Python 3.11, FastAPI, Pydantic v2, Uvicorn, Requests.

#### Track 4: Frontend & Web-GIS Lead (React, MapLibre & User Experience)
* **Domain Focus:** Geospatial User Interfaces, WebGL Rendering, Interactive Comparison, Data Visualization.
* **Core Responsibilities:**
  - **Web-GIS Canvas:** Builds responsive map viewer using **MapLibre GL** with raster tile server support and dynamic zoom/pan.
  - **Dual-Pane Split Swipe Slider:** Implements side-by-side swipe comparison tool for bi-temporal ($T_1$ vs $T_2$) and cross-modal (Optical vs SAR).
  - **Vector Layer Overlays:** Renders GeoJSON bounding boxes and segmentation polygon masks with hover tooltips displaying area metrics.
  - **Conversational AI Drawer:** Sleek chat interface with suggested operational prompts (*"Detect flood inundation"*, *"Quantify built-up expansion"*).
  - **Auditable Trace Inspector:** Real-time side drawer displaying live JSON telemetry steps for jury transparency.
* **Code Ownership:** `satquery-frontend/` (React, Vite, Tailwind CSS, MapLibre GL).
* **Primary Tech Stack:** React 18, TypeScript, MapLibre GL / OpenLayers, Tailwind CSS, Lucide Icons.

#### Track 5: Database & Geospatial Pipeline Lead (PostGIS, GeoTIFFs & Spatial Lake)
* **Domain Focus:** Spatial Databases, Raster Manipulation, Affine Projections, Band Mathematics.
* **Core Responsibilities:**
  - **Geospatial Data Lake:** Sets up **PostgreSQL + PostGIS** (or SpatiaLite) to store spatial metadata, pre-computed indices, and vector polygons.
  - **GeoTIFF Ingestion Engine:** Parses Cloud-Optimized GeoTIFFs (COG) via `Rasterio` and `GDAL`; extracts CRS projections (`EPSG:4326`, `EPSG:32643`) and affine transforms.
  - **Coordinate Projector:** Mathematically converts pixel masks $(x, y)$ into georeferenced GeoJSON polygons using the affine transform matrix (eliminating lat/long hallucinations).
  - **Multispectral & SAR Math:** Implements NDVI, NDWI band slicing, SAR radiometric calibration to $\sigma^0\text{ (dB)}$, and Lee speckle filtering.
* **Code Ownership:** `satquery/services/geospatial.py` & `satquery/db/`.
* **Primary Tech Stack:** Rasterio, GDAL, Shapely, PostgreSQL / PostGIS, SQLAlchemy / GeoAlchemy2.

#### Track 6: MLOps, Inference Optimization & Pitch Lead (Infra, QA & Presentation)
* **Domain Focus:** Model Quantization, GPU Resource Management, Live Demo Reliability, Pitch Deck Delivery.
* **Core Responsibilities:**
  - **Inference Acceleration:** Quantizes heavy models (VLM, SAM-2, ChangeFormer) via ONNX Runtime, TensorRT, or 4-bit/8-bit AWQ to achieve $<2.5\text{s}$ response times.
  - **GPU Memory Management:** Manages CUDA memory allocation, dynamic model offloading, and prevents Out-Of-Memory (OOM) errors during simultaneous tool execution.
  - **Zero-Downtime Demo Safeguards:** Builds offline local raster tile caches and fallback synthetic test beds to ensure 100% demo resilience during venue Wi-Fi failure.
  - **Automated Intelligence Dossier:** Implements automated PDF report generation (ReportLab) bundling query telemetry, previews, and polygon areas.
  - **Pitch Leadership:** Designs the official 8-slide AICTE submission deck, manages pitch timing, and directs the live presentation to the jury.
* **Deliverable Ownership:** Docker environment, ONNX acceleration pipeline, ReportLab PDF exporter, and Official 8-Slide PPT.
* **Primary Tech Stack:** Docker, ONNX Runtime, TensorRT, ReportLab, Shell/PowerShell.

---

# 10. Inter-Role Handshake Contracts & Data Exchange Matrix

To ensure the 6 members work seamlessly in parallel, strict data interfaces and handshakes are established:

| Sender Track | Receiver Track | Handshake Deliverable / Data Contract | Format / Protocol |
| :--- | :--- | :--- | :--- |
| **Track 2 (Research)** | **Track 1 (AI) & Track 5 (DB)** | Curated benchmark dataset splits, Cartosat/RISAT sample imagery, and ground-truth validation masks. | Folder structure + YAML manifest |
| **Track 5 (DB & GIS)** | **Track 1 (AI)** | Preprocessed, normalized NumPy raster slices (Bands, H, W) + calibrated SAR dB arrays. | NumPy array / In-memory tensor |
| **Track 5 (DB & GIS)** | **Track 3 (Backend)** | Spatial metadata (CRS, bounds, GSD, format) + Affine transform matrix for coordinate projection. | Pydantic `InputImageMetadata` |
| **Track 1 (AI)** | **Track 3 (Backend)** | Tool inference execution: textual answers, raw binary segmentation masks, and model confidence scores. | Python Dict / Tuple with telemetry |
| **Track 3 (Backend)** | **Track 5 (DB & GIS)** | Raw binary masks passed for affine transformation to polygon geometries and hectare area computation. | Shapely `Polygon` $\rightarrow$ GeoJSON |
| **Track 3 (Backend)** | **Track 4 (Frontend)** | Full response payload: text answer, GeoJSON FeatureCollections, confidence, and auditable JSON trace. | JSON over REST (`/api/query`) |
| **Track 6 (MLOps)** | **Track 1 (AI)** | Quantized ONNX/FP16 model graph exports and VRAM-optimized execution configurations. | ONNX / TorchScript model artifacts |
| **Track 6 (MLOps)** | **Track 4 & Track 3** | Docker Compose stack, local tile cache, and ReportLab PDF Dossier export endpoint. | Container image + PDF endpoint |

---

# 11. 36-Hour Hackathon Grand Finale Execution Schedule

```
Hours 00–06: Foundation, Schemas & Ingestion
  ├── Track 3 & 4: Scaffold FastAPI backend routes and React MapLibre map canvas.
  ├── Track 5: Implement Rasterio GeoTIFF ingestion, Affine parser, and PostGIS schema.
  ├── Track 1: Download pre-trained weights for RS-VLM, SAM-2, and ChangeFormer.
  ├── Track 2: Prepare curated test images (BigEarthNet, Cartosat, RISAT sample pairs).
  └── Track 6: Set up CUDA environment, Docker containers, and test harness.

Hours 06–14: Core AI Integration & Geospatial Pipeline
  ├── Track 1: Wire RS-VQA captioning, SAM-2 grounding, and ChangeFormer inference scripts.
  ├── Track 5: Implement SAR dB calibration, Lee speckle filter, and NDVI/NDWI band math.
  ├── Track 2: Benchmark initial model inferences against RSVQA and CDVQA ground-truth.
  ├── Track 4: Build MapLibre layer toggles, split-screen swipe slider, and chat drawer.
  ├── Track 3: Integrate specialist tool calls with Pydantic request/response schemas.
  └── Track 6: Apply FP16/INT8 quantization to models; verify latency < 2.5 seconds.

Hours 14–22: Agentic Orchestrator & End-to-End Handshake
  ├── Track 3: Implement intent classifier, task router, parameter bounds, and JSON trace logger.
  ├── Track 5: Wire raster mask-to-GeoJSON conversion with accurate hectare area calculation.
  ├── Track 4: Connect frontend chat input to `/api/query` and render live GeoJSON polygon layers.
  ├── Track 1: Finalize Optical-SAR cross-modal cross-attention fusion engine.
  ├── Track 2: Verify model responses against domain-specific remote sensing queries.
  └── Track 6: Implement automated PDF Intelligence Dossier export (ReportLab).

Hours 22–28: UI/UX Polish & Interactive Controls
  ├── Track 4: Finalize split comparison slider (T1/T2 & Optical/SAR), trace viewer modal, and tooltips.
  ├── Track 3: Implement execution trace telemetry export endpoint (`/api/trace/{id}`).
  ├── Track 1 & 5: Optimize polygon boundary simplification for fast vector rendering.
  ├── Track 2: Test edge cases (100% cloud cover optical + SAR, complex coastal shorelines).
  └── Track 6: Benchmark full pipeline throughput; eliminate memory leaks.

Hours 28–32: Stress Testing, Offline Fallback & Edge Cases
  ├── Track 6: Package local offline tile server and cached responses for zero-latency demo fallback.
  ├── Track 2 & 1: Run complete test suite on Cartosat-2S and RISAT-1A evaluation pairs.
  ├── Track 4 & 3: Fix UI layout glitches, loading spinners, and error notification toasts.
  └── Track 6: Assemble final 8-slide presentation deck with high-res architecture diagrams.

Hours 32–36: Live Rehearsals, Pitch Polish & Defense Prep
  ├── Track 6 & Track 3: Conduct 3 dry-run live pitches simulating ISRO jury interruptions.
  ├── Track 2: Review Jury Defense Battle Card; prepare technical answers on CRS and SAR physics.
  ├── All Members: Freeze codebase, verify backup presentation video, and prep live demo machine.
  └── Final Jury Pitch Delivery!
```

---

# 12. Official 8-Slide SIH Idea Submission PPT Blueprint

Follow this exact slide-by-slide structure for the official AICTE submission deck:

### Slide 1: Title Slide
* **Project Title:** SatQuery AI — Agentic Multimodal Remote Sensing Intelligence Platform
* **Problem Statement ID:** SIH26167 (ISRO / Space Applications Centre)
* **Team Details:** Team Name, College / Institution, Leader & 5 Members with assigned domains (AI, Research, Backend, Frontend, DB, MLOps).
* **Tagline:** *"Democratizing Earth Observation Intelligence through Agentic Vision-Language Orchestration and Cross-Modal Optical-SAR Fusion."*

### Slide 2: The Core Problem & Current Industry Pain Points
* **Siloed Remote Sensing Workflows:** Existing tools perform isolated tasks (classification, VQA, or change detection) requiring deep GIS expertise and manual parameter selection.
* **The Single-Sensor Optical Blindspot:** Optical satellites are blind during monsoon cloud cover and night—precisely when disaster intelligence (floods, landslides) is most urgent.
* **Generic VLM Failures:** Standard models (GPT-4V, LLaVA) lack remote sensing adaptation, cannot parse multi-band GeoTIFFs, hallucinate coordinates, and fail to process SAR backscatter.

### Slide 3: Proposed Solution & System Architecture
* Clear, high-level block diagram: **Multi-Modal Ingestion (GeoTIFF) $\rightarrow$ Agentic Task Orchestrator $\rightarrow$ Specialist Tool Registry $\rightarrow$ Coordinate Projector $\rightarrow$ Web-GIS & Auditable Trace**.
* Highlights: Multi-sensor synergy (Optical + SAR), zero coordinate hallucinations via affine mapping, and 100% observable execution traces.

### Slide 4: Technical Methodology & Algorithmic Innovation
* **Specialist Engine Registry:**
  - `RS-VQA-Engine`: Adapted on BigEarthNet-MM for scene captioning and visual QA.
  - `SpatialGrounding-Engine`: Grounding DINO + SAM-2 producing pixel-accurate vector polygons.
  - `BiTemporalChange-Engine`: ChangeFormer-V2 cross-attention difference head for CDVQA.
  - `OpticalSARFusion-Engine`: Dual-stream cross-modal fusion enabling all-weather vision through dense clouds.
* **The Affine Transform Formula:** Highlighting deterministic pixel-to-CRS conversion.

### Slide 5: Software Architecture & Technology Stack
* **6 Specialized Engineering Tracks:** Clear visual grid illustrating AI, Research, Backend, Frontend, Database, and MLOps ownership.
* **Stack:** Python 3.11, FastAPI, PostGIS, Rasterio, GDAL, PyTorch, React 18, MapLibre GL, ONNX Runtime.
* **Security & Observability:** Strict parameter bounds, auditable JSON execution trace schema, zero black-box dependencies.

### Slide 6: Quantitative Impact & Measurable Performance Benchmarks
* **Domain Adaptation Metric:** $> 82\%$ top-1 accuracy on RSVQA-HR and $> 0.65\text{ mIoU}$ on VRSBench.
* **Change Detection:** $> 0.80\text{ F1-Score}$ on CDVQA multitemporal difference benchmarks.
* **Operational Latency:** End-to-end agentic pipeline response time $< 2.5\text{ seconds}$ via ONNX quantization.
* **All-Weather Capability:** Delineates water bodies under 100% cloud cover using C-band SAR backscatter returns.

### Slide 7: Dual-Use Deployment Strategy & Scalability
* **Institutional Deployment:** Direct integration into **ISRO Bhuvan**, **MOSDAC**, and **NDMA** disaster monitoring portals.
* **Commercial Deep-Tech Spin-Off:** Automated crop insurance verification (PMFBY), highway infrastructure expansion tracking (NHAI), and ESG carbon credit auditing.
* **Cloud & Edge Readiness:** Deployable on sovereign government cloud (NIC MeghRaj / Bhuvan Cloud) or on-premise high-security servers.

### Slide 8: Team Competence & 36-Hour Hackathon Gantt Chart
* Concise breakdown of the 6 team members' roles (AI Lead, Research Lead, Backend Lead, Frontend Lead, DB/GIS Lead, MLOps/Pitch Lead).
* Clear milestone schedule showing progression from ingestion to final jury demonstration.

---

# 13. ISRO / SAC Jury Defense & Q&A Battle Card

### Q1: "Why not simply use GPT-4o or a large general-purpose VLM via API?"
* **Answer:** *"Generic VLMs only accept 3-channel RGB images (JPEG/PNG). They have zero understanding of SAR backscatter geometry ($\sigma^0\text{ in dB}$), multispectral NIR/SWIR wavelengths for vegetation/water indices, or Coordinate Reference Systems (CRS). Furthermore, generic models hallucinate geographical coordinates, whereas SatQuery AI deterministically derives coordinates via GeoTIFF Affine Transformation Matrices."*

### Q2: "How do you handle cloud cover during flood disaster analysis?"
* **Answer:** *"When optical sensors (Cartosat/Sentinel-2) are blinded by clouds, our Agentic Orchestrator invokes the `OpticalSARFusion-Engine`. Microwave radar from RISAT-1A / Sentinel-1 penetrates cloud cover and rain. Smooth flood water causes specular reflection, showing up as distinct low backscatter ($<-16\text{ dB}$), allowing us to map flood inundation regardless of atmospheric conditions."*

### Q3: "What makes your orchestrator 'Agentic' instead of just a hardcoded switch statement?"
* **Answer:** *"The orchestrator parses ambiguous, open-ended natural language intent, dynamically determines whether single, cross-modal, or bi-temporal pipelines are needed, inspects input raster metadata for CRS alignment, applies safety parameter bounding, coordinates multi-step executions (e.g., preprocessing $\rightarrow$ grounding $\rightarrow$ affine projection), and outputs an auditable execution trace with confidence scoring."*

### Q4: "How do you prove that your model didn't just memorize benchmark images?"
* **Answer:** *"Our system evaluates on unseen GeoTIFFs using strict geometric and radiometric checks. During the live demo, the jury can upload any custom GeoTIFF or modify parameters (such as the change threshold or speckle filter kernel), and watch SatQuery AI dynamically re-execute the pipeline and render verified vector layers in real time."*

---
*Master Plan and 6-Person Execution Roadmap updated and finalized for SIH 2026 Submission.*
