"""
Generate official SIH 2026 Idea Presentation PPTX strictly following the official format.
Modifies SIH2026-IDEA-Presentation-Format.pptx in place.
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

PPT_PATH = Path("SIH2026-IDEA-Presentation-Format.pptx")
prs = Presentation(str(PPT_PATH))

# Color palette
COLOR_NAVY = RGBColor(15, 37, 75)
COLOR_TEXT = RGBColor(35, 35, 35)
COLOR_WHITE = RGBColor(255, 255, 255)

# ==============================================================================
# SLIDE 1: TITLE PAGE
# ==============================================================================
slide1 = prs.slides[0]

# Subtitle 3
for s in slide1.shapes:
    if s.name == "Subtitle 3" and s.has_text_frame:
        tf = s.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        p.text = "SatQuery AI: Agentic Multimodal Remote Sensing Intelligence Platform"
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = COLOR_NAVY
    elif s.name == "TextBox 9" and s.has_text_frame:
        tf = s.text_frame
        tf.clear()
        lines = [
            ("Problem Statement ID: ", "SIH26167"),
            ("Problem Statement Title: ", "Agentic Multimodal AI for Remote Sensing Analysis (ISRO / SAC)"),
            ("Theme: ", "Space Technology / Disaster Management & Geospatial Intelligence"),
            ("PS Category: ", "Software"),
            ("Team ID: ", "[Registered Portal Team ID]"),
            ("Team Name: ", "SatQuery AI"),
            ("Team Roster: ", "Peter (Leader & Architect) | Pradipti (Research & Pitch) | Chhavi (AI/MLOps) | Vinayak (Frontend) | Achintya (Backend) | Misha (DB & GIS)")
        ]
        for idx, (label, val) in enumerate(lines):
            p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
            p.space_after = Pt(4)
            r1 = p.add_run()
            r1.text = label
            r1.font.bold = True
            r1.font.size = Pt(13)
            r1.font.color.rgb = COLOR_NAVY
            r2 = p.add_run()
            r2.text = val
            r2.font.bold = False
            r2.font.size = Pt(13)
            r2.font.color.rgb = COLOR_TEXT

# ==============================================================================
# HELPER FOR SLIDES 2 to 6
# ==============================================================================
def style_content_slide(slide, title_text, oval_name, sections):
    # 1. Update Title 1
    for s in slide.shapes:
        if "Title" in s.name and s.has_text_frame:
            tf = s.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.text = title_text
            p.font.size = Pt(24)
            p.font.bold = True
            p.font.color.rgb = COLOR_NAVY
        # Update Oval badge
        if s.name == oval_name and s.has_text_frame:
            tf = s.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.text = "SatQuery AI"
            p.alignment = PP_ALIGN.CENTER
            p.font.size = Pt(11)
            p.font.bold = True
            p.font.color.rgb = COLOR_WHITE

        # Update Content Box (TextBox 8)
        if s.name == "TextBox 8" and s.has_text_frame:
            s.left = Inches(0.75)
            s.top = Inches(1.35)
            s.width = Inches(11.8)
            s.height = Inches(5.45)
            tf = s.text_frame
            tf.word_wrap = True
            tf.clear()

            is_first = True
            for sec_title, bullets in sections:
                # Section heading
                p_head = tf.add_paragraph() if not is_first else tf.paragraphs[0]
                is_first = False
                p_head.space_before = Pt(5)
                p_head.space_after = Pt(2)
                r_head = p_head.add_run()
                r_head.text = sec_title
                r_head.font.bold = True
                r_head.font.size = Pt(12)
                r_head.font.color.rgb = COLOR_NAVY

                # Bullets
                for b_text in bullets:
                    p_b = tf.add_paragraph()
                    p_b.space_before = Pt(1)
                    p_b.space_after = Pt(2)
                    p_b.level = 0
                    
                    if ":" in b_text and b_text.startswith("- "):
                        parts = b_text.split(":", 1)
                        r_b1 = p_b.add_run()
                        r_b1.text = parts[0] + ":"
                        r_b1.font.bold = True
                        r_b1.font.size = Pt(10)
                        r_b1.font.color.rgb = COLOR_NAVY
                        
                        r_b2 = p_b.add_run()
                        r_b2.text = parts[1]
                        r_b2.font.bold = False
                        r_b2.font.size = Pt(10)
                        r_b2.font.color.rgb = COLOR_TEXT
                    else:
                        r_b = p_b.add_run()
                        r_b.text = b_text
                        r_b.font.size = Pt(10)
                        r_b.font.color.rgb = COLOR_TEXT


# ==============================================================================
# SLIDE 2: IDEA TITLE
# ==============================================================================
slide2_sections = [
    (
        "Proposed Solution (Describe your Idea/Solution/Prototype)",
        [
            "- SatQuery AI: An agentic, multimodal Earth Observation platform designed for ISRO / Space Applications Centre (SAC) converting natural language into verifiable geospatial intelligence across Optical (Cartosat-2S, Sentinel-2) and SAR (RISAT-1A, Sentinel-1) imagery.",
            "- Interactive Web-GIS Application: Features MapLibre GL dual-pane comparison swipe slider, dynamic GeoJSON vector polygon overlays, conversational AI chat drawer, and 1-click Intelligence Dossier PDF export."
        ]
    ),
    (
        "Detailed explanation of the proposed solution",
        [
            "- Deterministic Agentic Router (AgenticTaskRouter): Dynamically classifies intent across 5 tasks (Single VQA, Captioning, SAM-2 Text Grounding, Bi-Temporal Change Detection, and Cross-Modal Optical-SAR Fusion) and validates spatial co-registration between image pairs.",
            "- Cross-Modal Optical-SAR Fusion: Combines cloud-blind optical spectral bands with cloud-penetrating SAR microwave roughness (sigma0 in dB) for 24/7 all-weather flood inundation and border monitoring.",
            "- Zero Coordinate Hallucination: Directly projects pixel masks via Affine Transformation Matrices into EPSG:4326 GeoJSON polygons with exact physical hectare calculations ([lon, lat]^T = Affine * [x, y, 1]^T)."
        ]
    ),
    (
        "How it addresses the problem",
        [
            "- Solves Optical Cloud Blindspot: Optical sensors are blind during monsoon cloud cover - precisely when flood and cyclone disaster intelligence is most urgent. SAR radar penetrates clouds to delineate standing water.",
            "- Democratizes Complex GIS: Replaces cumbersome manual GIS desktop software with conversational English for disaster field officers.",
            "- 100% Observable Telemetry: Replaces opaque black-box LLMs with verifiable JSON execution traces detailing tools, bounds, and latency."
        ]
    ),
    (
        "Innovation and uniqueness of the solution",
        [
            "- Sovereign Dual-Use Architecture: Tailored for ISRO Bhuvan / MOSDAC / NDMA disaster response + commercial agritech PMFBY crop insurance verification and NHAI infrastructure auditing.",
            "- Sub-2.5s Lightweight Inference: 4-bit QLoRA fine-tuned Qwen2-VL-2B adapted for remote sensing, running in <6GB VRAM on consumer hardware."
        ]
    )
]
style_content_slide(prs.slides[1], "IDEA TITLE: SatQuery AI", "Oval 9", slide2_sections)

# ==============================================================================
# SLIDE 3: TECHNICAL APPROACH
# ==============================================================================
slide3_sections = [
    (
        "Technologies to be used (e.g. programming languages, frameworks, hardware)",
        [
            "- AI & MLOps Stack: Python 3.11, PyTorch 2.6, Qwen2-VL-2B (4-bit QLoRA), Grounding DINO + SAM-2 (Segment Anything Model 2), ChangeFormer-V2 cross-attention head, ONNX Runtime, GPUManager (dynamic CUDA VRAM management & cache clearing).",
            "- Geospatial & Ingestion Engine: Rasterio, GDAL, NumPy, SciPy (adaptive Lee speckle filter 5x5), Shapely, Affine transformation matrices.",
            "- Backend & Spatial Cache: FastAPI REST Gateway, Pydantic v2 universal schemas, SQLite spatial cache (satquery_cache.db), ReportLab PDF engine.",
            "- Web-GIS Frontend: React 18, TypeScript, MapLibre GL interactive canvas, Dual-Pane Split Swipe Slider, Tailwind CSS."
        ]
    ),
    (
        "Methodology and process for implementation (Flow Charts/Images/ working prototype)",
        [
            "- Step 1 (Multi-Modal Ingestion): Parses GeoTIFFs (Cartosat, RISAT, Sentinel) and benchmark PNGs; verifies CRS, GSD, and band layouts; converts SAR amplitude to calibrated backscatter sigma0 (dB).",
            "- Step 2 (Agentic Task Orchestration): AgenticTaskRouter parses natural language, determines task modality, verifies spatial co-registration between image footprints, and bounds parameters to strict ISRO guardrails (confidence, change threshold, filter kernel).",
            "- Step 3 (Dynamic Specialist Tool Dispatch): Routes to RS-VQA-Engine, SpatialGrounding-Engine, BiTemporalChange-Engine, or OpticalSARFusion-Engine with microsecond telemetry timing.",
            "- Step 4 (Deterministic Affine Coordinate Projection): Automatically projects binary prediction masks to Earth coordinates (EPSG:4326 GeoJSON polygons) and computes exact surface areas in hectares (ha).",
            "- Step 5 (Verifiable Output & Intelligence Dossier): Emits observable JSON trace (AuditableExecutionTrace), renders interactive vector overlays, and compiles automated Intelligence Dossier PDF reports."
        ]
    )
]
style_content_slide(prs.slides[2], "TECHNICAL APPROACH", "Oval 10", slide3_sections)

# ==============================================================================
# SLIDE 4: FEASIBILITY AND VIABILITY
# ==============================================================================
slide4_sections = [
    (
        "Analysis of the feasibility of the idea",
        [
            "- End-to-End Working Prototype Validated: 36/36 automated integration tests passing (100% pass rate) on GitHub repo (Shourya3113/SatQuery-AI-SIH-26) across API, Orchestration, Geospatial Affine, and SQLite Registry modules.",
            "- Lean Hardware Footprint: Modular specialist backends run within <6GB VRAM (operates smoothly on standard developer laptops and free Google Colab T4 GPUs, eliminating the 16GB VRAM barrier).",
            "- Sovereign & Air-Gapped Compatibility: Entire platform operates self-contained without external third-party proprietary APIs (OpenAI/Anthropic); ready for deployment on NIC MeghRaj or ISRO Bhuvan Cloud servers."
        ]
    ),
    (
        "Potential challenges and risks",
        [
            "- Risk 1 (Spatial Misalignment): Geometric distortion or CRS mismatches between multi-temporal passes or different sensor modalities (Optical vs SAR).",
            "- Risk 2 (SAR Speckle Noise): Coherent radar interference creating false high-frequency edges and spurious change detections.",
            "- Risk 3 (LLM Coordinate Hallucination): General generative models outputting fictitious geographic coordinates and arbitrary polygon boundaries."
        ]
    ),
    (
        "Strategies for overcoming these challenges",
        [
            "- Strategy 1 (Automated Preprocessing Guardrail): Ingestion pipeline verifies CRS (EPSG:32633, EPSG:4326), bounding box overlap, and spatial resolution prior to tool dispatch.",
            "- Strategy 2 (Adaptive Lee Speckle Filtering): Applied 5x5 window adaptive Lee filtering on calibrated SAR intensity to eliminate speckle noise while preserving sharp coastlines and urban edges.",
            "- Strategy 3 (Affine Matrix Coordinate Projection): Hardcoded affine coordinate math directly computes polygon vertices from raster transformations, making coordinate hallucination mathematically impossible."
        ]
    )
]
style_content_slide(prs.slides[3], "FEASIBILITY AND VIABILITY", "Oval 11", slide4_sections)

# ==============================================================================
# SLIDE 5: IMPACT AND BENEFITS
# ==============================================================================
slide5_sections = [
    (
        "Potential impact on the target audience",
        [
            "- ISRO / SAC & National Disaster Management Authority (NDMA): Instantaneous flood inundation and landslide mapping through dense clouds, cutting emergency damage assessment from days to <3 seconds.",
            "- Defense & Border Security Organizations: 24/7 all-weather structural change detection and vehicle/runway monitoring along borders without manual GIS registration.",
            "- District Administration & Relief Field Officers: Conversational natural-language interface enables field commanders to query satellite data without specialized GIS training."
        ]
    ),
    (
        "Benefits of the solution (social, economic, environmental, etc.)",
        [
            "- Social Benefit (Disaster Resilience): Rapid, verified disaster mapping directly saves lives during monsoon floods, cyclones, and cloudburst events by directing first responders to isolated regions.",
            "- Economic Benefit (Dual-Use Commercial Market):",
            "   - Agritech & Crop Insurance: Automates crop damage assessment for Pradhan Mantri Fasal Bima Yojana (PMFBY), preventing fraudulent claims and expediting payouts.",
            "   - Infrastructure Auditing: Dynamically monitors NHAI highway expansion, urban sprawl, and encroachments.",
            "   - Cost Reduction: Sovereign open-source stack eliminates expensive commercial GIS desktop licenses ($5,000+/seat).",
            "- Environmental Benefit: Continuous monitoring of deforestation, reservoir depletion, and wetland conservation through automated NDVI/NDWI indexing and directional change analysis."
        ]
    )
]
style_content_slide(prs.slides[4], "IMPACT AND BENEFITS", "Oval 11", slide5_sections)

# ==============================================================================
# SLIDE 6: RESEARCH AND REFERENCES
# ==============================================================================
slide6_sections = [
    (
        "Details / Links of the reference and research work",
        [
            "- Benchmark Datasets & Domain Adaptation:",
            "   - BigEarthNet-MM: 590,326 Sentinel-1 SAR and Sentinel-2 Multispectral patch pairs used for cross-modal contrastive representation learning.",
            "   - VRSBench: High-resolution visual grounding, scene captioning, and remote sensing VQA benchmarks.",
            "   - RSVQA & CDVQA: High/Low resolution VQA and Change Detection Visual Question Answering benchmarks.",
            "   - ISRO Sensor Calibration: Cartosat-2S (0.65m GSD pan-sharpened optical) & RISAT-1A / EOS-04 (C-band SAR backscatter calibration).",
            "- Foundational Literature & Model Backbones:",
            "   - Segment Anything Model 2 (SAM-2): Kirillov et al., Meta AI (2024) - Zero-shot promptable mask delineation.",
            "   - ChangeFormer-V2: Bandara & Patel (2022) - Bitemporal Transformer for remote sensing change detection.",
            "   - Qwen2-VL: Wang et al., Alibaba Cloud (2024) - Vision-Language model with dynamic resolution processing.",
            "- Project Repository & Verified Codebase:",
            "   - GitHub Repository: https://github.com/Shourya3113/SatQuery-AI-SIH-26",
            "   - Test Suite: 36/36 automated integration tests passing across REST API, Orchestrator, Affine Engine, and SQLite Registry."
        ]
    )
]
style_content_slide(prs.slides[5], "RESEARCH AND REFERENCES", "Oval 8", slide6_sections)

# ==============================================================================
# SLIDE 7: INSTRUCTIONS SLIDE DELETION (Per SIH 6-Slide Maximum Rule)
# ==============================================================================
if len(prs.slides) > 6:
    print(f"Removing instruction slide 7 (template explicitly allows deletion to comply with 6-slide max limit)...")
    rId = prs.slides._sldIdLst[6].rId
    prs.part.drop_rel(rId)
    del prs.slides._sldIdLst[6]

prs.save("SIH2026-IDEA-Presentation-Format.pptx")
print("Successfully generated SIH2026-IDEA-Presentation-Format.pptx with exactly 6 slides!")
