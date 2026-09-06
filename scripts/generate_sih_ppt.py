"""
Generate visually appealing, diagram-rich SIH 2026 Idea Presentation.
Maintains 100% compliance with official AICTE template and 6-slide limit.
Splits Slides 2-6 into 2 columns:
- Left Column: Mandatory template pointer headings and concise explanations.
- Right Column: High-resolution architecture & pipeline infographics.
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement

PPT_PATH = Path("SIH2026-IDEA-Presentation-Format.pptx")
BACKUP_PATH = Path("SIH2026-IDEA-Presentation-Format_Backup.pptx")
ASSETS_DIR = Path("assets")

# Restore fresh template
prs = Presentation(str(BACKUP_PATH))

# Color palette
COLOR_NAVY = RGBColor(15, 37, 75)
COLOR_TEXT = RGBColor(40, 40, 40)
COLOR_WHITE = RGBColor(255, 255, 255)


def format_header_paragraph(p, text):
    """Format section header paragraph without any stray bullets or indents."""
    pPr = p._p.get_or_add_pPr()
    for c in list(pPr):
        if any(t in c.tag for t in ('buChar', 'buAutoNum', 'buFont', 'buSzPct', 'buSzPts', 'buClr', 'buNone')):
            pPr.remove(c)
    pPr.insert(0, OxmlElement('a:buNone'))
    pPr.set('marL', '0')
    pPr.set('indent', '0')
    p.space_before = Pt(5)
    p.space_after = Pt(2)
    r = p.add_run()
    r.text = text
    r.font.bold = True
    r.font.size = Pt(10.5)
    r.font.color.rgb = COLOR_NAVY


def add_bullet_paragraph(tf, text):
    """Add a beautifully formatted bullet line with bold navy prefixes."""
    p = tf.add_paragraph()
    p.space_before = Pt(1)
    p.space_after = Pt(2)
    pPr = p._p.get_or_add_pPr()
    pPr.set('marL', '228600')
    pPr.set('indent', '-152400')
    
    clean_text = text.strip()
    is_sub_bullet = text.startswith("   -") or text.startswith("    -")
    font_size = Pt(8.3) if is_sub_bullet else Pt(8.8)
    prefix = "   • " if is_sub_bullet else "• "
    
    if ":" in clean_text and (clean_text.startswith("- ") or clean_text.startswith("• ")):
        parts = clean_text.split(":", 1)
        r1 = p.add_run()
        r1.text = prefix + parts[0].lstrip("-• ") + ":"
        r1.font.bold = True
        r1.font.size = font_size
        r1.font.color.rgb = COLOR_NAVY
        
        r2 = p.add_run()
        r2.text = parts[1]
        r2.font.bold = False
        r2.font.size = font_size
        r2.font.color.rgb = COLOR_TEXT
    else:
        r = p.add_run()
        r.text = prefix + clean_text.lstrip("-• ")
        r.font.size = font_size
        r.font.color.rgb = COLOR_TEXT


# ==============================================================================
# SLIDE 1: TITLE PAGE
# ==============================================================================
slide1 = prs.slides[0]

for s in slide1.shapes:
    if s.name == "Subtitle 3" and s.has_text_frame:
        tf = s.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        p.text = "SatQuery AI: Agentic Multimodal Remote Sensing Intelligence Platform"
        p.font.size = Pt(20)
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
            r1.font.size = Pt(12)
            r1.font.color.rgb = COLOR_NAVY
            r2 = p.add_run()
            r2.text = val
            r2.font.bold = False
            r2.font.size = Pt(12)
            r2.font.color.rgb = COLOR_TEXT


# ==============================================================================
# HELPER FOR TWO-COLUMN CONTENT SLIDES (SLIDES 2 to 6)
# ==============================================================================
def populate_content_slide(slide, title_text, sections, diagram_image_path):
    # 1. Title
    for s in slide.shapes:
        if "Title" in s.name and s.has_text_frame:
            tf = s.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.text = title_text
            p.font.size = Pt(23)
            p.font.bold = True
            p.font.color.rgb = COLOR_NAVY

        # 2. Team Badge (Oval shape)
        if "Oval" in s.name and s.has_text_frame:
            tf = s.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.text = "SatQuery AI"
            p.alignment = PP_ALIGN.CENTER
            p.font.size = Pt(11.5)
            p.font.bold = True
            p.font.color.rgb = COLOR_NAVY

        # 3. Left Column: Text Box with Mandatory Pointers
        if s.name == "TextBox 8" and s.has_text_frame:
            s.left = Inches(0.55)
            s.top = Inches(1.35)
            s.width = Inches(6.50)
            s.height = Inches(5.45)
            tf = s.text_frame
            tf.word_wrap = True
            tf.clear()

            for idx, (sec_title, bullets) in enumerate(sections):
                p_head = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
                format_header_paragraph(p_head, sec_title)

                for b_text in bullets:
                    add_bullet_paragraph(tf, b_text)

    # 4. Right Column: Embed High-Resolution Diagram
    if diagram_image_path and Path(diagram_image_path).exists():
        slide.shapes.add_picture(
            str(diagram_image_path),
            left=Inches(7.15),
            top=Inches(1.35),
            width=Inches(5.65),
            height=Inches(5.45)
        )


# ==============================================================================
# SLIDE 2: IDEA TITLE & PROPOSED SOLUTION
# ==============================================================================
slide2_sections = [
    (
        "Proposed Solution (Describe your Idea/Solution/Prototype)",
        [
            "- SatQuery AI: Agentic multimodal Earth Observation platform designed for ISRO/SAC converting natural language into verified geospatial intelligence across Optical (Cartosat, S-2) & SAR (RISAT, S-1) imagery.",
            "- Web-GIS Dashboard: Features MapLibre GL dual-pane swipe slider, dynamic GeoJSON vector delineation, and 1-click Intelligence Dossier PDF export."
        ]
    ),
    (
        "Detailed explanation of the proposed solution",
        [
            "- Agentic Task Router: Classifies intent across 5 tasks (Single VQA, Captioning, SAM-2 Grounding, ChangeFormer CDVQA, Optical-SAR Fusion).",
            "- Optical-SAR Cross-Modal Fusion: Combines cloud-blind optical spectral bands with cloud-penetrating SAR microwave roughness (sigma0 dB) for 24/7 all-weather intelligence.",
            "- Zero Coordinate Hallucination: Direct Affine Matrix projection from pixel masks to EPSG:4326 GeoJSON polygons with exact physical hectare calculations."
        ]
    ),
    (
        "How it addresses the problem",
        [
            "- Solves Cloud Blindspot: Overcomes optical sensor blindness during monsoon cloudbursts and cyclones using C-band radar wave penetration.",
            "- Eliminates GIS Complexity: Enables conversational natural-language querying for disaster field officers without desktop GIS expertise.",
            "- 100% Observable Telemetry: Generates verifiable JSON execution traces detailing tools, bounds, confidence, and latencies."
        ]
    ),
    (
        "Innovation and uniqueness of the solution",
        [
            "- Sovereign Dual-Use Architecture: ISRO Bhuvan / MOSDAC / NDMA disaster response + commercial PMFBY crop insurance & NHAI highway auditing.",
            "- Sub-2.5s Lightweight Inference: 4-bit QLoRA fine-tuned Qwen2-VL-2B adapted for remote sensing, operating in <6GB VRAM on consumer GPUs."
        ]
    )
]
populate_content_slide(prs.slides[1], "IDEA TITLE: SatQuery AI", slide2_sections, ASSETS_DIR / "slide2_architecture.png")

# ==============================================================================
# SLIDE 3: TECHNICAL APPROACH
# ==============================================================================
slide3_sections = [
    (
        "Technologies to be used (e.g. programming languages, frameworks, hardware)",
        [
            "- AI & MLOps Stack: Python 3.11, PyTorch 2.6, Qwen2-VL-2B (4-bit QLoRA), Grounding DINO + SAM-2, ChangeFormer-V2, ONNX Runtime, GPUManager.",
            "- Geospatial Engine: Rasterio, GDAL, NumPy, SciPy (adaptive 5x5 Lee speckle filter), Shapely, Affine transformation matrices.",
            "- Backend & Spatial Lake: FastAPI REST Gateway, Pydantic v2 schemas, SQLite spatial cache (satquery_cache.db), ReportLab PDF engine.",
            "- Web-GIS Frontend: React 18, TypeScript, MapLibre GL, Dual-Pane Split Swipe Slider, Tailwind CSS."
        ]
    ),
    (
        "Methodology and process for implementation (Flow Charts/Images/ working prototype)",
        [
            "- Stage 1 (Multi-Modal Ingestion): Parses GeoTIFFs & benchmark PNGs; verifies CRS, GSD, bands; converts SAR amplitude to calibrated sigma0 (dB).",
            "- Stage 2 (Agentic Orchestration): AgenticTaskRouter parses natural language, validates spatial co-registration between images, and bounds parameters.",
            "- Stage 3 (Dynamic Tool Execution): Routes to RS-VQA, SAM-2 Grounding, ChangeFormer CDVQA, or Optical-SAR Fusion with microsecond telemetry.",
            "- Stage 4 (Deterministic Affine Math): Projects binary masks to Earth coordinates [lon, lat]^T = Affine * [x, y, 1]^T, computing exact area in hectares.",
            "- Stage 5 (Trace & Dossier Synthesis): Emits observable JSON trace, renders interactive vector overlays, and compiles automated Intelligence Dossier PDFs."
        ]
    )
]
populate_content_slide(prs.slides[2], "TECHNICAL APPROACH", slide3_sections, ASSETS_DIR / "slide3_pipeline.png")

# ==============================================================================
# SLIDE 4: FEASIBILITY AND VIABILITY
# ==============================================================================
slide4_sections = [
    (
        "Analysis of the feasibility of the idea",
        [
            "- Working Prototype Validated: 36/36 automated integration tests passing (100% pass rate) on GitHub repo across API, Orchestrator, Affine Engine, and SQLite Registry.",
            "- Lean Hardware Footprint: Modular specialist backends run within <6GB VRAM (operates smoothly on standard developer laptops and free Google Colab T4 GPUs).",
            "- Sovereign & Air-Gapped Ready: Operates self-contained without external third-party proprietary APIs (OpenAI/Anthropic); deployable on NIC MeghRaj or Bhuvan Cloud."
        ]
    ),
    (
        "Potential challenges and risks",
        [
            "- Risk 1 (Spatial Misalignment): Geometric distortion or CRS mismatches between multi-temporal passes or different sensor modalities.",
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
populate_content_slide(prs.slides[3], "FEASIBILITY AND VIABILITY", slide4_sections, ASSETS_DIR / "slide4_feasibility.png")

# ==============================================================================
# SLIDE 5: IMPACT AND BENEFITS
# ==============================================================================
slide5_sections = [
    (
        "Potential impact on the target audience",
        [
            "- ISRO / SAC & National Disaster Management Authority (NDMA): Instantaneous flood and landslide mapping through dense clouds, cutting damage assessment from days to <3s.",
            "- Defense & Border Security Organizations: 24/7 all-weather structural change detection and vehicle/runway monitoring along borders without manual GIS registration.",
            "- District Administration & Field Officers: Conversational natural-language interface enables field commanders to query satellite data without specialized GIS training."
        ]
    ),
    (
        "Benefits of the solution (social, economic, environmental, etc.)",
        [
            "- Social Benefit (Disaster Resilience): Rapid, verified disaster mapping directly saves lives during monsoon floods, cyclones, and cloudburst events by directing first responders to isolated regions.",
            "- Economic Benefit (Dual-Use Commercial Market):",
            "   - Agritech & Crop Insurance: Automates crop damage assessment for PMFBY, preventing fraudulent claims and accelerating payouts.",
            "   - Infrastructure Auditing: Dynamically monitors NHAI highway expansion, urban sprawl, and encroachments.",
            "   - Cost Reduction: Sovereign open-source stack eliminates expensive commercial GIS desktop licenses ($5,000+/seat).",
            "- Environmental Benefit: Continuous monitoring of deforestation, reservoir depletion, and wetland conservation through automated NDVI/NDWI indexing and directional change analysis."
        ]
    )
]
populate_content_slide(prs.slides[4], "IMPACT AND BENEFITS", slide5_sections, ASSETS_DIR / "slide5_impact.png")

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
populate_content_slide(prs.slides[5], "RESEARCH AND REFERENCES", slide6_sections, ASSETS_DIR / "slide6_research.png")

# ==============================================================================
# SLIDE 7: INSTRUCTIONS SLIDE DELETION (Strict SIH 6-Slide Maximum Rule)
# ==============================================================================
if len(prs.slides) > 6:
    print(f"Removing instruction slide 7 to maintain strict 6-slide limit...")
    rId = prs.slides._sldIdLst[6].rId
    prs.part.drop_rel(rId)
    del prs.slides._sldIdLst[6]

prs.save(str(PPT_PATH))
print(f"Successfully generated visually enhanced {PPT_PATH} with 6 slides and embedded diagrams!")
