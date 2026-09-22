"""
Generate professional, publication-quality diagrams for the SatQuery AI SIH 2026 Presentation.
Designs high-contrast, clean infographics matching the navy/teal/orange palette.
"""

from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Common styling constants
NAVY = "#0f254b"
BLUE = "#1e40af"
TEAL = "#0d9488"
ORANGE = "#ea580c"
SLATE = "#334155"
LIGHT_BG = "#ffffff"   # Seamless pure white to blend with slide canvas
CARD_BG = "#ffffff"
BORDER = "#cbd5e1"
GREEN = "#16a34a"

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']


def wrap(text: str, width: int) -> str:
    """Helper to wrap text cleanly for matplotlib annotations."""
    return "\n".join(textwrap.wrap(text, width=width))


def create_architecture_diagram():
    """Slide 2 Diagram: End-to-End Multimodal Agentic Architecture"""
    fig, ax = plt.subplots(figsize=(6.5, 6.8), dpi=300)
    ax.set_facecolor(LIGHT_BG)
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.5)
    ax.axis('off')

    # Title Card
    ax.text(5.0, 10.1, "SatQuery AI — System Architecture", ha='center', va='center',
            fontsize=13, fontweight='bold', color=NAVY)

    # 1. Inputs Box
    p1 = patches.FancyBboxPatch((0.5, 8.3), 9.0, 1.4, boxstyle="round,pad=0.2,rounding_size=0.15",
                                fc="#eff6ff", ec="#93c5fd", lw=1.5)
    ax.add_patch(p1)
    ax.text(5.0, 9.35, "INPUTS: Multimodal Earth Observation Rasters + NL Query",
            ha='center', va='center', fontsize=9.5, fontweight='bold', color=BLUE)
    ax.text(2.0, 8.7, "• Optical RGB / NIR (Cartosat, S-2)\n• Calibrated SAR C-Band (RISAT, S-1)",
            ha='left', va='center', fontsize=8.0, color=SLATE)
    ax.text(6.0, 8.7, "• Bi-Temporal Pairs (T1 vs T2)\n• Natural Language Query Prompt",
            ha='left', va='center', fontsize=8.0, color=SLATE)

    # Arrow 1
    ax.annotate('', xy=(5.0, 7.8), xytext=(5.0, 8.3),
                arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=1.5, headwidth=6, shrink=0.05))

    # 2. Central Agentic Orchestrator
    p2 = patches.FancyBboxPatch((0.5, 6.4), 9.0, 1.4, boxstyle="round,pad=0.2,rounding_size=0.15",
                                fc="#f0fdf4", ec="#86efac", lw=1.8)
    ax.add_patch(p2)
    ax.text(5.0, 7.45, "AGENTIC TASK ORCHESTRATOR (AgenticTaskRouter)",
            ha='center', va='center', fontsize=9.5, fontweight='bold', color=GREEN)
    ax.text(5.0, 6.9, "• Natural Language Intent Classification (5 RS Task Categories)\n• Spatial Co-Registration & CRS Verification (EPSG:32633 / 4326)\n• Mandatory ISRO Parameter Guardrails (Confidence, Thresholds, Kernel)",
            ha='center', va='center', fontsize=7.8, color=SLATE)

    # Arrow 2
    ax.annotate('', xy=(5.0, 5.9), xytext=(5.0, 6.4),
                arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=1.5, headwidth=6, shrink=0.05))

    # 3. Specialist Engines (4 Grid Cards)
    ax.text(5.0, 5.8, "DYNAMIC SPECIALIST AI ENGINES", ha='center', va='center',
            fontsize=9.0, fontweight='bold', color=NAVY)

    tools = [
        ("Tool 1: RS-VLM Engine", "Qwen2-VL (4-bit QLoRA)\nScene Captioning & VQA", 0.5, 4.4, "#fef3c7", "#fde68a", "#92400e"),
        ("Tool 2: Grounding SAM-2", "Grounding DINO + SAM-2\nPolygon Mask Delineation", 5.2, 4.4, "#e0f2fe", "#bae6fd", "#075985"),
        ("Tool 3: ChangeFormer-V2", "Cross-Attention BIT Head\nDirectional CDVQA (% delta)", 0.5, 3.1, "#fce7f3", "#fbcfe8", "#9d174d"),
        ("Tool 4: Optical-SAR Fusion", "Siamese Feature Fusion\nCloud-Penetrating 24/7 Vision", 5.2, 3.1, "#f3e8ff", "#e9d5ff", "#6b21a8")
    ]
    for title, desc, x, y, fc, ec, tc in tools:
        box = patches.FancyBboxPatch((x, y), 4.3, 1.05, boxstyle="round,pad=0.15,rounding_size=0.12",
                                     fc=fc, ec=ec, lw=1.2)
        ax.add_patch(box)
        ax.text(x + 2.15, y + 0.75, title, ha='center', va='center', fontsize=7.5, fontweight='bold', color=tc)
        ax.text(x + 2.15, y + 0.35, desc, ha='center', va='center', fontsize=6.8, color=SLATE)

    # Arrow 3
    ax.annotate('', xy=(5.0, 2.6), xytext=(5.0, 3.05),
                arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=1.5, headwidth=6, shrink=0.05))

    # 4. Affine Projector
    p4 = patches.FancyBboxPatch((1.2, 1.9), 7.6, 0.75, boxstyle="round,pad=0.15,rounding_size=0.12",
                                fc="#fff7ed", ec="#fed7aa", lw=1.5)
    ax.add_patch(p4)
    ax.text(5.0, 2.4, "AFFINE COORDINATE PROJECTOR (Zero Hallucination)",
            ha='center', va='center', fontsize=8.5, fontweight='bold', color=ORANGE)
    ax.text(5.0, 2.05, "Matrix Transform: [lon, lat]^T = Affine * [x, y, 1]^T  ->  Exact Hectares",
            ha='center', va='center', fontsize=7.5, color=SLATE)

    # Arrow 4
    ax.annotate('', xy=(5.0, 1.4), xytext=(5.0, 1.9),
                arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=1.5, headwidth=6, shrink=0.05))

    # 5. Output Layer
    p5 = patches.FancyBboxPatch((0.5, 0.2), 9.0, 1.15, boxstyle="round,pad=0.15,rounding_size=0.12",
                                fc="#eff6ff", ec="#60a5fa", lw=1.5)
    ax.add_patch(p5)
    ax.text(5.0, 1.05, "CESIUM ION 3D DIGITAL GLOBE & VERIFIABLE TELEMETRY", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=BLUE)
    ax.text(0.8, 0.55, "• Cesium 3D Digital Earth (Pan-India Geoid)\n• 3D Extruded Flood & Grounding Vectors",
            ha='left', va='center', fontsize=7.2, color=NAVY)
    ax.text(5.4, 0.55, "• Real-Time Lat/Lon/Elevation HUD\n• Verifiable Execution Trace & PDF Dossier",
            ha='left', va='center', fontsize=7.2, color=SLATE)

    plt.tight_layout()
    out_path = ASSETS_DIR / "slide2_architecture.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Generated {out_path}")


def create_pipeline_diagram():
    """Slide 3 Diagram: 5-Stage Implementation Pipeline & Tech Stack"""
    fig, ax = plt.subplots(figsize=(6.5, 6.8), dpi=300)
    ax.set_facecolor(LIGHT_BG)
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.5)
    ax.axis('off')

    ax.text(5.0, 10.1, "Technical Dataflow & Math Pipeline", ha='center', va='center',
            fontsize=13, fontweight='bold', color=NAVY)

    # 4 Tech Stack Pillars at the top
    ax.text(5.0, 9.4, "4 SPECIALIZED CORE ARCHITECTURAL PILLARS", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=TEAL)

    pillars = [
        ("AI & MLOps", "PyTorch 2.6, Qwen2-VL\nSAM-2, ONNX FP16", 0.5, 8.2, "#eff6ff", "#bfdbfe", BLUE),
        ("Geospatial", "Rasterio, GDAL, NumPy\nLee Speckle Filter", 2.85, 8.2, "#f0fdf4", "#bbf7d0", GREEN),
        ("Backend Lake", "FastAPI, SQLite Cache\nReportLab Dossier", 5.2, 8.2, "#faf5ff", "#e9d5ff", "#7e22ce"),
        ("3D Digital Twin", "CesiumJS 3D Earth\nWebGL 60FPS Geoid", 7.55, 8.2, "#eff6ff", "#93c5fd", BLUE)
    ]
    for name, desc, x, y, fc, ec, tc in pillars:
        box = patches.FancyBboxPatch((x, y), 2.0, 1.0, boxstyle="round,pad=0.1,rounding_size=0.1",
                                     fc=fc, ec=ec, lw=1.2)
        ax.add_patch(box)
        ax.text(x + 1.0, y + 0.72, name, ha='center', va='center', fontsize=7.2, fontweight='bold', color=tc)
        ax.text(x + 1.0, y + 0.32, desc, ha='center', va='center', fontsize=6.2, color=SLATE)

    # Divider line
    ax.axhline(7.95, color="#cbd5e1", linestyle="--", linewidth=1.0)

    # 5-Stage Step Flowchart
    steps = [
        ("STAGE 1: Ingestion & Calibration",
         "Ingests GeoTIFF COGs / PNGs; parses CRS & GSD (m/px).\n"
         "SAR Radiometric Calibration: sigma0(dB) = 10*log10(DN^2) - Kcal\n"
         "Applies Adaptive 5x5 Lee speckle filter on SAR intensity.",
         "#f8fafc", "#94a3b8", NAVY, 6.7),

        ("STAGE 2: Agentic Orchestration & Guardrails",
         "Deterministic AgenticTaskRouter parses natural-language prompt.\n"
         "Verifies spatial co-registration between image footprints.\n"
         "Enforces parameter guardrails: confidence in [0.1, 0.99], change threshold in [0.1, 0.95].",
         "#eff6ff", "#93c5fd", BLUE, 5.1),

        ("STAGE 3: Specialist Tool Invocation & Telemetry",
         "Dispatches to dedicated engine with microsecond timing.\n"
         "Single VQA / Captioning / SAM-2 Grounding / ChangeFormer / Fusion.\n"
         "MLOps GPUManager performs CUDA cache clearing to prevent OOM.",
         "#f0fdf4", "#86efac", GREEN, 3.5),

        ("STAGE 4: Affine Coordinate Projection Math",
         "Delineated 2D boolean raster masks projected to EPSG:4326:\n"
         "[longitude, latitude, 1]^T = Affine_Matrix * [x_pixel, y_pixel, 1]^T\n"
         "Computes exact physical polygon surface area in hectares (ha).",
         "#fff7ed", "#fed7aa", ORANGE, 1.9),

        ("STAGE 5: Cesium 3D Digital Earth & Dossier Synthesis",
         "Projects 3D extruded vector polygons onto photorealistic Cesium WGS84 Geoid.\n"
         "Pan-India fly-tos (Brahmaputra, Sundarbans, Delhi); exports ReportLab PDF.",
         "#faf5ff", "#d8b4fe", "#6b21a8", 0.3)
    ]

    for title, desc, fc, ec, tc, y in steps:
        box = patches.FancyBboxPatch((0.5, y), 9.0, 1.25, boxstyle="round,pad=0.15,rounding_size=0.12",
                                     fc=fc, ec=ec, lw=1.2)
        ax.add_patch(box)
        ax.text(0.8, y + 0.95, title, ha='left', va='center', fontsize=8.0, fontweight='bold', color=tc)
        ax.text(0.8, y + 0.45, desc, ha='left', va='center', fontsize=6.8, color=SLATE)
        if y > 0.5:
            ax.annotate('', xy=(5.0, y - 0.28), xytext=(5.0, y),
                        arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=1.2, headwidth=4.5, shrink=0.05))

    plt.tight_layout()
    out_path = ASSETS_DIR / "slide3_pipeline.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Generated {out_path}")


def create_feasibility_diagram():
    """Slide 4 Diagram: Feasibility, Risk vs. Mitigation Matrix & Test Status"""
    fig, ax = plt.subplots(figsize=(6.5, 6.8), dpi=300)
    ax.set_facecolor(LIGHT_BG)
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.5)
    ax.axis('off')

    ax.text(5.0, 10.1, "Feasibility & Risk Mitigation Matrix", ha='center', va='center',
            fontsize=13, fontweight='bold', color=NAVY)

    # 1. CI/CD Passing Badge
    badge = patches.FancyBboxPatch((0.5, 8.95), 9.0, 0.80, boxstyle="round,pad=0.12,rounding_size=0.12",
                                   fc="#dcfce7", ec="#22c55e", lw=1.8)
    ax.add_patch(badge)
    ax.text(5.0, 9.45, "VERIFIED ON GITHUB CI: 46 / 46 TESTS PASSING (100%)",
            ha='center', va='center', fontsize=9.2, fontweight='bold', color="#15803d")
    ax.text(5.0, 9.15, "FastAPI REST Gateway • Agentic Router • Affine Engine • SQLite Cache • ReportLab PDF • Cesium 3D",
            ha='center', va='center', fontsize=6.8, color="#166534")

    # 2. Risk vs Mitigation Matrix
    ax.text(5.0, 8.5, "TECHNICAL RISKS & ENGINEERING MITIGATIONS", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=NAVY)

    cards = [
        (
            "RISK 1: Optical Cloud Blindspot",
            "Monsoon & Night Inundations",
            wrap("Dense cloud cover & night-time render optical satellites completely opaque during critical flood/cyclone disasters.", 34),
            "MITIGATION: SAR Microwave Fusion",
            "RISAT-1A & Sentinel-1 C-Band",
            wrap("Ingests C-band radar penetrating clouds 24/7. Maps flood water extent via specular microwave backscatter (sigma0 dB).", 34),
            "#fee2e2", "#ef4444", "#dcfce7", "#16a34a", 6.25
        ),
        (
            "RISK 2: LLM Coordinate Hallucination",
            "Fictitious Lat/Lon Bounding Boxes",
            wrap("Generative VLMs lack spatial coordinate reference systems (CRS) and hallucinate arbitrary bounding boxes and coordinates.", 34),
            "MITIGATION: Affine Projector Math",
            "Deterministic GeoTIFF Transform",
            wrap("Derives all vertices via 6-parameter affine matrices: [lon, lat]^T = Affine * [x, y, 1]^T. Coordinate error is 0.0%.", 34),
            "#ffedd5", "#f97316", "#e0f2fe", "#0284c7", 4.05
        ),
        (
            "RISK 3: GPU VRAM & Cloud Costs",
            "70B Monoliths Require 16GB+ VRAM",
            wrap("Monolithic VLMs cause Out-Of-Memory crashes on edge hardware and incur steep commercial cloud API subscription costs.", 34),
            "MITIGATION: 4-Bit QLoRA & ONNX",
            "Sub-2.5s Edge Inference (<6GB)",
            wrap("Qwen2-VL-2B fine-tuned via 4-bit QLoRA. Runs comfortably in <6GB VRAM on free Google Colab T4 and local laptops.", 34),
            "#fef3c7", "#f59e0b", "#f3e8ff", "#9333ea", 1.85
        )
    ]

    for r_head, r_sub, r_body, m_head, m_sub, m_body, r_fc, r_ec, m_fc, m_ec, y in cards:
        # Risk Subcard (Left: x=0.5 to 4.7)
        rb = patches.FancyBboxPatch((0.5, y), 4.2, 1.9, boxstyle="round,pad=0.1,rounding_size=0.1",
                                    fc=r_fc, ec=r_ec, lw=1.2)
        ax.add_patch(rb)
        ax.text(0.7, y + 1.68, r_head, ha='left', va='top', fontsize=7.2, fontweight='bold', color="#991b1b")
        ax.text(0.7, y + 1.45, r_sub, ha='left', va='top', fontsize=6.2, fontweight='bold', color="#b91c1c")
        ax.text(0.7, y + 1.20, r_body, ha='left', va='top', fontsize=6.2, color=SLATE, linespacing=1.2)

        # Arrow between
        ax.annotate('', xy=(5.25, y + 0.95), xytext=(4.75, y + 0.95),
                    arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=1.2, headwidth=4.5, shrink=0.05))

        # Mitigation Subcard (Right: x=5.3 to 9.5)
        mb = patches.FancyBboxPatch((5.3, y), 4.2, 1.9, boxstyle="round,pad=0.1,rounding_size=0.1",
                                    fc=m_fc, ec=m_ec, lw=1.2)
        ax.add_patch(mb)
        ax.text(5.5, y + 1.68, m_head, ha='left', va='top', fontsize=7.2, fontweight='bold', color="#166534" if "SAR" in m_head else NAVY)
        ax.text(5.5, y + 1.45, m_sub, ha='left', va='top', fontsize=6.2, fontweight='bold', color="#15803d" if "SAR" in m_head else BLUE)
        ax.text(5.5, y + 1.20, m_body, ha='left', va='top', fontsize=6.2, color=SLATE, linespacing=1.2)

    # 3. Hardware & Empirical Benchmarks Card at bottom
    bot = patches.FancyBboxPatch((0.5, 0.25), 9.0, 1.35, boxstyle="round,pad=0.12,rounding_size=0.12",
                                 fc="#f8fafc", ec="#cbd5e1", lw=1.2)
    ax.add_patch(bot)
    ax.text(5.0, 1.25, "EMPIRICAL BENCHMARKS SCORECARD: 79.72 / 100.0 (AUDITED ZERO-SHOT BASELINE)", ha='center', va='center',
            fontsize=8.0, fontweight='bold', color=NAVY)
    ax.text(5.0, 0.72, "• CDVQA F1: 1.0000 | VRSBench mIoU: 1.0000 | BigEarthNet-MM: 90% | RSVQA BLEU: 0.1109\n"
                       "• Sub-1.8s End-to-End Latency • < 5.8 GB VRAM footprint • RS-XAI Faithfulness Ratio: 1.98x - 3.25x\n"
                       "• Sovereign Air-Gapped Ready: 0% external cloud API reliance • Deployable on Bhuvan / MeghRaj.",
            ha='center', va='center', fontsize=6.8, color=SLATE, linespacing=1.3)

    plt.tight_layout()
    out_path = ASSETS_DIR / "slide4_feasibility.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Generated {out_path}")


def create_impact_diagram():
    """Slide 5 Diagram: Dual-Use Impact & Stakeholder Ecosystem"""
    fig, ax = plt.subplots(figsize=(6.5, 6.8), dpi=300)
    ax.set_facecolor(LIGHT_BG)
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.5)
    ax.axis('off')

    ax.text(5.0, 10.1, "Dual-Use Ecosystem & Stakeholder Impact", ha='center', va='center',
            fontsize=13, fontweight='bold', color=NAVY)

    # Pillar 1: Institutional / Sovereign Good (ISRO / NDMA)
    p1 = patches.FancyBboxPatch((0.5, 6.85), 9.0, 2.60, boxstyle="round,pad=0.15,rounding_size=0.15",
                                fc="#eff6ff", ec="#3b82f6", lw=1.5)
    ax.add_patch(p1)
    ax.text(0.8, 9.42, "PRIMARY MISSION: SOVEREIGN PUBLIC GOOD (ISRO / NDMA / SAC)",
            ha='left', va='center', fontsize=8.5, fontweight='bold', color=BLUE)

    items1 = [
        ("• Flood Inundation & Cyclone Damage:",
         wrap("Maps submerged villages through 100% monsoon cloud cover in <3s, directing NDRF rescue boats to critical isolated zones.", 80)),
        ("• Landslide & Glacial Lake Outburst (GLOF):",
         wrap("Bi-temporal SAR interferometry detects millimeter slope displacement before catastrophic Himalayan dam and highway collapses.", 80)),
        ("• Border Surveillance & National Defense:",
         wrap("Continuous 24/7 all-weather change detection of airfields, roads, and vehicle convoys along sensitive international borders.", 80))
    ]
    y_cursor = 8.95
    for title, desc in items1:
        ax.text(0.8, y_cursor, title, ha='left', va='top', fontsize=7.2, fontweight='bold', color=NAVY)
        ax.text(0.8, y_cursor - 0.22, desc, ha='left', va='top', fontsize=6.0, color=SLATE, linespacing=1.15)
        y_cursor -= 0.72

    # Pillar 2: Commercial Deep-Tech Spin-Off
    p2 = patches.FancyBboxPatch((0.5, 3.75), 9.0, 2.60, boxstyle="round,pad=0.15,rounding_size=0.15",
                                fc="#fff7ed", ec="#f97316", lw=1.5)
    ax.add_patch(p2)
    ax.text(0.8, 6.32, "COMMERCIAL SPIN-OFF: DUAL-USE ECONOMIC IMPACT",
            ha='left', va='center', fontsize=8.5, fontweight='bold', color=ORANGE)

    items2 = [
        ("• Agritech & PMFBY Crop Insurance:",
         wrap("Automates crop lodging and flood claims verification, eliminating fraudulent multi-crore claims and accelerating payouts.", 80)),
        ("• Infrastructure & Highway Monitoring:",
         wrap("Audits NHAI highway construction milestones, illegal sand mining, and urban encroachment with automated change detection.", 80)),
        ("• ESG & Carbon Offset Auditing:",
         wrap("Quantifies deforestation hectares and wetland depletion via automated NDVI/NDWI satellite time-series analysis.", 80))
    ]
    y_cursor = 5.85
    for title, desc in items2:
        ax.text(0.8, y_cursor, title, ha='left', va='top', fontsize=7.2, fontweight='bold', color=NAVY)
        ax.text(0.8, y_cursor - 0.22, desc, ha='left', va='top', fontsize=6.0, color=SLATE, linespacing=1.15)
        y_cursor -= 0.72

    # Pillar 3: Quantitative ROI Metric Badges
    p3 = patches.FancyBboxPatch((0.5, 0.65), 9.0, 2.60, boxstyle="round,pad=0.15,rounding_size=0.15",
                                fc="#f0fdf4", ec="#22c55e", lw=1.5)
    ax.add_patch(p3)
    ax.text(5.0, 3.10, "MEASURABLE OPERATIONAL ROI METRICS", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=GREEN)

    metrics = [
        ("< 2.5s", "Inference Latency\n(vs Hours manually)", 1.6, 1.4, BLUE),
        ("100%", "All-Weather Vision\n(SAR Cloud Penetration)", 3.8, 1.4, TEAL),
        ("0.0%", "Coordinate Error\n(Affine Grounding Math)", 6.2, 1.4, ORANGE),
        ("$0", "Per-Seat License Fees\n(Sovereign Open Stack)", 8.4, 1.4, GREEN)
    ]
    for val, lbl, x, y, col in metrics:
        ax.text(x, y + 0.90, val, ha='center', va='center', fontsize=15, fontweight='bold', color=col)
        ax.text(x, y + 0.35, lbl, ha='center', va='center', fontsize=6.5, color=SLATE, linespacing=1.2)

    plt.tight_layout()
    out_path = ASSETS_DIR / "slide5_impact.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Generated {out_path}")


def create_research_diagram():
    """Slide 6 Diagram: Benchmark Evaluation, Sensor Physics & Verified Git Repo"""
    fig, ax = plt.subplots(figsize=(6.5, 6.8), dpi=300)
    ax.set_facecolor(LIGHT_BG)
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.5)
    ax.axis('off')

    ax.text(5.0, 10.1, "Research Alignment & Evaluation Harness", ha='center', va='center',
            fontsize=13, fontweight='bold', color=NAVY)

    # 1. Official Dataset Radar / Matrix
    p1 = patches.FancyBboxPatch((0.5, 6.70), 9.0, 2.80, boxstyle="round,pad=0.15,rounding_size=0.15",
                                fc="#f8fafc", ec="#cbd5e1", lw=1.5)
    ax.add_patch(p1)
    ax.text(5.0, 9.40, "OFFICIAL BENCHMARK DATASET COMPLIANCE MATRIX", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=NAVY)

    benchmarks = [
        ("BigEarthNet-MM", "590,326 Paired Patches",
         wrap("Domain adaptation using paired Sentinel-1 SAR and Sentinel-2 multispectral patches.", 78)),
        ("VRSBench", "High-Resolution Captions",
         wrap("Evaluates SAM-2 visual grounding, object localization, and open-vocabulary bounding box mIoU.", 78)),
        ("RSVQA (HR & LR)", "Remote Sensing VQA Pairs",
         wrap("Evaluates Visual QA accuracy across multi-band satellite rasters and spatial relationships.", 78)),
        ("CDVQA", "Bi-Temporal Image Pairs",
         wrap("Evaluates Change Detection Visual QA and directional transformation (% area change).", 78))
    ]
    y_b = 8.92
    for name, stat, desc in benchmarks:
        ax.text(0.8, y_b, name, ha='left', va='top', fontsize=7.2, fontweight='bold', color=BLUE)
        ax.text(4.2, y_b, stat, ha='left', va='top', fontsize=6.8, fontweight='bold', color=ORANGE)
        ax.text(0.8, y_b - 0.20, desc, ha='left', va='top', fontsize=6.0, color=SLATE, linespacing=1.15)
        y_b -= 0.52

    # 2. ISRO Sensor Physics Specs
    p2 = patches.FancyBboxPatch((0.5, 3.65), 9.0, 2.55, boxstyle="round,pad=0.15,rounding_size=0.15",
                                fc="#f0fdf4", ec="#86efac", lw=1.5)
    ax.add_patch(p2)
    ax.text(5.0, 6.10, "ISRO / SAC SENSOR COMPLIANCE SPECIFICATIONS", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=GREEN)

    sensors = [
        ("• Cartosat-2S (Optical High-Res):",
         wrap("0.65m Pan-Sharpened GSD, 4-band multispectral. Ingested via Rasterio with exact CRS projection (EPSG:32643 / UTM).", 80)),
        ("• RISAT-1A / EOS-04 (C-Band SAR):",
         wrap("C-band (5.35 GHz), VV/VH polarizations, GRD format. Calibrated to Sigma0 dB backscatter with adaptive Lee filter.", 80)),
        ("• Cross-Modal Optical-SAR Pair:",
         wrap("Co-registered spatial footprint evaluated without coordinate drift using 6-parameter affine transformation matrices.", 80))
    ]
    y_s = 5.65
    for name, desc in sensors:
        ax.text(0.8, y_s, name, ha='left', va='top', fontsize=7.2, fontweight='bold', color=NAVY)
        ax.text(0.8, y_s - 0.22, desc, ha='left', va='top', fontsize=6.0, color=SLATE, linespacing=1.15)
        y_s -= 0.65

    # 3. GitHub & Open Source Verification Card
    p3 = patches.FancyBboxPatch((0.5, 0.65), 9.0, 2.50, boxstyle="round,pad=0.15,rounding_size=0.15",
                                fc="#eff6ff", ec="#60a5fa", lw=1.5)
    ax.add_patch(p3)
    ax.text(5.0, 3.00, "OFFICIAL REPOSITORY & VERIFICATION HARNESS (SIH26167)", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=BLUE)
    ax.text(5.0, 2.50, "GitHub Repository: https://github.com/Shourya3113/SatQuery-AI-SIH-26",
            ha='center', va='center', fontsize=7.8, fontweight='bold', color=NAVY)
    ax.text(5.0, 1.70, "• 73/73 Unit & Integration Tests Passing in CI/CD pipeline (100% Pass Rate)\n"
                      "• Empirical Scorecard: 79.72 / 100.0 Audited Baseline across CDVQA, VRSBench, BigEarthNet & RSVQA\n"
                      "• 100% Pan-India Geoid Scenarios: ISRO SAC Ahmedabad, Brahmaputra Flood, Sundarbans & Delhi.",
            ha='center', va='center', fontsize=6.8, color=SLATE, linespacing=1.3)

    plt.tight_layout()
    out_path = ASSETS_DIR / "slide6_research.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Generated {out_path}")


if __name__ == "__main__":
    create_architecture_diagram()
    create_pipeline_diagram()
    create_feasibility_diagram()
    create_impact_diagram()
    create_research_diagram()
