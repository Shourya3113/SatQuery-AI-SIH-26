"""
SatQuery AI - Publication-Grade Intelligence Dossier Generator
Owner: Achintya (Backend Lead) & Peter (Chief Architect)

Generates a comprehensive, verifiable ReportLab PDF Intelligence Dossier
from a SatQuery AI QueryResponse conforming to ISRO SIH26167 standards:
- Section 1: Executive Briefing & Query Objective
- Section 2: Sensor Ingestion & Metadata Audit
- Section 3: Geospatial Detections & Vector Metrics
- Section 4: Visual Evidence & Explainability Verification (RS-XAI)
- Section 5: Auditable Execution Trace & Pipeline Steps
"""

import io
import base64
import datetime
from pathlib import Path
from typing import Optional, Any, Dict, List

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib import colors

from core.schemas import QueryResponse, XAIExplanation


def _get_custom_styles():
    """Builds a cohesive, professional typography system for the dossier."""
    base = getSampleStyleSheet()

    header_style = ParagraphStyle(
        "DossierHeader",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0B192C"),
        alignment=TA_CENTER
    )

    sub_header_style = ParagraphStyle(
        "DossierSubHeader",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        alignment=TA_CENTER
    )

    section_heading = ParagraphStyle(
        "DossierSectionHeading",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E3E62"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    subsection_heading = ParagraphStyle(
        "DossierSubSectionHeading",
        parent=base["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "DossierBody",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1E293B"),
        alignment=TA_LEFT
    )

    body_bold = ParagraphStyle(
        "DossierBodyBold",
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    code_style = ParagraphStyle(
        "DossierCode",
        parent=base["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0F172A")
    )

    table_cell = ParagraphStyle(
        "DossierTableCell",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1E293B")
    )

    table_header = ParagraphStyle(
        "DossierTableHeader",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.white
    )

    callout_text = ParagraphStyle(
        "DossierCalloutText",
        parent=base["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#334155")
    )

    return {
        "header": header_style,
        "sub_header": sub_header_style,
        "section": section_heading,
        "sub_section": subsection_heading,
        "body": body_style,
        "body_bold": body_bold,
        "code": code_style,
        "table_cell": table_cell,
        "table_header": table_header,
        "callout": callout_text
    }


def generate_report(query_response: QueryResponse, output_path: str):
    """
    Generate an Intelligence Dossier PDF from a QueryResponse.
    Fully incorporates Section 4: Visual Evidence & Explainability Verification (RS-XAI).
    """
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    document = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    st = _get_custom_styles()
    story = []

    # =========================================================================
    # Header Banner
    # =========================================================================
    story.append(Paragraph("SATQUERY AI — INTELLIGENCE DOSSIER", st["header"]))
    story.append(Paragraph(
        "Autonomous Agentic Multimodal Remote Sensing Platform | ISRO SIH26167",
        st["sub_header"]
    ))
    story.append(Spacer(1, 8))

    doc_meta_data = [
        [
            Paragraph(f"<b>Trace ID:</b> {query_response.trace_id}", st["table_cell"]),
            Paragraph(f"<b>Date/Time:</b> {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}", st["table_cell"]),
            Paragraph(f"<b>Task:</b> {query_response.task_category.value if hasattr(query_response.task_category, 'value') else query_response.task_category}", st["table_cell"]),
            Paragraph(f"<b>Confidence:</b> {(query_response.confidence_score * 100):.1f}%", st["table_cell"]),
        ]
    ]
    t_meta = Table(doc_meta_data, colWidths=[160, 120, 160, 80])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    # =========================================================================
    # Section 1: Executive Briefing & Query Objective
    # =========================================================================
    story.append(Paragraph("1. Executive Briefing & Query Objective", st["section"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E3E62"), spaceAfter=6))

    query_box_data = [
        [Paragraph("<b>User Query:</b>", st["body_bold"]), Paragraph(f"\"{query_response.query}\"", st["body"])],
        [Paragraph("<b>AI Synthesis:</b>", st["body_bold"]), Paragraph(query_response.text_response, st["body"])],
    ]
    t_query = Table(query_box_data, colWidths=[100, 420])
    t_query.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_query)
    story.append(Spacer(1, 10))

    # =========================================================================
    # Section 2: Sensor Ingestion & Metadata Audit
    # =========================================================================
    story.append(Paragraph("2. Sensor Ingestion & Spatial Metadata Audit", st["section"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E3E62"), spaceAfter=6))

    input_audit = query_response.execution_trace.input_audit if query_response.execution_trace else {}
    alignment = input_audit.get("spatial_alignment", {})
    co_reg = alignment.get("co_registered", False)
    target_crs = alignment.get("crs", "Local Pixel / EPSG:4326")
    filenames = input_audit.get("filenames", [])
    modalities = input_audit.get("modalities", [])
    formats = input_audit.get("formats", [])
    resolutions = alignment.get("spatial_resolutions_m", [])

    rows_meta = [
        [
            Paragraph("Filename", st["table_header"]),
            Paragraph("Modality", st["table_header"]),
            Paragraph("Format", st["table_header"]),
            Paragraph("Spatial GSD", st["table_header"]),
            Paragraph("CRS Alignment", st["table_header"])
        ]
    ]

    for i, fname in enumerate(filenames):
        mod = modalities[i] if i < len(modalities) else "Unknown"
        fmt = formats[i] if i < len(formats) else "TIF"
        res = f"{resolutions[i]:.1f}m" if i < len(resolutions) and resolutions[i] else "N/A"
        crs_str = target_crs or "Local"
        rows_meta.append([
            Paragraph(fname, st["table_cell"]),
            Paragraph(mod, st["table_cell"]),
            Paragraph(fmt, st["table_cell"]),
            Paragraph(res, st["table_cell"]),
            Paragraph(crs_str, st["table_cell"]),
        ])

    if len(rows_meta) == 1:
        rows_meta.append([Paragraph("No input imagery metadata logged.", st["table_cell"]), "", "", "", ""])

    t_inputs = Table(rows_meta, colWidths=[150, 140, 60, 70, 100])
    t_inputs.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3E62")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_inputs)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"<b>Spatial Co-Registration Verified:</b> {str(co_reg).upper()} | "
        f"<b>Coordinate System:</b> {target_crs or 'Local Pixel'}",
        st["body"]
    ))
    story.append(Spacer(1, 10))

    # =========================================================================
    # Section 3: Geospatial Detections & Vector Metrics
    # =========================================================================
    story.append(Paragraph("3. Geospatial Detections & Vector Layer Metrics", st["section"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E3E62"), spaceAfter=6))

    v_layers = query_response.vector_layers or []
    rows_vec = [
        [
            Paragraph("Layer Identifier", st["table_header"]),
            Paragraph("Feature Type", st["table_header"]),
            Paragraph("Polygons", st["table_header"]),
            Paragraph("Total Area (Hectares)", st["table_header"]),
            Paragraph("Change Dynamics", st["table_header"])
        ]
    ]

    for vl in v_layers:
        metrics = vl.metrics or {}
        area_ha = metrics.get("total_area_hectares", metrics.get("area_hectares", 0.0))
        chg_pct = metrics.get("change_percentage", None)
        chg_str = f"{chg_pct:.1f}% Δ" if chg_pct is not None else "Static"

        rows_vec.append([
            Paragraph(vl.layer_name, st["table_cell"]),
            Paragraph(vl.feature_type, st["table_cell"]),
            Paragraph(str(vl.feature_count), st["table_cell"]),
            Paragraph(f"{area_ha:.2f} ha", st["table_cell"]),
            Paragraph(chg_str, st["table_cell"]),
        ])

    if len(rows_vec) == 1:
        rows_vec.append([
            Paragraph("No vector layers generated (VQA / Scene Captioning execution).", st["table_cell"]),
            "", "", "", ""
        ])

    t_vec = Table(rows_vec, colWidths=[150, 90, 60, 120, 100])
    t_vec.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3E62")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_vec)
    story.append(Spacer(1, 10))

    # =========================================================================
    # Section 4: Visual Evidence & Explainability Verification (RS-XAI)
    # =========================================================================
    xai: Optional[XAIExplanation] = query_response.xai_explanation
    if xai:
        story.append(Paragraph("4. Visual Evidence & Explainability Verification (RS-XAI)", st["section"]))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E3E62"), spaceAfter=6))

        story.append(Paragraph(
            f"<b>Attribution Technique:</b> {xai.method} | "
            f"<b>Hardware Tier:</b> {xai.hardware_tier or 'Tactical Edge'} | "
            f"<b>Execution Latency:</b> {xai.runtime_ms or 0.0:.2f} ms",
            st["body"]
        ))
        story.append(Spacer(1, 6))

        # Heatmap Image Display
        if xai.heatmap_overlay_base64:
            try:
                raw_b64 = xai.heatmap_overlay_base64.split(",", 1)[-1]
                img_bytes = base64.b64decode(raw_b64)
                img_io = io.BytesIO(img_bytes)
                rl_img = RLImage(img_io, width=280, height=180)
                rl_img.hAlign = "CENTER"

                img_wrapper = Table([[rl_img]], colWidths=[520])
                img_wrapper.setStyle(TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0F172A")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(img_wrapper)
                story.append(Spacer(1, 4))
                story.append(Paragraph(
                    "<i>Figure 4.1: Spatially constrained token energy attribution overlay (EPSG:4326 georeferenced boundary aligned).</i>",
                    st["callout"]
                ))
                story.append(Spacer(1, 8))
            except Exception as e:
                story.append(Paragraph(f"Heatmap visualization rendering skipped: {e}", st["callout"]))

        # Modality Attribution Table (Shapley)
        if xai.modality_attribution:
            story.append(Paragraph("<b>4.1 Multi-Modal Cooperative Shapley Attribution:</b>", st["sub_section"]))
            rows_shapley = [
                [Paragraph("Input Modality", st["table_header"]), Paragraph("Attribution (%)", st["table_header"]), Paragraph("Mathematical Formulation & Interpretation", st["table_header"])]
            ]
            for mod_name, val in xai.modality_attribution.items():
                pct = val * 100.0
                rows_shapley.append([
                    Paragraph(mod_name.upper(), st["table_cell"]),
                    Paragraph(f"{pct:.1f}%", st["body_bold"]),
                    Paragraph(f"Model marginal reliance across coalitions for {mod_name} under defined value function.", st["table_cell"])
                ])
            t_shap = Table(rows_shapley, colWidths=[120, 100, 300])
            t_shap.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3E62")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(t_shap)
            story.append(Spacer(1, 6))

        # Physical Scattering Rationale
        if xai.physics_rationale:
            story.append(Paragraph("<b>4.2 Physics & Microwave Scattering Breakdown:</b>", st["sub_section"]))
            pr = xai.physics_rationale
            scat_dist = pr.get("scattering_distribution_pct", {})
            rows_phys = [
                [Paragraph("Scattering Mechanism", st["table_header"]), Paragraph("Distribution (%)", st["table_header"]), Paragraph("Physical Interpretation (Woodhouse 2006)", st["table_header"])]
            ]
            if scat_dist:
                rows_phys.append([
                    Paragraph("Surface / Specular", st["table_cell"]),
                    Paragraph(f"{scat_dist.get('surface_specular_pct', 0.0)}%", st["body_bold"]),
                    Paragraph("σ⁰ < -16 dB: Smooth calm water, specular radar reflection away from sensor.", st["table_cell"])
                ])
                rows_phys.append([
                    Paragraph("Volume Canopy", st["table_cell"]),
                    Paragraph(f"{scat_dist.get('volume_canopy_pct', 0.0)}%", st["body_bold"]),
                    Paragraph("-16 dB ≤ σ⁰ ≤ -6 dB: Diffuse vegetation canopy and agricultural crops.", st["table_cell"])
                ])
                rows_phys.append([
                    Paragraph("Double-Bounce Urban", st["table_cell"]),
                    Paragraph(f"{scat_dist.get('double_bounce_urban_pct', 0.0)}%", st["body_bold"]),
                    Paragraph("σ⁰ > -6 dB: Dihedral corner reflection off vertical built-up structures.", st["table_cell"])
                ])
                t_phys = Table(rows_phys, colWidths=[130, 90, 300])
                t_phys.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3E62")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]))
                story.append(t_phys)
                story.append(Spacer(1, 6))

        # Spectral Sensitivity
        if xai.spectral_sensitivity:
            story.append(Paragraph("<b>4.3 Analytical Spectral Sensitivity Derivatives:</b>", st["sub_section"]))
            rows_sens = [
                [Paragraph("Spectral Band", st["table_header"]), Paragraph("Gradient Sensitivity", st["table_header"]), Paragraph("Analytical Derivative Rationale", st["table_header"])]
            ]
            for band, score in xai.spectral_sensitivity.items():
                rows_sens.append([
                    Paragraph(band.replace("_", " ").title(), st["table_cell"]),
                    Paragraph(f"{score:.4f}", st["body_bold"]),
                    Paragraph("Closed-form partial derivative ∂(Index)/∂(Band) reflecting physical reflection/absorption.", st["table_cell"])
                ])
            t_sens = Table(rows_sens, colWidths=[130, 100, 290])
            t_sens.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3E62")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(t_sens)
            story.append(Spacer(1, 6))

        # Summary & Limitations
        story.append(Paragraph(f"<b>Summary:</b> {xai.summary}", st["body"]))
        if xai.limitations:
            lims = " • ".join(xai.limitations)
            story.append(Paragraph(f"<i><b>Scientific Caveats:</b> {lims}</i>", st["callout"]))
        story.append(Spacer(1, 10))

    # =========================================================================
    # Section 5: Auditable Execution Trace & Pipeline Steps
    # =========================================================================
    story.append(Paragraph("5. Auditable Execution Trace & Pipeline Steps", st["section"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E3E62"), spaceAfter=6))

    pipe_steps = query_response.execution_trace.orchestration.get("pipeline_steps", []) if query_response.execution_trace else []
    rows_steps = [
        [
            Paragraph("Step #", st["table_header"]),
            Paragraph("Tool / Specialist Engine", st["table_header"]),
            Paragraph("Duration (ms)", st["table_header"]),
            Paragraph("Status", st["table_header"])
        ]
    ]

    for step in pipe_steps:
        s_num = str(step.get("step_number", "-"))
        s_tool = step.get("tool_name", "SpecialistTool")
        s_dur = f"{step.get('duration_ms', 0.0):.2f} ms"
        s_stat = step.get("status", "SUCCESS")
        rows_steps.append([
            Paragraph(s_num, st["table_cell"]),
            Paragraph(s_tool, st["table_cell"]),
            Paragraph(s_dur, st["table_cell"]),
            Paragraph(s_stat, st["table_cell"]),
        ])

    if len(rows_steps) == 1:
        rows_steps.append([Paragraph("No orchestration steps logged.", st["table_cell"]), "", "", ""])

    t_steps = Table(rows_steps, colWidths=[50, 250, 110, 110])
    t_steps.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3E62")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_steps)
    story.append(Spacer(1, 14))

    # =========================================================================
    # Footer Notice
    # =========================================================================
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94A3B8"), spaceAfter=4))
    story.append(Paragraph(
        "Confidential & Sovereign Intelligence Dossier | SatQuery AI — ISRO SIH26167 "
        "| Generated automatically via Peter's Central Agentic Task Router.",
        st["callout"]
    ))

    # Build Document
    document.build(story)