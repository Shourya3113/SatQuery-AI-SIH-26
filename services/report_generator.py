"""
SatQuery AI - Intelligence Dossier Generator
Owner: Achintya (Backend Lead)

Generates a PDF report from a SatQuery AI query result.
"""

from html import escape

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors


MODEL_NAMES = {
    "vqa_engine": "Qwen2-VL-2B-Instruct",
    "grounding_engine": "Grounding DINO Tiny + SAM ViT-Base",
    "change_engine": "BIT Change Detection",
    "fusion_engine": "Optical-SAR Physics-Based Fusion",
}


def generate_report(query_response, output_path):
    """
    Generate an Intelligence Dossier PDF from a QueryResponse.
    """

    document = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    story = []

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    story.append(
        Paragraph(
            "SATQUERY AI - INTELLIGENCE DOSSIER",
            title_style
        )
    )

    story.append(Spacer(1, 20))

    # ---------------------------------------------------------
    # Query
    # ---------------------------------------------------------

    story.append(
        Paragraph("<b>Query</b>", styles["Heading2"])
    )

    story.append(
        Paragraph(
            escape(str(query_response.query)),
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 12))

    # ---------------------------------------------------------
    # Task Category
    # ---------------------------------------------------------

    story.append(
        Paragraph("<b>Task Category</b>", styles["Heading2"])
    )

    story.append(
        Paragraph(
            escape(str(query_response.task_category)),
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 12))

    # ---------------------------------------------------------
    # Analysis Result
    # ---------------------------------------------------------

    story.append(
        Paragraph("<b>Analysis Result</b>", styles["Heading2"])
    )

    story.append(
        Paragraph(
            escape(str(query_response.text_response)),
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 12))

    # ---------------------------------------------------------
    # Confidence
    # ---------------------------------------------------------

    story.append(
        Paragraph("<b>Confidence Score</b>", styles["Heading2"])
    )

    confidence = float(query_response.confidence_score)

    story.append(
        Paragraph(
            f"{confidence:.4f} ({confidence * 100:.2f}%)",
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 12))

    # ---------------------------------------------------------
    # Execution / Models
    # ---------------------------------------------------------

    story.append(
        Paragraph("<b>Execution Details</b>", styles["Heading2"])
    )

    pipeline_steps = (
        query_response.execution_trace
        .orchestration
        .get("pipeline_steps", [])
    )

    execution_rows = [
        ["Tool", "Model", "Status", "Duration (ms)"]
    ]

    for step in pipeline_steps:
        tool_name = step.get("tool_name", "Unknown")
        model_name = MODEL_NAMES.get(tool_name, tool_name)

        execution_rows.append(
            [
                tool_name,
                model_name,
                step.get("status", "UNKNOWN"),
                str(step.get("duration_ms", "-")),
            ]
        )

    if len(execution_rows) > 1:
        table = Table(
            execution_rows,
            colWidths=[105, 180, 70, 75],
            repeatRows=1,
        )

        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        story.append(table)

    story.append(Spacer(1, 12))

    # ---------------------------------------------------------
    # Overall execution latency
    # ---------------------------------------------------------

    execution_duration = (
        query_response.execution_trace
        .results
        .get("execution_duration_ms")
    )

    if execution_duration is not None:
        story.append(
            Paragraph(
                f"<b>Total Execution Time:</b> "
                f"{execution_duration:.2f} ms",
                styles["BodyText"]
            )
        )

        story.append(Spacer(1, 12))

    # ---------------------------------------------------------
    # Trace ID
    # ---------------------------------------------------------

    story.append(
        Paragraph("<b>Trace ID</b>", styles["Heading2"])
    )

    story.append(
        Paragraph(
            escape(str(query_response.trace_id)),
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 20))

    # ---------------------------------------------------------
    # Benchmark / Reference Information
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "<b>Benchmark & Model References</b>",
            styles["Heading2"]
        )
    )

    references = [
        "Qwen2-VL-2B-Instruct — Visual Question Answering / image understanding.",
        "Grounding DINO Tiny — text-guided object grounding.",
        "SAM ViT-Base — image segmentation.",
        "BIT — bi-temporal remote sensing change detection.",
        "CLIP ViT-B/16 + LoRA — remote sensing representation adaptation.",
    ]

    for reference in references:
        story.append(
            Paragraph(
                f"• {escape(reference)}",
                styles["BodyText"]
            )
        )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Generated by SatQuery AI",
            styles["BodyText"]
        )
    )

    document.build(story)