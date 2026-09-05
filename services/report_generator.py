"""
SatQuery AI - Intelligence Dossier Generator
Owner: Achintya (Backend Lead)

Generates a PDF report from a SatQuery AI query result.
"""

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER


def generate_report(query_response, output_path):
    """
    Generate an Intelligence Dossier PDF from a QueryResponse.
    """

    # Create the PDF document
    document = SimpleDocTemplate(
        output_path,
        pagesize=A4
    )

    # Load default ReportLab styles
    styles = getSampleStyleSheet()

    # Content that will be placed inside the PDF
    story = []

    # Title
    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    story.append(
        Paragraph("SATQUERY AI - INTELLIGENCE DOSSIER", title_style)
    )

    story.append(Spacer(1, 20))

    # Query information
    story.append(
        Paragraph("<b>Query</b>", styles["Heading2"])
    )

    story.append(
        Paragraph(query_response.query, styles["BodyText"])
    )

    story.append(Spacer(1, 12))

    # Task category
    story.append(
        Paragraph("<b>Task Category</b>", styles["Heading2"])
    )

    story.append(
        Paragraph(
            str(query_response.task_category),
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 12))

    # AI response
    story.append(
        Paragraph("<b>Analysis Result</b>", styles["Heading2"])
    )

    story.append(
        Paragraph(
            query_response.text_response,
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 12))

    # Confidence
    story.append(
        Paragraph("<b>Confidence Score</b>", styles["Heading2"])
    )

    story.append(
        Paragraph(
            str(query_response.confidence_score),
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 12))

    # Trace ID
    story.append(
        Paragraph("<b>Trace ID</b>", styles["Heading2"])
    )

    story.append(
        Paragraph(
            query_response.trace_id,
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 20))

    # Build the PDF
    document.build(story)