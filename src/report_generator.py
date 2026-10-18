"""
Report generation module for EduPredict.
Generates professional PDF and CSV reports with student identity,
academic metrics, risk factors, suggestions, and mandatory educational disclaimers.
"""

import io
from typing import Dict, Any, List
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable
)

DISCLAIMER_TEXT = (
    "Disclaimer: EduPredict predictions, risk assessments, and guidance suggestions "
    "are algorithmic estimates generated for early warning and academic advising purposes only. "
    "They do not constitute guaranteed academic outcomes or official academic grading."
)


def generate_student_pdf(record: Dict[str, Any]) -> bytes:
    """
    Generates a professional, beautifully styled PDF performance report
    for an individual student prediction record. Returns raw PDF bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=12
    )
    section_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1F2937"),
        spaceBefore=10,
        spaceAfter=6
    )
    cell_bold = ParagraphStyle(
        "CellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#111827")
    )
    cell_normal = ParagraphStyle(
        "CellNormal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#374151")
    )
    bullet_style = ParagraphStyle(
        "BulletText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1F2937"),
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=4
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#6B7280"),
        alignment=1
    )

    story = []

    # Title & Branding
    story.append(Paragraph("🎓 EduPredict – Student Academic Performance Report", title_style))
    story.append(Paragraph("AI-Driven Student Performance Prediction and Early Warning System", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=14))

    # Student Identity Card
    student_name = record.get("student_name", "N/A")
    roll_number = str(record.get("roll_number", "N/A"))
    pred_date = record.get("prediction_date", "N/A")
    rec_id = str(record.get("id", "Pending / New"))

    identity_data = [
        [
            Paragraph("<b>Student Full Name:</b>", cell_bold),
            Paragraph(student_name, cell_normal),
            Paragraph("<b>Roll Number:</b>", cell_bold),
            Paragraph(roll_number, cell_bold)
        ],
        [
            Paragraph("<b>Assessment Date:</b>", cell_bold),
            Paragraph(pred_date, cell_normal),
            Paragraph("<b>Record ID:</b>", cell_bold),
            Paragraph(rec_id, cell_normal)
        ]
    ]

    t_identity = Table(identity_data, colWidths=[120, 150, 100, 160])
    t_identity.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F3F4F6")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#D1D5DB")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_identity)
    story.append(Spacer(1, 12))

    # Prediction Overview Summary
    story.append(Paragraph("📊 Prediction & Risk Summary", section_style))

    category = record.get("predicted_category", "N/A")
    risk_level = record.get("risk_level", "N/A")
    score = record.get("performance_score", 0)
    confidence = record.get("confidence", 0.0)

    # Color badge based on category
    cat_bg = colors.HexColor("#DCFCE7") if category in ["Excellent", "Good"] else colors.HexColor("#FEF3C7")
    risk_bg = colors.HexColor("#FEE2E2") if risk_level == "High Risk" else (
        colors.HexColor("#FEF3C7") if risk_level == "Medium Risk" else colors.HexColor("#DCFCE7")
    )

    summary_data = [
        [
            Paragraph("<b>Predicted Performance</b>", cell_bold),
            Paragraph("<b>Risk Level</b>", cell_bold),
            Paragraph("<b>Performance Index</b>", cell_bold),
            Paragraph("<b>Model Confidence</b>", cell_bold)
        ],
        [
            Paragraph(f"<b>{category}</b>", ParagraphStyle("C", parent=cell_bold, fontSize=11, textColor=colors.HexColor("#15803D") if category in ["Excellent", "Good"] else colors.HexColor("#B45309"))),
            Paragraph(f"<b>{risk_level}</b>", ParagraphStyle("R", parent=cell_bold, fontSize=11, textColor=colors.HexColor("#B91C1C") if risk_level == "High Risk" else colors.HexColor("#15803D"))),
            Paragraph(f"<b>{score} / 100</b>", cell_bold),
            Paragraph(f"<b>{confidence}%</b>", cell_bold)
        ]
    ]

    t_summary = Table(summary_data, colWidths=[135, 135, 130, 130])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ('BACKGROUND', (0, 1), (0, 1), cat_bg),
        ('BACKGROUND', (1, 1), (1, 1), risk_bg),
        ('BACKGROUND', (2, 1), (-1, 1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 12))

    # Academic Input Summary
    story.append(Paragraph("📘 Academic Profile & Input Factors", section_style))
    inputs = record.get("academic_inputs", {})
    input_table_data = [
        [
            Paragraph("<b>Factor</b>", cell_bold),
            Paragraph("<b>Recorded Value</b>", cell_bold),
            Paragraph("<b>Factor</b>", cell_bold),
            Paragraph("<b>Recorded Value</b>", cell_bold)
        ],
        [
            Paragraph("Hours Studied (weekly)", cell_normal),
            Paragraph(f"{inputs.get('Hours_Studied', 'N/A')} hrs", cell_normal),
            Paragraph("Physical Activity", cell_normal),
            Paragraph(f"{inputs.get('Physical_Activity', 'N/A')} hrs/wk", cell_normal)
        ],
        [
            Paragraph("Attendance", cell_normal),
            Paragraph(f"{inputs.get('Attendance', 'N/A')}%", cell_normal),
            Paragraph("Internet Access", cell_normal),
            Paragraph(str(inputs.get("Internet_Access", "N/A")), cell_normal)
        ],
        [
            Paragraph("Sleep Hours (daily)", cell_normal),
            Paragraph(f"{inputs.get('Sleep_Hours', 'N/A')} hrs", cell_normal),
            Paragraph("Parental Involvement", cell_normal),
            Paragraph(str(inputs.get("Parental_Involvement", "N/A")), cell_normal)
        ],
        [
            Paragraph("Previous Exam Scores", cell_normal),
            Paragraph(f"{inputs.get('Previous_Scores', 'N/A')}%", cell_normal),
            Paragraph("Access to Resources", cell_normal),
            Paragraph(str(inputs.get("Access_to_Resources", "N/A")), cell_normal)
        ],
        [
            Paragraph("Tutoring Sessions", cell_normal),
            Paragraph(str(inputs.get("Tutoring_Sessions", "N/A")), cell_normal),
            Paragraph("Motivation Level", cell_normal),
            Paragraph(str(inputs.get("Motivation_Level", "N/A")), cell_normal)
        ]
    ]

    t_inputs = Table(input_table_data, colWidths=[150, 115, 150, 115])
    t_inputs.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_inputs)
    story.append(Spacer(1, 12))

    # Identified Risk Drivers
    story.append(Paragraph("⚠ Identified Risk Factors & Early Warnings", section_style))
    reasons = record.get("risk_reasons", [])
    if isinstance(reasons, str):
        reasons = [reasons]
    for r in reasons:
        story.append(Paragraph(f"• {r}", bullet_style))

    story.append(Spacer(1, 10))

    # Actionable Suggestions
    story.append(Paragraph("💡 Recommended Academic Action Plan", section_style))
    suggestions = record.get("suggestions", [])
    if isinstance(suggestions, str):
        suggestions = [suggestions]
    for s in suggestions:
        story.append(Paragraph(f"• {s}", bullet_style))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#D1D5DB"), spaceAfter=10))

    # Mandatory Educational Disclaimer
    story.append(Paragraph(DISCLAIMER_TEXT, disclaimer_style))

    doc.build(story)
    return buffer.getvalue()


def generate_student_csv(record: Dict[str, Any]) -> str:
    """Generates a clean CSV export for an individual student record."""
    inputs = record.get("academic_inputs", {})
    reasons = "; ".join(record.get("risk_reasons", [])) if isinstance(record.get("risk_reasons"), list) else str(record.get("risk_reasons", ""))
    suggestions = "; ".join(record.get("suggestions", [])) if isinstance(record.get("suggestions"), list) else str(record.get("suggestions", ""))

    data = [{
        "Record ID": record.get("id", "N/A"),
        "Student Name": record.get("student_name", ""),
        "Roll Number": str(record.get("roll_number", "")),
        "Prediction Date": record.get("prediction_date", ""),
        "Predicted Performance": record.get("predicted_category", ""),
        "Risk Level": record.get("risk_level", ""),
        "Performance Score": record.get("performance_score", 0),
        "Confidence (%)": record.get("confidence", 0.0),
        "Hours Studied": inputs.get("Hours_Studied", ""),
        "Attendance (%)": inputs.get("Attendance", ""),
        "Sleep Hours": inputs.get("Sleep_Hours", ""),
        "Previous Scores": inputs.get("Previous_Scores", ""),
        "Tutoring Sessions": inputs.get("Tutoring_Sessions", ""),
        "Physical Activity": inputs.get("Physical_Activity", ""),
        "Internet Access": inputs.get("Internet_Access", ""),
        "Parental Involvement": inputs.get("Parental_Involvement", ""),
        "Access to Resources": inputs.get("Access_to_Resources", ""),
        "Motivation Level": inputs.get("Motivation_Level", ""),
        "Risk Factors": reasons,
        "Suggestions": suggestions,
        "Disclaimer": DISCLAIMER_TEXT
    }]

    df = pd.DataFrame(data)
    return df.to_csv(index=False)


def generate_batch_csv(records: List[Dict[str, Any]]) -> str:
    """Generates a consolidated CSV string for multiple student records."""
    if not records:
        return ""

    rows = []
    for r in records:
        inputs = r.get("academic_inputs", {})
        reasons = "; ".join(r.get("risk_reasons", [])) if isinstance(r.get("risk_reasons"), list) else str(r.get("risk_reasons", ""))
        suggestions = "; ".join(r.get("suggestions", [])) if isinstance(r.get("suggestions"), list) else str(r.get("suggestions", ""))

        rows.append({
            "Record ID": r.get("id", ""),
            "Student Name": r.get("student_name", ""),
            "Roll Number": str(r.get("roll_number", "")),
            "Prediction Date": r.get("prediction_date", ""),
            "Predicted Performance": r.get("predicted_category", ""),
            "Risk Level": r.get("risk_level", ""),
            "Performance Score": r.get("performance_score", 0),
            "Confidence (%)": r.get("confidence", 0.0),
            "Hours Studied": inputs.get("Hours_Studied", ""),
            "Attendance (%)": inputs.get("Attendance", ""),
            "Sleep Hours": inputs.get("Sleep_Hours", ""),
            "Previous Scores": inputs.get("Previous_Scores", ""),
            "Tutoring Sessions": inputs.get("Tutoring_Sessions", ""),
            "Physical Activity": inputs.get("Physical_Activity", ""),
            "Internet Access": inputs.get("Internet_Access", ""),
            "Parental Involvement": inputs.get("Parental_Involvement", ""),
            "Access to Resources": inputs.get("Access_to_Resources", ""),
            "Motivation Level": inputs.get("Motivation_Level", ""),
            "Risk Factors": reasons,
            "Suggestions": suggestions,
            "Disclaimer": DISCLAIMER_TEXT
        })

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)
