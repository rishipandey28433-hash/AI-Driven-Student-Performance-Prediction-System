"""
Unit tests for EduPredict report generator.
Verifies PDF generation, CSV exports, student identity inclusion,
and educational disclaimer compliance.
"""

import unittest
from src.report_generator import (
    generate_student_pdf,
    generate_student_csv,
    generate_batch_csv,
    DISCLAIMER_TEXT
)


class TestReportGenerator(unittest.TestCase):

    def setUp(self):
        self.sample_record = {
            "id": 1,
            "student_name": "Tony Stark",
            "roll_number": "0077",
            "prediction_date": "2026-10-18 14:00:00",
            "predicted_category": "Excellent",
            "risk_level": "Low Risk",
            "confidence": 96.5,
            "performance_score": 94,
            "academic_inputs": {
                "Hours_Studied": 28,
                "Attendance": 92,
                "Sleep_Hours": 7,
                "Previous_Scores": 88,
                "Tutoring_Sessions": 0,
                "Physical_Activity": 5,
                "Internet_Access": "Yes",
                "Parental_Involvement": "High",
                "Access_to_Resources": "High",
                "Motivation_Level": "High"
            },
            "risk_reasons": ["No major risk factors detected."],
            "suggestions": ["Maintain consistent study habits."]
        }

    def test_generate_student_pdf_valid(self):
        pdf_bytes = generate_student_pdf(self.sample_record)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1500)
        # PDF magic header
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))

    def test_generate_student_csv_contains_identity_and_disclaimer(self):
        csv_str = generate_student_csv(self.sample_record)
        self.assertIn("Tony Stark", csv_str)
        self.assertIn("0077", csv_str)
        self.assertIn("Excellent", csv_str)
        self.assertIn("Low Risk", csv_str)
        self.assertIn(DISCLAIMER_TEXT, csv_str)

    def test_generate_batch_csv_contains_all_records(self):
        records = [
            self.sample_record,
            {
                "id": 2,
                "student_name": "Peter Parker",
                "roll_number": "0088",
                "prediction_date": "2026-10-18 14:05:00",
                "predicted_category": "Good",
                "risk_level": "Low Risk",
                "confidence": 85.0,
                "performance_score": 75,
                "academic_inputs": {"Hours_Studied": 15, "Attendance": 80},
                "risk_reasons": [],
                "suggestions": []
            }
        ]
        csv_str = generate_batch_csv(records)
        self.assertIn("Tony Stark", csv_str)
        self.assertIn("0077", csv_str)
        self.assertIn("Peter Parker", csv_str)
        self.assertIn("0088", csv_str)
        self.assertIn(DISCLAIMER_TEXT, csv_str)


if __name__ == "__main__":
    unittest.main()
