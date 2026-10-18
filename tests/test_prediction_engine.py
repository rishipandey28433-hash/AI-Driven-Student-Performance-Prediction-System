"""
Unit tests for EduPredict prediction engine.
Verifies student identity validation, feature isolation,
risk scoring, suggestion generation, and ML model inference.
"""

import unittest
from unittest.mock import patch
from src.prediction_engine import (
    validate_student_identity,
    calculate_risk,
    generate_suggestions,
    calculate_performance_score,
    predict_single_student,
    MODEL_FEATURE_NAMES,
    ModelArtifacts
)


class TestPredictionEngine(unittest.TestCase):

    def test_validate_student_identity(self):
        # Empty name
        valid, msg = validate_student_identity("", "001")
        self.assertFalse(valid)
        self.assertIn("name", msg.lower())

        # Whitespace name
        valid, msg = validate_student_identity("   ", "001")
        self.assertFalse(valid)

        # None name
        valid, msg = validate_student_identity(None, "001")
        self.assertFalse(valid)

        # Empty roll number
        valid, msg = validate_student_identity("Alice", "")
        self.assertFalse(valid)
        self.assertIn("roll number", msg.lower())

        # Whitespace roll number
        valid, msg = validate_student_identity("Alice", "   ")
        self.assertFalse(valid)

        # Preserved leading zeros
        valid, msg = validate_student_identity("Alice Smith", "00042")
        self.assertTrue(valid)
        self.assertEqual(msg, "")

    def test_calculate_risk(self):
        # High risk
        risk_level, score, reasons = calculate_risk(
            hours_studied=5,
            attendance=45,
            sleep_hours=4,
            previous_scores=40,
            motivation_level="Low",
            access_to_resources="Low"
        )
        self.assertEqual(risk_level, "High Risk")
        self.assertGreaterEqual(score, 5)
        self.assertIn("Low attendance (<60%)", reasons)

        # Low risk
        risk_level_low, score_low, reasons_low = calculate_risk(
            hours_studied=25,
            attendance=95,
            sleep_hours=8,
            previous_scores=85,
            motivation_level="High",
            access_to_resources="High"
        )
        self.assertEqual(risk_level_low, "Low Risk")
        self.assertEqual(score_low, 0)
        self.assertIn("No major risk factors detected.", reasons_low)

    def test_generate_suggestions(self):
        # Suggestions for low attendance and study hours
        suggs = generate_suggestions(
            attendance=50,
            hours_studied=8,
            previous_scores=55,
            motivation_level="Low",
            access_to_resources="Low",
            sleep_hours=5
        )
        self.assertGreaterEqual(len(suggs), 3)

        # Suggestions for healthy student
        good_suggs = generate_suggestions(
            attendance=90,
            hours_studied=25,
            previous_scores=85,
            motivation_level="High",
            access_to_resources="High",
            sleep_hours=8
        )
        self.assertEqual(len(good_suggs), 1)
        self.assertIn("Maintain your consistent study habits", good_suggs[0])

    def test_calculate_performance_score_bounds(self):
        # Max score clamp
        score_high = calculate_performance_score(100, 40, 100, 0)
        self.assertLessEqual(score_high, 100)

        # Min score clamp
        score_low = calculate_performance_score(0, 0, 0, 10)
        self.assertGreaterEqual(score_low, 0)

    def test_predict_single_student_end_to_end(self):
        academic_inputs = {
            "Hours_Studied": 18,
            "Attendance": 82,
            "Sleep_Hours": 7,
            "Previous_Scores": 72,
            "Tutoring_Sessions": 1,
            "Physical_Activity": 4,
            "Internet_Access": "Yes",
            "Parental_Involvement": "Medium",
            "Access_to_Resources": "Medium",
            "Motivation_Level": "Medium"
        }

        result = predict_single_student(
            student_name="Bruce Wayne",
            roll_number="0070",
            academic_inputs=academic_inputs
        )

        self.assertEqual(result["student_name"], "Bruce Wayne")
        self.assertEqual(result["roll_number"], "0070")
        self.assertIn(result["predicted_category"], ["Excellent", "Good", "Average", "Poor"])
        self.assertIn(result["risk_level"], ["Low Risk", "Medium Risk", "High Risk"])
        self.assertGreater(result["confidence"], 0.0)
        self.assertIsInstance(result["class_probabilities"], dict)
        self.assertGreater(len(result["suggestions"]), 0)

    def test_student_identity_never_passed_to_ml_model(self):
        # Verify that the ML model feature inputs NEVER include roll_number or student_name
        artifacts = ModelArtifacts.get_instance()
        self.assertNotIn("student_name", MODEL_FEATURE_NAMES)
        self.assertNotIn("roll_number", MODEL_FEATURE_NAMES)
        self.assertNotIn("name", MODEL_FEATURE_NAMES)
        self.assertNotIn("roll", MODEL_FEATURE_NAMES)
        self.assertEqual(len(MODEL_FEATURE_NAMES), 10)

    def test_below_average_rule_for_low_hours_and_attendance(self):
        # Test 1: hours = 2, attendance = 9
        res1 = predict_single_student(
            student_name="Test Student",
            roll_number="001",
            academic_inputs={"Hours_Studied": 2, "Attendance": 9}
        )
        self.assertEqual(res1["predicted_category"], "Below Average")
        self.assertEqual(res1["risk_level"], "High Risk")

        # Test 2: hours = 1, attendance = 5
        res2 = predict_single_student(
            student_name="Test Student 2",
            roll_number="002",
            academic_inputs={"Hours_Studied": 1, "Attendance": 5}
        )
        self.assertEqual(res2["predicted_category"], "Below Average")
        self.assertEqual(res2["risk_level"], "High Risk")

        # Test 3: hours = 2, attendance = 10
        res3 = predict_single_student(
            student_name="Test Student 3",
            roll_number="003",
            academic_inputs={"Hours_Studied": 2, "Attendance": 10}
        )
        self.assertEqual(res3["predicted_category"], "Below Average")
        self.assertEqual(res3["risk_level"], "High Risk")

        # Test 4: User screenshot case (hours=5, attendance=9, previous_scores=2, sleep=2)
        res_screenshot = predict_single_student(
            student_name="Screenshot Student",
            roll_number="005",
            academic_inputs={"Hours_Studied": 5, "Attendance": 9, "Previous_Scores": 2, "Sleep_Hours": 2}
        )
        self.assertEqual(res_screenshot["predicted_category"], "Below Average")
        self.assertEqual(res_screenshot["risk_level"], "High Risk")

        # Test 5: Normal student should not be affected
        res4 = predict_single_student(
            student_name="Good Student",
            roll_number="004",
            academic_inputs={"Hours_Studied": 20, "Attendance": 85}
        )
        self.assertIn(res4["predicted_category"], ["Good", "Excellent", "Average"])
        self.assertNotEqual(res4["predicted_category"], "Below Average")


if __name__ == "__main__":
    unittest.main()
