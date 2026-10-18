"""
End-to-End Integration Test for EduPredict Student Identity & Prediction Record Management.
Simulates the complete lifecycle:
- DB initialization and restart persistence
- Student identification and prediction association
- Leading zeros preservation in roll numbers
- Historical tracking across multiple predictions
- PDF and CSV report downloads with disclaimers
- Batch CSV predictions with duplicate and error handling
- Safe record deletion
"""

import os
import io
import tempfile
import unittest
import pandas as pd
from src.database import (
    init_db,
    save_prediction,
    get_all_predictions,
    get_latest_predictions_per_student,
    get_student_history,
    get_student,
    delete_prediction,
    get_database_stats
)
from src.prediction_engine import (
    predict_single_student,
    validate_student_identity,
    ModelArtifacts
)
from src.report_generator import (
    generate_student_pdf,
    generate_student_csv,
    generate_batch_csv,
    DISCLAIMER_TEXT
)
from src.batch_prediction import (
    read_csv_preserving_identity,
    process_batch_predictions,
    save_batch_records
)


class TestIntegrationWorkflow(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "integration_test.db")
        init_db(self.db_path)
        self.artifacts = ModelArtifacts.get_instance()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_student_prediction_lifecycle(self):
        # 1. Validation check
        valid, err = validate_student_identity("", "0042")
        self.assertFalse(valid)
        valid, err = validate_student_identity("Harry Potter", "")
        self.assertFalse(valid)
        valid, err = validate_student_identity("Harry Potter", "0007")
        self.assertTrue(valid)

        # 2. First Prediction for Harry Potter (Term 1)
        term1_inputs = {
            "Hours_Studied": 10,
            "Attendance": 65,
            "Sleep_Hours": 6,
            "Previous_Scores": 55,
            "Tutoring_Sessions": 1,
            "Physical_Activity": 2,
            "Internet_Access": "Yes",
            "Parental_Involvement": "Medium",
            "Access_to_Resources": "Medium",
            "Motivation_Level": "Medium"
        }
        res1 = predict_single_student(
            student_name="Harry Potter",
            roll_number="0007",
            academic_inputs=term1_inputs,
            artifacts=self.artifacts
        )
        res1["prediction_date"] = "2026-03-01 10:00:00"
        id1 = save_prediction(res1, db_path=self.db_path)
        res1["id"] = id1

        # Check DB state
        student = get_student("0007", db_path=self.db_path)
        self.assertEqual(student["full_name"], "Harry Potter")
        self.assertEqual(student["roll_number"], "0007")

        # 3. Second Prediction for Harry Potter (Term 2 - Improved Habits)
        term2_inputs = {
            "Hours_Studied": 25,
            "Attendance": 92,
            "Sleep_Hours": 8,
            "Previous_Scores": 85,
            "Tutoring_Sessions": 3,
            "Physical_Activity": 4,
            "Internet_Access": "Yes",
            "Parental_Involvement": "High",
            "Access_to_Resources": "High",
            "Motivation_Level": "High"
        }
        res2 = predict_single_student(
            student_name="Harry Potter",
            roll_number="0007",
            academic_inputs=term2_inputs,
            artifacts=self.artifacts
        )
        res2["prediction_date"] = "2026-06-01 10:00:00"
        id2 = save_prediction(res2, db_path=self.db_path)
        res2["id"] = id2

        self.assertNotEqual(id1, id2)

        # Verify historical preservation: both predictions exist
        history = get_student_history("0007", db_path=self.db_path)
        self.assertEqual(len(history), 2)
        # Term 2 should be newest
        self.assertEqual(history[0]["prediction_date"], "2026-06-01 10:00:00")
        self.assertGreater(history[0]["performance_score"], history[1]["performance_score"])

        # Latest predictions per student should return only Term 2
        latest = get_latest_predictions_per_student(db_path=self.db_path)
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0]["id"], id2)

        # 4. Report Generation
        pdf_bytes = generate_student_pdf(res2)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))
        self.assertGreater(len(pdf_bytes), 2000)

        csv_text = generate_student_csv(res2)
        self.assertIn("Harry Potter", csv_text)
        self.assertIn("0007", csv_text)
        self.assertIn(DISCLAIMER_TEXT, csv_text)

        # 5. Batch Predictions
        batch_csv_content = """Roll_Number,Student_Name,Hours_Studied,Attendance,Sleep_Hours,Previous_Scores,Tutoring_Sessions,Physical_Activity,Internet_Access,Parental_Involvement,Access_to_Resources,Motivation_Level
0008,Ron Weasley,14,75,7,60,1,3,Yes,Medium,Medium,Medium
0009,Hermione Granger,35,98,7,95,0,3,Yes,High,High,High
0008,Ron Weasley (Retest),16,80,7,65,2,3,Yes,Medium,Medium,Medium
"""
        batch_df = read_csv_preserving_identity(io.BytesIO(batch_csv_content.encode("utf-8")))
        results_df, summary, batch_records = process_batch_predictions(batch_df, artifacts=self.artifacts)

        self.assertEqual(summary["valid_count"], 3)
        self.assertIn("0008", summary["duplicate_roll_numbers"])

        # Save batch to DB
        saved_count = save_batch_records(batch_records, db_path=self.db_path)
        self.assertEqual(saved_count, 3)

        # Verify total database stats
        stats = get_database_stats(db_path=self.db_path)
        # Students: Harry (0007), Ron (0008), Hermione (0009) = 3 students
        self.assertEqual(stats["total_students"], 3)
        # Predictions: 2 (Harry) + 3 (Batch) = 5 predictions
        self.assertEqual(stats["total_predictions"], 5)

        # 6. Persistence across restart simulation
        # Re-initialize DB using the same file path and test queries
        init_db(self.db_path)
        restarted_stats = get_database_stats(db_path=self.db_path)
        self.assertEqual(restarted_stats["total_students"], 3)
        self.assertEqual(restarted_stats["total_predictions"], 5)

        # 7. Safe Deletion of single record
        del_success = delete_prediction(id1, db_path=self.db_path)
        self.assertTrue(del_success)
        remaining_history = get_student_history("0007", db_path=self.db_path)
        self.assertEqual(len(remaining_history), 1)
        self.assertEqual(remaining_history[0]["id"], id2)


if __name__ == "__main__":
    unittest.main()
