"""
Unit tests for EduPredict batch prediction module.
Verifies CSV ingestion, preservation of leading zeros, column detection,
duplicate detection, and ML feature isolation.
"""

import io
import os
import tempfile
import unittest
import pandas as pd
from src.batch_prediction import (
    read_csv_preserving_identity,
    process_batch_predictions,
    save_batch_records
)
from src.database import init_db, get_student_history, get_database_stats


class TestBatchPrediction(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "batch_test.db")
        init_db(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_read_csv_preserves_leading_zeros(self):
        csv_data = """Roll_Number,Student_Name,Hours_Studied,Attendance,Sleep_Hours,Previous_Scores,Tutoring_Sessions,Physical_Activity,Internet_Access,Parental_Involvement,Access_to_Resources,Motivation_Level
0012,John Doe,15,80,7,65,2,3,Yes,Medium,Medium,Medium
0045,Jane Smith,22,90,8,85,1,4,Yes,High,High,High
"""
        buffer = io.BytesIO(csv_data.encode("utf-8"))
        df = read_csv_preserving_identity(buffer)

        self.assertEqual(df["Roll_Number"].iloc[0], "0012")
        self.assertEqual(df["Roll_Number"].iloc[1], "0045")
        self.assertEqual(df["Student_Name"].iloc[0], "John Doe")

    def test_process_batch_predictions_success_and_feature_isolation(self):
        csv_data = """Roll_Number,Student_Name,Hours_Studied,Attendance,Sleep_Hours,Previous_Scores,Tutoring_Sessions,Physical_Activity,Internet_Access,Parental_Involvement,Access_to_Resources,Motivation_Level
0001,Clark Kent,25,95,8,90,1,5,Yes,High,High,High
0002,Barry Allen,12,70,6,58,2,2,Yes,Medium,Medium,Low
"""
        df = read_csv_preserving_identity(io.BytesIO(csv_data.encode("utf-8")))
        results_df, summary, records = process_batch_predictions(df)

        self.assertEqual(summary["total_rows"], 2)
        self.assertEqual(summary["valid_count"], 2)
        self.assertEqual(summary["invalid_count"], 0)
        self.assertEqual(len(results_df), 2)
        self.assertEqual(results_df.iloc[0]["Roll Number"], "0001")
        self.assertEqual(results_df.iloc[0]["Student Name"], "Clark Kent")
        self.assertEqual(results_df.iloc[1]["Roll Number"], "0002")
        self.assertEqual(results_df.iloc[1]["Student Name"], "Barry Allen")

        # Save to database
        saved = save_batch_records(records, db_path=self.db_path)
        self.assertEqual(saved, 2)

        stats = get_database_stats(db_path=self.db_path)
        self.assertEqual(stats["total_students"], 2)
        self.assertEqual(stats["total_predictions"], 2)

    def test_batch_prediction_reports_duplicates_and_missing(self):
        csv_data = """Roll_Number,Student_Name,Hours_Studied,Attendance,Sleep_Hours,Previous_Scores,Tutoring_Sessions,Physical_Activity,Internet_Access,Parental_Involvement,Access_to_Resources,Motivation_Level
0099,Diana Prince,20,85,7,75,2,3,Yes,High,Medium,High
,No Roll Student,15,80,7,65,2,3,Yes,Medium,Medium,Medium
0099,Duplicate Diana,20,85,7,75,2,3,Yes,High,Medium,High
"""
        df = read_csv_preserving_identity(io.BytesIO(csv_data.encode("utf-8")))
        results_df, summary, records = process_batch_predictions(df)

        self.assertIn("0099", summary["duplicate_roll_numbers"])
        # Row 2 is missing roll number
        self.assertGreaterEqual(summary["invalid_count"], 1)
        self.assertTrue(any("Roll Number is missing" in err["reasons"] for err in summary["errors"]))


if __name__ == "__main__":
    unittest.main()
