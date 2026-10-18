"""
Unit tests for EduPredict persistent database module.
Verifies table initialization, student upsert, leading zeros preservation,
multiple historical predictions, parameterized search/filter, and record deletion.
"""

import os
import tempfile
import unittest
import sqlite3
from src.database import (
    init_db,
    upsert_student,
    get_student,
    save_prediction,
    get_all_predictions,
    get_latest_predictions_per_student,
    get_student_history,
    get_prediction_by_id,
    delete_prediction,
    delete_student,
    get_database_stats,
    predictions_to_dataframe
)


class TestDatabaseModule(unittest.TestCase):

    def setUp(self):
        # Create a temporary database for isolation
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_edupredict.db")
        init_db(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_db_creates_tables_and_indexes(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        self.assertIn("students", tables)
        self.assertIn("predictions", tables)
        conn.close()

    def test_student_upsert_and_leading_zeros(self):
        # Roll numbers with leading zeros must be preserved as strings
        roll = "00042"
        name = "Alex Mercer"
        upsert_student(roll, name, db_path=self.db_path)

        student = get_student(roll, db_path=self.db_path)
        self.assertIsNotNone(student)
        self.assertEqual(student["roll_number"], "00042")
        self.assertEqual(student["full_name"], "Alex Mercer")

        # Update name
        upsert_student(roll, "Alex J. Mercer", db_path=self.db_path)
        updated = get_student(roll, db_path=self.db_path)
        self.assertEqual(updated["full_name"], "Alex J. Mercer")
        self.assertEqual(updated["roll_number"], "00042")

    def test_save_prediction_associates_with_student(self):
        rec = {
            "roll_number": "007",
            "student_name": "James Bond",
            "prediction_date": "2026-10-18 10:00:00",
            "predicted_category": "Excellent",
            "risk_level": "Low Risk",
            "confidence": 95.5,
            "performance_score": 88,
            "academic_inputs": {"Hours_Studied": 25, "Attendance": 90},
            "risk_reasons": ["No major risk factors detected."],
            "suggestions": ["Keep up the good work."],
            "class_probabilities": {"Excellent": 95.5, "Good": 4.5}
        }
        pred_id = save_prediction(rec, db_path=self.db_path)
        self.assertGreater(pred_id, 0)

        # Check that student was also created in student master table
        student = get_student("007", db_path=self.db_path)
        self.assertIsNotNone(student)
        self.assertEqual(student["full_name"], "James Bond")

        # Fetch prediction by id
        fetched = get_prediction_by_id(pred_id, db_path=self.db_path)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["roll_number"], "007")
        self.assertEqual(fetched["predicted_category"], "Excellent")
        self.assertEqual(fetched["academic_inputs"]["Hours_Studied"], 25)

    def test_multiple_predictions_for_same_student_not_overwritten(self):
        roll = "CS-101"
        name = "Sarah Connor"

        rec1 = {
            "roll_number": roll,
            "student_name": name,
            "prediction_date": "2026-09-01 10:00:00",
            "predicted_category": "Average",
            "risk_level": "Medium Risk",
            "confidence": 75.0,
            "performance_score": 60,
            "academic_inputs": {"Hours_Studied": 10, "Attendance": 70}
        }
        rec2 = {
            "roll_number": roll,
            "student_name": name,
            "prediction_date": "2026-10-01 10:00:00",
            "predicted_category": "Good",
            "risk_level": "Low Risk",
            "confidence": 88.0,
            "performance_score": 78,
            "academic_inputs": {"Hours_Studied": 20, "Attendance": 88}
        }

        id1 = save_prediction(rec1, db_path=self.db_path)
        id2 = save_prediction(rec2, db_path=self.db_path)

        self.assertNotEqual(id1, id2)

        # Retrieve history for Sarah Connor
        history = get_student_history(roll, db_path=self.db_path)
        self.assertEqual(len(history), 2)
        # History is ordered by date DESC, so rec2 should be first
        self.assertEqual(history[0]["predicted_category"], "Good")
        self.assertEqual(history[1]["predicted_category"], "Average")

        # Latest predictions per student should return only rec2
        latest = get_latest_predictions_per_student(db_path=self.db_path)
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0]["predicted_category"], "Good")

    def test_search_and_filter_queries(self):
        records = [
            {
                "roll_number": "001",
                "student_name": "Alice Smith",
                "predicted_category": "Excellent",
                "risk_level": "Low Risk",
                "prediction_date": "2026-10-10 10:00:00"
            },
            {
                "roll_number": "002",
                "student_name": "Bob Jones",
                "predicted_category": "Poor",
                "risk_level": "High Risk",
                "prediction_date": "2026-10-11 10:00:00"
            },
            {
                "roll_number": "003",
                "student_name": "Charlie Brown",
                "predicted_category": "Average",
                "risk_level": "Medium Risk",
                "prediction_date": "2026-10-12 10:00:00"
            }
        ]
        for r in records:
            save_prediction(r, db_path=self.db_path)

        # Search by student name
        results = get_all_predictions(search="Alice", db_path=self.db_path)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["roll_number"], "001")

        # Search by roll number
        results = get_all_predictions(search="002", db_path=self.db_path)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["student_name"], "Bob Jones")

        # Filter by risk level
        high_risk = get_all_predictions(risk_level="High Risk", db_path=self.db_path)
        self.assertEqual(len(high_risk), 1)
        self.assertEqual(high_risk[0]["student_name"], "Bob Jones")

        # Filter by category
        excellent = get_all_predictions(category="Excellent", db_path=self.db_path)
        self.assertEqual(len(excellent), 1)
        self.assertEqual(excellent[0]["student_name"], "Alice Smith")

    def test_parameterized_query_safety(self):
        # Ensure special SQL characters do not break query execution
        naughty_name = "Robert'); DROP TABLE students;--"
        naughty_roll = "00099' OR '1'='1"

        rec = {
            "roll_number": naughty_roll,
            "student_name": naughty_name,
            "predicted_category": "Average",
            "risk_level": "Low Risk"
        }
        pred_id = save_prediction(rec, db_path=self.db_path)
        self.assertGreater(pred_id, 0)

        # Search using tricky string
        results = get_all_predictions(search="DROP TABLE", db_path=self.db_path)
        self.assertEqual(len(results), 1)

        # Verify students table still exists intact
        stats = get_database_stats(db_path=self.db_path)
        self.assertEqual(stats["total_students"], 1)

    def test_delete_prediction_and_student(self):
        rec = {
            "roll_number": "DEL-1",
            "student_name": "Delete Me",
            "predicted_category": "Average",
            "risk_level": "Low Risk"
        }
        pred_id = save_prediction(rec, db_path=self.db_path)
        self.assertTrue(delete_prediction(pred_id, db_path=self.db_path))
        self.assertIsNone(get_prediction_by_id(pred_id, db_path=self.db_path))

    def test_predictions_to_dataframe(self):
        rec = {
            "roll_number": "0088",
            "student_name": "Diana Prince",
            "prediction_date": "2026-10-18 12:00:00",
            "predicted_category": "Excellent",
            "risk_level": "Low Risk",
            "confidence": 98.0,
            "performance_score": 92,
            "academic_inputs": {"Hours_Studied": 30, "Attendance": 95}
        }
        save_prediction(rec, db_path=self.db_path)
        records = get_all_predictions(db_path=self.db_path)
        df = predictions_to_dataframe(records)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["Roll Number"], "0088")
        self.assertEqual(df.iloc[0]["Student Name"], "Diana Prince")


if __name__ == "__main__":
    unittest.main()
