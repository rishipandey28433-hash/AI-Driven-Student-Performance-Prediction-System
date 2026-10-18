"""
Database module for EduPredict application.
Provides persistent SQLite storage for student identity and prediction records.
All SQL queries are strictly parameterized for safety and data integrity.
Uses proper connection lifecycle management for clean resource handling on all operating systems.
"""

import os
import json
import sqlite3
from datetime import datetime
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Generator
import pandas as pd

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "edupredict.db"
)


def get_db_path(custom_path: Optional[str] = None) -> str:
    """Returns database file path, ensuring directory exists."""
    path = custom_path or os.environ.get("EDUPREDICT_DB_PATH", DEFAULT_DB_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Creates a SQLite connection with foreign keys and row factory enabled."""
    path = get_db_path(db_path)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@contextmanager
def get_db_cursor(db_path: Optional[str] = None) -> Generator[sqlite3.Cursor, None, None]:
    """Context manager that yields a cursor and guarantees commit and connection closure."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[str] = None) -> None:
    """
    Initializes database schema if tables do not exist.
    Creates students and predictions tables along with indexes.
    """
    with get_db_cursor(db_path) as cursor:
        # Students master table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                roll_number TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)

        # Predictions historical table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_number TEXT NOT NULL,
                student_name TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                predicted_category TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                confidence REAL,
                performance_score INTEGER,
                academic_inputs TEXT NOT NULL,
                risk_reasons TEXT,
                suggestions TEXT,
                class_probabilities TEXT,
                FOREIGN KEY (roll_number) REFERENCES students (roll_number) ON DELETE CASCADE
            );
        """)

        # Performance indexes for searching and filtering
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_roll ON predictions(roll_number);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_date ON predictions(prediction_date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_cat ON predictions(predicted_category);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_risk ON predictions(risk_level);")


def upsert_student(roll_number: str, full_name: str, db_path: Optional[str] = None) -> None:
    """
    Inserts a new student or updates existing student name without overwriting history.
    Roll number is preserved as a string (retaining leading zeros).
    """
    clean_roll = str(roll_number).strip()
    clean_name = str(full_name).strip()
    if not clean_roll or not clean_name:
        raise ValueError("Student full name and roll number cannot be empty.")

    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_db_cursor(db_path) as cursor:
        cursor.execute("""
            INSERT INTO students (roll_number, full_name, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(roll_number) DO UPDATE SET
                full_name = excluded.full_name,
                updated_at = excluded.updated_at;
        """, (clean_roll, clean_name, now_iso, now_iso))


def get_student(roll_number: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetches student master profile by roll number."""
    clean_roll = str(roll_number).strip()
    with get_db_cursor(db_path) as cursor:
        cursor.execute("SELECT * FROM students WHERE roll_number = ?;", (clean_roll,))
        row = cursor.fetchone()
        return dict(row) if row else None


def save_prediction(record: Dict[str, Any], db_path: Optional[str] = None) -> int:
    """
    Saves a student prediction record.
    Preserves existing predictions without overwriting history.
    """
    roll_number = str(record.get("roll_number", "")).strip()
    student_name = str(record.get("student_name", "")).strip()

    if not roll_number:
        raise ValueError("Roll number is required for saving prediction.")
    if not student_name:
        raise ValueError("Student name is required for saving prediction.")

    # Ensure student master record exists
    upsert_student(roll_number, student_name, db_path=db_path)

    prediction_date = record.get("prediction_date")
    if not prediction_date:
        prediction_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Serialize structured fields to JSON
    academic_inputs = record.get("academic_inputs", {})
    academic_json = json.dumps(academic_inputs) if isinstance(academic_inputs, dict) else str(academic_inputs)

    risk_reasons = record.get("risk_reasons", [])
    reasons_json = json.dumps(risk_reasons) if isinstance(risk_reasons, list) else str(risk_reasons)

    suggestions = record.get("suggestions", [])
    suggestions_json = json.dumps(suggestions) if isinstance(suggestions, list) else str(suggestions)

    class_probs = record.get("class_probabilities", {})
    probs_json = json.dumps(class_probs) if isinstance(class_probs, dict) else str(class_probs)

    predicted_category = str(record.get("predicted_category", "Unknown"))
    risk_level = str(record.get("risk_level", "Unknown"))
    confidence = float(record.get("confidence", 0.0)) if record.get("confidence") is not None else None
    performance_score = int(record.get("performance_score", 0)) if record.get("performance_score") is not None else None

    with get_db_cursor(db_path) as cursor:
        cursor.execute("""
            INSERT INTO predictions (
                roll_number, student_name, prediction_date,
                predicted_category, risk_level, confidence,
                performance_score, academic_inputs, risk_reasons,
                suggestions, class_probabilities
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            roll_number, student_name, prediction_date,
            predicted_category, risk_level, confidence,
            performance_score, academic_json, reasons_json,
            suggestions_json, probs_json
        ))
        return cursor.lastrowid


def _parse_row(row: sqlite3.Row) -> Dict[str, Any]:
    """Helper to convert sqlite3.Row to dict and parse JSON fields."""
    item = dict(row)
    for json_col, default in [
        ("academic_inputs", {}),
        ("risk_reasons", []),
        ("suggestions", []),
        ("class_probabilities", {})
    ]:
        val = item.get(json_col)
        if isinstance(val, str):
            try:
                item[json_col] = json.loads(val)
            except Exception:
                item[json_col] = default
        elif val is None:
            item[json_col] = default
    return item


def get_all_predictions(
    search: Optional[str] = None,
    category: Optional[str] = None,
    risk_level: Optional[str] = None,
    sort_order: str = "desc",
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves all prediction records with optional search, filtering, and sorting.
    Queries are fully parameterized.
    """
    query = "SELECT * FROM predictions WHERE 1=1"
    params: List[Any] = []

    if search:
        search_pattern = f"%{search.strip()}%"
        query += " AND (student_name LIKE ? OR roll_number LIKE ?)"
        params.extend([search_pattern, search_pattern])

    if category and category != "All":
        query += " AND predicted_category = ?"
        params.append(category)

    if risk_level and risk_level != "All":
        query += " AND risk_level = ?"
        params.append(risk_level)

    order_direction = "ASC" if str(sort_order).lower() == "asc" else "DESC"
    query += f" ORDER BY prediction_date {order_direction}, id {order_direction};"

    with get_db_cursor(db_path) as cursor:
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [_parse_row(r) for r in rows]


def get_latest_predictions_per_student(
    search: Optional[str] = None,
    category: Optional[str] = None,
    risk_level: Optional[str] = None,
    sort_order: str = "desc",
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Returns only the latest prediction for each student using window functions.
    Supports search, category filtering, and risk filtering.
    """
    query = """
        WITH ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY roll_number
                       ORDER BY prediction_date DESC, id DESC
                   ) as rn
            FROM predictions
        )
        SELECT id, roll_number, student_name, prediction_date,
               predicted_category, risk_level, confidence,
               performance_score, academic_inputs, risk_reasons,
               suggestions, class_probabilities
        FROM ranked
        WHERE rn = 1
    """
    params: List[Any] = []

    if search:
        search_pattern = f"%{search.strip()}%"
        query += " AND (student_name LIKE ? OR roll_number LIKE ?)"
        params.extend([search_pattern, search_pattern])

    if category and category != "All":
        query += " AND predicted_category = ?"
        params.append(category)

    if risk_level and risk_level != "All":
        query += " AND risk_level = ?"
        params.append(risk_level)

    order_direction = "ASC" if str(sort_order).lower() == "asc" else "DESC"
    query += f" ORDER BY prediction_date {order_direction}, id {order_direction};"

    with get_db_cursor(db_path) as cursor:
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [_parse_row(r) for r in rows]


def get_student_history(roll_number: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves chronological prediction history for a specific student."""
    clean_roll = str(roll_number).strip()
    with get_db_cursor(db_path) as cursor:
        cursor.execute("""
            SELECT * FROM predictions
            WHERE roll_number = ?
            ORDER BY prediction_date DESC, id DESC;
        """, (clean_roll,))
        rows = cursor.fetchall()
        return [_parse_row(r) for r in rows]


def get_prediction_by_id(prediction_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetches a specific prediction record by its primary key ID."""
    with get_db_cursor(db_path) as cursor:
        cursor.execute("SELECT * FROM predictions WHERE id = ?;", (prediction_id,))
        row = cursor.fetchone()
        return _parse_row(row) if row else None


def delete_prediction(prediction_id: int, db_path: Optional[str] = None) -> bool:
    """Safely removes a single prediction record by ID using parameterized query."""
    with get_db_cursor(db_path) as cursor:
        cursor.execute("DELETE FROM predictions WHERE id = ?;", (prediction_id,))
        return cursor.rowcount > 0


def delete_student(roll_number: str, db_path: Optional[str] = None) -> bool:
    """Removes a student and cascades deletion to all their prediction records."""
    clean_roll = str(roll_number).strip()
    with get_db_cursor(db_path) as cursor:
        cursor.execute("DELETE FROM predictions WHERE roll_number = ?;", (clean_roll,))
        cursor.execute("DELETE FROM students WHERE roll_number = ?;", (clean_roll,))
        return cursor.rowcount > 0


def get_database_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Computes high-level statistics for reporting and dashboard widgets."""
    with get_db_cursor(db_path) as cursor:
        cursor.execute("SELECT COUNT(*) FROM students;")
        total_students = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions;")
        total_predictions = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(performance_score) FROM predictions WHERE performance_score IS NOT NULL;")
        avg_score_row = cursor.fetchone()[0]
        avg_score = round(avg_score_row, 1) if avg_score_row is not None else 0.0

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE risk_level = 'High Risk';")
        high_risk_count = cursor.fetchone()[0]

        cursor.execute("SELECT predicted_category, COUNT(*) FROM predictions GROUP BY predicted_category;")
        category_counts = {row[0]: row[1] for row in cursor.fetchall()}

        cursor.execute("SELECT risk_level, COUNT(*) FROM predictions GROUP BY risk_level;")
        risk_counts = {row[0]: row[1] for row in cursor.fetchall()}

        return {
            "total_students": total_students,
            "total_predictions": total_predictions,
            "avg_score": avg_score,
            "high_risk_count": high_risk_count,
            "category_counts": category_counts,
            "risk_counts": risk_counts
        }


def predictions_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Flattens a list of prediction records into a clean pandas DataFrame for display and CSV export."""
    if not records:
        return pd.DataFrame(columns=[
            "ID", "Roll Number", "Student Name", "Prediction Date",
            "Predicted Performance", "Risk Level", "Performance Score",
            "Confidence (%)", "Hours Studied", "Attendance (%)",
            "Sleep Hours", "Previous Scores", "Tutoring Sessions",
            "Physical Activity", "Internet Access", "Parental Involvement",
            "Access to Resources", "Motivation Level"
        ])

    flat_list = []
    for r in records:
        inputs = r.get("academic_inputs", {})
        flat_list.append({
            "ID": r.get("id"),
            "Roll Number": str(r.get("roll_number", "")),
            "Student Name": r.get("student_name", ""),
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
            "Motivation Level": inputs.get("Motivation_Level", "")
        })

    df = pd.DataFrame(flat_list)
    return df
