"""
Batch prediction module for EduPredict.
Handles CSV ingestion, student identity extraction, identity validation,
strict isolation of identity from ML features, duplicate detection,
and bulk prediction execution.
"""

import io
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from .prediction_engine import (
    MODEL_FEATURE_NAMES,
    ModelArtifacts,
    calculate_risk,
    generate_suggestions,
    calculate_performance_score
)
from .database import save_prediction


NAME_CANDIDATES = [
    "student_name", "student name", "name", "full_name",
    "full name", "student", "student_fullname"
]

ROLL_CANDIDATES = [
    "roll_number", "roll number", "roll_no", "roll no",
    "rollno", "id", "student_id", "student id", "registration_no"
]

FEATURE_ALIASES = {
    "Hours_Studied": ["hours_studied", "hours studied", "study_hours", "study hours", "hours"],
    "Attendance": ["attendance", "attendance (%)", "attendance_percent", "attendance_rate"],
    "Sleep_Hours": ["sleep_hours", "sleep hours", "sleep"],
    "Previous_Scores": ["previous_scores", "previous scores", "previous score", "prev_scores", "prev_score"],
    "Tutoring_Sessions": ["tutoring_sessions", "tutoring sessions", "tutoring"],
    "Physical_Activity": ["physical_activity", "physical activity", "activity"],
    "Internet_Access": ["internet_access", "internet access", "internet"],
    "Parental_Involvement": ["parental_involvement", "parental involvement"],
    "Access_to_Resources": ["access_to_resources", "access to resources", "resources"],
    "Motivation_Level": ["motivation_level", "motivation level", "motivation"]
}


def detect_column(df_columns: List[str], candidates: List[str]) -> Optional[str]:
    """Finds best matching column name case-insensitively."""
    lower_map = {col.lower().strip().replace("-", "_"): col for col in df_columns}
    for cand in candidates:
        cand_key = cand.lower().strip().replace("-", "_")
        if cand_key in lower_map:
            return lower_map[cand_key]
    return None


def read_csv_preserving_identity(file_source: Any) -> pd.DataFrame:
    """
    Reads CSV file ensuring all object/identifier columns (especially roll numbers)
    are parsed as strings to strictly preserve leading zeros (e.g., '0042').
    """
    if isinstance(file_source, bytes):
        file_source = io.BytesIO(file_source)

    # First read sample to check column names
    preview = pd.read_csv(file_source, nrows=5)
    file_source.seek(0) if hasattr(file_source, "seek") else None

    # Determine roll number column
    roll_col = detect_column(preview.columns.tolist(), ROLL_CANDIDATES)
    name_col = detect_column(preview.columns.tolist(), NAME_CANDIDATES)

    dtype_dict = {}
    if roll_col:
        dtype_dict[roll_col] = str
    if name_col:
        dtype_dict[name_col] = str

    df = pd.read_csv(file_source, dtype=dtype_dict, keep_default_na=False)
    return df


def process_batch_predictions(
    df: pd.DataFrame,
    artifacts: Optional[ModelArtifacts] = None
) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Validates and runs batch prediction on uploaded DataFrame.
    Strictly isolates student identity from ML prediction features.

    Returns:
        results_df: DataFrame with student info and prediction outcomes.
        summary: Dict with counts, duplicates, and validation errors.
        valid_records: List of structured prediction dicts ready for DB save.
    """
    if artifacts is None:
        artifacts = ModelArtifacts.get_instance()

    cols = df.columns.tolist()
    name_col = detect_column(cols, NAME_CANDIDATES)
    roll_col = detect_column(cols, ROLL_CANDIDATES)

    # Map academic feature columns
    feature_map = {}
    for feat, aliases in FEATURE_ALIASES.items():
        found = detect_column(cols, aliases)
        if found:
            feature_map[feat] = found

    # Check for missing essential academic features
    missing_academic_cols = [f for f in MODEL_FEATURE_NAMES if f not in feature_map]

    total_rows = len(df)
    errors = []
    seen_rolls = {}
    duplicate_rolls = set()

    # Track duplicate roll numbers in batch
    if roll_col:
        for idx, val in enumerate(df[roll_col]):
            clean_r = str(val).strip()
            if clean_r:
                if clean_r in seen_rolls:
                    duplicate_rolls.add(clean_r)
                    seen_rolls[clean_r].append(idx + 1)
                else:
                    seen_rolls[clean_r] = [idx + 1]

    valid_rows_list = []
    valid_records = []

    for idx, row in df.iterrows():
        row_num = idx + 1
        row_errors = []

        # 1. Identity validation
        if not name_col:
            student_name = f"Student {row_num}"
        else:
            raw_name = row[name_col]
            student_name = str(raw_name).strip() if pd.notna(raw_name) else ""
            if not student_name:
                row_errors.append("Student Name is missing or empty.")

        if not roll_col:
            student_roll = f"GEN-{row_num:04d}"
        else:
            raw_roll = row[roll_col]
            student_roll = str(raw_roll).strip() if pd.notna(raw_roll) else ""
            if not student_roll:
                row_errors.append("Roll Number is missing or empty.")

        # Check missing required academic features
        if missing_academic_cols:
            row_errors.append(f"Missing required ML features in dataset: {', '.join(missing_academic_cols)}")

        # 2. Academic feature extraction and conversion
        academic_data = {}
        for feat in MODEL_FEATURE_NAMES:
            if feat in feature_map:
                val = row[feature_map[feat]]
                if pd.isna(val) or str(val).strip() == "":
                    row_errors.append(f"Missing value for feature '{feat}'.")
                    continue

                if feat in ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores", "Physical_Activity"]:
                    try:
                        academic_data[feat] = float(val)
                    except ValueError:
                        row_errors.append(f"Invalid numeric value '{val}' for '{feat}'.")
                elif feat == "Tutoring_Sessions":
                    try:
                        academic_data[feat] = int(float(val))
                    except ValueError:
                        row_errors.append(f"Invalid integer value '{val}' for '{feat}'.")
                else:
                    academic_data[feat] = str(val).strip().capitalize()

        if row_errors:
            errors.append({
                "row": row_num,
                "student_name": student_name or "(Unknown)",
                "roll_number": student_roll or "(Unknown)",
                "reasons": "; ".join(row_errors)
            })
            continue

        # 3. Model Inference (ML feature isolation)
        try:
            internet_encoded = artifacts.label_encoders["Internet_Access"].transform([academic_data["Internet_Access"]])[0]
            parental_encoded = artifacts.label_encoders["Parental_Involvement"].transform([academic_data["Parental_Involvement"]])[0]
            resources_encoded = artifacts.label_encoders["Access_to_Resources"].transform([academic_data["Access_to_Resources"]])[0]
            motivation_encoded = artifacts.label_encoders["Motivation_Level"].transform([academic_data["Motivation_Level"]])[0]

            feat_df = pd.DataFrame([{
                "Hours_Studied": academic_data["Hours_Studied"],
                "Attendance": academic_data["Attendance"],
                "Sleep_Hours": academic_data["Sleep_Hours"],
                "Previous_Scores": academic_data["Previous_Scores"],
                "Tutoring_Sessions": academic_data["Tutoring_Sessions"],
                "Physical_Activity": academic_data["Physical_Activity"],
                "Internet_Access": internet_encoded,
                "Parental_Involvement": parental_encoded,
                "Access_to_Resources": resources_encoded,
                "Motivation_Level": motivation_encoded
            }])[MODEL_FEATURE_NAMES]

            scaled = artifacts.scaler.transform(feat_df)
            pred = artifacts.model.predict(scaled)
            pred_label = str(artifacts.target_encoder.inverse_transform(pred)[0])

            proba = artifacts.model.predict_proba(scaled)[0]
            classes = artifacts.target_encoder.classes_
            class_probs = {str(c): round(float(p) * 100, 1) for c, p in zip(classes, proba)}
            confidence = round(float(max(proba)) * 100, 1)

            risk_level, risk_score, reasons = calculate_risk(
                hours_studied=academic_data["Hours_Studied"],
                attendance=academic_data["Attendance"],
                sleep_hours=academic_data["Sleep_Hours"],
                previous_scores=academic_data["Previous_Scores"],
                motivation_level=academic_data["Motivation_Level"],
                access_to_resources=academic_data["Access_to_Resources"]
            )

            suggestions = generate_suggestions(
                attendance=academic_data["Attendance"],
                hours_studied=academic_data["Hours_Studied"],
                previous_scores=academic_data["Previous_Scores"],
                motivation_level=academic_data["Motivation_Level"],
                access_to_resources=academic_data["Access_to_Resources"],
                sleep_hours=academic_data["Sleep_Hours"]
            )

            perf_score = calculate_performance_score(
                attendance=academic_data["Attendance"],
                hours_studied=academic_data["Hours_Studied"],
                previous_scores=academic_data["Previous_Scores"],
                risk_score=risk_score
            )

            # Record dictionary for database
            rec = {
                "student_name": student_name,
                "roll_number": student_roll,
                "predicted_category": pred_label,
                "risk_level": risk_level,
                "risk_score": risk_score,
                "confidence": confidence,
                "performance_score": perf_score,
                "academic_inputs": academic_data,
                "risk_reasons": reasons,
                "suggestions": suggestions,
                "class_probabilities": class_probs
            }
            valid_records.append(rec)

            # Row dictionary for DataFrame display
            valid_rows_list.append({
                "Roll Number": student_roll,
                "Student Name": student_name,
                "Predicted Performance": pred_label,
                "Risk Level": risk_level,
                "Performance Score": perf_score,
                "Confidence (%)": confidence,
                "Top Suggestion": suggestions[0] if suggestions else "Maintain study routine",
                "Attendance (%)": academic_data["Attendance"],
                "Hours Studied": academic_data["Hours_Studied"],
                "Previous Scores": academic_data["Previous_Scores"],
                "Sleep Hours": academic_data["Sleep_Hours"],
                "Motivation Level": academic_data["Motivation_Level"]
            })

        except Exception as e:
            errors.append({
                "row": row_num,
                "student_name": student_name,
                "roll_number": student_roll,
                "reasons": f"Inference error: {str(e)}"
            })

    results_df = pd.DataFrame(valid_rows_list)
    summary = {
        "total_rows": total_rows,
        "valid_count": len(valid_records),
        "invalid_count": len(errors),
        "errors": errors,
        "duplicate_roll_numbers": list(duplicate_rolls),
        "name_column_found": name_col,
        "roll_column_found": roll_col
    }

    return results_df, summary, valid_records


def save_batch_records(records: List[Dict[str, Any]], db_path: Optional[str] = None) -> int:
    """Saves all validated batch prediction records into persistent database."""
    saved_count = 0
    for rec in records:
        try:
            save_prediction(rec, db_path=db_path)
            saved_count += 1
        except Exception:
            continue
    return saved_count
