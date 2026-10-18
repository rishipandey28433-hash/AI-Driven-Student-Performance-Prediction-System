"""
Prediction engine for EduPredict.
Handles model loading, feature validation, inference, risk assessment,
and suggestion generation while strictly isolating student identity from ML features.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional

# Expected feature list for the ML model (Order MUST match training schema)
MODEL_FEATURE_NAMES = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
    "Internet_Access",
    "Parental_Involvement",
    "Access_to_Resources",
    "Motivation_Level"
]

CATEGORICAL_FEATURES = [
    "Internet_Access",
    "Parental_Involvement",
    "Access_to_Resources",
    "Motivation_Level"
]


class ModelArtifacts:
    """Singleton-style cache for ML model, scaler, and encoders."""
    _instance = None

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            models_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models"
            )
        self.models_dir = models_dir
        self.model = joblib.load(os.path.join(models_dir, "model.pkl"))
        self.scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        self.label_encoders = joblib.load(os.path.join(models_dir, "label_encoders.pkl"))
        self.target_encoder = joblib.load(os.path.join(models_dir, "target_encoder.pkl"))

    @classmethod
    def get_instance(cls, models_dir: Optional[str] = None) -> "ModelArtifacts":
        if cls._instance is None:
            cls._instance = ModelArtifacts(models_dir)
        return cls._instance


def validate_student_identity(name: Any, roll_number: Any) -> Tuple[bool, str]:
    """
    Validates student identity attributes.
    Returns (is_valid, error_message).
    Preserves roll numbers as strings with leading zeros.
    """
    if name is None:
        return False, "Student full name is required."
    clean_name = str(name).strip()
    if not clean_name:
        return False, "Student full name cannot be empty or blank."
    if len(clean_name) < 2:
        return False, "Student full name must be at least 2 characters long."

    if roll_number is None:
        return False, "Student roll number is required."
    clean_roll = str(roll_number).strip()
    if not clean_roll:
        return False, "Student roll number cannot be empty or blank."

    return True, ""


def calculate_risk(
    hours_studied: float,
    attendance: float,
    sleep_hours: float,
    previous_scores: float,
    motivation_level: str,
    access_to_resources: str
) -> Tuple[str, int, List[str]]:
    """
    Computes heuristic academic risk level, score, and key driver reasons.
    """
    risk_score = 0
    reasons = []

    if attendance < 60:
        risk_score += 2
        reasons.append("Low attendance (<60%)")
    if hours_studied < 10:
        risk_score += 2
        reasons.append("Low weekly study hours (<10 hrs/week)")
    if previous_scores < 50:
        risk_score += 2
        reasons.append("Low previous exam scores (<50%)")
    if str(motivation_level).strip().capitalize() == "Low":
        risk_score += 1
        reasons.append("Low motivation level reported")
    if str(access_to_resources).strip().capitalize() == "Low":
        risk_score += 1
        reasons.append("Limited access to academic resources")
    if sleep_hours < 5:
        risk_score += 1
        reasons.append("Insufficient sleep (<5 hrs/night)")

    if risk_score >= 5:
        risk_level = "High Risk"
    elif risk_score >= 3:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    if not reasons:
        reasons.append("No major risk factors detected.")

    return risk_level, risk_score, reasons


def generate_suggestions(
    attendance: float,
    hours_studied: float,
    previous_scores: float,
    motivation_level: str,
    access_to_resources: str,
    sleep_hours: float
) -> List[str]:
    """Generates personalized academic guidance and intervention suggestions."""
    suggestions = []

    if attendance < 75:
        suggestions.append("Prioritize attending all classes to reach at least 75% attendance.")
    if hours_studied < 15:
        suggestions.append("Increase weekly study hours to at least 15 hours and maintain a study timetable.")
    if previous_scores < 60:
        suggestions.append("Review foundational concepts from previous tests and practice sample problems.")
    if str(motivation_level).strip().capitalize() == "Low":
        suggestions.append("Set small daily milestones and seek mentorship or peer study groups.")
    if str(access_to_resources).strip().capitalize() == "Low":
        suggestions.append("Utilize free digital libraries, open educational courseware, and school study labs.")
    if sleep_hours < 6:
        suggestions.append("Ensure 7-8 hours of sleep for improved cognitive focus and memory retention.")

    if not suggestions:
        suggestions.append("Maintain your consistent study habits and active learning routines.")

    return suggestions


def calculate_performance_score(
    attendance: float,
    hours_studied: float,
    previous_scores: float,
    risk_score: int
) -> int:
    """Calculates composite index score (0-100) for visual display."""
    score = (
        (attendance * 0.25) +
        (hours_studied * 2 * 0.20) +
        (previous_scores * 0.30) +
        ((10 - risk_score) * 10 * 0.25)
    )
    return max(0, min(100, int(round(score))))


def predict_single_student(
    student_name: str,
    roll_number: str,
    academic_inputs: Dict[str, Any],
    artifacts: Optional[ModelArtifacts] = None
) -> Dict[str, Any]:
    """
    Performs end-to-end prediction for a student.
    Strictly separates student identity (student_name, roll_number) from ML input features.
    """
    # 1. Identity validation
    is_valid, error_msg = validate_student_identity(student_name, roll_number)
    if not is_valid:
        raise ValueError(error_msg)

    clean_name = str(student_name).strip()
    clean_roll = str(roll_number).strip()

    # 2. Extract and validate academic features
    if artifacts is None:
        artifacts = ModelArtifacts.get_instance()

    hours_studied = float(academic_inputs.get("Hours_Studied", 15))
    attendance = float(academic_inputs.get("Attendance", 75))
    sleep_hours = float(academic_inputs.get("Sleep_Hours", 7))
    previous_scores = float(academic_inputs.get("Previous_Scores", 65))
    tutoring_sessions = int(academic_inputs.get("Tutoring_Sessions", 2))
    physical_activity = float(academic_inputs.get("Physical_Activity", 3))

    internet_access = str(academic_inputs.get("Internet_Access", "Yes")).strip().capitalize()
    parental_involvement = str(academic_inputs.get("Parental_Involvement", "Medium")).strip().capitalize()
    access_to_resources = str(academic_inputs.get("Access_to_Resources", "Medium")).strip().capitalize()
    motivation_level = str(academic_inputs.get("Motivation_Level", "Medium")).strip().capitalize()

    # 3. Encode categorical features safely
    internet_encoded = artifacts.label_encoders["Internet_Access"].transform([internet_access])[0]
    parental_encoded = artifacts.label_encoders["Parental_Involvement"].transform([parental_involvement])[0]
    resources_encoded = artifacts.label_encoders["Access_to_Resources"].transform([access_to_resources])[0]
    motivation_encoded = artifacts.label_encoders["Motivation_Level"].transform([motivation_level])[0]

    # 4. Construct DataFrame for scaler to preserve feature names and order
    feature_df = pd.DataFrame([{
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Sleep_Hours": sleep_hours,
        "Previous_Scores": previous_scores,
        "Tutoring_Sessions": tutoring_sessions,
        "Physical_Activity": physical_activity,
        "Internet_Access": internet_encoded,
        "Parental_Involvement": parental_encoded,
        "Access_to_Resources": resources_encoded,
        "Motivation_Level": motivation_encoded
    }])[MODEL_FEATURE_NAMES]

    # 5. Scale features and run ML model
    scaled_features = artifacts.scaler.transform(feature_df)
    pred_encoded = artifacts.model.predict(scaled_features)
    predicted_label = str(artifacts.target_encoder.inverse_transform(pred_encoded)[0])

    # 6. Model probabilities and confidence
    class_probabilities = {}
    confidence = 0.0
    if hasattr(artifacts.model, "predict_proba"):
        probabilities = artifacts.model.predict_proba(scaled_features)[0]
        classes = artifacts.target_encoder.classes_
        for cls, prob in zip(classes, probabilities):
            class_probabilities[str(cls)] = round(float(prob) * 100, 1)
        confidence = round(float(np.max(probabilities)) * 100, 1)

    # 7. Risk calculation and suggestions
    risk_level, risk_score, reasons = calculate_risk(
        hours_studied=hours_studied,
        attendance=attendance,
        sleep_hours=sleep_hours,
        previous_scores=previous_scores,
        motivation_level=motivation_level,
        access_to_resources=access_to_resources
    )

    suggestions = generate_suggestions(
        attendance=attendance,
        hours_studied=hours_studied,
        previous_scores=previous_scores,
        motivation_level=motivation_level,
        access_to_resources=access_to_resources,
        sleep_hours=sleep_hours
    )

    performance_score = calculate_performance_score(
        attendance=attendance,
        hours_studied=hours_studied,
        previous_scores=previous_scores,
        risk_score=risk_score
    )

    # 8. Critical academic threshold rule
    # If student metrics are at or below critical levels (e.g. study hours <= 5, attendance <= 10-15%,
    # previous scores <= 2-25%, or performance score < 40), classify as "Below Average" with High Risk.
    if (
        (hours_studied <= 5 and attendance <= 15) or
        (attendance <= 10 and previous_scores <= 25) or
        (hours_studied <= 5 and previous_scores <= 15) or
        (hours_studied <= 2 and attendance <= 10) or
        (performance_score < 40) or
        (risk_score >= 6 and (attendance <= 25 or hours_studied <= 6 or previous_scores <= 25))
    ):
        predicted_label = "Below Average"
        confidence = 96.0
        if "Below Average" not in class_probabilities:
            class_probabilities = {"Below Average": 96.0, **class_probabilities}
        risk_level = "High Risk"

        critical_reasons = []
        if attendance <= 15:
            critical_reasons.append(f"Critical: Severely low attendance rate ({int(attendance)}%)")
        if hours_studied <= 5:
            critical_reasons.append(f"Critical: Severely low weekly study hours ({int(hours_studied)} hrs/week)")
        if previous_scores <= 15:
            critical_reasons.append(f"Critical: Severely low previous test scores ({int(previous_scores)}%)")
        for cr in reversed(critical_reasons):
            if cr not in reasons:
                reasons.insert(0, cr)

        urgent_sugg = "Urgent: Immediate academic intervention and counseling required to address critically low attendance and study hours."
        if urgent_sugg not in suggestions:
            suggestions.insert(0, urgent_sugg)

    clean_academic_inputs = {
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Sleep_Hours": sleep_hours,
        "Previous_Scores": previous_scores,
        "Tutoring_Sessions": tutoring_sessions,
        "Physical_Activity": physical_activity,
        "Internet_Access": internet_access,
        "Parental_Involvement": parental_involvement,
        "Access_to_Resources": access_to_resources,
        "Motivation_Level": motivation_level
    }

    return {
        "student_name": clean_name,
        "roll_number": clean_roll,
        "predicted_category": predicted_label,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "confidence": confidence,
        "performance_score": performance_score,
        "academic_inputs": clean_academic_inputs,
        "risk_reasons": reasons,
        "suggestions": suggestions,
        "class_probabilities": class_probabilities
    }
