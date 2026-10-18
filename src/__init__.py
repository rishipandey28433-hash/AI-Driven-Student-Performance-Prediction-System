"""
EduPredict core package.
Contains database persistence, prediction engine, reporting, and batch processing.
"""

from .database import (
    init_db,
    save_prediction,
    get_all_predictions,
    get_latest_predictions_per_student,
    get_student_history,
    delete_prediction,
    delete_student,
    get_database_stats,
    predictions_to_dataframe
)
from .prediction_engine import (
    predict_single_student,
    validate_student_identity,
    calculate_risk,
    generate_suggestions,
    calculate_performance_score,
    ModelArtifacts
)
from .report_generator import (
    generate_student_pdf,
    generate_student_csv,
    generate_batch_csv
)
from .batch_prediction import (
    process_batch_predictions,
    save_batch_records,
    read_csv_preserving_identity
)

__all__ = [
    "init_db",
    "save_prediction",
    "get_all_predictions",
    "get_latest_predictions_per_student",
    "get_student_history",
    "delete_prediction",
    "delete_student",
    "get_database_stats",
    "predictions_to_dataframe",
    "predict_single_student",
    "validate_student_identity",
    "calculate_risk",
    "generate_suggestions",
    "calculate_performance_score",
    "ModelArtifacts",
    "generate_student_pdf",
    "generate_student_csv",
    "generate_batch_csv",
    "process_batch_predictions",
    "save_batch_records",
    "read_csv_preserving_identity"
]
