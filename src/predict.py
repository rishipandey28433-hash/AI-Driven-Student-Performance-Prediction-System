import joblib
import numpy as np

# Load saved model files
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")

# Sample input format:
# Hours_Studied, Attendance, Sleep_Hours, Previous_Scores,
# Tutoring_Sessions, Physical_Activity,
# Internet_Access, Parental_Involvement, Access_to_Resources, Motivation_Level

# IMPORTANT:
# Encoded values used here are sample values only
sample_input = np.array([[20, 85, 7, 78, 2, 4, 1, 2, 2, 2]])

# Scale input
sample_scaled = scaler.transform(sample_input)

# Predict
prediction = model.predict(sample_scaled)

# Decode output label
predicted_label = target_encoder.inverse_transform(prediction)

print("Predicted Performance:", predicted_label[0])