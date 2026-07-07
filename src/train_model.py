import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("data/StudentPerformanceFactors.csv")

# =========================
# 2. CREATE PERFORMANCE CATEGORY FROM Exam_Score
# =========================
def convert_score_to_performance(score):
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Average"
    else:
        return "Poor"

df["Performance"] = df["Exam_Score"].apply(convert_score_to_performance)

# =========================
# 3. SELECT FEATURES
# =========================
selected_features = [
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

# Keep only required columns
df = df[selected_features + ["Performance"]]

# =========================
# 4. HANDLE CATEGORICAL COLUMNS
# =========================
label_encoders = {}

categorical_cols = [
    "Internet_Access",
    "Parental_Involvement",
    "Access_to_Resources",
    "Motivation_Level"
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# =========================
# 5. SPLIT FEATURES AND TARGET
# =========================
X = df.drop("Performance", axis=1)
y = df["Performance"]

# Encode target
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# =========================
# 6. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# =========================
# 7. SCALE NUMERICAL FEATURES
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 8. TRAIN MODEL
# =========================
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# =========================
# 9. EVALUATE MODEL
# =========================
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(acc * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# =========================
# 10. SAVE MODEL FILES
# =========================
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")
joblib.dump(target_encoder, "models/target_encoder.pkl")

print("\nModel, scaler, and encoders saved successfully in models/ folder.")