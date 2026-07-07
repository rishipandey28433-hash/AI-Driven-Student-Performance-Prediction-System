import pandas as pd

# Load dataset
df = pd.read_csv("data/StudentPerformanceFactors.csv")

print("===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== NUMERICAL SUMMARY =====")
print(df.describe())

# Check target column
if "Exam_Score" in df.columns:
    print("\nExam_Score column found successfully.")
else:
    print("\nExam_Score column not found. Please check the dataset column names.")