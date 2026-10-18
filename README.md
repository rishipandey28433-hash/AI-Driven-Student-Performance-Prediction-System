# 🎓 EduPredict – AI-Driven Student Performance Prediction and Early Warning System

EduPredict is an enterprise-grade academic analytics and early warning web application that predicts a student’s performance category, identifies risk indicators, and provides actionable intervention plans.

EduPredict includes a **Student Identity and Prediction Record Management** subsystem with persistent SQLite storage, historical progression tracking, batch CSV prediction, and PDF/CSV reporting with educational disclaimers.

---

## 📌 Project Objective
The primary objective of EduPredict is to provide educators, counselors, and academic advisors with an **early warning academic support system**. By recognizing risk patterns early (low attendance, deficient study hours, low motivation, sleep deprivation), institutions can proactively intervene with personalized guidance before examinations occur.

---

## 🚀 Key Features

### 1. Student Identity & Form Validation
- **Required Identity Fields:** Student Full Name and unique Roll Number (Student ID).
- **Leading-Zero Preservation:** Roll numbers are stored strictly as text strings (e.g., `0042`, `CS-007`) so numerical leading zeros are never lost.
- **Strict Input Validation:** Enforces non-empty names and roll numbers, trims whitespace, and prevents predictions without valid identification.
- **Model Feature Isolation:** Student identities are kept as application-level metadata and **never** passed into the machine learning feature pipeline.

### 2. Multi-Record Prediction History & Longitudinal Tracking
- **Non-Destructive Persistence:** Prior predictions are never overwritten when new predictions are recorded for a student.
- **Historical Timeline:** Track a student's performance score and risk trajectory over multiple semesters or review periods.
- **Search & Filtering:** Search by student name or roll number; filter by predicted category (`Excellent`, `Good`, `Average`, `Poor`) or risk level (`Low`, `Medium`, `High`); sort chronologically.
- **Latest vs. All Records:** Switch seamlessly between latest snapshot per student and full historical ledger.
- **Record Deletion with Confirmation:** Authorized users can permanently remove individual records with an explicit safety confirmation step.

### 3. Persistent Local Storage (SQLite)
- **Local SQLite Engine:** Data persists reliably across application restarts in `data/edupredict.db`.
- **Parameterized SQL:** All database transactions use strictly parameterized queries to guarantee SQL injection immunity.
- **Relational Integrity:** Foreign key constraints with cascading logic between master `students` and historical `predictions` tables.
- **Privacy & Git Exclusion:** Database files (`*.db`, `*.sqlite`) are gitignored to ensure real student records are never committed to version control.

### 4. Professional Reporting & Downloads
- **Individual PDF Reports:** Styled formal performance reports generated using ReportLab, containing student identity, prediction summary badges, full academic profile, risk factors, suggestions, and mandatory educational disclaimers.
- **Individual & Batch CSV Exports:** Instant CSV export for single student records or filtered search result tables.
- **Educational Disclaimer:** Every report includes the mandatory notice:
  > *"Disclaimer: EduPredict predictions, risk assessments, and guidance suggestions are algorithmic estimates generated for early warning and academic advising purposes only. They do not constitute guaranteed academic outcomes or official academic grading."*

### 5. High-Throughput Batch Prediction
- **CSV Ingestion:** Ingest multi-student CSV datasets with automatic detection of name and roll number columns.
- **Format Integrity:** Preserves leading zeros in roll numbers across bulk records.
- **Validation & Duplicate Alerts:** Flags missing student identities, missing academic factors, and duplicate roll numbers within the batch.
- **Batch Export & Database Commit:** Export enriched batch predictions or save all valid records to the persistent database with one click.

---

## 🧠 Technologies Used
- **Python 3.11**
- **Streamlit** – Web interface & interactive dashboard
- **Scikit-learn** – Random Forest Classifier (200 estimators)
- **Pandas & NumPy** – High-performance data manipulation & tabular processing
- **SQLite3** – ACID-compliant local persistent database
- **ReportLab** – Professional PDF report compilation
- **Joblib** – Model artifact serialization

---

## 📂 Project Structure
```text
EduPredict/
│
├── app/
│   └── app.py                      # Streamlit application with multi-view navigation
│
├── src/
│   ├── __init__.py                 # Core package exports
│   ├── database.py                 # SQLite storage, parameterized queries, search & stats
│   ├── prediction_engine.py        # Feature validation, ML inference, risk & suggestions
│   ├── report_generator.py         # PDF and CSV report compilation with disclaimers
│   ├── batch_prediction.py         # CSV ingestion, leading-zero preservation & validation
│   ├── explore_data.py             # Data exploration script
│   ├── train_model.py              # ML model training script
│   └── predict.py                  # Sample CLI inference script
│
├── models/
│   ├── model.pkl                   # Trained Random Forest classifier
│   ├── scaler.pkl                  # Fitted StandardScaler
│   ├── label_encoders.pkl          # Categorical feature encoders
│   └── target_encoder.pkl          # Performance category target encoder
│
├── data/
│   ├── StudentPerformanceFactors.csv  # Base training dataset
│   └── edupredict.db               # SQLite persistent database (gitignored)
│
├── tests/
│   ├── test_database.py            # Unit tests for SQLite persistence & queries
│   ├── test_prediction_engine.py   # Unit tests for validation & ML inference
│   ├── test_report_generator.py    # Unit tests for PDF/CSV generation
│   ├── test_batch_prediction.py    # Unit tests for CSV batch processing
│   └── test_integration_workflow.py # Full end-to-end integration test
│
├── main.py                         # Application launcher
├── requirements.txt                # Project dependencies
├── .gitignore                      # Git exclusion rules (DBs, logs, caches)
└── README.md                       # System documentation
```

---

## ⚙️ Installation & Setup

1. **Clone or Navigate to the Repository:**
   ```bash
   cd AI-Driven-Student-Performance-Prediction-and-Early-Warning-System
   ```

2. **Install Required Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify the Installation with Automated Tests:**
   ```bash
   python -m unittest discover -s tests -p "test_*.py"
   ```

---

## 🚀 Running the Application

Launch the Streamlit web application using either:

```bash
python main.py
```
or
```bash
streamlit run app/app.py
```

The application will start locally at `http://localhost:8501`.

---

## 💾 Database Management & Maintenance

### Storage Location
The SQLite database file is stored at:
```text
data/edupredict.db
```
The database and tables are created automatically on the first application launch.

### Backup Procedure
To create a backup of all student records and historical predictions:
1. Ensure the application is idle.
2. Copy `data/edupredict.db` to a secure backup directory (e.g., `backup/edupredict_2026-10-18.db`).

### Restore Procedure
1. Stop the application.
2. Replace `data/edupredict.db` with your backup copy.
3. Restart the application.

### Deletion & Reset Procedure
- **Single Record Deletion:** Open the **Student Records & History** view, select the record, expand the **Danger Zone**, check the confirmation box, and click **Permanently Delete Record**.
- **Complete System Reset:** Stop the application and delete `data/edupredict.db`. The application will generate a fresh empty database upon next launch.

---

## 🔒 Privacy & Data Protection Guarantees
- **Local Execution:** 100% of prediction calculations, database operations, and PDF exports run locally. No student records are transmitted to third-party APIs.
- **Git Safety:** Database files (`*.db`, `*.sqlite`, `*.sqlite3`) are excluded in `.gitignore` to prevent confidential student data from entering version control.
- **Strict Isolation:** Student names and roll numbers serve strictly for record retrieval and are never exposed to the ML feature vector.