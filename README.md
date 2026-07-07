# 🎓 EduPredict – AI-Driven Student Performance Prediction and Early Warning System

EduPredict is a machine learning-based web application that predicts a student’s academic performance using study habits, attendance, previous scores, motivation, and resource-related factors.  
It also identifies risk level, highlights possible reasons behind poor performance, and provides personalized improvement suggestions.

---

## 📌 Project Objective
The goal of this project is to build an **early warning academic support system** that helps identify students who may be at risk of low performance and provides actionable suggestions for improvement.

---

## 🚀 Features
- Predicts student performance as:
  - **Excellent**
  - **Good**
  - **Average**
  - **Poor**
- Detects **risk level**:
  - Low Risk
  - Medium Risk
  - High Risk
- Displays **top reasons** behind poor academic performance
- Generates **personalized improvement suggestions**
- Provides a **simple interactive Streamlit web interface**
- Shows **student input summary and visual analysis chart**

---

## 🧠 Technologies Used
- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Streamlit**
- **Joblib**

---

## 📂 Project Structure
```bash id="vlqaqj"
EduPredict/
│
├── data/
│   └── StudentPerformanceFactors.csv
│
├── src/
│   ├── explore_data.py
│   ├── train_model.py
│   └── predict.py
│
├── app/
│   └── app.py
│
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── target_encoder.pkl
│
├── notebooks/
├── main.py
├── requirements.txt
└── README.md