import streamlit as st
import numpy as np
import joblib
import pandas as pd

# =========================
# LOAD MODEL FILES
# =========================
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="EduPredict", page_icon="🎓", layout="wide")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🎓 EduPredict")
st.sidebar.write("AI-Driven Student Performance Prediction and Early Warning System")
st.sidebar.markdown("---")
st.sidebar.write("### Features")
st.sidebar.write("✔ Performance Prediction")
st.sidebar.write("✔ Risk Level Detection")
st.sidebar.write("✔ Improvement Suggestions")
st.sidebar.write("✔ Student Input Analysis")

st.sidebar.markdown("---")
st.sidebar.info("Fill the student details and click **Predict Performance**.")

# =========================
# MAIN TITLE
# =========================
st.title("🎓 EduPredict – Student Performance Prediction")
st.write("Predict student academic performance, identify risk level, and get improvement suggestions.")

st.markdown("---")

# =========================
# INPUT SECTION
# =========================
st.subheader("📘 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    hours_studied = st.slider("Hours Studied (per week)", 0, 40, 15)
    attendance = st.slider("Attendance (%)", 0, 100, 75)
    sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
    previous_scores = st.slider("Previous Scores", 0, 100, 65)
    tutoring_sessions = st.slider("Tutoring Sessions", 0, 10, 2)

with col2:
    physical_activity = st.slider("Physical Activity (hours/week)", 0, 15, 3)
    internet_access = st.selectbox("Internet Access", ["Yes", "No"])
    parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
    access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
    motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])

# =========================
# PREDICT BUTTON
# =========================
if st.button("🚀 Predict Performance"):

    # Encode categorical inputs
    internet_encoded = label_encoders["Internet_Access"].transform([internet_access])[0]
    parental_encoded = label_encoders["Parental_Involvement"].transform([parental_involvement])[0]
    resources_encoded = label_encoders["Access_to_Resources"].transform([access_to_resources])[0]
    motivation_encoded = label_encoders["Motivation_Level"].transform([motivation_level])[0]

    input_data = np.array([[
        hours_studied,
        attendance,
        sleep_hours,
        previous_scores,
        tutoring_sessions,
        physical_activity,
        internet_encoded,
        parental_encoded,
        resources_encoded,
        motivation_encoded
    ]])

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_label = target_encoder.inverse_transform(prediction)[0]

    # =========================
    # RISK LOGIC
    # =========================
    risk_score = 0
    reasons = []

    if attendance < 60:
        risk_score += 2
        reasons.append("Low attendance")
    if hours_studied < 10:
        risk_score += 2
        reasons.append("Less study hours")
    if previous_scores < 50:
        risk_score += 2
        reasons.append("Low previous scores")
    if motivation_level == "Low":
        risk_score += 1
        reasons.append("Low motivation")
    if access_to_resources == "Low":
        risk_score += 1
        reasons.append("Limited study resources")
    if sleep_hours < 5:
        risk_score += 1
        reasons.append("Insufficient sleep")

    if risk_score >= 5:
        risk_level = "High Risk"
    elif risk_score >= 3:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    # =========================
    # SUGGESTIONS
    # =========================
    suggestions = []

    if attendance < 75:
        suggestions.append("Try to improve attendance to at least 75% or above.")
    if hours_studied < 15:
        suggestions.append("Increase weekly study hours and maintain a study schedule.")
    if previous_scores < 60:
        suggestions.append("Revise previous weak topics and solve more practice questions.")
    if motivation_level == "Low":
        suggestions.append("Set small weekly goals and track progress to improve motivation.")
    if access_to_resources == "Low":
        suggestions.append("Use online learning resources, notes, and library materials.")
    if sleep_hours < 6:
        suggestions.append("Maintain a healthy sleep routine for better concentration.")

    if not suggestions:
        suggestions.append("Keep maintaining your current study habits and consistency.")

    if not reasons:
        reasons.append("No major risk factors detected.")

    # =========================
    # PERFORMANCE SCORE FOR DISPLAY
    # =========================
    performance_score = (
        (attendance * 0.25) +
        (hours_studied * 2 * 0.20) +
        (previous_scores * 0.30) +
        ((10 - risk_score) * 10 * 0.25)
    )
    performance_score = max(0, min(100, int(performance_score)))

    # =========================
    # DISPLAY RESULTS
    # =========================
    st.markdown("---")
    st.subheader("📊 Prediction Result")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Predicted Performance", predicted_label)

    with c2:
        st.metric("Risk Level", risk_level)

    with c3:
        st.metric("Performance Score", f"{performance_score}/100")

    st.progress(performance_score / 100)

    # =========================
    # INPUT SUMMARY TABLE
    # =========================
    st.subheader("🧾 Student Input Summary")
    summary_df = pd.DataFrame({
        "Feature": [
            "Hours Studied", "Attendance", "Sleep Hours", "Previous Scores",
            "Tutoring Sessions", "Physical Activity", "Internet Access",
            "Parental Involvement", "Access to Resources", "Motivation Level"
        ],
        "Value": [
            hours_studied, attendance, sleep_hours, previous_scores,
            tutoring_sessions, physical_activity, internet_access,
            parental_involvement, access_to_resources, motivation_level
        ]
    })
    st.dataframe(summary_df, use_container_width=True)

    # =========================
    # REASONS AND SUGGESTIONS
    # =========================
    colA, colB = st.columns(2)

    with colA:
        st.subheader("⚠ Top Reasons")
        for r in reasons[:3]:
            st.write(f"- {r}")

    with colB:
        st.subheader("💡 Suggestions")
        for s in suggestions:
            st.write(f"- {s}")

    # =========================
    # SIMPLE VISUAL ANALYSIS
    # =========================
    st.subheader("📈 Visual Analysis")

    chart_df = pd.DataFrame({
        "Metric": ["Attendance", "Study Hours x2", "Previous Scores", "Sleep Hours x10"],
        "Value": [attendance, hours_studied * 2, previous_scores, sleep_hours * 10]
    })

    st.bar_chart(chart_df.set_index("Metric"))

    # =========================
    # FINAL MESSAGE
    # =========================
    st.markdown("---")
    if predicted_label == "Excellent":
        st.success("The student is expected to perform very well academically.")
    elif predicted_label == "Good":
        st.success("The student is likely to perform well with minor improvements.")
    elif predicted_label == "Average":
        st.warning("The student has average predicted performance and may benefit from improvement strategies.")
    else:
        st.error("The student appears to be at academic risk and needs early intervention.")