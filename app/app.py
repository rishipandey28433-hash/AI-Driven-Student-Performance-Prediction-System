"""
EduPredict - AI-Driven Student Performance Prediction and Early Warning System.
Main Streamlit Application.
Features:
1. Professional Student Identification & Prediction Form
2. Multi-record student history with SQLite persistent storage
3. Search, filter, sorting, and drill-down historical progression
4. PDF & CSV report generation with educational disclaimers
5. Batch CSV prediction with leading-zero preservation and duplicate detection
6. Local privacy and data governance
"""

import os
import sys
import io
import pandas as pd
import numpy as np
import streamlit as st

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.database import (
    init_db,
    save_prediction,
    get_all_predictions,
    get_latest_predictions_per_student,
    get_student_history,
    get_student,
    get_prediction_by_id,
    delete_prediction,
    get_database_stats,
    predictions_to_dataframe,
    get_db_path
)
from src.prediction_engine import (
    predict_single_student,
    validate_student_identity,
    ModelArtifacts
)
from src.report_generator import (
    generate_student_pdf,
    generate_student_csv,
    generate_batch_csv,
    DISCLAIMER_TEXT
)
from src.batch_prediction import (
    read_csv_preserving_identity,
    process_batch_predictions,
    save_batch_records
)

# Initialize database tables on startup
init_db()

# Load model artifacts safely
try:
    artifacts = ModelArtifacts.get_instance()
except Exception as e:
    st.error(f"Failed to load machine learning model artifacts: {e}")
    st.stop()

# ==============================================================================
# STREAMLIT PAGE CONFIG & PROFESSIONAL THEME
# ==============================================================================
st.set_page_config(
    page_title="EduPredict – Student Performance & Early Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling - Full Dark & Light Mode Theme Adaptive
st.markdown("""
<style>
    /* Metric container styling - Theme adaptive */
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.25) !important;
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(128, 128, 128, 0.45) !important;
    }
    div[data-testid="stMetric"] label {
        color: var(--text-color) !important;
        font-weight: 500 !important;
        opacity: 0.85;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-color) !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
        color: var(--text-color) !important;
    }
    
    /* Custom card classes - Theme adaptive */
    .edu-card {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        padding: 18px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }
    .student-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-excellent { background-color: rgba(34, 197, 94, 0.2); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.4); }
    .badge-good { background-color: rgba(59, 130, 246, 0.2); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.4); }
    .badge-average { background-color: rgba(234, 179, 8, 0.2); color: #eab308; border: 1px solid rgba(234, 179, 8, 0.4); }
    .badge-poor { background-color: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.4); }
    
    .badge-low-risk { background-color: rgba(34, 197, 94, 0.2); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.4); }
    .badge-med-risk { background-color: rgba(234, 179, 8, 0.2); color: #eab308; border: 1px solid rgba(234, 179, 8, 0.4); }
    .badge-high-risk { background-color: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.4); }
    
    .disclaimer-box {
        font-size: 0.85rem;
        color: var(--text-color);
        opacity: 0.85;
        background-color: var(--secondary-background-color);
        border-left: 3px solid #3b82f6;
        padding: 12px 16px;
        margin-top: 25px;
        border-radius: 0 6px 6px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.title("🎓 EduPredict")
st.sidebar.caption("AI-Driven Student Early Warning & Performance System")
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "🔮 New Student Prediction",
        "📁 Student Records & History",
        "📊 Batch Prediction (CSV)",
        "ℹ️ System Documentation & Storage"
    ],
    index=0
)

st.sidebar.markdown("---")
# Quick Database Summary in Sidebar
try:
    stats = get_database_stats()
    st.sidebar.markdown("### 📈 Database Overview")
    st.sidebar.write(f"• **Enrolled Students:** {stats['total_students']}")
    st.sidebar.write(f"• **Recorded Predictions:** {stats['total_predictions']}")
    st.sidebar.write(f"• **Flagged High Risk:** {stats['high_risk_count']}")
    st.sidebar.write(f"• **Average Score:** {stats['avg_score']}/100")
except Exception:
    pass

st.sidebar.markdown("---")
st.sidebar.caption("🔒 Local Persistence • Privacy Preserved • SQLite")


# ==============================================================================
# 1. VIEW: NEW STUDENT PREDICTION
# ==============================================================================
if nav_choice == "🔮 New Student Prediction":
    st.title("🔮 Student Performance Prediction")
    st.markdown(
        "Evaluate academic risk factors and predict student performance category "
        "using AI-driven early intervention models."
    )
    st.markdown("---")

    # SECTION 1: STUDENT INFORMATION FORM
    st.subheader("👤 Student Information")
    st.caption("Enter the student's unique identity details. Identity fields are strictly kept separate from prediction features.")

    col_id1, col_id2 = st.columns(2)
    with col_id1:
        student_name_input = st.text_input(
            "Student Full Name *",
            value="",
            placeholder="e.g. Johnathan Smith",
            help="Required: Enter the student's legal or registered full name."
        )

    with col_id2:
        roll_number_input = st.text_input(
            "Roll Number / Student ID *",
            value="",
            placeholder="e.g. 0042 or CS-2026-089",
            help="Required & Unique: Preserves leading zeros. Multiple predictions for the same roll number build historical tracking."
        )

    # Check if student already exists in DB to give helpful contextual feedback
    clean_roll_preview = str(roll_number_input).strip()
    if clean_roll_preview:
        existing_student = get_student(clean_roll_preview)
        if existing_student:
            st.info(
                f"ℹ️ **Existing Student Found:** Recognized as **{existing_student['full_name']}** "
                f"(Roll #{clean_roll_preview}). This prediction will be stored as a new dated historical record."
            )

    st.markdown("---")

    # SECTION 2: ACADEMIC & BEHAVIORAL FACTORS FORM
    st.subheader("📘 Academic & Behavioral Factors")

    col_ac1, col_ac2, col_ac3 = st.columns(3)

    with col_ac1:
        st.markdown("**📚 Study & Attendance**")
        hours_studied = st.slider("Hours Studied (weekly)", min_value=0, max_value=40, value=15, step=1)
        attendance = st.slider("Attendance Rate (%)", min_value=0, max_value=100, value=75, step=1)
        previous_scores = st.slider("Previous Scores (%)", min_value=0, max_value=100, value=65, step=1)
        tutoring_sessions = st.slider("Tutoring Sessions (monthly)", min_value=0, max_value=10, value=2, step=1)

    with col_ac2:
        st.markdown("**🏃 Health & Lifestyle**")
        sleep_hours = st.slider("Daily Sleep Hours", min_value=0, max_value=12, value=7, step=1)
        physical_activity = st.slider("Physical Activity (hours/week)", min_value=0, max_value=15, value=3, step=1)
        internet_access = st.selectbox("Internet Access at Home", ["Yes", "No"], index=0)

    with col_ac3:
        st.markdown("**🏡 Environment & Support**")
        parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"], index=1)
        access_to_resources = st.selectbox("Access to Study Resources", ["Low", "Medium", "High"], index=1)
        motivation_level = st.selectbox("Student Motivation Level", ["Low", "Medium", "High"], index=1)

    st.markdown("---")

    # Prediction Action
    btn_predict = st.button("🚀 Run Performance Prediction", type="primary", use_container_width=True)

    if btn_predict:
        # Validate student identity
        is_valid, validation_error = validate_student_identity(student_name_input, roll_number_input)
        if not is_valid:
            st.error(f"❌ Validation Error: {validation_error}")
            st.warning("Prediction cannot proceed until student name and roll number are provided.")
            st.stop()

        academic_inputs = {
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

        with st.spinner("Analyzing student factors and generating prediction..."):
            try:
                result = predict_single_student(
                    student_name=student_name_input,
                    roll_number=roll_number_input,
                    academic_inputs=academic_inputs,
                    artifacts=artifacts
                )
                # Save to persistent database
                pred_id = save_prediction(result)
                result["id"] = pred_id
                st.session_state["active_result"] = result
                st.success(f"✅ Prediction generated and saved successfully! (Record ID: #{pred_id})")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.stop()

    # DISPLAY ACTIVE RESULT IF AVAILABLE
    if "active_result" in st.session_state:
        res = st.session_state["active_result"]

        st.markdown("### 📊 Prediction Results & Early Warning Analysis")

        # Prominent Student Identity Banner
        st.markdown(f"""
        <div class="edu-card" style="border-left: 5px solid #3b82f6;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <h2 style="margin: 0; font-size: 1.6rem; font-weight: 700; color: var(--text-color);">{res['student_name']}</h2>
                    <p style="margin: 6px 0 0 0; color: var(--text-color); opacity: 0.85; font-size: 0.95rem;">
                        <b>Roll Number:</b> <code style="padding: 2px 8px; border-radius: 4px;">{res['roll_number']}</code> &nbsp;•&nbsp; 
                        <b>Record ID:</b> #{res.get('id', 'N/A')} &nbsp;•&nbsp; 
                        <b>Assessment Time:</b> Just now
                    </p>
                </div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <span class="student-badge badge-{res['predicted_category'].lower()}">Predicted: {res['predicted_category']}</span>
                    <span class="student-badge badge-{'high' if 'High' in res['risk_level'] else ('med' if 'Medium' in res['risk_level'] else 'low')}-risk">{res['risk_level']}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 4 Metrics Columns
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Predicted Category", res["predicted_category"])
        with m2:
            st.metric("Risk Level", res["risk_level"])
        with m3:
            st.metric("Performance Index", f"{res['performance_score']}/100")
        with m4:
            st.metric("Model Confidence", f"{res['confidence']}%")

        st.progress(res["performance_score"] / 100)

        # Class Probability Breakdown
        if res.get("class_probabilities"):
            with st.expander("🔍 Model Probability Breakdown", expanded=False):
                prob_cols = st.columns(len(res["class_probabilities"]))
                for col, (cls_name, prob_val) in zip(prob_cols, res["class_probabilities"].items()):
                    col.metric(f"Probability: {cls_name}", f"{prob_val}%")

        # Academic Inputs Summary Table
        st.subheader("🧾 Recorded Academic Profile")
        inputs_data = res["academic_inputs"]
        summary_df = pd.DataFrame([
            {"Factor": "Hours Studied (weekly)", "Value": f"{inputs_data['Hours_Studied']} hrs"},
            {"Factor": "Attendance Rate", "Value": f"{inputs_data['Attendance']}%"},
            {"Factor": "Daily Sleep Hours", "Value": f"{inputs_data['Sleep_Hours']} hrs"},
            {"Factor": "Previous Exam Scores", "Value": f"{inputs_data['Previous_Scores']}%"},
            {"Factor": "Tutoring Sessions", "Value": f"{inputs_data['Tutoring_Sessions']}"},
            {"Factor": "Physical Activity", "Value": f"{inputs_data['Physical_Activity']} hrs/wk"},
            {"Factor": "Internet Access", "Value": f"{inputs_data['Internet_Access']}"},
            {"Factor": "Parental Involvement", "Value": f"{inputs_data['Parental_Involvement']}"},
            {"Factor": "Access to Resources", "Value": f"{inputs_data['Access_to_Resources']}"},
            {"Factor": "Motivation Level", "Value": f"{inputs_data['Motivation_Level']}"}
        ])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Top Reasons and Suggestions
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.subheader("⚠ Identified Risk Factors")
            for reason in res["risk_reasons"]:
                st.warning(f"• {reason}")

        with col_res2:
            st.subheader("💡 Recommended Academic Action Plan")
            for suggestion in res["suggestions"]:
                st.info(f"• {suggestion}")

        # Visual Analysis
        st.subheader("📈 Factor Analysis Chart")
        chart_data = pd.DataFrame({
            "Metric": ["Attendance (%)", "Study Hours (x2)", "Previous Scores (%)", "Sleep Hours (x10)"],
            "Score": [
                inputs_data["Attendance"],
                inputs_data["Hours_Studied"] * 2,
                inputs_data["Previous_Scores"],
                inputs_data["Sleep_Hours"] * 10
            ]
        })
        st.bar_chart(chart_data.set_index("Metric"))

        # Report Downloads Section
        st.markdown("---")
        st.subheader("📄 Export & Download Student Report")
        col_down1, col_down2 = st.columns(2)

        with col_down1:
            pdf_bytes = generate_student_pdf(res)
            st.download_button(
                label="📥 Download Official PDF Report",
                data=pdf_bytes,
                file_name=f"EduPredict_Report_{res['roll_number']}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        with col_down2:
            csv_str = generate_student_csv(res)
            st.download_button(
                label="📥 Download Student CSV Record",
                data=csv_str,
                file_name=f"EduPredict_Record_{res['roll_number']}.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Mandatory Disclaimer
        st.markdown(f'<div class="disclaimer-box">{DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)


# ==============================================================================
# 2. VIEW: STUDENT RECORDS & PREDICTION HISTORY
# ==============================================================================
elif nav_choice == "📁 Student Records & History":
    st.title("📁 Student Records & Prediction History")
    st.markdown(
        "Browse, search, filter, and inspect historical student predictions. "
        "Each prediction is preserved as a dated record across application restarts."
    )
    st.markdown("---")

    # KPI Summary Cards
    stats = get_database_stats()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Enrolled Students", stats["total_students"])
    k2.metric("Total Predictions", stats["total_predictions"])
    k3.metric("High Risk Flagged", stats["high_risk_count"])
    k4.metric("Avg Performance Index", f"{stats['avg_score']}/100")

    st.markdown("---")

    # Search and Filter Toolbar
    f1, f2, f3, f4 = st.columns([3, 2, 2, 2])
    with f1:
        search_query = st.text_input("🔍 Search Student Name or Roll Number", placeholder="Type name or roll number...")
    with f2:
        category_filter = st.selectbox("Category Filter", ["All", "Excellent", "Good", "Average", "Poor"], index=0)
    with f3:
        risk_filter = st.selectbox("Risk Level Filter", ["All", "Low Risk", "Medium Risk", "High Risk"], index=0)
    with f4:
        sort_order = st.selectbox("Sort by Date", ["Newest First", "Oldest First"], index=0)

    # View Mode Toggle
    view_mode = st.radio(
        "Display Mode",
        ["📌 Latest Prediction per Student", "📜 All Historical Predictions"],
        horizontal=True
    )

    order_param = "desc" if sort_order == "Newest First" else "asc"

    if view_mode == "📌 Latest Prediction per Student":
        records = get_latest_predictions_per_student(
            search=search_query if search_query else None,
            category=category_filter,
            risk_level=risk_filter,
            sort_order=order_param
        )
    else:
        records = get_all_predictions(
            search=search_query if search_query else None,
            category=category_filter,
            risk_level=risk_filter,
            sort_order=order_param
        )

    st.write(f"Showing **{len(records)}** matching records.")

    if records:
        df_display = predictions_to_dataframe(records)

        # Simplified columns for clean table display
        table_cols = [
            "ID", "Roll Number", "Student Name", "Prediction Date",
            "Predicted Performance", "Risk Level", "Performance Score", "Confidence (%)"
        ]
        st.dataframe(df_display[table_cols], use_container_width=True, hide_index=True)

        # Download Table CSV
        batch_csv = generate_batch_csv(records)
        st.download_button(
            label="📥 Export Filtered Table to CSV",
            data=batch_csv,
            file_name="EduPredict_Filtered_Records.csv",
            mime="text/csv"
        )

        st.markdown("---")

        # Record Inspector & Timeline Progression
        st.subheader("🔍 Record Inspector & Student History Drill-down")

        record_options = {
            f"Record #{r['id']} – {r['student_name']} (Roll: {r['roll_number']}) [{r['prediction_date']}]": r['id']
            for r in records
        }

        selected_label = st.selectbox("Select a Record to Inspect:", list(record_options.keys()))
        selected_id = record_options[selected_label]
        target_record = get_prediction_by_id(selected_id)

        if target_record:
            st.markdown(f"""
            <div class="edu-card" style="border-left: 5px solid #3b82f6;">
                <h3 style="margin: 0; font-size: 1.3rem; font-weight: 700; color: var(--text-color);">{target_record['student_name']} (Roll #{target_record['roll_number']})</h3>
                <p style="margin: 6px 0 0 0; color: var(--text-color); opacity: 0.85; font-size: 0.95rem;">
                   <b>Prediction Date:</b> {target_record['prediction_date']} &nbsp;•&nbsp; 
                   <b>Category:</b> {target_record['predicted_category']} &nbsp;•&nbsp; 
                   <b>Risk:</b> {target_record['risk_level']} &nbsp;•&nbsp; 
                   <b>Performance Score:</b> {target_record['performance_score']}/100 &nbsp;•&nbsp; 
                   <b>Confidence:</b> {target_record['confidence']}%</p>
            </div>
            """, unsafe_allow_html=True)

            # Student Historical Progression Timeline
            student_history = get_student_history(target_record["roll_number"])
            if len(student_history) > 1:
                st.markdown(f"#### 📈 Historical Progression for {target_record['student_name']} ({len(student_history)} records)")
                history_df = pd.DataFrame([
                    {
                        "Date": h["prediction_date"],
                        "Performance Score": h["performance_score"],
                        "Category": h["predicted_category"],
                        "Risk Level": h["risk_level"]
                    }
                    for h in reversed(student_history)
                ])
                st.line_chart(history_df.set_index("Date")["Performance Score"])

            # Download buttons for selected record
            col_rec_d1, col_rec_d2 = st.columns(2)
            with col_rec_d1:
                pdf_bytes_hist = generate_student_pdf(target_record)
                st.download_button(
                    label=f"📥 Download PDF Report for Record #{target_record['id']}",
                    data=pdf_bytes_hist,
                    file_name=f"EduPredict_Report_{target_record['roll_number']}_{target_record['id']}.pdf",
                    mime="application/pdf",
                    key=f"dl_pdf_{target_record['id']}"
                )
            with col_rec_d2:
                csv_bytes_hist = generate_student_csv(target_record)
                st.download_button(
                    label=f"📥 Download CSV for Record #{target_record['id']}",
                    data=csv_bytes_hist,
                    file_name=f"EduPredict_Record_{target_record['roll_number']}_{target_record['id']}.csv",
                    mime="text/csv",
                    key=f"dl_csv_{target_record['id']}"
                )

            # Record Deletion Zone with Confirmation
            with st.expander("⚠ Remove / Delete Record (Danger Zone)", expanded=False):
                st.warning("Removing a prediction record is permanent and cannot be undone.")
                confirm_del = st.checkbox(
                    f"I confirm that I want to delete Record #{target_record['id']} for {target_record['student_name']}.",
                    key=f"del_chk_{target_record['id']}"
                )
                if st.button("🗑️ Permanently Delete Record", type="secondary", disabled=not confirm_del):
                    success = delete_prediction(target_record["id"])
                    if success:
                        st.success(f"Record #{target_record['id']} was permanently deleted.")
                        st.rerun()
                    else:
                        st.error("Failed to delete record. Please check database permissions.")

    else:
        st.info("No student records found matching the specified filters or search query.")

    st.markdown(f'<div class="disclaimer-box">{DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)


# ==============================================================================
# 3. VIEW: BATCH PREDICTION (CSV)
# ==============================================================================
elif nav_choice == "📊 Batch Prediction (CSV)":
    st.title("📊 Batch Student Prediction")
    st.markdown(
        "Upload a CSV file of multiple students to evaluate performance categories in bulk. "
        "Student identity attributes (name, roll number) are preserved with leading zeros and isolated from ML features."
    )
    st.markdown("---")

    # Sample Template Generator
    sample_csv_data = """Roll_Number,Student_Name,Hours_Studied,Attendance,Sleep_Hours,Previous_Scores,Tutoring_Sessions,Physical_Activity,Internet_Access,Parental_Involvement,Access_to_Resources,Motivation_Level
0010,Alice Walker,18,85,7,72,2,3,Yes,Medium,Medium,Medium
0020,Bob Roberts,8,55,5,45,0,1,No,Low,Low,Low
0030,Catherine Hall,28,95,8,92,1,4,Yes,High,High,High
"""
    st.download_button(
        label="📥 Download Sample Batch CSV Template",
        data=sample_csv_data,
        file_name="EduPredict_Batch_Template.csv",
        mime="text/csv",
        help="Use this template to format your batch prediction CSV correctly."
    )

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Student Batch CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read CSV preserving leading zeros
            raw_bytes = uploaded_file.getvalue()
            batch_df = read_csv_preserving_identity(raw_bytes)
            st.write(f"📁 Successfully loaded **{len(batch_df)}** rows from `{uploaded_file.name}`.")

            # Process predictions
            results_df, summary, records = process_batch_predictions(batch_df, artifacts=artifacts)

            # Display Validation & Identification Summary
            st.subheader("📋 Ingestion & Validation Summary")
            v1, v2, v3 = st.columns(3)
            v1.metric("Total Rows Ingested", summary["total_rows"])
            v2.metric("Valid Predictions", summary["valid_count"])
            v3.metric("Invalid / Skipped Rows", summary["invalid_count"])

            if summary["duplicate_roll_numbers"]:
                st.warning(
                    f"⚠ **Duplicate Roll Numbers Detected in File:** {', '.join(summary['duplicate_roll_numbers'])}. "
                    "Each entry will be evaluated individually without overwriting."
                )

            if summary["errors"]:
                with st.expander("⚠ View Row Validation Errors", expanded=True):
                    for err in summary["errors"]:
                        st.error(f"Row {err['row']} ({err['student_name']} - Roll: {err['roll_number']}): {err['reasons']}")

            # Display Batch Results
            if not results_df.empty:
                st.subheader("📊 Batch Prediction Results")
                st.dataframe(results_df, use_container_width=True)

                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    batch_export_csv = generate_batch_csv(records)
                    st.download_button(
                        label="📥 Download Enriched Batch Predictions (CSV)",
                        data=batch_export_csv,
                        file_name=f"EduPredict_Batch_Results_{uploaded_file.name}",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True
                    )

                with col_b2:
                    if st.button("💾 Save All Valid Batch Predictions to Database", use_container_width=True):
                        saved_count = save_batch_records(records)
                        st.success(f"Successfully saved {saved_count} records to the persistent database!")

        except Exception as e:
            st.error(f"Error processing batch CSV: {e}")

    st.markdown(f'<div class="disclaimer-box">{DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)


# ==============================================================================
# 4. VIEW: SYSTEM DOCUMENTATION & STORAGE
# ==============================================================================
elif nav_choice == "ℹ️ System Documentation & Storage":
    st.title("ℹ️ System Documentation & Storage Architecture")
    st.markdown("Detailed reference for storage paths, backup procedures, model mechanics, and data privacy policies.")
    st.markdown("---")

    st.subheader("💾 Persistent SQLite Storage")
    db_file_path = get_db_path()
    st.code(f"Database File Location: {db_file_path}", language="bash")

    st.markdown("""
    EduPredict stores student identities and prediction records in a local **SQLite** database.
    - **ACID Compliant:** Safe concurrent transactions and foreign key integrity.
    - **Fully Parameterized Queries:** Protects against SQL injection attacks.
    - **String Preservation:** Roll numbers are stored as exact strings so leading zeros (e.g., `0042`) are never truncated.
    - **Historical Tracking:** New predictions for existing roll numbers are stored as distinct dated entries, allowing longitudinal analysis.
    """)

    st.markdown("#### 📦 Backup & Recovery Procedure")
    st.markdown("""
    To back up the prediction records:
    1. Locate the database file at `data/edupredict.db`.
    2. Make a copy of `edupredict.db` while the application is idle.
    3. To restore, place your backup copy back into `data/edupredict.db`.
    """)

    st.markdown("#### 🗑️ Reset & Deletion Procedure")
    st.markdown("""
    To delete records:
    - **Individual Record:** Navigate to **Student Records & History**, select the record, expand the Danger Zone, and confirm deletion.
    - **Full Database Reset:** Stop the application and remove the `data/edupredict.db` file. The application will automatically initialize a fresh, empty schema on the next launch.
    """)

    st.markdown("---")
    st.subheader("🔒 Privacy & Security Guarantees")
    st.markdown("""
    - **Zero External Telemetry:** All inference, database operations, and PDF/CSV reports run 100% locally. No student records are ever transmitted over the Internet.
    - **Git Version Control Exclusion:** `*.db` and `*.sqlite` patterns are strictly included in `.gitignore` to guarantee real student information is never pushed to public repositories.
    - **Model Feature Isolation:** Student identities (names, roll numbers) are used strictly for record attribution and are **never** passed into the scikit-learn machine learning pipeline.
    """)

    st.markdown("---")
    st.subheader("🧠 Machine Learning Model Specifications")
    st.markdown("""
    - **Model Type:** Random Forest Classifier (200 trees, scikit-learn).
    - **Input Features (10):**
      1. `Hours_Studied`: Weekly study hours (0-40)
      2. `Attendance`: Class attendance percentage (0-100)
      3. `Sleep_Hours`: Daily sleep duration (0-12)
      4. `Previous_Scores`: Historical examination marks (0-100)
      5. `Tutoring_Sessions`: Monthly tutoring lessons (0-10)
      6. `Physical_Activity`: Weekly exercise hours (0-15)
      7. `Internet_Access`: Home internet connectivity (Yes/No)
      8. `Parental_Involvement`: Guardian engagement (Low/Medium/High)
      9. `Access_to_Resources`: Study material availability (Low/Medium/High)
      10. `Motivation_Level`: Student drive (Low/Medium/High)
    - **Target Performance Classes:** `Excellent`, `Good`, `Average`, `Poor`.
    - **Confidence Score:** Derived from model class probability distribution via `predict_proba`.
    """)

    st.markdown(f'<div class="disclaimer-box">{DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)