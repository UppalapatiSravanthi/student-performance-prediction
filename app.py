"""
Student Performance Prediction and Analysis System
Streamlit Web Application
--------------------------------------------------
A modern, end-to-end Data Science and Machine Learning dashboard
for analyzing, predicting, and monitoring student academic performance.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Student Performance Prediction & Analysis System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E293B;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-excellent {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-good {
        background-color: #E0F2FE;
        color: #075985;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-average {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-poor {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
    }
    .recommendation-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 12px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Local src imports
from src.data_preprocessing import load_data, clean_data, FEATURE_COLUMNS
from src.eda import (
    get_basic_metrics,
    get_missing_values_report,
    get_statistical_summary,
    get_gender_breakdown,
    get_attendance_brackets_analysis,
    get_study_hours_brackets_analysis,
    get_correlation_matrix,
    identify_at_risk_students
)
from src.prediction import (
    load_artifacts,
    predict_single_student,
    batch_predict,
    get_performance_category
)

@st.cache_data
def get_cached_dataset():
    """Loads and returns dataset."""
    if not os.path.exists(DATA_PATH):
        from generate_dataset import save_dataset
        save_dataset(DATA_PATH)
    return load_data(DATA_PATH)

@st.cache_resource
def get_cached_artifacts():
    """Loads and caches trained models and preprocessor."""
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.pkl")
    if not os.path.exists(metadata_path):
        from train_model import train_and_evaluate_models
        train_and_evaluate_models(DATA_PATH, MODELS_DIR)
    return load_artifacts(MODELS_DIR)

# Sidebar Navigation
st.sidebar.title("Navigation")
st.sidebar.image("https://img.icons8.com/color/96/graduation-cap.png", width=70)
selected_page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Data Analysis",
        "Visualization",
        "Prediction",
        "Student Risk Analysis",
        "Model Performance",
        "Download"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Student Performance System**\n\n"
    "An AI-powered academic performance predictor and risk early warning system."
)

# -------------------------------------------------------------------------------------------------
# PAGE 1: HOME
# -------------------------------------------------------------------------------------------------
if selected_page == "Home":
    st.markdown('<div class="main-header">Student Performance Prediction and Analysis System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">An End-to-End Machine Learning Solution for Academic Analytics, Grade Prediction, and Early Risk Detection</div>',
        unsafe_allow_html=True
    )
    
    df = get_cached_dataset()
    kpis = get_basic_metrics(df)
    
    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Total Students</div><div class="metric-val">{kpis["total_students"]:,}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Average Marks</div><div class="metric-val">{kpis["avg_marks"]}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Pass Percentage</div><div class="metric-val" style="color:#16A34A;">{kpis["pass_pct"]}%</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Fail Percentage</div><div class="metric-val" style="color:#DC2626;">{kpis["fail_pct"]}%</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Avg Study Hours</div><div class="metric-val">{kpis["avg_study_hours"]}h/d</div></div>', unsafe_allow_html=True)
        
    st.markdown("### System Architecture & Workflow")
    st.write(
        """
        This system guides academic institutions, educators, and mentors through a full data science pipeline:
        """
    )
    
    wf1, wf2, wf3, wf4 = st.columns(4)
    with wf1:
        st.info("📊 **1. Data & Analytics**\n\nComprehensive exploration of demographic, study behavior, and test score variables.")
    with wf2:
        st.info("📈 **2. Visual Discovery**\n\nInteractive correlation, distribution, and performance factor visualizations.")
    with wf3:
        st.info("🤖 **3. ML Predictions**\n\nDual model architecture: Regression for expected marks and Classification for Pass/Fail.")
    with wf4:
        st.info("⚠️ **4. Risk Interventions**\n\nEarly-warning indicator alerting mentors about students with attendance or score risks.")

    st.markdown("### Dataset Overview Quick Look")
    st.dataframe(df.head(6), use_container_width=True)

# -------------------------------------------------------------------------------------------------
# PAGE 2: DATA ANALYSIS
# -------------------------------------------------------------------------------------------------
elif selected_page == "Data Analysis":
    st.title("Data Analysis & Summary")
    st.markdown("Inspect raw student records, missing value diagnostics, and subgroup statistics.")
    
    df = get_cached_dataset()
    
    tab1, tab2, tab3, tab4 = st.tabs(["Raw Dataset", "Missing Values & Quality", "Statistical Summary", "Subgroup Analysis"])
    
    with tab1:
        st.subheader("Dataset Records")
        n_rows = st.slider("Number of rows to view:", min_value=5, max_value=len(df), value=15, step=5)
        
        # Simple filters
        filt_col1, filt_col2 = st.columns(2)
        with filt_col1:
            gender_filter = st.multiselect("Filter by Gender:", options=["All", "Male", "Female"], default=["All"])
        with filt_col2:
            result_filter = st.multiselect("Filter by Result:", options=["All", "Pass", "Fail"], default=["All"])
            
        filtered_df = df.copy()
        if "All" not in gender_filter and len(gender_filter) > 0:
            filtered_df = filtered_df[filtered_df["Gender"].isin(gender_filter)]
        if "All" not in result_filter and len(result_filter) > 0:
            filtered_df = filtered_df[filtered_df["Result"].isin(result_filter)]
            
        st.dataframe(filtered_df.head(n_rows), use_container_width=True)
        st.caption(f"Showing {min(n_rows, len(filtered_df))} of {len(filtered_df)} filtered records.")
        
    with tab2:
        st.subheader("Missing Values Report")
        missing_rep = get_missing_values_report(df)
        st.dataframe(missing_rep, use_container_width=True)
        
        has_nulls = df.isnull().sum().sum() > 0
        if has_nulls:
            st.warning("Notice: Synthetic realistic missing entries were detected in some features. The pipeline uses median imputation for numerical data and mode imputation for categorical data to ensure zero information loss and no bias.")
        else:
            st.success("No missing values found in the active dataset.")
            
    with tab3:
        st.subheader("Numerical Features Descriptive Statistics")
        stat_summary = get_statistical_summary(df)
        st.dataframe(stat_summary, use_container_width=True)
        
    with tab4:
        st.subheader("Performance by Demographic & Behavior Groups")
        col_sub1, col_sub2 = st.columns(2)
        
        with col_sub1:
            st.markdown("#### Gender Performance Breakdown")
            gender_df = get_gender_breakdown(df)
            st.dataframe(gender_df, use_container_width=True)
            
            st.markdown("#### Attendance Brackets vs Final Marks")
            att_df = get_attendance_brackets_analysis(df)
            st.dataframe(att_df, use_container_width=True)
            
        with col_sub2:
            st.markdown("#### Study Hours Brackets vs Performance")
            study_df = get_study_hours_brackets_analysis(df)
            st.dataframe(study_df, use_container_width=True)
            
            st.markdown("#### Extracurricular Activities Impact")
            extra_df = df.groupby("Extracurricular_Activities").agg(
                Count=("Student_ID", "count"),
                Avg_Marks=("Final_Marks", "mean"),
                Pass_Rate=("Result", lambda x: (x == "Pass").mean() * 100.0)
            ).reset_index().round(2)
            st.dataframe(extra_df, use_container_width=True)

# -------------------------------------------------------------------------------------------------
# PAGE 3: VISUALIZATION
# -------------------------------------------------------------------------------------------------
elif selected_page == "Visualization":
    st.title("Interactive Visual Explorations")
    st.markdown("Explore key trends, distributions, and multi-variable correlations.")
    
    df = clean_data(get_cached_dataset())
    sns.set_theme(style="whitegrid", palette="muted")
    
    viz_choice = st.selectbox(
        "Select Visualization:",
        [
            "1. Study Hours vs Final Marks",
            "2. Attendance vs Final Marks",
            "3. Previous Marks vs Final Marks",
            "4. Gender-Wise Performance Comparison",
            "5. Pass vs Fail Distribution",
            "6. Feature Correlation Heatmap",
            "7. Final Marks Distribution (Histogram & KDE)",
            "8. Box Plots: Extracurricular & Performance"
        ]
    )
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    if viz_choice == "1. Study Hours vs Final Marks":
        sns.regplot(
            data=df, x="Study_Hours_Per_Day", y="Final_Marks",
            scatter_kws={"alpha": 0.45, "color": "#2563EB"},
            line_kws={"color": "#DC2626", "linewidth": 2.5},
            ax=ax
        )
        ax.set_title("Study Hours per Day vs. Final Marks", fontsize=14, fontweight="bold")
        ax.set_xlabel("Study Hours per Day")
        ax.set_ylabel("Final Marks (0 - 100)")
        st.pyplot(fig)
        st.info("Insight: Consistent daily study hours demonstrate a strong positive correlation with final exam performance.")
        
    elif viz_choice == "2. Attendance vs Final Marks":
        sns.regplot(
            data=df, x="Attendance_Percentage", y="Final_Marks",
            scatter_kws={"alpha": 0.45, "color": "#059669"},
            line_kws={"color": "#EA580C", "linewidth": 2.5},
            ax=ax
        )
        ax.set_title("Classroom Attendance vs. Final Marks", fontsize=14, fontweight="bold")
        ax.set_xlabel("Attendance Percentage (%)")
        ax.set_ylabel("Final Marks (0 - 100)")
        st.pyplot(fig)
        st.info("Insight: Higher attendance percentages significantly reduce the likelihood of scoring below the 50-mark threshold.")

    elif viz_choice == "3. Previous Marks vs Final Marks":
        sns.scatterplot(
            data=df, x="Previous_Marks", y="Final_Marks", hue="Result",
            palette={"Pass": "#16A34A", "Fail": "#DC2626"}, alpha=0.65, ax=ax
        )
        ax.set_title("Previous Academic Marks vs. Final Marks", fontsize=14, fontweight="bold")
        ax.set_xlabel("Previous Semester Marks")
        ax.set_ylabel("Final Marks")
        st.pyplot(fig)
        st.info("Insight: Past academic foundations provide strong predictive signals for current semester results.")

    elif viz_choice == "4. Gender-Wise Performance Comparison":
        sns.boxplot(data=df, x="Gender", y="Final_Marks", palette=["#3B82F6", "#EC4899"], ax=ax, width=0.45)
        ax.set_title("Marks Distribution by Gender", fontsize=14, fontweight="bold")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Final Marks")
        st.pyplot(fig)
        st.info("Insight: Gender distributions show balanced academic potential with comparable medians and spreads across groups.")

    elif viz_choice == "5. Pass vs Fail Distribution":
        res_counts = df["Result"].value_counts()
        colors = ["#16A34A", "#DC2626"]
        ax.pie(
            res_counts, labels=res_counts.index, autopct="%1.1f%%",
            startangle=140, colors=colors, explode=(0.04, 0.08),
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        ax.set_title("Overall Student Result Ratio (Pass / Fail)", fontsize=14, fontweight="bold")
        st.pyplot(fig)
        st.info("Insight: Healthy pass rates exist overall, but targeted interventions can recover the failing percentage.")

    elif viz_choice == "6. Feature Correlation Heatmap":
        plt.close(fig)
        fig_heat, ax_heat = plt.subplots(figsize=(9, 6.5))
        corr = get_correlation_matrix(df)
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax_heat, cbar=True)
        ax_heat.set_title("Correlation Heatmap of Numerical Features", fontsize=14, fontweight="bold")
        st.pyplot(fig_heat)
        st.info("Insight: Previous Marks, Assignment Score, and Internal Marks exhibit the highest positive coefficients with Final Marks.")

    elif viz_choice == "7. Final Marks Distribution (Histogram & KDE)":
        sns.histplot(df["Final_Marks"], kde=True, color="#6366F1", bins=25, ax=ax)
        ax.axvline(40.0, color="#DC2626", linestyle="--", linewidth=2, label="Pass Threshold (40 Marks)")
        ax.axvline(df["Final_Marks"].mean(), color="#16A34A", linestyle="-.", linewidth=2, label=f'Mean ({df["Final_Marks"].mean():.1f})')
        ax.set_title("Distribution of Final Marks with KDE Curve", fontsize=14, fontweight="bold")
        ax.set_xlabel("Final Marks")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)
        st.info("Insight: The distribution approximates a normal bell curve centered around 66 marks.")

    elif viz_choice == "8. Box Plots: Extracurricular & Performance":
        sns.boxplot(data=df, x="Extracurricular_Activities", y="Final_Marks", palette=["#F59E0B", "#10B981"], ax=ax, width=0.45)
        ax.set_title("Final Marks by Extracurricular Participation", fontsize=14, fontweight="bold")
        ax.set_xlabel("Extracurricular Activities")
        ax.set_ylabel("Final Marks")
        st.pyplot(fig)
        st.info("Insight: Students who engage in extracurricular activities maintain strong academic balance without negative impact.")

# -------------------------------------------------------------------------------------------------
# PAGE 4: PREDICTION
# -------------------------------------------------------------------------------------------------
elif selected_page == "Prediction":
    st.title("Student Performance Prediction")
    st.markdown("Enter student academic and behavioral parameters to forecast final marks, result, and category.")
    
    artifacts = get_cached_artifacts()
    
    with st.form("student_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 📚 Academic Metrics")
            previous_marks = st.slider("Previous Marks (0 - 100):", min_value=0.0, max_value=100.0, value=65.0, step=0.5)
            assignment_score = st.slider("Assignment Score (0 - 100):", min_value=0.0, max_value=100.0, value=70.0, step=0.5)
            internal_marks = st.slider("Internal Assessment (0 - 30):", min_value=0.0, max_value=30.0, value=20.0, step=0.5)
            
        with col2:
            st.markdown("##### ⏱️ Study & Behavioral Factors")
            study_hours = st.slider("Daily Study Hours:", min_value=0.5, max_value=12.0, value=4.5, step=0.5)
            attendance = st.slider("Attendance Percentage (%):", min_value=30.0, max_value=100.0, value=82.0, step=0.5)
            sleep_hours = st.slider("Sleep Hours Per Day:", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
            
        with col3:
            st.markdown("##### 👤 Demographic & Lifestyle")
            gender = st.selectbox("Gender:", ["Male", "Female"])
            age = st.number_input("Age:", min_value=15, max_value=35, value=19)
            extracurricular = st.selectbox("Extracurricular Activities:", ["Yes", "No"])
            
        submit_btn = st.form_submit_button("🔮 Predict Student Performance", use_container_width=True)
        
    if submit_btn:
        input_data = {
            "Gender": gender,
            "Age": age,
            "Study_Hours_Per_Day": study_hours,
            "Attendance_Percentage": attendance,
            "Previous_Marks": previous_marks,
            "Assignment_Score": assignment_score,
            "Internal_Marks": internal_marks,
            "Sleep_Hours": sleep_hours,
            "Extracurricular_Activities": extracurricular
        }
        
        prediction = predict_single_student(input_data, artifacts)
        
        st.markdown("### Prediction Results")
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        
        with res_col1:
            st.metric("Predicted Final Marks", f"{prediction['predicted_marks']} / 100")
            
        with res_col2:
            res_val = prediction["predicted_result"]
            badge_color = "#16A34A" if res_val == "Pass" else "#DC2626"
            st.markdown(
                f'<div style="text-align:center; padding:10px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px;">'
                f'<div style="font-size:0.8rem; color:#64748B;">PREDICTED RESULT</div>'
                f'<div style="font-size:1.8rem; font-weight:700; color:{badge_color};">{res_val}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            
        with res_col3:
            cat = prediction["performance_category"]
            cat_classes = {
                "Excellent": "badge-excellent",
                "Good": "badge-good",
                "Average": "badge-average",
                "Poor": "badge-poor"
            }
            css_cls = cat_classes.get(cat, "badge-average")
            st.markdown(
                f'<div style="text-align:center; padding:10px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px;">'
                f'<div style="font-size:0.8rem; color:#64748B;">PERFORMANCE LEVEL</div>'
                f'<div style="margin-top:6px;"><span class="{css_cls}" style="font-size:1.2rem;">{cat}</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )
            
        with res_col4:
            prob = prediction["pass_probability"]
            st.metric("Pass Probability", f"{prob}%")
            
        # Risk alert
        if prediction["is_at_risk"]:
            st.error("⚠️ **Risk Alert**: This student meets at-risk criteria due to low predicted marks or attendance levels. Special attention is advised.")
        else:
            st.success("✅ **Status**: This student is currently on a safe academic trajectory.")
            
        # Recommendations
        st.markdown("#### 💡 Tailored Academic Guidance & Recommendations")
        for rec in prediction["recommendations"]:
            st.markdown(f'<div class="recommendation-box">📌 {rec}</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------
# PAGE 5: STUDENT RISK ANALYSIS
# -------------------------------------------------------------------------------------------------
elif selected_page == "Student Risk Analysis":
    st.title("Student Risk Analysis & Early Intervention")
    st.markdown("Identify vulnerable students early to provide timely academic support and counseling.")
    
    df = get_cached_dataset()
    
    col1, col2 = st.columns(2)
    with col1:
        att_threshold = st.slider("Attendance Threshold (%):", min_value=50.0, max_value=90.0, value=75.0, step=1.0)
    with col2:
        marks_threshold = st.slider("Marks Risk Threshold:", min_value=35.0, max_value=60.0, value=50.0, step=1.0)
        
    risk_df = identify_at_risk_students(df, attendance_thresh=att_threshold, marks_thresh=marks_threshold)
    
    # Summary of Risk
    tot_risk = len(risk_df)
    crit_count = int((risk_df["Risk_Level"] == "Critical").sum())
    pct_risk = round((tot_risk / len(df)) * 100.0, 1)
    
    rk1, rk2, rk3 = st.columns(3)
    with rk1:
        st.metric("Total Flagged Students", tot_risk)
    with rk2:
        st.metric("Critical Risk Cases", crit_count)
    with rk3:
        st.metric("Cohort Percentage At Risk", f"{pct_risk}%")
        
    st.subheader(f"Flagged Students List ({tot_risk} records)")
    display_cols = [
        "Student_ID", "Gender", "Attendance_Percentage", "Study_Hours_Per_Day",
        "Previous_Marks", "Internal_Marks", "Final_Marks", "Result", "Risk_Level"
    ]
    st.dataframe(risk_df[display_cols], use_container_width=True)
    
    st.markdown("### 📋 Recommended Action Plan for Teachers and Counselors")
    st.markdown("""
    1. **Personalized Mentorship Sessions**: Assign academic mentors to students flagged as `Critical` for weekly reviews.
    2. **Attendance Remediation**: Contact students with attendance below 75% to address underlying personal or health issues.
    3. **Remedial Tutorials**: Schedule small-group review sessions for students scoring poorly in internal tests.
    4. **Peer Study Circles**: Pair struggling students with peers in the `Excellent` category to promote collaborative learning.
    """)

# -------------------------------------------------------------------------------------------------
# PAGE 6: MODEL PERFORMANCE
# -------------------------------------------------------------------------------------------------
elif selected_page == "Model Performance":
    st.title("Machine Learning Models & Evaluation")
    st.markdown("Compare regression and classification algorithms and explore feature importance.")
    
    artifacts = get_cached_artifacts()
    metadata = artifacts.get("metadata", {})
    
    tab_reg, tab_clf, tab_feat = st.tabs(["Regression Models (Final Marks)", "Classification Model (Pass/Fail)", "Feature Importance"])
    
    with tab_reg:
        st.subheader("Model Comparison: Linear Regression vs Random Forest Regressor")
        reg_df = metadata.get("reg_comparison", pd.DataFrame())
        st.dataframe(reg_df, use_container_width=True)
        
        st.markdown(f"**Best Performing Model**: `{metadata.get('best_reg_name', 'Random Forest Regressor')}`")
        
        # Actual vs Predicted Plot
        actuals = metadata.get("test_actual_marks", [])
        pred_rf = metadata.get("test_pred_rf", [])
        pred_lr = metadata.get("test_pred_lr", [])
        
        if len(actuals) > 0 and len(pred_rf) > 0:
            st.markdown("#### Actual vs Predicted Marks (Test Set)")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(actuals, pred_rf, alpha=0.6, color="#2563EB", label="Random Forest Predictions")
            ax.scatter(actuals, pred_lr, alpha=0.3, color="#F59E0B", label="Linear Regression Predictions")
            min_val = min(min(actuals), min(pred_rf))
            max_val = max(max(actuals), max(pred_rf))
            ax.plot([min_val, max_val], [min_val, max_val], color="#DC2626", linestyle="--", linewidth=2, label="Ideal Fit (y = x)")
            ax.set_xlabel("Actual Final Marks")
            ax.set_ylabel("Predicted Final Marks")
            ax.set_title("Actual vs. Predicted Final Marks on Test Data", fontsize=13, fontweight="bold")
            ax.legend()
            st.pyplot(fig)
            
    with tab_clf:
        st.subheader("Random Forest Classifier Performance (Pass / Fail)")
        clf_metrics = metadata.get("metrics_clf", {})
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Accuracy", f"{clf_metrics.get('Accuracy', 0)}%")
        with c2:
            st.metric("Precision", f"{clf_metrics.get('Precision', 0)}%")
        with c3:
            st.metric("Recall", f"{clf_metrics.get('Recall', 0)}%")
        with c4:
            st.metric("F1-Score", f"{clf_metrics.get('F1_Score', 0)}%")
            
        cm = metadata.get("cm", None)
        if cm is not None:
            st.markdown("#### Confusion Matrix")
            fig_cm, ax_cm = plt.subplots(figsize=(5, 3.8))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fail (0)", "Pass (1)"],
                yticklabels=["Fail (0)", "Pass (1)"],
                ax=ax_cm
            )
            ax_cm.set_xlabel("Predicted Label")
            ax_cm.set_ylabel("Actual Label")
            ax_cm.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
            st.pyplot(fig_cm)
            
    with tab_feat:
        st.subheader("Feature Importance Analysis (Random Forest)")
        feat_df = metadata.get("feature_importance", pd.DataFrame())
        
        if not feat_df.empty:
            fig_feat, ax_feat = plt.subplots(figsize=(8, 4.5))
            sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis", ax=ax_feat)
            ax_feat.set_title("Feature Importance Ranking in Marks Prediction", fontsize=13, fontweight="bold")
            ax_feat.set_xlabel("Relative Importance")
            ax_feat.set_ylabel("Feature Name")
            st.pyplot(fig_feat)
            st.dataframe(feat_df, use_container_width=True)

# -------------------------------------------------------------------------------------------------
# PAGE 7: DOWNLOAD
# -------------------------------------------------------------------------------------------------
elif selected_page == "Download":
    st.title("Data and Reports Export Center")
    st.markdown("Download original datasets, prediction outputs, and risk assessments for offline analysis.")
    
    df = get_cached_dataset()
    artifacts = get_cached_artifacts()
    
    st.subheader("1. Original Student Dataset")
    st.write(f"Complete dataset containing {len(df)} records.")
    csv_raw = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Raw Dataset (student_data.csv)",
        data=csv_raw,
        file_name="student_data.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("---")
    st.subheader("2. Full Batch Predictions & Categorization")
    st.write("Augmented dataset with predicted marks, pass/fail result, and performance category.")
    pred_df = batch_predict(df, artifacts)
    csv_pred = pred_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Predictions Report (student_predictions.csv)",
        data=csv_pred,
        file_name="student_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("---")
    st.subheader("3. At-Risk Students Action List")
    st.write("Targeted list of students flagged for low attendance or marks risk.")
    risk_df = identify_at_risk_students(df, attendance_thresh=75.0, marks_thresh=50.0)
    csv_risk = risk_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download At-Risk Students Report (at_risk_students.csv)",
        data=csv_risk,
        file_name="at_risk_students.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("---")
    st.subheader("4. Model Performance Metrics Report")
    st.write("Evaluation benchmarks for Linear Regression, Random Forest Regressor, and Classifier.")
    metadata = artifacts.get("metadata", {})
    reg_df = metadata.get("reg_comparison", pd.DataFrame())
    csv_metrics = reg_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Model Evaluation Metrics (model_evaluation_metrics.csv)",
        data=csv_metrics,
        file_name="model_evaluation_metrics.csv",
        mime="text/csv",
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #94A3B8; font-size: 0.85rem;">'
    'Student Performance Prediction and Analysis System • Built with Python, Scikit-Learn & Streamlit'
    '</div>',
    unsafe_allow_html=True
)
