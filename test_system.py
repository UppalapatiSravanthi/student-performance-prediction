"""
Comprehensive End-to-End System Test Script
Tests:
  1. Data Loading and Preprocessing
  2. EDA and Metric Calculations
  3. Single Prediction & Recommendations
  4. Batch Prediction & Risk Classification
  5. Artifact Integrity
"""

import os
import sys

# Ensure project root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.data_preprocessing import load_data, clean_data, prepare_data_for_modeling
from src.eda import get_basic_metrics, identify_at_risk_students, get_correlation_matrix
from src.prediction import load_artifacts, predict_single_student, batch_predict
from src.evaluation import evaluate_regression_model, evaluate_classification_model

def run_tests():
    print(">>> Starting System Verification...")
    
    # 1. Dataset verification
    data_path = os.path.join(current_dir, "data", "student_data.csv")
    assert os.path.exists(data_path), f"Dataset missing at {data_path}"
    df = load_data(data_path)
    print(f"[PASS] Dataset loaded successfully: {len(df)} records, {len(df.columns)} columns.")
    assert len(df) >= 1000, "Dataset contains fewer than 1000 records."
    
    # 2. Preprocessing verification
    df_clean = clean_data(df)
    assert df_clean.isnull().sum().sum() == 0, "Missing values remain after clean_data"
    print("[PASS] Missing values handled and cleaned.")
    
    # 3. Model artifacts verification
    models_dir = os.path.join(current_dir, "models")
    artifacts = load_artifacts(models_dir)
    assert "marks_model" in artifacts and "result_model" in artifacts and "scaler" in artifacts
    print("[PASS] Model artifacts loaded successfully.")
    
    # 4. Single student prediction
    sample_student = {
        "Gender": "Female",
        "Age": 20,
        "Study_Hours_Per_Day": 6.5,
        "Attendance_Percentage": 92.0,
        "Previous_Marks": 78.0,
        "Assignment_Score": 82.0,
        "Internal_Marks": 25.0,
        "Sleep_Hours": 7.5,
        "Extracurricular_Activities": "Yes"
    }
    pred_res = predict_single_student(sample_student, artifacts)
    print(f"[PASS] Single Prediction Output: {pred_res['predicted_marks']} marks, Result: {pred_res['predicted_result']}, Category: {pred_res['performance_category']}")
    assert "predicted_marks" in pred_res and "recommendations" in pred_res
    
    # At-risk student prediction test
    at_risk_student = {
        "Gender": "Male",
        "Age": 22,
        "Study_Hours_Per_Day": 1.5,
        "Attendance_Percentage": 52.0,
        "Previous_Marks": 32.0,
        "Assignment_Score": 38.0,
        "Internal_Marks": 9.0,
        "Sleep_Hours": 4.5,
        "Extracurricular_Activities": "No"
    }
    pred_risk = predict_single_student(at_risk_student, artifacts)
    print(f"[PASS] At-Risk Student Prediction: {pred_risk['predicted_marks']} marks, Result: {pred_risk['predicted_result']}, Category: {pred_risk['performance_category']}, Risk: {pred_risk['is_at_risk']}")
    assert pred_risk["is_at_risk"] == True
    
    # 5. Batch prediction test
    df_batch = batch_predict(df.head(20), artifacts)
    assert "Predicted_Marks" in df_batch.columns and "Predicted_Result" in df_batch.columns
    print(f"[PASS] Batch Prediction verified on sample cohort of {len(df_batch)} students.")
    
    # 6. EDA tests
    kpis = get_basic_metrics(df)
    assert kpis["total_students"] == len(df)
    at_risk_df = identify_at_risk_students(df, attendance_thresh=75.0, marks_thresh=50.0)
    print(f"[PASS] Risk analysis flagged {len(at_risk_df)} students at risk.")
    
    print("\n[ALL TESTS PASSED SUCCESSFULLY! The pipeline is completely functional.]")

if __name__ == "__main__":
    run_tests()
