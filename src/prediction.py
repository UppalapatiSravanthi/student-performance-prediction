"""
Prediction and Inference Module
Loads trained model artifacts and provides single-student and batch
prediction capabilities with performance categorization and recommendations.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List

PERFORMANCE_CATEGORIES = {
    "Excellent": {"min": 80.0, "color": "#28a745", "badge": "success"},
    "Good": {"min": 65.0, "color": "#17a2b8", "badge": "info"},
    "Average": {"min": 50.0, "color": "#ffc107", "badge": "warning"},
    "Poor": {"min": 0.0, "color": "#dc3545", "badge": "danger"}
}

def load_artifacts(models_dir: str) -> Dict[str, Any]:
    """
    Loads saved model binaries and scaler from models directory.
    """
    marks_model_path = os.path.join(models_dir, "marks_model.pkl")
    linear_model_path = os.path.join(models_dir, "linear_model.pkl")
    result_model_path = os.path.join(models_dir, "result_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    metadata_path = os.path.join(models_dir, "model_metadata.pkl")
    
    if not (os.path.exists(marks_model_path) and os.path.exists(result_model_path) and os.path.exists(scaler_path)):
        raise FileNotFoundError(
            f"Model artifacts not found in '{models_dir}'. Please run train_model.py first."
        )
        
    marks_model = joblib.load(marks_model_path)
    result_model = joblib.load(result_model_path)
    scaler = joblib.load(scaler_path)
    
    linear_model = joblib.load(linear_model_path) if os.path.exists(linear_model_path) else None
    metadata = joblib.load(metadata_path) if os.path.exists(metadata_path) else {}
    
    return {
        "marks_model": marks_model,
        "linear_model": linear_model,
        "result_model": result_model,
        "scaler": scaler,
        "metadata": metadata
    }

def get_performance_category(marks: float) -> str:
    """
    Categorizes student score into Excellent, Good, Average, or Poor.
    """
    if marks >= 80.0:
        return "Excellent"
    elif marks >= 65.0:
        return "Good"
    elif marks >= 50.0:
        return "Average"
    else:
        return "Poor"

def generate_recommendations(input_dict: Dict[str, Any], predicted_marks: float) -> List[str]:
    """
    Produces actionable advice tailored to student input features and predicted score.
    """
    recommendations = []
    
    study_hrs = float(input_dict.get("Study_Hours_Per_Day", 0))
    attendance = float(input_dict.get("Attendance_Percentage", 0))
    assignment = float(input_dict.get("Assignment_Score", 0))
    internal = float(input_dict.get("Internal_Marks", 0))
    sleep = float(input_dict.get("Sleep_Hours", 7))
    extracurricular = input_dict.get("Extracurricular_Activities", "No")
    
    if study_hrs < 3.0:
        recommendations.append("Increase daily focused study time to at least 3.5 - 5.0 hours.")
    elif study_hrs > 8.0:
        recommendations.append("Study hours are high. Avoid burnout by adopting structured breaks (Pomodoro technique).")
        
    if attendance < 75.0:
        recommendations.append("Attendance is under 75%. Regular classroom attendance is critical to grasp core concepts.")
        
    if assignment < 60.0:
        recommendations.append("Dedicate more effort to homework and assignment submissions to strengthen fundamentals.")
        
    if internal < 18.0:
        recommendations.append("Seek instructor consultation or tutorial sessions to address weak topics in internal tests.")
        
    if sleep < 6.0:
        recommendations.append("Maintain at least 7 hours of sleep. Insufficient sleep severely degrades cognitive recall.")
    elif sleep > 9.0:
        recommendations.append("Optimize sleep routine to 7-8 hours to increase daytime alertness and study productivity.")
        
    if extracurricular == "No":
        recommendations.append("Participating in co-curricular or physical activities helps relieve academic stress.")
        
    if predicted_marks >= 80.0:
        recommendations.append("Outstanding performance! Maintain this consistency and consider mentoring peers.")
    elif predicted_marks < 50.0:
        recommendations.append("Immediate academic intervention recommended. Create a personalized remediation study schedule.")
        
    return recommendations

def predict_single_student(
    input_dict: Dict[str, Any], 
    artifacts: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Performs inference for an individual student record.
    
    Parameters:
        input_dict (dict): Feature inputs from UI or API.
        artifacts (dict): Loaded models and scaler.
        
    Returns:
        dict: Predictions, probabilities, categories, and suggestions.
    """
    from src.data_preprocessing import transform_single_input
    
    scaler = artifacts["scaler"]
    marks_model = artifacts["marks_model"]
    result_model = artifacts["result_model"]
    
    # Preprocess and scale
    scaled_df = transform_single_input(input_dict, scaler)
    
    # Marks prediction
    raw_marks_pred = marks_model.predict(scaled_df)[0]
    final_marks_pred = float(np.clip(raw_marks_pred, 0.0, 100.0).round(1))
    
    # Result prediction (Pass/Fail)
    raw_result_pred = result_model.predict(scaled_df)[0]
    result_str = "Pass" if raw_result_pred == 1 else "Fail"
    
    # Class probability
    if hasattr(result_model, "predict_proba"):
        probs = result_model.predict_proba(scaled_df)[0]
        pass_prob = round(float(probs[1]) * 100.0, 1)
        fail_prob = round(float(probs[0]) * 100.0, 1)
    else:
        pass_prob = 100.0 if result_str == "Pass" else 0.0
        fail_prob = 100.0 - pass_prob
        
    category = get_performance_category(final_marks_pred)
    recommendations = generate_recommendations(input_dict, final_marks_pred)
    
    # Risk assessment
    att = float(input_dict.get("Attendance_Percentage", 100))
    is_at_risk = (att < 75.0) or (final_marks_pred < 50.0) or (result_str == "Fail")
    
    return {
        "predicted_marks": final_marks_pred,
        "predicted_result": result_str,
        "pass_probability": pass_prob,
        "fail_probability": fail_prob,
        "performance_category": category,
        "is_at_risk": is_at_risk,
        "recommendations": recommendations
    }

def batch_predict(df: pd.DataFrame, artifacts: Dict[str, Any]) -> pd.DataFrame:
    """
    Generates predictions for a dataframe of students.
    """
    from src.data_preprocessing import clean_data, encode_features, FEATURE_COLUMNS
    
    scaler = artifacts["scaler"]
    marks_model = artifacts["marks_model"]
    result_model = artifacts["result_model"]
    
    df_clean = clean_data(df)
    df_encoded = encode_features(df_clean)
    
    X = df_encoded[FEATURE_COLUMNS]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURE_COLUMNS, index=df.index)
    
    preds_marks = np.clip(marks_model.predict(X_scaled), 0.0, 100.0).round(1)
    preds_result_num = result_model.predict(X_scaled)
    preds_result = ["Pass" if p == 1 else "Fail" for p in preds_result_num]
    
    df_out = df.copy()
    df_out["Predicted_Marks"] = preds_marks
    df_out["Predicted_Result"] = preds_result
    df_out["Performance_Category"] = [get_performance_category(m) for m in preds_marks]
    
    # Risk identification
    df_out["At_Risk"] = (
        (df_out["Attendance_Percentage"] < 75.0) | 
        (df_out["Predicted_Marks"] < 50.0) | 
        (df_out["Predicted_Result"] == "Fail")
    ).map({True: "Yes", False: "No"})
    
    return df_out
