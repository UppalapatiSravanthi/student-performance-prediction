"""
Model Evaluation Module
Computes regression metrics (MAE, MSE, RMSE, R2) and classification
metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
Provides utility functions to format comparison tables and residual analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Any, Tuple

def evaluate_regression_model(y_true, y_pred, model_name: str = "Model") -> Dict[str, Any]:
    """
    Computes regression evaluation metrics: MAE, MSE, RMSE, and R2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "Model": model_name,
        "MAE": round(float(mae), 3),
        "MSE": round(float(mse), 3),
        "RMSE": round(float(rmse), 3),
        "R2_Score": round(float(r2), 4)
    }

def evaluate_classification_model(y_true, y_pred, model_name: str = "Model") -> Dict[str, Any]:
    """
    Computes classification evaluation metrics: Accuracy, Precision, Recall, F1-Score.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "Model": model_name,
        "Accuracy": round(float(acc) * 100.0, 2),
        "Precision": round(float(prec) * 100.0, 2),
        "Recall": round(float(rec) * 100.0, 2),
        "F1_Score": round(float(f1) * 100.0, 2)
    }

def get_confusion_matrix_data(y_true, y_pred) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Returns confusion matrix array and structured counts.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Fail is 0, Pass is 1
    tn, fp, fn, tp = cm.ravel()
    details = {
        "True_Fail (TN)": int(tn),
        "False_Pass (FP)": int(fp),
        "False_Fail (FN)": int(fn),
        "True_Pass (TP)": int(tp)
    }
    return cm, details

def get_actual_vs_predicted_df(y_true, y_pred, n_samples: int = 100) -> pd.DataFrame:
    """
    Builds a comparison dataframe of actual vs predicted marks with residuals.
    """
    df_comp = pd.DataFrame({
        "Actual_Marks": np.array(y_true).round(1),
        "Predicted_Marks": np.array(y_pred).round(1)
    })
    df_comp["Residual"] = (df_comp["Actual_Marks"] - df_comp["Predicted_Marks"]).round(2)
    df_comp["Absolute_Error"] = df_comp["Residual"].abs().round(2)
    
    if n_samples is not None and len(df_comp) > n_samples:
        return df_comp.head(n_samples)
    return df_comp

def compare_models(metrics_list: list) -> pd.DataFrame:
    """
    Combines a list of evaluation metric dictionaries into a comparative DataFrame.
    """
    return pd.DataFrame(metrics_list)
