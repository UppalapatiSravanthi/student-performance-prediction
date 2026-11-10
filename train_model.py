"""
Model Training Script for Student Performance Prediction and Analysis System
Trains:
  1. Linear Regression (predicts Final_Marks)
  2. Random Forest Regressor (predicts Final_Marks)
  3. Random Forest Classifier (predicts Pass/Fail)
Evaluates and logs metrics: MAE, MSE, RMSE, R2, Accuracy, Precision, Recall, F1.
Saves model artifacts, scaler, and training metadata to models/ directory.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Local imports
from src.data_preprocessing import load_data, prepare_data_for_modeling, FEATURE_COLUMNS
from src.evaluation import (
    evaluate_regression_model,
    evaluate_classification_model,
    get_confusion_matrix_data,
    compare_models
)

def train_and_evaluate_models(data_path: str, models_dir: str):
    """
    Orchestrates the model training and evaluation lifecycle.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    print("=" * 70)
    print("  STUDENT PERFORMANCE PREDICTION SYSTEM - MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    # 1. Load Data
    print(f"\n[1/5] Loading dataset from: {data_path}")
    df = load_data(data_path)
    print(f"      Total records loaded: {len(df)}")
    
    # 2. Preprocess & Train-Test Split (Prevent Data Leakage)
    print("\n[2/5] Preprocessing features and splitting (80% Train, 20% Test)...")
    data_dict = prepare_data_for_modeling(df, test_size=0.2, random_state=42)
    
    X_train = data_dict["X_train_scaled"]
    X_test = data_dict["X_test_scaled"]
    y_train_reg = data_dict["y_train_reg"]
    y_test_reg = data_dict["y_test_reg"]
    y_train_clf = data_dict["y_train_clf"]
    y_test_clf = data_dict["y_test_clf"]
    scaler = data_dict["scaler"]
    
    print(f"      Training samples: {len(X_train)}")
    print(f"      Testing samples:  {len(X_test)}")
    print(f"      Features: {FEATURE_COLUMNS}")
    
    # 3. Train Regression Models (Final Marks Prediction)
    print("\n[3/5] Training Regression Models (Final_Marks prediction)...")
    
    # Model 1: Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train_reg)
    y_pred_lr = lr_model.predict(X_test)
    metrics_lr = evaluate_regression_model(y_test_reg, y_pred_lr, model_name="Linear Regression")
    
    # Model 2: Random Forest Regressor
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, y_train_reg)
    y_pred_rf_reg = rf_reg.predict(X_test)
    metrics_rf_reg = evaluate_regression_model(y_test_reg, y_pred_rf_reg, model_name="Random Forest Regressor")
    
    # Compare Regression Models
    reg_comparison = compare_models([metrics_lr, metrics_rf_reg])
    print("\n--- Regression Model Comparison ---")
    print(reg_comparison.to_string(index=False))
    
    # Choose best regression model (based on lower RMSE / higher R2)
    if metrics_rf_reg["R2_Score"] >= metrics_lr["R2_Score"]:
        best_reg_model = rf_reg
        best_reg_name = "Random Forest Regressor"
        y_best_pred_reg = y_pred_rf_reg
    else:
        best_reg_model = lr_model
        best_reg_name = "Linear Regression"
        y_best_pred_reg = y_pred_lr
        
    print(f"\n      Selected Best Marks Regressor: {best_reg_name}")
    
    # 4. Train Classification Model (Pass/Fail Prediction)
    print("\n[4/5] Training Classification Model (Pass/Fail prediction)...")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train_clf)
    y_pred_clf = rf_clf.predict(X_test)
    metrics_clf = evaluate_classification_model(y_test_clf, y_pred_clf, model_name="Random Forest Classifier")
    
    cm, cm_counts = get_confusion_matrix_data(y_test_clf, y_pred_clf)
    
    print("\n--- Classification Performance (Pass/Fail) ---")
    for k, v in metrics_clf.items():
        if k != "Model":
            print(f"      {k:12}: {v}%")
    print(f"      Confusion Matrix Counts: {cm_counts}")
    
    # Feature Importance (Random Forest Regressor)
    feature_importance_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": rf_reg.feature_importances_
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    # 5. Persist Model Artifacts
    print(f"\n[5/5] Saving model artifacts to: '{models_dir}'...")
    joblib.dump(best_reg_model, os.path.join(models_dir, "marks_model.pkl"))
    joblib.dump(lr_model, os.path.join(models_dir, "linear_model.pkl"))
    joblib.dump(rf_reg, os.path.join(models_dir, "rf_reg_model.pkl"))
    joblib.dump(rf_clf, os.path.join(models_dir, "result_model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    
    # Save training metadata for dashboard
    metadata = {
        "best_reg_name": best_reg_name,
        "reg_comparison": reg_comparison,
        "metrics_lr": metrics_lr,
        "metrics_rf_reg": metrics_rf_reg,
        "metrics_clf": metrics_clf,
        "cm": cm,
        "cm_counts": cm_counts,
        "feature_importance": feature_importance_df,
        "test_actual_marks": list(y_test_reg.values),
        "test_pred_lr": list(y_pred_lr),
        "test_pred_rf": list(y_pred_rf_reg),
        "test_pred_best": list(y_best_pred_reg),
        "feature_columns": FEATURE_COLUMNS
    }
    joblib.dump(metadata, os.path.join(models_dir, "model_metadata.pkl"))
    
    print("      [OK] Saved marks_model.pkl")
    print("      [OK] Saved linear_model.pkl")
    print("      [OK] Saved rf_reg_model.pkl")
    print("      [OK] Saved result_model.pkl")
    print("      [OK] Saved scaler.pkl")
    print("      [OK] Saved model_metadata.pkl")
    print("\n" + "=" * 70)
    print("  [SUCCESS] MODEL TRAINING COMPLETED WITHOUT ERRORS!")
    print("=" * 70)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(current_dir, "data", "student_data.csv")
    models_directory = os.path.join(current_dir, "models")
    
    if not os.path.exists(dataset_file):
        print(f"[!] Dataset not found at '{dataset_file}'. Generating now...")
        from generate_dataset import save_dataset
        save_dataset(dataset_file)
        
    train_and_evaluate_models(dataset_file, models_directory)
