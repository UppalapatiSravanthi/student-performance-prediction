"""
Data Preprocessing Module
Handles data loading, cleaning, missing value imputation,
encoding categorical variables, scaling, and train-test splitting.
Prevents data leakage by fitting transformers on train data only.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any, List

FEATURE_COLUMNS: List[str] = [
    "Gender",
    "Age",
    "Study_Hours_Per_Day",
    "Attendance_Percentage",
    "Previous_Marks",
    "Assignment_Score",
    "Internal_Marks",
    "Sleep_Hours",
    "Extracurricular_Activities"
]

NUMERICAL_COLUMNS: List[str] = [
    "Age",
    "Study_Hours_Per_Day",
    "Attendance_Percentage",
    "Previous_Marks",
    "Assignment_Score",
    "Internal_Marks",
    "Sleep_Hours"
]

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads student dataset from a CSV file.
    
    Parameters:
        filepath (str): Path to student_data.csv.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        raise FileNotFoundError(f"Error reading file at {filepath}: {str(e)}")

def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Returns the count of missing values per column."""
    return df.isnull().sum()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans data by handling missing values and dropping unneeded identifiers.
    Imputes numerical columns with median and categorical with mode.
    """
    df_clean = df.copy()
    
    # Impute numerical missing values using median to resist outliers
    for col in NUMERICAL_COLUMNS:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            
    # Impute categorical missing values using mode
    for col in ["Gender", "Extracurricular_Activities"]:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            
    return df_clean

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical columns into numerical values:
    Gender: Female -> 0, Male -> 1
    Extracurricular_Activities: No -> 0, Yes -> 1
    Result: Fail -> 0, Pass -> 1 (if present)
    """
    df_encoded = df.copy()
    
    gender_map = {"Female": 0, "Male": 1}
    extra_map = {"No": 0, "Yes": 1}
    result_map = {"Fail": 0, "Pass": 1}
    
    if "Gender" in df_encoded.columns:
        df_encoded["Gender"] = df_encoded["Gender"].map(gender_map).fillna(0).astype(int)
    if "Extracurricular_Activities" in df_encoded.columns:
        df_encoded["Extracurricular_Activities"] = df_encoded["Extracurricular_Activities"].map(extra_map).fillna(0).astype(int)
    if "Result" in df_encoded.columns:
        df_encoded["Result"] = df_encoded["Result"].map(result_map).fillna(0).astype(int)
        
    return df_encoded

def prepare_data_for_modeling(
    df: pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Prepares training and testing sets with feature scaling.
    Prevents data leakage: fit scaler ONLY on X_train.
    
    Returns a dictionary containing:
        - X_train, X_test
        - y_train_reg, y_test_reg (Final_Marks)
        - y_train_clf, y_test_clf (Result)
        - scaler (fitted StandardScaler)
        - feature_names (list of features)
    """
    # 1. Clean missing values
    df_clean = clean_data(df)
    
    # 2. Encode categorical variables
    df_encoded = encode_features(df_clean)
    
    # 3. Separate features and targets
    X = df_encoded[FEATURE_COLUMNS].copy()
    y_reg = df_encoded["Final_Marks"].copy()
    y_clf = df_encoded["Result"].copy()
    
    # 4. Train-Test Split (stratify on Result for balanced classification split)
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        X, y_reg, y_clf, test_size=test_size, random_state=random_state, stratify=y_clf
    )
    
    # 5. Fit Scaler ONLY on Training Data (Preventing Data Leakage)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=FEATURE_COLUMNS, 
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=FEATURE_COLUMNS, 
        index=X_test.index
    )
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train_reg": y_train_reg,
        "y_test_reg": y_test_reg,
        "y_train_clf": y_train_clf,
        "y_test_clf": y_test_clf,
        "scaler": scaler,
        "feature_names": FEATURE_COLUMNS,
        "numerical_columns": NUMERICAL_COLUMNS
    }

def transform_single_input(input_dict: Dict[str, Any], scaler: StandardScaler) -> pd.DataFrame:
    """
    Transforms a single student input dictionary into a scaled dataframe
    ready for model inference.
    
    Parameters:
        input_dict (dict): Keys matching FEATURE_COLUMNS with raw values.
        scaler (StandardScaler): Pre-fitted scaler.
        
    Returns:
        pd.DataFrame: Scaled single-row dataframe.
    """
    df_single = pd.DataFrame([input_dict])
    df_encoded = encode_features(df_single)
    df_ordered = df_encoded[FEATURE_COLUMNS]
    scaled_values = scaler.transform(df_ordered)
    return pd.DataFrame(scaled_values, columns=FEATURE_COLUMNS)
