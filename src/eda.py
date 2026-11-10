"""
Exploratory Data Analysis (EDA) Module
Provides analytical functions, aggregation logic, and statistical summaries
for student academic data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

def get_basic_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes key performance indicators (KPIs) for the dataset.
    """
    total_students = len(df)
    avg_marks = float(df["Final_Marks"].mean())
    median_marks = float(df["Final_Marks"].median())
    highest_mark = float(df["Final_Marks"].max())
    lowest_mark = float(df["Final_Marks"].min())
    
    pass_count = int((df["Result"] == "Pass").sum())
    fail_count = int((df["Result"] == "Fail").sum())
    
    pass_pct = (pass_count / total_students) * 100.0 if total_students > 0 else 0.0
    fail_pct = (fail_count / total_students) * 100.0 if total_students > 0 else 0.0
    
    avg_study_hours = float(df["Study_Hours_Per_Day"].mean())
    avg_attendance = float(df["Attendance_Percentage"].mean())
    
    return {
        "total_students": total_students,
        "avg_marks": round(avg_marks, 2),
        "median_marks": round(median_marks, 2),
        "highest_mark": round(highest_mark, 2),
        "lowest_mark": round(lowest_mark, 2),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_pct": round(pass_pct, 2),
        "fail_pct": round(fail_pct, 2),
        "avg_study_hours": round(avg_study_hours, 2),
        "avg_attendance": round(avg_attendance, 2)
    }

def get_missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a detailed summary of missing values in the dataframe.
    """
    null_counts = df.isnull().sum()
    null_pct = (df.isnull().sum() / len(df)) * 100.0
    data_types = df.dtypes
    
    report = pd.DataFrame({
        "Feature": df.columns,
        "Missing Values": null_counts.values,
        "Percentage (%)": null_pct.round(2).values,
        "Data Type": [str(t) for t in data_types.values]
    })
    return report

def get_statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns rounded descriptive statistics for all numerical features.
    """
    numerical_df = df.select_dtypes(include=[np.number])
    summary = numerical_df.describe().T
    summary = summary[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    return summary.round(2)

def get_gender_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes performance differences based on gender.
    """
    breakdown = df.groupby("Gender").agg(
        Total_Students=("Student_ID", "count"),
        Average_Marks=("Final_Marks", "mean"),
        Average_Attendance=("Attendance_Percentage", "mean"),
        Average_Study_Hours=("Study_Hours_Per_Day", "mean"),
        Pass_Rate=("Result", lambda x: (x == "Pass").mean() * 100.0)
    ).reset_index()
    return breakdown.round(2)

def get_attendance_brackets_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorizes attendance into brackets and evaluates corresponding marks.
    """
    bins = [0, 60, 75, 85, 100]
    labels = ["Critical (<60%)", "Low (60-75%)", "Moderate (75-85%)", "High (85-100%)"]
    
    df_temp = df.copy()
    df_temp["Attendance_Bracket"] = pd.cut(df_temp["Attendance_Percentage"], bins=bins, labels=labels)
    
    bracket_summary = df_temp.groupby("Attendance_Bracket", observed=False).agg(
        Student_Count=("Student_ID", "count"),
        Avg_Final_Marks=("Final_Marks", "mean"),
        Pass_Rate=("Result", lambda x: (x == "Pass").mean() * 100.0)
    ).reset_index()
    
    return bracket_summary.round(2)

def get_study_hours_brackets_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorizes study hours per day into groups and inspects performance.
    """
    bins = [0, 2.5, 5.0, 7.5, 12]
    labels = ["Low (<2.5 hrs)", "Moderate (2.5-5 hrs)", "Good (5-7.5 hrs)", "Intensive (>7.5 hrs)"]
    
    df_temp = df.copy()
    df_temp["Study_Bracket"] = pd.cut(df_temp["Study_Hours_Per_Day"], bins=bins, labels=labels)
    
    study_summary = df_temp.groupby("Study_Bracket", observed=False).agg(
        Student_Count=("Student_ID", "count"),
        Avg_Final_Marks=("Final_Marks", "mean"),
        Pass_Rate=("Result", lambda x: (x == "Pass").mean() * 100.0)
    ).reset_index()
    
    return study_summary.round(2)

def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Pearson correlation matrix for numerical features.
    """
    numerical_df = df.select_dtypes(include=[np.number])
    return numerical_df.corr().round(3)

def identify_at_risk_students(
    df: pd.DataFrame, 
    attendance_thresh: float = 75.0, 
    marks_thresh: float = 50.0
) -> pd.DataFrame:
    """
    Filters students at academic risk based on attendance and marks criteria.
    Assigns Risk Levels: Critical, Moderate, or Mild.
    """
    df_risk = df.copy()
    
    is_low_att = df_risk["Attendance_Percentage"] < attendance_thresh
    is_low_marks = df_risk["Final_Marks"] < marks_thresh
    is_fail = df_risk["Result"] == "Fail"
    
    # Filter only at risk
    at_risk_mask = is_low_att | is_low_marks | is_fail
    at_risk_df = df_risk[at_risk_mask].copy()
    
    # Define risk severity
    def assign_risk_level(row):
        att_bad = row["Attendance_Percentage"] < attendance_thresh
        marks_bad = row["Final_Marks"] < marks_thresh
        is_failing = row["Result"] == "Fail"
        
        if (att_bad and marks_bad) or is_failing:
            return "Critical"
        elif marks_bad:
            return "High Risk (Academic)"
        elif att_bad:
            return "Moderate Risk (Attendance)"
        else:
            return "Mild Risk"
            
    at_risk_df["Risk_Level"] = at_risk_df.apply(assign_risk_level, axis=1)
    
    return at_risk_df.sort_values(by="Final_Marks", ascending=True)
