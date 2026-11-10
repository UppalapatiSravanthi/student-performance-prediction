"""
Dataset Generation Script for Student Performance Prediction and Analysis System
Generates a realistic student dataset with 1,200 records.
Includes demographic, academic, behavioral, and performance metrics.
"""

import os
import numpy as np
import pandas as pd

def generate_student_dataset(num_records: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic student academic performance dataset.
    
    Parameters:
        num_records (int): Number of student records to generate (default: 1200).
        random_state (int): Seed for reproducibility.
        
    Returns:
        pd.DataFrame: Cleaned and structured student dataset.
    """
    np.random.seed(random_state)
    
    # 1. Student ID
    student_ids = [f"STU{1000 + i}" for i in range(1, num_records + 1)]
    
    # 2. Gender (Approx. 52% Male, 48% Female)
    genders = np.random.choice(["Male", "Female"], size=num_records, p=[0.52, 0.48])
    
    # 3. Age (17 to 24, centered around 19-20)
    ages = np.random.choice([17, 18, 19, 20, 21, 22, 23, 24], size=num_records, 
                            p=[0.05, 0.25, 0.35, 0.20, 0.08, 0.04, 0.02, 0.01])
    
    # 4. Study Hours Per Day (Between 1.0 and 10.0 hours, gamma-distributed)
    study_hours = np.random.gamma(shape=3.5, scale=1.2, size=num_records)
    study_hours = np.clip(study_hours, 1.0, 10.0).round(1)
    
    # 5. Attendance Percentage (Between 45.0% and 100.0%)
    base_attendance = np.random.beta(a=5, b=2, size=num_records) * 55 + 45
    study_bonus_att = (study_hours / 10.0) * 6.0
    attendance = np.clip(base_attendance + study_bonus_att, 45.0, 100.0).round(1)
    
    # 6. Previous Marks (Out of 100, normally distributed around 58)
    previous_marks = np.random.normal(loc=58.0, scale=16.0, size=num_records)
    previous_marks = np.clip(previous_marks, 20.0, 98.0).round(1)
    
    # 7. Assignment Score (Out of 100)
    assignment_score = (
        0.45 * previous_marks + 
        0.30 * (study_hours * 10) + 
        0.15 * attendance + 
        np.random.normal(0, 7, num_records)
    )
    assignment_score = np.clip(assignment_score, 20.0, 100.0).round(1)
    
    # 8. Internal Marks (Out of 30, scaled assessment)
    internal_marks = (
        0.18 * previous_marks + 
        0.10 * assignment_score + 
        np.random.normal(0, 2.5, num_records)
    )
    internal_marks = np.clip(internal_marks, 5.0, 30.0).round(1)
    
    # 9. Sleep Hours (4.0 to 10.0 hours)
    sleep_hours = np.random.normal(loc=6.8, scale=1.2, size=num_records)
    sleep_hours = np.clip(sleep_hours, 4.0, 10.0).round(1)
    
    # 10. Extracurricular Activities ('Yes' or 'No')
    extracurricular = np.random.choice(["Yes", "No"], size=num_records, p=[0.42, 0.58])
    
    # 11. Final Marks calculation (Out of 100)
    # Balanced realistic weightings:
    # 0.30*Prev + 0.25*Assign + 0.65*Internal + 0.12*Att + 1.2*StudyHrs + sleep + noise
    sleep_effect = np.where(sleep_hours < 5.0, -4.0, np.where(sleep_hours > 8.5, -2.0, 1.5))
    extra_effect = np.where(extracurricular == "Yes", 2.0, 0.0)
    
    calculated_final_marks = (
        0.28 * previous_marks +
        0.24 * assignment_score +
        0.65 * internal_marks +
        0.12 * attendance +
        1.35 * study_hours +
        sleep_effect +
        extra_effect +
        np.random.normal(-2, 4.0, num_records)
    )
    final_marks = np.clip(calculated_final_marks, 15.0, 100.0).round(1)
    
    # 12. Result (Pass if Final_Marks >= 40.0 else Fail)
    result = np.where(final_marks >= 40.0, "Pass", "Fail")
    
    # Assemble DataFrame
    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Gender": genders,
        "Age": ages,
        "Study_Hours_Per_Day": study_hours,
        "Attendance_Percentage": attendance,
        "Previous_Marks": previous_marks,
        "Assignment_Score": assignment_score,
        "Internal_Marks": internal_marks,
        "Sleep_Hours": sleep_hours,
        "Extracurricular_Activities": extracurricular,
        "Final_Marks": final_marks,
        "Result": result
    })
    
    # Introduce a small number of realistic missing values (e.g. 15 missing in sleep, 10 in assignment)
    # to demonstrate real data cleaning and imputation capabilities in the pipeline
    missing_sleep_idx = np.random.choice(df.index, size=15, replace=False)
    missing_assignment_idx = np.random.choice(df.index, size=10, replace=False)
    
    df.loc[missing_sleep_idx, "Sleep_Hours"] = np.nan
    df.loc[missing_assignment_idx, "Assignment_Score"] = np.nan
    
    return df

def save_dataset(filepath: str = "data/student_data.csv"):
    """Generates and writes the dataset to the specified CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = generate_student_dataset(num_records=1200, random_state=42)
    df.to_csv(filepath, index=False)
    print(f"[SUCCESS] Generated {len(df)} student records saved to '{filepath}'")
    print(f"Summary of Result: {df['Result'].value_counts().to_dict()}")
    print(f"Average Final Marks: {df['Final_Marks'].mean():.2f}")
    return df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "data", "student_data.csv")
    save_dataset(output_path)
