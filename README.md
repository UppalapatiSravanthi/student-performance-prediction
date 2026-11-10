# Student Performance Prediction and Analysis System Using Python and Machine Learning

[![Python Version](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11%20|%203.12%20|%203.13%20|%203.14-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Dashboard](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Project Title
**Student Performance Prediction and Analysis System Using Python and Machine Learning**

---

## 2. Abstract
The **Student Performance Prediction and Analysis System** is an end-to-end, data-driven web application designed to forecast academic outcomes and identify at-risk students before final evaluations occur. Utilizing supervised machine learning algorithms—including **Linear Regression**, **Random Forest Regressor**, and **Random Forest Classifier**—the system predicts numerical final marks, pass/fail status, and qualitative performance tiers (Excellent, Good, Average, Poor) from multidimensional academic, behavioral, and demographic features. Featuring an interactive, multi-tab **Streamlit** dashboard, the application provides educators, mentors, and academic administrators with exploratory data analysis, visual diagnostic tools, risk early-warning systems, and customized pedagogical intervention strategies.

---

## 3. Introduction
Educational institutions consistently seek reliable methodologies to assess student progress, prevent academic attrition, and boost course completion rates. Traditional evaluation techniques evaluate performance post-facto—after exams are graded and semester terms conclude—leaving minimal opportunity for remediation. By integrating machine learning models into academic workflows, educators can transition from reactive record-keeping to proactive student mentorship. This project implements a full-stack Data Science solution encompassing dataset generation, robust data preprocessing, exploratory data analysis (EDA), predictive modeling with leakage prevention, and an interactive graphical dashboard.

---

## 4. Problem Statement
Academic failure and poor course retention remain persistent challenges in modern educational systems. Key bottlenecks in traditional workflows include:
1. **Late Interventions**: Educators discover struggling students only after midterm or semester final results are released.
2. **Siloed Academic Metrics**: Attendance, assignment consistency, previous academic foundation, and lifestyle factors (e.g., sleep and extracurricular balance) are evaluated independently rather than holistically.
3. **Absence of Diagnostic Analytics**: Mentors lack intuitive visual platforms that synthesize complex feature correlations into clear risk classifications and automated advisory guidance.

---

## 5. Objectives
- **Build a Realistic Academic Repository**: Generate and curate a multi-attribute dataset with 1,200 student records reflecting diverse academic patterns.
- **Implement a Robust Preprocessing Pipeline**: Handle missing data through statistical imputation, categorical encoding, and feature scaling while strictly preventing data leakage.
- **Dual-Tier Machine Learning Engine**:
  - Predict continuous numerical marks using **Linear Regression** and **Random Forest Regressor**.
  - Classify categorical examination outcome (**Pass** vs. **Fail**) using **Random Forest Classifier**.
- **Model Evaluation and Benchmarking**: Contrast models using standard metrics: MAE, MSE, RMSE, R² Score, Accuracy, Precision, Recall, and F1-Score.
- **Early Warning & Risk Detection**: Automatically detect vulnerable students based on attendance (< 75%) and predicted marks (< 50).
- **Interactive Web Dashboard**: Deploy an easy-to-use, multi-page Streamlit dashboard for real-time inference, exploratory data analysis, and report generation.

---

## 6. Existing System vs. Proposed System

| Dimension | Existing System | Proposed System |
|---|---|---|
| **Approach** | Manual, retrospective evaluation | Automated, predictive machine learning |
| **Timing** | Post-examination failure reporting | Early semester proactive risk detection |
| **Factors Considered** | Isolated exam marks | Holistic: Attendance, study hours, sleep, assignments, previous marks |
| **Output** | Raw grades and marks sheets | Predicted marks, Pass/Fail probability, risk tier, tailored advice |
| **Visualization** | Static, tabular spreadsheets | Interactive charts, scatter trendlines, distribution curves, heatmaps |
| **Remediation** | Generalized or ad-hoc assistance | Automated, personalized pedagogical recommendations |

---

## 7. Technologies Used
- **Programming Language**: Python (3.9+)
- **Data Manipulation & Analysis**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (LinearRegression, RandomForestRegressor, RandomForestClassifier, StandardScaler)
- **Data Visualization**: Matplotlib, Seaborn
- **Model Persistence**: Joblib
- **Web Dashboard**: Streamlit
- **Development Environment**: Visual Studio Code (VS Code), PowerShell / Terminal

---

## 8. System Requirements

### Hardware Requirements
- **Processor**: Intel Core i3 / AMD Ryzen 3 or higher
- **RAM**: Minimum 4 GB (8 GB recommended)
- **Storage**: Minimum 500 MB free disk space

### Software Requirements
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: Version 3.9 or newer (tested on Python 3.10 – 3.14)
- **Web Browser**: Google Chrome, Mozilla Firefox, Microsoft Edge, or Safari

---

## 9. Dataset Description
The dataset contains 1,200 synthetic student records with 12 attributes:

| Column Name | Data Type | Description | Value Range |
|---|---|---|---|
| `Student_ID` | String | Unique student identifier | `STU1001` - `STU2200` |
| `Gender` | String | Student gender | `Male`, `Female` |
| `Age` | Integer | Student age in years | `17` - `24` |
| `Study_Hours_Per_Day` | Float | Dedicated daily study duration | `1.0` - `10.0` hours |
| `Attendance_Percentage` | Float | Semester lecture attendance rate | `45.0%` - `100.0%` |
| `Previous_Marks` | Float | Academic marks from past examination | `20.0` - `98.0` |
| `Assignment_Score` | Float | Continuous assessment score | `20.0` - `100.0` |
| `Internal_Marks` | Float | Mid-term internal assessment mark | `5.0` - `30.0` |
| `Sleep_Hours` | Float | Average nocturnal sleep duration | `4.0` - `10.0` hours |
| `Extracurricular_Activities` | String | Active participation in clubs/sports | `Yes`, `No` |
| `Final_Marks` | Float | Final examination score | `15.0` - `100.0` |
| `Result` | String | Academic examination outcome | `Pass` (≥ 40) / `Fail` (< 40) |

---

## 10. Methodology & Data Science Workflow

```
[1. Data Generation] --> [2. Data Ingestion] --> [3. Cleaning & Imputation]
                                                          |
                                                          v
[6. ML Training] <--- [5. Feature Scaling] <--- [4. Train-Test Split]
        |
        +---> Linear Regression (Continuous Marks)
        +---> Random Forest Regressor (Continuous Marks)
        +---> Random Forest Classifier (Pass / Fail)
        |
        v
[7. Model Evaluation] (MAE, RMSE, R², Accuracy, F1, Confusion Matrix)
        |
        v
[8. Streamlit Dashboard] (Home, EDA, Viz, Prediction, Risk, Evaluation, Export)
```

1. **Data Collection & Ingestion**: Loading raw student records via Pandas.
2. **Data Cleaning & Missing Value Handling**: Missing values in numerical attributes are imputed with the median to resist skewness; categorical values are imputed with the mode.
3. **Categorical Feature Encoding**: Binary mapping for `Gender` (Female=0, Male=1), `Extracurricular_Activities` (No=0, Yes=1), and `Result` (Fail=0, Pass=1).
4. **Data Leakage Prevention**: Data is partitioned into 80% training and 20% testing sets *before* feature scaling. The `StandardScaler` is fitted strictly on `X_train` and transforms `X_test`.
5. **Model Training & Hyperparameter Tuning**: Fitting Linear Regression and Random Forest ensembles with controlled depth and estimator parameters.
6. **Persistence**: Serializing models, fitted scalers, and evaluation metrics using Joblib.
7. **Frontend Deployment**: Serving interactive predictions and analytics via Streamlit.

---

## 11. Algorithms Used

### 1. Linear Regression
Models the linear relationship between the independent academic/behavioral predictors and the dependent continuous target (`Final_Marks`):
$$\hat{y} = \beta_0 + \sum_{i=1}^{p} \beta_i x_i$$
Serves as an interpretable parametric baseline.

### 2. Random Forest Regressor
An ensemble learning algorithm that constructs 100 decision trees during training. It handles non-linear interactions, feature collinearity, and outputs the mean prediction of individual trees:
$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

### 3. Random Forest Classifier
Ensemble of decision trees trained with bootstrap aggregation and Gini impurity splitting to predict the probability of class labels ($Result \in \{Pass, Fail\}$):
$$Gini = 1 - \sum_{k=1}^{K} p_k^2$$

---

## 12. Results and Evaluation Metrics

### Continuous Marks Prediction (Regression)

| Model Name | MAE | MSE | RMSE | R² Score |
|---|---|---|---|---|
| **Linear Regression** | **3.454** | **19.231** | **4.385** | **0.8663** |
| **Random Forest Regressor** | 3.706 | 22.389 | 4.732 | 0.8444 |

*Note: Linear Regression achieves an $R^2$ of ~86.6%, indicating that over 86% of the variance in student final marks is accurately captured by the feature set.*

### Outcome Classification (Pass / Fail)

| Metric | Random Forest Classifier Result |
|---|---|
| **Accuracy** | **93.75%** |
| **Precision** | **95.11%** |
| **Recall** | **98.17%** |
| **F1-Score** | **96.61%** |

### Confusion Matrix (Test Set: 240 Students)
- **True Failures Detected (TN)**: 11
- **False Pass (FP)**: 11
- **False Fail (FN)**: 4
- **True Passes Detected (TP)**: 214

---

## 13. Project Folder Structure

```
student_performance_project/
│
├── app.py                      # Main Streamlit web application
├── generate_dataset.py         # Synthetic student dataset generator (1,200 records)
├── train_model.py              # ML training, evaluation, and serialization script
├── test_system.py              # Automated unit and integration verification script
├── requirements.txt            # Project dependencies
├── README.md                   # Comprehensive documentation
│
├── data/
│   └── student_data.csv        # 1,200 student records CSV
│
├── models/
│   ├── marks_model.pkl         # Serialized best marks regressor
│   ├── linear_model.pkl        # Serialized Linear Regression model
│   ├── rf_reg_model.pkl        # Serialized Random Forest Regressor
│   ├── result_model.pkl        # Serialized Random Forest Classifier
│   ├── scaler.pkl              # Fitted StandardScaler transformer
│   └── model_metadata.pkl      # Training metrics, feature importance, test outputs
│
└── src/
    ├── __init__.py             # Python package marker
    ├── data_preprocessing.py   # Data loading, cleaning, imputation, scaling, splitting
    ├── eda.py                  # Analytical KPIs, summaries, and risk identification logic
    ├── evaluation.py           # Evaluation metrics (MAE, RMSE, R², Accuracy, F1, Confusion Matrix)
    └── prediction.py           # Inference engine, categorization, and recommendation generator
```

---

## 14. Advantages
- **Early Identification**: Detects struggling students weeks before final examinations.
- **Holistic Assessment**: Combines behavioral routines (study and sleep) with historical assessment data.
- **Actionable Guidance**: Generates personalized tips for students and educators.
- **Leakage-Free ML**: Strictly maintains data hygiene between train and test splits.
- **User-Friendly Dashboard**: No coding required to interact with models, inspect charts, or download reports.

---

## 15. Limitations
- Synthetic dataset reflects simulated patterns; real-world deployment requires integration with institutional student information systems (SIS).
- Extreme social, psychological, or health emergencies outside the recorded features cannot be captured.
- Outlier grading variances across different academic departments are not modeled.

---

## 16. Future Enhancements
- Integration with institutional Learning Management Systems (Canvas, Moodle, Blackboard) via automated APIs.
- Time-series tracking of student performance across sequential semesters.
- Deep learning architectures (LSTMs or Transformers) for multi-semester trajectory forecasting.
- Automated email alerts to mentors and students when a student is flagged as `Critical Risk`.

---

## 17. Step-by-Step Installation Instructions (VS Code & Windows)

### Step 1: Open VS Code and Terminal
Open Visual Studio Code, press ``Ctrl + ` `` to open the integrated terminal.

### Step 2: Navigate to Project Directory
```powershell
cd C:\Users\trive\.gemini\antigravity\scratch\student_performance_project
```

### Step 3: Create a Virtual Environment
Isolate project dependencies by creating a dedicated virtual environment:
```powershell
python -m venv venv
```

### Step 4: Activate Virtual Environment
Activate the virtual environment in PowerShell:
```powershell
venv\Scripts\activate
```
*(If PowerShell displays an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and rerun the activate command).*

### Step 5: Install Dependencies
Install all required libraries specified in `requirements.txt`:
```powershell
pip install -r requirements.txt
```

---

## 18. Execution Steps

### Step 1: Generate Dataset
```powershell
python generate_dataset.py
```
*Creates `data/student_data.csv` with 1,200 records.*

### Step 2: Train Machine Learning Models
```powershell
python train_model.py
```
*Preprocesses data, trains regression and classification models, outputs evaluation metrics, and saves trained models into `models/`.*

### Step 3: Verify the Pipeline (Optional but Recommended)
```powershell
python test_system.py
```
*Runs automated integrity tests on preprocessing, predictions, and model loading.*

### Step 4: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
*Launches the application in your default web browser at `http://localhost:8501`.*

---

## 19. Sample Outputs & Testing Walkthrough

### Test Case 1: High-Performing Student
- **Inputs**: Study Hours: 7.0, Attendance: 95%, Previous Marks: 85, Assignment: 90, Internal: 27, Sleep: 7.5, Extracurricular: Yes
- **Predicted Marks**: ~88 - 92
- **Predicted Result**: Pass (Pass Probability: 99%)
- **Performance Tier**: Excellent
- **System Advice**: *"Outstanding performance! Maintain consistency and consider mentoring peers."*

### Test Case 2: At-Risk Student
- **Inputs**: Study Hours: 1.5, Attendance: 54%, Previous Marks: 32, Assignment: 35, Internal: 9, Sleep: 4.5, Extracurricular: No
- **Predicted Marks**: ~28 - 34
- **Predicted Result**: Fail (Fail Probability: 92%)
- **Performance Tier**: Poor
- **System Advice**: *"Immediate academic intervention recommended. Increase daily study hours to at least 4.0 and consult course instructors."*

---

## 20. Key Presentation Points for Viva / Project Demonstration
1. **Explain the Dual ML Task**: Emphasize why both Regression (to predict the exact mark) and Classification (to determine Pass/Fail probability) are deployed simultaneously.
2. **Preventing Data Leakage**: Highlight how `StandardScaler` is fitted exclusively on `X_train` and applied to `X_test` and single-student inference.
3. **Model Comparison**: Explain the benchmark comparison between Linear Regression ($R^2 \approx 0.866$) and Random Forest Regressor ($R^2 \approx 0.844$).
4. **Early Risk Warning**: Showcase the Student Risk Analysis page and demonstrate how varying the attendance threshold isolates vulnerable students.
5. **Full Modularity**: Point out the clean architecture separating preprocessing (`data_preprocessing.py`), analysis (`eda.py`), evaluation (`evaluation.py`), inference (`prediction.py`), and presentation (`app.py`).
