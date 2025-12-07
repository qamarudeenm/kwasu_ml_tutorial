# Student Performance Prediction Project

This project implements an end-to-end Machine Learning pipeline to predict student performance based on study habits and attendance. It includes data preprocessing, feature engineering, model training, evaluation, and a REST API for serving predictions.

## 1. Data Preprocessing
**Script:** `src/preprocessing.py`

The raw data (`student_performance.csv`) contains student records.
- **Missing Values**: Numerical columns are filled with the **mean**, and categorical columns with the **mode**.
- **Categorical Encoding**: The `grade` target variable is encoded into numerical labels (0, 1, 2, 3, 4) using `LabelEncoder`. This encoder is saved to `models/label_encoder.pkl` to decode predictions later.

## 2. Feature Engineering
**Script:** `src/features.py`

To ensure models perform optimally, features are scaled.
- **Scaling**: We use `StandardScaler` to normalize numerical features (`weekly_self_study_hours`, `attendance_percentage`, `class_participation`). This ensures that features with larger ranges don't dominate the model.
- The scaler is saved to `models/scaler.pkl` to transform new data during inference.

## 3. Model Training
**Script:** `src/train.py`

We train two separate models for different prediction tasks:
1.  **Linear Regression**: Predicts the exact `total_score` (continuous variable).
2.  **Logistic Regression**: Predicts the `grade` (categorical variable: A, B, C, D, F).

The data is split into **Train (80%)** and **Test (20%)** sets to ensure we evaluate on unseen data.

## 4. Model Evaluation & Interpretation
**Script:** `src/evaluate.py`

### Linear Regression (Total Score)
- **Mean Squared Error (MSE): 2.04**: On average, the squared difference between predicted and actual scores is low, indicating good precision.
- **R² Score: 0.96**: The model explains **96%** of the variance in the student scores. This is a very high score, suggesting that study hours and attendance are strong predictors of performance.

### Logistic Regression (Grade)
- **Accuracy: 0.70**: The model correctly predicts the exact letter grade 70% of the time.
- **Accuracy: 0.70**: The model correctly predicts the exact letter grade 70% of the time.

### Detailed Analysis (F1 Score & Confusion Matrix)
The **Classification Report** reveals a critical insight that Accuracy hides:

1.  **Class 0 (Grade A)**: **F1 Score ~0.85**.
    - The model is excellent at identifying top students. This is expected because the dataset is dominated by high-performing students (109k examples).
2.  **Middle Classes (Grades B, C, D)**: **F1 Score ~0.50**.
    - The model struggles to distinguish between these grades, likely confusing them with neighbors (e.g., predicting a 'C' as a 'B').
3.  **Class 4 (Grade F)**: **F1 Score 0.00**.
    - **CRITICAL FAILURE**: The model completely fails to identify failing students.
    - **Reason**: There are only ~1,200 failing students vs ~110,000 'A' students. The model has learned to ignore this minority class to maximize overall accuracy.

### Conclusion on Model Performance
- **Is this model good?** It depends on the goal.
    - If the goal is to **predict top performers**, it is **Good**.
    - If the goal is to **identify students at risk of failing** (Early Warning System), it is **USELESS** despite 70% accuracy.
- **Fix**: To fix this, we would need to use techniques like **Oversampling (SMOTE)** or **Class Weights** to force the model to pay attention to the failing students.

## 5. Serving Layer (Deployment)
**Script:** `app/main.py`

The models are deployed using **FastAPI**, providing a high-performance, easy-to-use API.
- **`/predict/score`**: Accepts student data and returns the predicted `total_score`.
- **`/predict/grade`**: Accepts student data and returns the predicted `grade` (e.g., "A").

### Usage
Start the server:
```bash
uvicorn app.main:app --reload --port 3000
```

Make a prediction:
```bash
curl -X POST "http://127.0.0.1:3000/predict/score" \
     -H "Content-Type: application/json" \
     -d '{"weekly_self_study_hours": 20, "attendance_percentage": 90, "class_participation": 8}'
```
