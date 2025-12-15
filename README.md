# Student Performance Prediction Project

This project implements an end-to-end Machine Learning pipeline to predict student performance based on study habits and attendance. It includes data preprocessing, feature engineering, model training, evaluation, and a REST API for serving predictions.

## 1. Getting Started

### Clone the Repository
To get started with this project, clone the repository to your local machine:
```bash
git clone https://github.com/qamarudeenm/kwasu_ml_tutorial
cd kwasu_ml_tutorial
```

## 2. Environment Setup

### 2.1 Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

**For Windows:**
```bash
python -m venv venv
```

**For macOS and Linux:**
```bash
python3 -m venv venv
```

### 2.2 Activate the Virtual Environment

**For Windows:**
```bash
venv\Scripts\activate
```

**For macOS and Linux:**
```bash
source venv/bin/activate
```

### 2.3 Install Dependencies
Once the virtual environment is active, install the required packages:
```bash
pip install -r requirements.txt
```

## 3. Running the Project

### Jupyter Notebooks
This project contains several notebooks for learning and experimentation. Ensure your virtual environment is active before running Jupyter.

**CNN Model:**
To explore the Convolutional Neural Network model:
```bash
jupyter notebook Cnn_Model.ipynb
```

**Preprocessing Lecture:**
To learn about data preprocessing and feature engineering:
```bash
jupyter notebook src/preprocessing_feature_engineering/preprocessing_step.ipynb
```

**Homework:**
To work on the assignments:
```bash
cd home_work
jupyter notebook
```

## 4. Data Preprocessing
**Script:** `src/preprocessing.py`

The raw data (`student_performance.csv`) contains student records.
- **Missing Values**: Numerical columns are filled with the **mean**, and categorical columns with the **mode**.
- **Categorical Encoding**: The `grade` target variable is encoded into numerical labels (0, 1, 2, 3, 4) using `LabelEncoder`. This encoder is saved to `models/label_encoder.pkl` to decode predictions later.

## 5. Feature Engineering
**Script:** `src/features.py`

To ensure models perform optimally, features are scaled.
- **Scaling**: We use `StandardScaler` to normalize numerical features (`weekly_self_study_hours`, `attendance_percentage`, `class_participation`). This ensures that features with larger ranges don't dominate the model.
- The scaler is saved to `models/scaler.pkl` to transform new data during inference.

## 6. Model Training
**Script:** `src/train.py`

We train two separate models for different prediction tasks:
1.  **Linear Regression**: Predicts the exact `total_score` (continuous variable).
2.  **Logistic Regression**: Predicts the `grade` (categorical variable: A, B, C, D, F).

The data is split into **Train (80%)** and **Test (20%)** sets to ensure we evaluate on unseen data.

## 7. Model Evaluation & Interpretation
**Script:** `src/evaluate.py`

### Linear Regression (Total Score)
- **Mean Squared Error (MSE): 80.94**: The average squared difference between predicted and actual scores is around 81, which means the model's predictions are off by about $\sqrt{81} \approx 9$ points on average.
- **R² Score: 0.66**: The model explains **66%** of the variance in the student scores. This indicates a moderate fit; while study hours and attendance are important, other factors likely influence the score.

### Logistic Regression (Grade)
- **Accuracy: 0.70**: The model correctly predicts the exact letter grade 70% of the time.

### Detailed Analysis (F1 Score & Confusion Matrix)
The **Confusion Matrix** and **Classification Report** reveal critical insights that Accuracy hides:

**Confusion Matrix:**
```
[[97143 12162   581     2     0]  <- True Class 0 (A)
 [18503 26293  6553   207     0]  <- True Class 1 (B)
 [ 1720 11705 12833  2122     0]  <- True Class 2 (C)
 [   23   969  4803  3132     0]  <- True Class 3 (D)
 [    0    17   402   830     0]] <- True Class 4 (F)
```

1.  **Class 0 (Grade A)**: **F1 Score 0.85**.
    - The model performs well here, correctly identifying 97,143 out of 109,888 'A' students.
    - **Implication**: The dataset is heavily imbalanced with 'A' students being the majority, so the model is biased towards predicting 'A'.
2.  **Class 4 (Grade F)**: **F1 Score 0.00**.
    - **CRITICAL FAILURE**: The model fails to correctly classify *any* of the failing students (0 true positives).
    - It confuses them with Class 3 (830 times) and Class 2 (402 times).
    - **Reason**: With only ~1,200 failing students vs ~110,000 'A' students, the model ignores this minority class to maximize overall accuracy.

### Conclusion on Model Performance
- **Is this model good?**
    - For predicting **top performers**, it is decent.
    - For an **Early Warning System** (identifying at-risk students), it is **completely ineffective**.
- **Fix**: To address this, techniques like **Oversampling (SMOTE)**, **Class Weights**, or collecting more data for lower grades are necessary.

## 8. Serving Layer (Deployment)
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
