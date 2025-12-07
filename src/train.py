import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import joblib
import os

def train_models():
    if not os.path.exists('dataset/student_performance_final.csv'):
        print("Final dataset not found! Run features.py first.")
        exit(1)
        
    df = pd.read_csv('dataset/student_performance_final.csv')
    
    # Define features and targets
    # Features are all columns except targets
    target_cols = ['total_score', 'grade', 'grade_encoded']
    X = df.drop(columns=target_cols)
    
    # Target for Regression
    y_reg = df['total_score']
    
    # Target for Classification
    y_cls = df['grade_encoded']
    
    # Split data (using same seed for reproducibility)
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=42
    )
    
    # Train Linear Regression
    print("Training Linear Regression...")
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_reg_train)
    
    # Train Logistic Regression
    print("Training Logistic Regression...")
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_cls_train)
    
    # Save models
    os.makedirs('models', exist_ok=True)
    joblib.dump(lin_reg, 'models/linear_regression.pkl')
    joblib.dump(log_reg, 'models/logistic_regression.pkl')
    
    # Save test data for evaluation
    X_test.to_csv('dataset/X_test.csv', index=False)
    y_reg_test.to_csv('dataset/y_reg_test.csv', index=False)
    y_cls_test.to_csv('dataset/y_cls_test.csv', index=False)
    
    print("Models trained and saved.")

if __name__ == "__main__":
    train_models()
