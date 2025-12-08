import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import joblib
import os

def evaluate_models():
    if not os.path.exists('models/linear_regression.pkl'):
        print("Models not found! Run train.py first.")
        exit(1)
        
    # Load test data
    X_test = pd.read_csv('dataset/X_test.csv')
    y_reg_test = pd.read_csv('dataset/y_reg_test.csv').values.ravel()
    y_cls_test = pd.read_csv('dataset/y_cls_test.csv').values.ravel()
    
    # Load models
    lin_reg = joblib.load('models/linear_regression.pkl')
    log_reg = joblib.load('models/logistic_regression.pkl')
    
    # Evaluate Linear Regression
    print("--- Linear Regression Evaluation ---")
    y_reg_pred = lin_reg.predict(X_test)
    mse = mean_squared_error(y_reg_test, y_reg_pred)
    r2 = r2_score(y_reg_test, y_reg_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    
    # Evaluate Logistic Regression
    print("\n--- Logistic Regression Evaluation ---")
    y_cls_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_cls_test, y_cls_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_cls_test, y_cls_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_cls_test, y_cls_pred))

if __name__ == "__main__":
    evaluate_models()
