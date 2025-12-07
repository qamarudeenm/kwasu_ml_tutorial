import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def scale_features(df, target_cols=['total_score', 'grade', 'grade_encoded']):
    # Select only numerical columns for scaling, excluding targets
    feature_cols = [col for col in df.columns if col not in target_cols and df[col].dtype in ['int64', 'float64']]
    
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Save scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return df

if __name__ == "__main__":
    if not os.path.exists('dataset/student_performance_processed.csv'):
        print("Processed dataset not found! Run preprocessing.py first.")
        exit(1)

    df = pd.read_csv('dataset/student_performance_processed.csv')
    df = scale_features(df)
    print("Feature scaling complete.")
    print(df.head())
    df.to_csv('dataset/student_performance_final.csv', index=False)
