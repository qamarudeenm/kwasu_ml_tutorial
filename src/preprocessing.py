import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    # Drop student_id as it's not a feature
    if 'student_id' in df.columns:
        df = df.drop(columns=['student_id'])
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print("Missing values found. Filling with mean for numerical and mode for categorical.")
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    
    # Encode 'grade' column
    le = LabelEncoder()
    df['grade_encoded'] = le.fit_transform(df['grade'])
    
    # Save the label encoder for later use
    os.makedirs('models', exist_ok=True)
    joblib.dump(le, 'models/label_encoder.pkl')
    
    return df

if __name__ == "__main__":
    # Ensure dataset exists
    if not os.path.exists('dataset/student_performance.csv'):
        print("Dataset not found!")
        exit(1)
        
    df = load_data('dataset/student_performance.csv')
    print(df)
    df = preprocess_data(df)
    print("Data preprocessing complete.")
    print(df.head())
    
    # Save processed data
    df.to_csv('dataset/student_performance_processed.csv', index=False)
