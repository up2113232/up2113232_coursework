import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def clean_data(df):
    
    # Clean the gaming anxiety dataset
    
    # Parameters:
    # df (pd.DataFrame): Raw dataset
    
    # Returns:
    # pd.DataFrame: Cleaned dataset
    
    if df is None:
        return None
    
    df_clean = df.copy()
    
    # Check for missing values
    print("Missing values per column:")
    print(df_clean.isnull().sum())
    
    # Handle missing values if any
    if df_clean.isnull().sum().sum() > 0:
        # For numerical columns, fill with median
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

    # Remove rows with ANY missing values
    rows_before = len(df_clean)
    df_clean = df_clean.dropna()
    rows_after = len(df_clean)
    rows_removed = rows_before - rows_after
    print(f"Removed {rows_removed} rows with missing values")

    
    
    # Remove duplicates to help with ML training
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    print(f"Removed {duplicates_removed} duplicate rows")
    
    return df_clean

def encode_features(data):
    
    # Encode categorical features for ML models
         
    label_encoders = {}
    
    for col in ['GADE', 'Game', 'earnings', 'whyplay', 'League', 
            'Gender', 'Work', 'Playstyle']:
        
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le

    return data

def split_data(X, y, test_size=0.2, random_state=42):
    
    # Split data into train and test sets
    
    # Parameters:
    # X : Features
    # y : Targets
    # test_size: Percentage for testing set
    # random_state: Random seed so we can recreate results
    
    # Returns:
    # X_train, X_test, y_train, y_test
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_features(X_train, X_test):
    
    # Scale features using StandardScaler
    
    # Parameters:
    # X_train : Training features
    # X_test (): Test features
    
    # Returns:
    # Scaled X_train, X_test, and the scaler object
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


