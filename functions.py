import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def get_data(file_path='gaming_anxiety_data.csv'):
    
    # Load and return the gaming anxiety dataset
    
    # Parameters:
    # file_path (str): Path to the CSV file
    
    # Returns:
    # pd.DataFrame: Loaded dataset
    
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

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
    
    # Remove duplicates to help with ML training
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    print(f"Removed {duplicates_removed} duplicate rows")
    
    return df_clean
