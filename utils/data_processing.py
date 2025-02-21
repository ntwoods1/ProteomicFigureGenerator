import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def validate_data(df):
    """Validate uploaded data format and content."""
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check for required columns
    required_cols = ["Description"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        validation_results["valid"] = False
        validation_results["errors"].append(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Check for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        validation_results["valid"] = False
        validation_results["errors"].append("At least 2 numeric columns required for analysis")
        
    return validation_results

def normalize_data(df, method="log2"):
    """Normalize data using specified method."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_norm = df.copy()
    
    if method == "log2":
        df_norm[numeric_cols] = np.log2(df_norm[numeric_cols])
    elif method == "zscore":
        scaler = StandardScaler()
        df_norm[numeric_cols] = scaler.fit_transform(df_norm[numeric_cols])
    
    return df_norm

def handle_missing_values(df, method="mean"):
    """Handle missing values using specified method."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_clean = df.copy()
    
    imputer = SimpleImputer(strategy=method)
    df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
    
    return df_clean

def detect_outliers(df, threshold=3):
    """Detect outliers using z-score method."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = pd.DataFrame(index=df.index)
    
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col]))
        outliers[col] = z_scores > threshold
        
    return outliers

def batch_correct(df, batch_col, data_cols):
    """Perform batch effect correction using ComBat-like approach."""
    # Simplified batch correction using mean-centering
    df_corrected = df.copy()
    batches = df[batch_col].unique()
    
    for batch in batches:
        batch_mask = df[batch_col] == batch
        for col in data_cols:
            batch_mean = df_corrected.loc[batch_mask, col].mean()
            df_corrected.loc[batch_mask, col] -= batch_mean
            
    return df_corrected
