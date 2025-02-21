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

def filter_proteins(df, min_detection_rate=0.5, min_samples=2):
    """Filter proteins based on detection rate and minimum sample count."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Calculate detection rate for each protein
    detection_rates = (df[numeric_cols].notna().sum(axis=1) / len(numeric_cols))
    min_samples_detected = df[numeric_cols].notna().sum(axis=1)

    # Apply filters
    mask = (detection_rates >= min_detection_rate) & (min_samples_detected >= min_samples)
    filtered_df = df[mask].copy()

    return filtered_df, {
        "total_proteins": len(df),
        "filtered_proteins": len(filtered_df),
        "removed_proteins": len(df) - len(filtered_df)
    }

def normalize_data(df, method="log2", center_scale=True):
    """Normalize data using specified method."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_norm = df.copy()

    if method == "log2":
        # Add small constant to avoid log(0)
        min_positive = df_norm[numeric_cols].replace(0, np.nan).min().min()
        offset = min_positive * 0.01 if min_positive > 0 else 0.01
        df_norm[numeric_cols] = np.log2(df_norm[numeric_cols] + offset)
    elif method == "zscore":
        scaler = StandardScaler()
        df_norm[numeric_cols] = scaler.fit_transform(df_norm[numeric_cols])
    elif method == "median":
        if center_scale:
            df_norm[numeric_cols] = df_norm[numeric_cols].subtract(
                df_norm[numeric_cols].median(axis=1), axis=0
            )

    return df_norm

def handle_missing_values(df, method="knn", min_valid_values=0.5):
    """Handle missing values using specified method."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_clean = df.copy()

    # Remove rows with too many missing values
    valid_counts = df_clean[numeric_cols].notna().sum(axis=1)
    df_clean = df_clean[valid_counts >= (len(numeric_cols) * min_valid_values)]

    if method == "knn":
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=3)
        df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
    elif method == "mean":
        imputer = SimpleImputer(strategy="mean")
        df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
    elif method == "median":
        imputer = SimpleImputer(strategy="median")
        df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
    elif method == "constant":
        imputer = SimpleImputer(strategy="constant", fill_value=0)
        df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])

    return df_clean

def calculate_quality_metrics(df):
    """Calculate quality control metrics for the dataset."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    metrics = {
        "total_proteins": len(df),
        "missing_values_per_sample": df[numeric_cols].isnull().sum().to_dict(),
        "missing_values_percentage": (df[numeric_cols].isnull().sum() / len(df) * 100).to_dict(),
        "cv_per_sample": df[numeric_cols].std() / df[numeric_cols].mean() * 100,
        "detection_rates": (df[numeric_cols].notna().sum(axis=1) / len(numeric_cols)).describe().to_dict()
    }

    return metrics

def detect_outliers(df, method="zscore", threshold=3):
    """Detect outliers using specified method."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = pd.DataFrame(index=df.index)

    if method == "zscore":
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
            outliers[col] = z_scores > threshold
    elif method == "iqr":
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers[col] = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))

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