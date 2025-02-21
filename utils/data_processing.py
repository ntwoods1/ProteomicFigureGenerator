import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from statsmodels.nonparametric.smoothers_lowess import lowess

def analyze_dataset_structure(df):
    """
    Analyze the structure of proteomics dataset to identify cell lines, conditions, and replicates.
    """
    # Find all quantity columns
    quantity_cols = [col for col in df.columns if col.endswith("PG.Quantity")]

    # Parse sample information from column names
    dataset_info = {
        "quantity_columns": quantity_cols,
        "cell_lines": set(),
        "conditions": set(),
        "replicates": {}
    }

    for col in quantity_cols:
        try:
            # Remove the bracketed number and .PG.Quantity suffix
            sample_info = col.split("]")[1].split(".PG.Quantity")[0].strip()
            parts = sample_info.split("_")

            if len(parts) >= 3:  # Ensure we have enough parts
                cell_line = parts[0]
                # Treatment is everything between cell line and replicate
                treatment = "_".join(parts[1:-1])
                replicate = parts[-1]

                dataset_info["cell_lines"].add(cell_line)
                dataset_info["conditions"].add(treatment)

                # Group replicates by cell line and treatment
                group_key = f"{cell_line}_{treatment}"
                if group_key not in dataset_info["replicates"]:
                    dataset_info["replicates"][group_key] = []
                dataset_info["replicates"][group_key].append(col)  # Store full column name

        except Exception as e:
            print(f"Error parsing column name {col}: {str(e)}")
            continue

    # Convert sets to sorted lists for better display
    dataset_info["cell_lines"] = sorted(list(dataset_info["cell_lines"]))
    dataset_info["conditions"] = sorted(list(dataset_info["conditions"]))

    # Add summary statistics
    dataset_info["summary"] = {
        "num_cell_lines": len(dataset_info["cell_lines"]),
        "num_conditions": len(dataset_info["conditions"]),
        "num_quantity_columns": len(quantity_cols)
    }

    return dataset_info

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

    # Check for peptide count columns
    peptide_cols = [col for col in df.columns if col.endswith("PG.NrOfStrippedSequencesIdentified")]
    if not peptide_cols:
        validation_results["warnings"].append("No peptide count columns found (ending with 'PG.NrOfStrippedSequencesIdentified')")

    return validation_results

def filter_by_peptide_count(df, min_peptides=1):
    """Filter proteins based on the number of peptides used for identification."""
    peptide_cols = [col for col in df.columns if col.endswith("PG.NrOfStrippedSequencesIdentified")]

    if not peptide_cols:
        raise ValueError("No peptide count columns found (ending with 'PG.NrOfStrippedSequencesIdentified')")

    # Get maximum peptide count for each protein across all samples
    max_peptides = df[peptide_cols].max(axis=1)

    # Filter based on threshold
    mask = max_peptides >= min_peptides
    filtered_df = df[mask].copy()

    # Calculate statistics
    stats_dict = {
        "total_proteins": len(df),
        "proteins_passing_filter": len(filtered_df),
        "proteins_removed": len(df) - len(filtered_df),
        "peptide_threshold": min_peptides,
        "max_peptides_found": max_peptides.max()
    }

    return filtered_df, stats_dict

def calculate_and_filter_cv(df, cv_cutoff=None, dataset_info=None):
    """Calculate coefficient of variation (CV) for replicate samples and filter proteins based on CV cutoff."""
    if dataset_info is None:
        dataset_info = analyze_dataset_structure(df)
    if cv_cutoff is None:
        return df, {"proteins_passing_cv": len(df), "proteins_removed_cv": 0, "average_cv": 0}

    filtered_df = df.copy()
    cv_values = []

    # Process each replicate group separately
    for group, replicate_cols in dataset_info["replicates"].items():
        if len(replicate_cols) > 1:  # Only calculate CV if we have multiple replicates
            # Extract data for this group's replicates
            group_data = df[replicate_cols]

            # Calculate CV for this group
            group_cv = (group_data.std(axis=1) / group_data.mean(axis=1) * 100)
            cv_values.extend(group_cv.dropna().tolist())

            # Update mask for this group
            if cv_cutoff is not None:
                mask = group_cv <= cv_cutoff
                filtered_df = filtered_df[mask]

    if not cv_values:
        return df, {"proteins_passing_cv": len(df), "proteins_removed_cv": 0, "average_cv": 0}

    # Calculate statistics
    stats = {
        "proteins_passing_cv": len(filtered_df),
        "proteins_removed_cv": len(df) - len(filtered_df),
        "average_cv": float(np.mean(cv_values))
    }

    return filtered_df, stats

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

def normalize_data(df, method="log2", center_scale=True, center_method="zscore", quantity_only=True):
    """Normalize data using specified method."""
    df_norm = df.copy()

    # Select columns to normalize
    if quantity_only:
        numeric_cols = [col for col in df.columns if col.endswith("PG.Quantity")]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns

    if not numeric_cols:
        raise ValueError("No numeric columns found for normalization")

    # First apply column-wise normalization
    if method == "log2":
        # Add small constant to avoid log(0)
        min_positive = df_norm[numeric_cols].replace(0, np.nan).min().min()
        offset = min_positive * 0.01 if min_positive > 0 else 0.01
        df_norm[numeric_cols] = np.log2(df_norm[numeric_cols] + offset)
    elif method == "zscore":
        scaler = StandardScaler()
        df_norm[numeric_cols] = scaler.fit_transform(df_norm[numeric_cols])
    elif method == "median":
        df_norm[numeric_cols] = df_norm[numeric_cols].subtract(
            df_norm[numeric_cols].median(axis=1), axis=0
        )
    elif method == "loess":
        # Apply LOESS normalization to each column
        for col in numeric_cols:
            y = df_norm[col].values
            x = np.arange(len(y))
            # Remove NaN values for LOESS
            mask = ~np.isnan(y)
            if sum(mask) > 2:  # Need at least 3 points for LOESS
                y_smoothed = lowess(y[mask], x[mask], frac=0.3, it=3, return_sorted=False)
                # Normalize to the smoothed curve
                df_norm.loc[mask, col] = y[mask] - y_smoothed + np.median(y[mask])

    # Then apply row centering if requested
    if center_scale and center_method:
        if center_method == "zscore":
            # Center on 0 with standard deviation of 1
            row_means = df_norm[numeric_cols].mean(axis=1)
            row_stds = df_norm[numeric_cols].std(axis=1)
            df_norm[numeric_cols] = df_norm[numeric_cols].subtract(row_means, axis=0).div(row_stds, axis=0)
        elif center_method == "scale100":
            # Center on 100
            row_means = df_norm[numeric_cols].mean(axis=1)
            scaling_factor = 100 / row_means
            df_norm[numeric_cols] = df_norm[numeric_cols].multiply(scaling_factor, axis=0)

    return df_norm

def handle_missing_values(df, method="knn", min_valid_values=0.5):
    """Handle missing values using specified method."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_clean = df.copy()

    # Remove rows with too many missing values
    valid_counts = df_clean[numeric_cols].notna().sum(axis=1)
    df_clean = df_clean[valid_counts >= (len(numeric_cols) * min_valid_values)]

    if method == "knn":
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
        "cv_per_sample": (df[numeric_cols].std() / df[numeric_cols].mean() * 100).to_dict(),
        "detection_rates": (df[numeric_cols].notna().sum(axis=1) / len(numeric_cols)).describe().to_dict()
    }

    return metrics

def detect_outliers(df, method="zscore", threshold=3):
    """Detect outliers using specified method."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = pd.DataFrame(index=df.index, columns=numeric_cols)

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