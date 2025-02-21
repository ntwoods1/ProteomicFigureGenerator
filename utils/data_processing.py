import pandas as pd
import numpy as np

def analyze_dataset_structure(df):
    """
    Analyze the structure of proteomics dataset to identify cell lines, conditions, and replicates.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        dict: Dictionary containing dataset structure information
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
        # Extract information from column name
        # Expected format: [N] CellLine_Treatment_...Rep{X}
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
                dataset_info["replicates"][group_key].append(replicate)

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

def calculate_and_filter_cv(df, cv_cutoff=None, dataset_info=None):
    """
    Calculate coefficient of variation (CV) for replicate samples and filter proteins based on CV cutoff.

    Args:
        df (pd.DataFrame): Input dataframe
        cv_cutoff (float): Maximum allowed CV percentage (e.g., 30 for 30%)
        dataset_info (dict): Dataset structure information containing replicate groupings

    Returns:
        tuple: (filtered_df, cv_stats)
    """
    if dataset_info is None:
        dataset_info = analyze_dataset_structure(df)
    if cv_cutoff is None or dataset_info is None:
        return df, {"proteins_passing_cv": len(df), "proteins_removed_cv": 0, "average_cv": 0}

    filtered_df = df.copy()
    cv_values = []

    # Get replicate groups from dataset_info
    for group, replicate_cols in dataset_info["replicates"].items():
        if len(replicate_cols) > 1:  # Only calculate CV if we have multiple replicates
            # Find the corresponding quantity columns
            quantity_cols = [col for col in df.columns if col.endswith("PG.Quantity") and any(rep in col for rep in replicate_cols)]

            if quantity_cols:
                # Calculate CV for this group
                group_data = df[quantity_cols]
                cv = (group_data.std(axis=1) / group_data.mean(axis=1) * 100)
                cv_values.extend(cv.dropna().tolist())

    if not cv_values:
        return df, {"proteins_passing_cv": len(df), "proteins_removed_cv": 0, "average_cv": 0}

    # Calculate max CV for each protein across all replicate groups
    max_cv = pd.Series(cv_values).max()

    # Apply CV filter if cutoff is provided
    if cv_cutoff is not None:
        mask = max_cv <= cv_cutoff
        filtered_df = filtered_df[mask]

    # Calculate statistics
    stats = {
        "proteins_passing_cv": len(filtered_df),
        "proteins_removed_cv": len(df) - len(filtered_df),
        "average_cv": float(np.mean(cv_values))
    }

    return filtered_df, stats