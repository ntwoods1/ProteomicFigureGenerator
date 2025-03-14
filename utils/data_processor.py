import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

class DataProcessor:
    @staticmethod
    def _check_sklearn_dependency() -> bool:
        """Check if scikit-learn is available"""
        try:
            from sklearn.impute import KNNImputer
            return True
        except ImportError:
            warnings.warn(
                "scikit-learn is not installed. KNN imputation will not be available. "
                "Install it using: pip install scikit-learn"
            )
            return False

    @staticmethod
    def validate_data(df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate uploaded proteomics data
        """
        try:
            # Check if dataframe is empty
            if df.empty:
                return False, "The uploaded file is empty"

            # Check for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return False, "Data must contain at least two numeric columns"

            # Check for missing values
            missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
            if missing_pct > 50:
                return False, f"Too many missing values ({missing_pct:.1f}%)"

            return True, "Data validation successful"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess proteomics data
        """
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Fill remaining NaN values with median of respective columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Log2 transform numeric columns
        df[numeric_cols] = np.log2(df[numeric_cols].replace(0, np.nan))
        
        return df
    
    @staticmethod
    def filter_by_peptide_count(df: pd.DataFrame, peptide_cols: List[str], min_peptides: int, filter_method: str = "Any", original_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Filter proteins based on peptide count
        
        Args:
            df: Dataframe with proteomics data
            peptide_cols: List of columns containing peptide counts
            min_peptides: Minimum number of peptides required
            filter_method: How to apply the filter ("Any", "All", or "Average")
            original_df: Optional original full dataframe containing peptide columns
            
        Returns:
            Filtered dataframe
        """
        # If no peptide columns provided, return original dataframe
        if not peptide_cols:
            return df
        
        # Use original_df if provided, otherwise use df
        peptide_df = original_df if original_df is not None else df
        
        # Filter columns that actually exist in the peptide dataframe
        valid_peptide_cols = [col for col in peptide_cols if col in peptide_df.columns]
        
        if not valid_peptide_cols:
            # Try to find similar peptide columns
            for col in df.columns:
                if "NrOfPrecursorsMeasured" in col or "NrOfStrippedSequencesIdentified" in col:
                    valid_peptide_cols.append(col)
            
            if not valid_peptide_cols:
                return df
        
        # Make sure there's a common index to align dataframes
        protein_col = next((col for col in df.columns if col == 'PG.Genes'), None)
        if protein_col and protein_col in peptide_df.columns:
            # Use protein ID as index for alignment
            df_index = df[protein_col].copy()
            peptide_df_index = peptide_df[protein_col].copy()
        else:
            # No common protein ID column, use default index
            df_index = df.index
            peptide_df_index = peptide_df.index
        
        if "Any" in filter_method:
            # Keep row if ANY peptide count column meets threshold
            # Make sure we only use columns that exist in the peptide dataframe
            cols_to_use = [col for col in valid_peptide_cols if col in peptide_df.columns]
            if not cols_to_use:
                return df.copy()
            
            mask = peptide_df[cols_to_use].ge(min_peptides).any(axis=1)
            
            # Extract the indices of rows to keep
            indices_to_keep = peptide_df_index[mask]
            
            # Apply filtering on the original dataframe
            if protein_col and protein_col in df.columns:
                return df[df[protein_col].isin(indices_to_keep)].copy()
            else:
                # If using default index
                return df[df.index.isin(indices_to_keep)].copy()
            
        elif "All" in filter_method:
            # Keep row if ALL peptide count columns meet threshold
            # Make sure we only use columns that exist in the peptide dataframe
            cols_to_use = [col for col in valid_peptide_cols if col in peptide_df.columns]
            if not cols_to_use:
                return df.copy()
            
            mask = peptide_df[cols_to_use].ge(min_peptides).all(axis=1)
            
            # Extract the indices of rows to keep
            indices_to_keep = peptide_df_index[mask]
            
            # Apply filtering on the original dataframe
            if protein_col and protein_col in df.columns:
                return df[df[protein_col].isin(indices_to_keep)].copy()
            else:
                # If using default index
                return df[df.index.isin(indices_to_keep)].copy()
            
        elif "Average" in filter_method:
            # Keep row if AVERAGE peptide count meets threshold
            # Make sure we only use columns that exist in the peptide dataframe
            cols_to_use = [col for col in valid_peptide_cols if col in peptide_df.columns]
            if not cols_to_use:
                return df.copy()
            
            avg_peptides = peptide_df[cols_to_use].mean(axis=1)
            mask = avg_peptides >= min_peptides
            
            # Extract the indices of rows to keep
            indices_to_keep = peptide_df_index[mask]
            
            # Apply filtering on the original dataframe
            if protein_col and protein_col in df.columns:
                return df[df[protein_col].isin(indices_to_keep)].copy()
            else:
                # If using default index
                return df[df.index.isin(indices_to_keep)].copy()
            
        else:
            # Default to "Any" method
            # Make sure we only use columns that exist in the peptide dataframe
            cols_to_use = [col for col in valid_peptide_cols if col in peptide_df.columns]
            if not cols_to_use:
                return df.copy()
            
            mask = peptide_df[cols_to_use].ge(min_peptides).any(axis=1)
            
            # Extract the indices of rows to keep
            indices_to_keep = peptide_df_index[mask]
            
            # Apply filtering on the original dataframe
            if protein_col and protein_col in df.columns:
                return df[df[protein_col].isin(indices_to_keep)].copy()
            else:
                # If using default index
                return df[df.index.isin(indices_to_keep)].copy()
    
    @staticmethod
    def impute_missing_values(df: pd.DataFrame, method: str, group_selections: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """
        Impute missing values in proteomics data
        
        Args:
            df: Dataframe with proteomics data
            method: Imputation method (Mean, Median, Min, KNN, Group-wise Mean)
            group_selections: Dictionary mapping group names to column lists (for group-specific imputation)
        
        Returns:
            Dataframe with imputed values
        """
        imputed_df = df.copy()
        
        if method == "None":
            return imputed_df
        
        # Get numeric columns
        numeric_cols = imputed_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == "KNN":
            if not DataProcessor._check_sklearn_dependency():
                warnings.warn("Falling back to mean imputation as scikit-learn is not available")
                method = "Mean"
            else:
                try:
                    from sklearn.impute import KNNImputer
                    imputer = KNNImputer(n_neighbors=5)
                    imputed_values = imputer.fit_transform(imputed_df[numeric_cols])
                    imputed_df[numeric_cols] = pd.DataFrame(imputed_values, columns=numeric_cols, index=imputed_df.index)
                    
                    # Double-check for any remaining NaNs after KNN
                    for col in numeric_cols:
                        if imputed_df[col].isna().any():
                            col_mean = imputed_df[col].mean()
                            if pd.isna(col_mean):
                                col_mean = 0
                            imputed_df[col] = imputed_df[col].fillna(col_mean)
                    return imputed_df
                except Exception as e:
                    warnings.warn(f"KNN imputation failed: {str(e)}. Falling back to mean imputation.")
                    method = "Mean"
        
        if method == "Mean":
            # Impute with mean of each column
            for col in numeric_cols:
                col_mean = imputed_df[col].mean()
                # Handle case where entire column might be NaN
                if pd.isna(col_mean):
                    col_mean = 0  # Use 0 as fallback if entire column is NaN
                imputed_df[col] = imputed_df[col].fillna(col_mean)
                
        elif method == "Median":
            # Impute with median of each column
            for col in numeric_cols:
                col_median = imputed_df[col].median()
                # Handle case where entire column might be NaN
                if pd.isna(col_median):
                    col_median = 0  # Use 0 as fallback if entire column is NaN
                imputed_df[col] = imputed_df[col].fillna(col_median)
                
        elif method == "Min":
            # Impute with minimum value of each row (scaled by factor)
            # For each row with missing values, calculate row min and use min/2
            for idx in imputed_df.index:
                row_data = imputed_df.loc[idx, numeric_cols]
                if row_data.isna().any():  # Only process rows with missing values
                    row_min = row_data.min()
                    # Handle case where entire row might be NaN
                    if pd.isna(row_min):
                        row_min = 0  # Use 0 as fallback if entire row is NaN
                    else:
                        row_min = row_min / 2  # Use min/2 for missing value imputation
                        
                    # Apply the row-specific minimum to missing values in this row
                    for col in numeric_cols:
                        if pd.isna(imputed_df.loc[idx, col]):
                            imputed_df.loc[idx, col] = row_min
                
        elif method == "Group-wise Mean" and group_selections:
            # Impute with mean of each group
            for group_name, columns in group_selections.items():
                for col in columns:
                    if col in numeric_cols:
                        col_mean = imputed_df[col].mean()
                        if pd.isna(col_mean):
                            col_mean = 0
                        imputed_df[col] = imputed_df[col].fillna(col_mean)
            
            # Check for any columns that might not be in any group and handle them with global mean
            ungrouped_cols = [col for col in numeric_cols if not any(col in group_cols for group_cols in group_selections.values())]
            for col in ungrouped_cols:
                if imputed_df[col].isna().any():
                    col_mean = imputed_df[col].mean()
                    if pd.isna(col_mean):
                        col_mean = 0
                    imputed_df[col] = imputed_df[col].fillna(col_mean)
        
        # Final check for any remaining NaNs and handle with zeros as last resort
        has_remaining_nan = imputed_df[numeric_cols].isna().any().any()
        if has_remaining_nan:
            imputed_df[numeric_cols] = imputed_df[numeric_cols].fillna(0)
            
        return imputed_df
    
    @staticmethod
    def filter_by_valid_values(df: pd.DataFrame, group_selections: Dict[str, List[str]], min_valid_pct: float) -> pd.DataFrame:
        """
        Filter proteins based on percentage of valid values in each group
        
        Args:
            df: Dataframe with proteomics data
            group_selections: Dictionary mapping group names to column lists
            min_valid_pct: Minimum percentage of valid values required (0-1)
            
        Returns:
            Filtered dataframe
        """
        protein_col = next((col for col in df.columns if col == 'PG.Genes'), None)
        if not protein_col:
            # Try to find any protein ID column
            protein_col = next((col for col in df.columns if 'protein' in col.lower() or 'gene' in col.lower()), None)
        
        # If we can't find a protein column, just use the first column
        if not protein_col:
            protein_col = df.columns[0]
            
        filtered_df = df.copy()
        
        # Check each group separately
        for group_name, columns in group_selections.items():
            # Calculate valid values percentage for each protein in this group
            valid_counts = filtered_df[columns].notna().sum(axis=1)
            valid_pct = valid_counts / len(columns)
            
            # Filter proteins with enough valid values
            filtered_df = filtered_df[valid_pct >= min_valid_pct]
            
        return filtered_df
    
    @staticmethod
    def filter_by_cv(df: pd.DataFrame, group_selections: Dict[str, List[str]], max_cv: float) -> pd.DataFrame:
        """
        Filter proteins based on coefficient of variation (CV) within groups
        
        Args:
            df: Dataframe with proteomics data
            group_selections: Dictionary mapping group names to column lists
            max_cv: Maximum CV allowed (0-1)
            
        Returns:
            Filtered dataframe
        """
        filtered_df = df.copy()
        
        # Create a mask of proteins to keep
        keep_mask = pd.Series(True, index=filtered_df.index)
        
        # Check each group separately
        for group_name, columns in group_selections.items():
            # Calculate CV for each protein in this group
            # CV = std / mean
            group_data = filtered_df[columns]
            cv = group_data.std(axis=1) / group_data.mean(axis=1).abs()
            
            # Update mask to only keep proteins with acceptable CV
            keep_mask = keep_mask & (cv <= max_cv)
            
        return filtered_df[keep_mask]
    
    @staticmethod
    def normalize_data(df: pd.DataFrame, method: str, columns: List[str]) -> pd.DataFrame:
        """
        Normalize proteomics data using various methods
        
        Args:
            df: Dataframe with proteomics data
            method: Normalization method (Log2, Median, Z-score, Quantile, Total Intensity)
            columns: Columns to normalize
            
        Returns:
            Normalized dataframe
        """
        normalized_df = df.copy()
        
        if method == "Log2":
            # Apply log2 transformation
            normalized_df[columns] = np.log2(normalized_df[columns].replace(0, np.nan))
            
        elif method == "Median":
            # Median normalization (subtract the median of each sample)
            for col in columns:
                col_median = normalized_df[col].median()
                normalized_df[col] = normalized_df[col] - col_median
                
        elif method == "Z-score":
            # Z-score normalization ((x - mean) / std)
            for col in columns:
                col_mean = normalized_df[col].mean()
                col_std = normalized_df[col].std()
                normalized_df[col] = (normalized_df[col] - col_mean) / col_std
                
        elif method == "Quantile":
            # Quantile normalization
            normalized_df[columns] = DataProcessor._quantile_normalize(normalized_df[columns])
            
        elif method == "Total Intensity":
            # Total intensity normalization (divide by sum of all values in column)
            for col in columns:
                total_intensity = normalized_df[col].sum()
                if total_intensity > 0:  # Avoid division by zero
                    normalized_df[col] = normalized_df[col] / total_intensity
            
        return normalized_df
    
    @staticmethod
    def _quantile_normalize(df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform quantile normalization on a dataframe
        
        Args:
            df: Dataframe with numeric columns to normalize
            
        Returns:
            Normalized dataframe
        """
        # Get rank of each value within its column
        rank_mean = df.stack().groupby(df.rank(method='first').stack().astype(int)).mean()
        
        # Replace each value with the mean of the corresponding rank
        return df.rank(method='min').stack().astype(int).map(rank_mean).unstack()