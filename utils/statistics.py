import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple, List, Optional
import warnings

class StatisticalAnalysis:
    @staticmethod
    def _check_statsmodels_dependency() -> bool:
        """Check if statsmodels is available"""
        try:
            import statsmodels
            return True
        except ImportError:
            warnings.warn(
                "statsmodels is not installed. Some statistical functions may not be available. "
                "Install it using: pip install statsmodels"
            )
            return False

    @staticmethod
    def _check_sklearn_dependency() -> bool:
        """Check if scikit-learn is available"""
        try:
            import sklearn
            return True
        except ImportError:
            warnings.warn(
                "scikit-learn is not installed. Some analysis functions may not be available. "
                "Install it using: pip install scikit-learn"
            )
            return False

    @staticmethod
    def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate descriptive statistics"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats_df = df[numeric_cols].describe()
        stats_df.loc['variance'] = df[numeric_cols].var()
        stats_df.loc['skewness'] = df[numeric_cols].skew()
        return stats_df

    @staticmethod
    def ttest_analysis(group1: pd.Series, group2: pd.Series) -> Dict[str, float]:
        """Perform t-test between two groups"""
        t_stat, p_val = stats.ttest_ind(
            group1.dropna(),
            group2.dropna(),
            equal_var=False
        )
        return {
            't_statistic': t_stat,
            'p_value': p_val
        }

    @staticmethod
    def calculate_power(control_values, treatment_values, alpha=0.05, effect_size=None) -> float:
        """Calculate statistical power for t-test comparison between two groups"""
        if not StatisticalAnalysis._check_statsmodels_dependency():
            return 0.0

        try:
            from statsmodels.stats.power import TTestIndPower

            # Remove NaN values
            control_values = np.array(control_values)[~np.isnan(np.array(control_values))]
            treatment_values = np.array(treatment_values)[~np.isnan(np.array(treatment_values))]

            # If either group has no valid values, return 0 power
            if len(control_values) == 0 or len(treatment_values) == 0:
                return 0.0

            # Calculate effect size (Cohen's d) if not provided
            if effect_size is None:
                mean1, mean2 = np.mean(control_values), np.mean(treatment_values)
                std1, std2 = np.std(control_values, ddof=1), np.std(treatment_values, ddof=1)

                # Pooled standard deviation
                n1, n2 = len(control_values), len(treatment_values)
                pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))

                # Avoid division by zero
                if pooled_std == 0:
                    return 0.0

                effect_size = abs((mean2 - mean1) / pooled_std)

            # Initialize power analysis
            power_analysis = TTestIndPower()

            # Calculate power
            n1, n2 = len(control_values), len(treatment_values)
            power = power_analysis.power(effect_size=effect_size, 
                                      nobs1=n1, 
                                      ratio=n2/n1 if n1 > 0 else 1, 
                                      alpha=alpha)

            return power
        except Exception as e:
            warnings.warn(f"Error calculating statistical power: {str(e)}")
            return 0.0

    @staticmethod
    def anova_analysis(groups: Dict[str, pd.Series]) -> Dict[str, float]:
        """Perform one-way ANOVA"""
        group_data = [group.dropna() for group in groups.values()]
        f_stat, p_val = stats.f_oneway(*group_data)
        return {
            'f_statistic': f_stat,
            'p_value': p_val
        }
    
    @staticmethod
    def correlation_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate correlation matrix and p-values"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()

        # Calculate p-values
        p_values = pd.DataFrame(np.zeros_like(corr_matrix), 
                              columns=corr_matrix.columns, 
                              index=corr_matrix.index)

        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                if i != j:
                    stat, p = stats.pearsonr(df[numeric_cols.iloc[i]].dropna(), 
                                          df[numeric_cols.iloc[j]].dropna())
                    p_values.iloc[i,j] = p

        return corr_matrix, p_values

    @staticmethod
    def calculate_study_power(df, control_cols, treatment_cols, alpha=0.05, fc_threshold=1.0, max_proteins=500):
        """
        Calculate overall statistical power for the study based on observed data
        
        Args:
            df: DataFrame containing protein intensity data
            control_cols: Column names for control group samples
            treatment_cols: Column names for treatment group samples
            alpha: Significance level (default: 0.05)
            fc_threshold: Log2 fold change threshold used for significance
            max_proteins: Maximum number of proteins to use in calculation (for performance)
            
        Returns:
            dict: Contains mean_power, median_power, power_by_effect_size, and sample_size_info
        """
        import numpy as np
        import pandas as pd
        
        total_proteins = len(df)
        
        # Calculate log2FC for all proteins
        control_mean = df[control_cols].mean(axis=1)
        treatment_mean = df[treatment_cols].mean(axis=1)
        epsilon = 1e-10
        ratio = (treatment_mean + epsilon) / (control_mean + epsilon)
        log2fc = np.log2(ratio)
        
        # Calculate p-values using t-test
        p_values = []
        indices_with_data = []
        effect_size_list = []
        
        # Calculate effect sizes and p-values for all proteins with sufficient data
        for i, (index, row) in enumerate(df.iterrows()):
            control_values = row[control_cols].values.astype(float)
            treatment_values = row[treatment_cols].values.astype(float)
            
            # Remove NaN values
            control_values = control_values[~np.isnan(control_values)]
            treatment_values = treatment_values[~np.isnan(treatment_values)]
            
            # Only process proteins with enough data
            if len(control_values) >= 2 and len(treatment_values) >= 2:
                # Calculate effect size (Cohen's d)
                mean1, mean2 = np.mean(control_values), np.mean(treatment_values)
                std1, std2 = np.std(control_values, ddof=1), np.std(treatment_values, ddof=1)
                
                # Pooled standard deviation
                n1, n2 = len(control_values), len(treatment_values)
                pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
                
                # Skip if pooled_std is zero
                if pooled_std > 0:
                    effect_size = abs((mean2 - mean1) / pooled_std)
                    effect_size_list.append((i, effect_size))
                    indices_with_data.append(i)
                    
                    try:
                        # Calculate p-value
                        from scipy import stats
                        _, p_val = stats.ttest_ind(
                            treatment_values, 
                            control_values, 
                            equal_var=False,
                            nan_policy='omit'
                        )
                        p_values.append((i, p_val))
                    except:
                        pass
        
        # Filter proteins based solely on fold change
        significant_proteins = []
        
        # Create dictionaries for faster lookup
        effect_size_dict = dict(effect_size_list)
        p_value_dict = dict(p_values)
        
        # Calculate log2FC for protein selection
        log2fc_dict = {}
        for i in indices_with_data:
            row = df.iloc[i]
            control_values = row[control_cols].values.astype(float)
            treatment_values = row[treatment_cols].values.astype(float)
            
            # Remove NaN values
            control_values = control_values[~np.isnan(control_values)]
            treatment_values = treatment_values[~np.isnan(treatment_values)]
            
            if len(control_values) > 0 and len(treatment_values) > 0:
                mean_control = np.mean(control_values)
                mean_treatment = np.mean(treatment_values)
                epsilon = 1e-10  # Small value to prevent log(0)
                ratio = (mean_treatment + epsilon) / (mean_control + epsilon)
                log2fc_dict[i] = np.log2(ratio)
        
        # Select proteins that pass only the fold change threshold, regardless of p-value
        for i in indices_with_data:
            if i in log2fc_dict and abs(log2fc_dict[i]) >= fc_threshold:
                significant_proteins.append(i)
        
        # If we have significant proteins, use them
        if significant_proteins:
            indices = significant_proteins
            sampling_method = "significant_log2fc_only"
        else:
            # If no proteins pass the fold change threshold, use all proteins with data
            indices = indices_with_data
            sampling_method = "all"
        
        # Calculate power for sampled proteins
        powers = []
        effect_sizes = []
        
        for i in indices:
            row = df.iloc[i]
            control_values = row[control_cols].values.astype(float)
            treatment_values = row[treatment_cols].values.astype(float)
            
            # Remove NaN values
            control_values = control_values[~np.isnan(control_values)]
            treatment_values = treatment_values[~np.isnan(treatment_values)]
            
            # Skip if not enough values
            if len(control_values) < 2 or len(treatment_values) < 2:
                continue
                
            # Calculate effect size (Cohen's d)
            mean1, mean2 = np.mean(control_values), np.mean(treatment_values)
            std1, std2 = np.std(control_values, ddof=1), np.std(treatment_values, ddof=1)
            
            # Pooled standard deviation
            n1, n2 = len(control_values), len(treatment_values)
            pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
            
            # Skip if pooled_std is zero
            if pooled_std == 0:
                continue
                
            effect_size = abs((mean2 - mean1) / pooled_std)
            effect_sizes.append(effect_size)
            
            # Calculate power
            from statsmodels.stats.power import TTestIndPower
            power_analysis = TTestIndPower()
            power = power_analysis.power(
                effect_size=effect_size, 
                nobs1=n1, 
                ratio=n2/n1 if n1 > 0 else 1, 
                alpha=alpha
            )
            powers.append(power)
        
        # Calculate mean and median power
        valid_powers = [p for p in powers if p is not None and not np.isnan(p)]
        mean_power = np.mean(valid_powers) if valid_powers else 0
        median_power = np.median(valid_powers) if valid_powers else 0
        
        # Calculate how power varies with effect size
        power_by_effect_size = []
        if effect_sizes and powers:
            combined = list(zip(effect_sizes, powers))
            combined.sort(key=lambda x: x[0])  # Sort by effect size
            
            # Group into bins by effect size
            bins = {}
            for es, pwr in combined:
                bin_key = round(es * 2) / 2  # Round to nearest 0.5
                if bin_key not in bins:
                    bins[bin_key] = []
                bins[bin_key].append(pwr)
            
            # Calculate average power for each bin
            for es_bin, bin_powers in sorted(bins.items()):
                power_by_effect_size.append({
                    'effect_size': es_bin,
                    'mean_power': np.mean(bin_powers),
                    'count': len(bin_powers)
                })
        
        # Include information about sampling
        sample_size_info = {
            'total_proteins': total_proteins,
            'proteins_sampled': len(valid_powers),
            'sampling_method': sampling_method,
            'control_samples': len(control_cols),
            'treatment_samples': len(treatment_cols)
        }
        
        return {
            'mean_power': mean_power,
            'median_power': median_power,
            'power_by_effect_size': power_by_effect_size,
            'sample_size_info': sample_size_info,
            'fc_threshold': fc_threshold
        }
        
    @staticmethod
    def permutation_test(df: pd.DataFrame, control_cols: List[str], 
                       treatment_cols: List[str], original_p_values: List[float], 
                       n_permutations: int = 1000) -> List[float]:
        """
        Perform permutation test for multiple hypothesis correction
        
        Args:
            df: DataFrame with proteomics data
            control_cols: List of column names for control group
            treatment_cols: List of column names for treatment group
            original_p_values: List of original p-values from t-test
            n_permutations: Number of permutations to perform (default: 1000)
            
        Returns:
            List of corrected p-values
        """
        import streamlit as st
        
        # Create a copy of the data
        data = df.copy()
        
        # Initialize counts for each protein (how many times random p-value <= original p-value)
        counts = np.zeros(len(data))
        
        # Get all sample columns to permute
        all_cols = control_cols + treatment_cols
        n_control = len(control_cols)
        
        # Create a Streamlit progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run permutations with progress bar
        for perm_idx in range(n_permutations):
            # Update progress
            progress = (perm_idx + 1) / n_permutations
            progress_bar.progress(progress)
            status_text.text(f"Running permutation {perm_idx+1}/{n_permutations}")
            
            # Process proteins in batches for better performance
            batch_size = min(100, len(data))  # Process up to 100 proteins at once
            for batch_start in range(0, len(data), batch_size):
                batch_end = min(batch_start + batch_size, len(data))
                batch_indices = range(batch_start, batch_end)
                
                for i in batch_indices:
                    row = data.iloc[i]
                    # Get all values across both groups for this protein
                    all_values = row[all_cols].values.astype(float)
                    all_values = all_values[~np.isnan(all_values)]
                    
                    if len(all_values) >= 3:  # Need at least 3 values for meaningful permutation
                        # Randomly assign values to control and treatment
                        np.random.shuffle(all_values)
                        permuted_control = all_values[:n_control]
                        permuted_treatment = all_values[n_control:n_control+len(treatment_cols)]
                        
                        # Only compute if we have enough values in each group
                        if len(permuted_control) > 0 and len(permuted_treatment) > 0:
                            try:
                                # Perform t-test on permuted data
                                _, perm_pval = stats.ttest_ind(
                                    permuted_treatment, 
                                    permuted_control, 
                                    equal_var=False,
                                    nan_policy='omit'
                                )
                                
                                # Increment count if permuted p-value <= original p-values[i]
                                if perm_pval <= original_p_values[i]:
                                    counts[i] += 1
                            except:
                                pass
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Calculate adjusted p-values (count + 1) / (n_permutations + 1)
        # Adding 1 to numerator and denominator prevents p-values of 0
        corrected_p_values = (counts + 1) / (n_permutations + 1)
        
        return corrected_p_values