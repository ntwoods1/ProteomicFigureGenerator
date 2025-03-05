# Statistical Methods Documentation

This document describes the statistical methods implemented in the Proteomics Data Analysis application.

## Data Processing Pipeline

### 1. Data Filtering
- **Peptide Count Filter**: Removes proteins identified by fewer than a user-specified number of peptides
- **Valid Values Filter**: Filters based on percentage of valid values, either globally or within replicate groups
- **CV Threshold**: Removes proteins with high coefficient of variation within replicate groups

### 2. Normalization Methods
- **Log2 Transformation**: Standard log2 transformation with small constant addition to handle zeros
- **Z-score**: Standardization to mean=0, sd=1
- **Median**: Centering based on median values
- **LOESS**: Local regression smoothing normalization

### 3. Missing Value Handling
- **None**: Keep missing values as is
- **Constant**: Replace with a constant value (typically 0)
- **Mean/Median**: Replace with mean or median of the group
- **KNN**: K-nearest neighbor imputation
- **Half Minimum**: Replace with half of the minimum value in the row

## Statistical Analysis

### 1. Student's t-test
- Compares means between two groups
- Assumes normal distribution and equal variances
- Used in both bar charts and volcano plots
- Returns two-tailed p-values

### 2. Multiple Testing Correction
- **Bonferroni Correction**: Controls family-wise error rate (FWER)
  - Most conservative approach
  - Multiplies p-values by number of tests
  - Strong control of false positives
- **Benjamini-Hochberg (FDR)**: Controls false discovery rate
  - Less stringent than Bonferroni
  - Better balance of false positives and power
  - Particularly useful for large-scale testing

### 3. ANOVA
- One-way ANOVA for comparing multiple groups
- Tests whether means of all groups are equal
- Used when comparing three or more conditions
- Returns single p-value for overall comparison

### 4. Coefficient of Variation (CV)
- Measures relative variability
- Calculated as: (standard deviation / mean) * 100
- Used for quality control and filtering
- Applied within replicate groups

## Visualization Methods

### 1. Volcano Plots
- X-axis: Log2 fold change
- Y-axis: -Log10(p-value)
- Significance thresholds customizable
- Multiple testing correction options

### 2. PCA Analysis
- Principal Component Analysis for dimensionality reduction
- Visualizes major sources of variation
- Automatic scaling of input data
- Explained variance ratio displayed

### 3. Heatmaps
- Hierarchical clustering options
- Standardization across rows/columns
- Statistical significance markers
- Custom color schemes

### 4. Bar Charts
- Error bars: Standard deviation or Standard error of mean
- Individual replicate visualization with jittered points
- Statistical significance testing
- Fold change calculations (regular or log2)

## Data Processing Workflow

1. **Initial Data Loading**
   - Parse column names to identify samples, conditions, and replicates
   - Extract gene names from protein descriptions
   - Validate data format and required columns

2. **Filtering Process**
   - Apply peptide count filter first
   - Then filter by valid values
   - Finally apply CV threshold
   - Track protein numbers at each step

3. **Statistical Analysis**
   - Perform group comparisons
   - Apply multiple testing correction
   - Calculate fold changes
   - Generate comprehensive statistics table

4. **Result Generation**
   - Create interactive visualizations
   - Export statistical results
   - Generate downloadable plots and data tables