# Proteomics Data Analysis Web Application

A cutting-edge web application for proteomics data analysis, leveraging advanced computational techniques to transform complex scientific datasets into meaningful insights through interactive and statistically robust visualizations.

## Features

### Data Processing Pipeline
- Peptide count filtering with configurable thresholds
- Valid values filtering with group-wise options
- CV threshold filtering with interactive visualization
- Multiple options for handling missing values (KNN, mean, median, constant)
- Flexible data normalization methods (log2, z-score, median, LOESS)

### Statistical Analysis
- Multiple testing correction methods
  - Bonferroni correction for strong FWER control
  - Benjamini-Hochberg procedure for FDR control
- Student's t-test with control group comparison
- ANOVA for multiple group analysis
- Coefficient of variation analysis within replicate groups

### Interactive Visualizations
- Volcano plots
  - Customizable significance thresholds
  - Multiple testing correction options
  - Interactive point selection
  - Downloadable results
- Principal Component Analysis (PCA)
  - Interactive sample selection
  - Explained variance visualization
  - Customizable confidence ellipses
- Interactive Heatmaps
  - Hierarchical clustering options
  - Statistical significance markers
  - Group mean visualization
- Protein Bar Charts
  - Individual replicate visualization with jittered points
  - Error bars (SD or SEM)
  - Regular or log2 fold change display
  - Statistical analysis with downloadable results

## Quick Start

### Running on Replit
1. Fork this repository to your Replit account
2. Click the Run button
3. The Streamlit interface will automatically open in a new tab

### Local Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/proteomics-analysis.git
cd proteomics-analysis
```

2. Install required packages:
```bash
pip install streamlit pandas numpy scipy scikit-learn statsmodels plotly seaborn matplotlib upsetplot
```

3. Run the application:
```bash
streamlit run main.py
```

## Installation Instructions

### Prerequisites
1. Install Python 3.11 or later from [python.org](https://www.python.org/downloads/)
2. During Python installation on Windows:
   - ✅ Check "Add Python to PATH"
   - ✅ Check "pip package installer"

### Windows Setup
1. Open Command Prompt as Administrator
2. Create and activate a virtual environment:
```cmd
python -m venv venv
venv\Scripts\activate
```

3. Install required packages:
```cmd
python -m pip install --upgrade pip
pip install pandas>=2.0.0  # Install pandas 2.0 or later for best compatibility
pip install streamlit pandas numpy scipy scikit-learn statsmodels plotly seaborn matplotlib upsetplot openpyxl
```

4. Run the application:
```cmd
streamlit run main.py
```

The application will open automatically in your default web browser.

### macOS/Linux Setup
1. Open Terminal
2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
python -m pip install --upgrade pip
pip install pandas>=2.0.0  # Install pandas 2.0 or later for best compatibility
pip install streamlit pandas numpy scipy scikit-learn statsmodels plotly seaborn matplotlib upsetplot openpyxl
```

4. Run the application:
```bash
streamlit run main.py
```

## Troubleshooting

### Common Issues
1. **"Python not found" error**:
   - Ensure Python is added to PATH during installation
   - Restart your command prompt/terminal after installation

2. **Package installation errors**:
   - Try updating pip: `python -m pip install --upgrade pip`
   - Install packages one by one if bulk installation fails
   - For pandas warnings, ensure you have pandas 2.0 or later installed

3. **Port already in use**:
   - Close other Streamlit applications
   - Check if any other service is using port 5000

4. **Pandas related warnings**:
   - If you see downcasting warnings, check your pandas version with `pip show pandas`
   - Update pandas if needed: `pip install --upgrade pandas`
   - The application is configured to handle downcasting warnings automatically

### Pandas Configuration
The application uses pandas 2.0+ features and includes automatic configuration to suppress downcasting warnings. If you still see warnings:
1. Verify pandas version: `pip show pandas`
2. Update pandas if below 2.0: `pip install --upgrade pandas`
3. Ensure you're running the latest version of the application from the repository


## Input Data Format

The application expects Excel files (.xlsx) with the following columns:

### Required Columns
- Description: Protein descriptions (used for gene name extraction)
- PG.Quantity columns: Quantitative measurements for each sample
- PG.NrOfStrippedSequencesIdentified or PG.NrOfPrecursorsMeasured: Peptide counts

### Column Naming Convention
Sample columns should follow this format:
```
[Number]_CellLine_Treatment_ReplicateNumber.PG.Quantity
```
Example: `[1]_HeLa_Control_1.PG.Quantity`

## Usage Guide

1. **Data Upload**
   - Upload one or more Excel files containing proteomics data
   - The app automatically detects sample groups and replicates

2. **Data Processing**
   - Configure filtering parameters in the sidebar
   - Monitor data quality through interactive visualizations
   - View detailed statistics about filtered proteins

3. **Analysis**
   - Generate volcano plots for differential expression analysis
   - Explore data structure through PCA
   - Create custom heatmaps for selected proteins
   - Generate statistical bar charts with error bars

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this tool in your research, please cite:
```
To be added