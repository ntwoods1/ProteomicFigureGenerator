# Proteomics Data Analysis Platform

A Streamlit-based web application for advanced proteomics data analysis, offering researchers a comprehensive platform for complex dataset exploration and visualization.

## Features

- 📊 Interactive web interface built with Streamlit
- 🔍 Flexible sample group configuration with multi-column selection
- 📈 Dynamic PCA visualization with adjustable confidence intervals (90-100%)
- 📁 Support for multiple file formats (including Excel)
- 📉 Interactive statistical analysis tools with user-controlled visualization parameters

## Project Structure

```
├── main.py                     # Main application entry point
├── pages/                      # Streamlit multipage app structure
│   ├── 1_📥_Data_Upload.py    # Data upload and initial processing
│   ├── 2_🧪_Data_Processing.py # Data processing and transformation
│   └── 3_📈_Visualization.py   # Data visualization components
├── utils/                      # Utility modules
│   ├── data_processor.py      # Data processing utilities
│   ├── statistics.py          # Statistical analysis functions
│   └── visualizations.py      # Visualization utilities
├── .streamlit/                 # Streamlit configuration
│   └── config.toml            # Streamlit settings
└── pyproject.toml             # Project dependencies and metadata
```

## Features in Detail

### Data Upload (📥)
- Support for CSV and Excel file formats
- Automated column type detection
- Flexible group configuration for experimental design

### Data Processing (🧪)
- Peptide count filtering with customizable thresholds
- Missing value handling with multiple imputation methods
- Coefficient of variation (CV) filtering
- Multiple normalization options (Log2, Median, Z-score, etc.)

### Visualization (📈)
- Interactive volcano plots with customizable thresholds
- PCA analysis with confidence ellipses
- Protein intensity distribution analysis
- Protein rank plots
- Expression bar plots with statistical annotations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/proteomics-analysis-platform.git
cd proteomics-analysis-platform
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Environment Setup

### Required Dependencies
Core dependencies are managed through `pyproject.toml` and include:
- streamlit>=1.43.1
- pandas>=2.2.3
- numpy>=2.2.3
- matplotlib>=3.10.1
- plotly>=6.0.0
- scipy>=1.15.2
- seaborn>=0.13.2
- openpyxl>=3.1.5

### Additional Requirements
- Python 3.11 or higher
- statsmodels (for statistical analysis)

## Usage

1. Start the application:
```bash
streamlit run main.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Follow the intuitive workflow:
   - Upload your proteomics data
   - Configure analysis parameters
   - Explore visualizations and results

## Streamlit Configuration

The application uses a custom Streamlit configuration located in `.streamlit/config.toml`:
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000

[theme]
primaryColor = "#0066cc"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

This configuration ensures:
- Proper server setup for both local and deployed environments
- Consistent theming across the application
- Accessibility from external networks when deployed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Inspired by the needs of the proteomics research community
- Special thanks to all contributors

## Support

If you encounter any issues or have questions, please:
1. Check the [Issues](../../issues) page
2. Review existing pull requests
3. Create a new issue if needed

## Known Issues

- The statistics module requires additional dependencies (statsmodels) which must be installed separately
- Some visualization features may require specific Python versions (3.11+)
- Large datasets may require additional memory allocation

---

Built with ❤️ for the proteomics research community