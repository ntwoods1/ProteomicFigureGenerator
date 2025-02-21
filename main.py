import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import base64
import tempfile
from utils import data_processing as dp
from utils import visualization as viz
from utils import statistics as stats

# Configure Streamlit page with better error handling
try:
    st.set_page_config(
        page_title="Proteomics Data Analysis",
        page_icon="🧬",
        layout="wide"
    )
except Exception as e:
    st.error(f"Error initializing page: {str(e)}")

# Title and introduction
st.title("🧬 Proteomics Data Analysis Platform")
st.markdown("""
This platform provides comprehensive tools for proteomics data analysis, including:
- Data validation and normalization
- Statistical analysis
- Interactive visualizations
- Publication-ready figure export
""")

# Sidebar configuration
st.sidebar.header("Data Upload & Configuration")

# File upload with error handling
try:
    uploaded_files = st.sidebar.file_uploader(
        "Upload Proteomics Data (Excel/CSV)",
        accept_multiple_files=True,
        type=["xlsx", "csv"],
        key="file_uploader"
    )
except Exception as e:
    st.sidebar.error(f"Error during file upload: {str(e)}")
    st.sidebar.info("If you continue to experience issues, try refreshing the page.")
    uploaded_files = None

# Initialize session state with error handling
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}

# Data processing options
if uploaded_files:
    with st.sidebar.expander("Data Processing Options"):
        normalization_method = st.selectbox(
            "Normalization Method",
            ["log2", "zscore", "none"],
            help="Choose method for data normalization"
        )
        
        missing_value_method = st.selectbox(
            "Missing Value Handling",
            ["mean", "median", "zero"],
            help="Choose method for handling missing values"
        )
        
        outlier_threshold = st.slider(
            "Outlier Detection Threshold",
            1.0, 5.0, 3.0,
            help="Z-score threshold for outlier detection"
        )
    
    # Load and process datasets
    for file in uploaded_files:
        try:
            if file.name.endswith('.csv'):
                data = pd.read_csv(file)
            else:
                data = pd.read_excel(file)
            
            # Validate data
            validation_results = dp.validate_data(data)
            
            if validation_results["valid"]:
                # Store raw data
                st.session_state.datasets[file.name] = data
                
                # Process data
                processed_data = data.copy()
                if normalization_method != "none":
                    processed_data = dp.normalize_data(processed_data, method=normalization_method)
                processed_data = dp.handle_missing_values(processed_data, method=missing_value_method)
                
                # Store processed data
                st.session_state.processed_data[file.name] = processed_data
                
                st.sidebar.success(f"Successfully processed: {file.name}")
            else:
                st.sidebar.error(f"Validation failed for {file.name}:")
                for error in validation_results["errors"]:
                    st.sidebar.error(f"- {error}")
                
        except Exception as e:
            st.sidebar.error(f"Error processing {file.name}: {str(e)}")

    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Overview",
        "Statistical Analysis",
        "Volcano Plot",
        "PCA Analysis",
        "Heatmap"
    ])

    # Data Overview Tab
    with tab1:
        st.header("Data Overview")
        selected_dataset = st.selectbox(
            "Select Dataset",
            list(st.session_state.processed_data.keys())
        )
        
        if selected_dataset:
            data = st.session_state.processed_data[selected_dataset]
            st.write("### Processed Data Preview")
            st.dataframe(data.head())
            
            st.write("### Data Statistics")
            st.dataframe(data.describe())
            
            # Outlier detection
            outliers = dp.detect_outliers(data, threshold=outlier_threshold)
            if outliers.any().any():
                st.write("### Outlier Detection")
                st.write("Cells highlighted in red indicate potential outliers")
                styled_data = data.style.background_gradient(
                    cmap='Reds',
                    subset=pd.IndexSlice[outliers.index, outliers.columns[outliers.any()]]
                )
                st.dataframe(styled_data)

    # Statistical Analysis Tab
    with tab2:
        st.header("Statistical Analysis")
        if len(st.session_state.processed_data) >= 2:
            datasets = list(st.session_state.processed_data.keys())
            dataset1 = st.selectbox("Select first dataset", datasets)
            dataset2 = st.selectbox("Select second dataset", datasets, index=1)
            
            if dataset1 != dataset2:
                data1 = st.session_state.processed_data[dataset1]
                data2 = st.session_state.processed_data[dataset2]
                
                stat_method = st.selectbox(
                    "Statistical Test",
                    ["ttest", "mannwhitney"]
                )
                
                results = stats.perform_differential_analysis(
                    pd.concat([data1, data2]),
                    data1.columns,
                    data2.columns,
                    method=stat_method
                )
                
                st.write("### Statistical Analysis Results")
                st.dataframe(results)
                
                # Download results
                csv = results.to_csv(index=False)
                st.download_button(
                    "Download Results CSV",
                    csv,
                    "statistical_analysis.csv",
                    "text/csv"
                )

    # Volcano Plot Tab
    with tab3:
        st.header("Volcano Plot")
        if len(st.session_state.processed_data) > 0:
            selected_dataset = st.selectbox(
                "Select Dataset for Volcano Plot",
                list(st.session_state.processed_data.keys()),
                key="volcano_dataset"
            )
            
            data = st.session_state.processed_data[selected_dataset]
            
            # Column selection
            cols = list(data.select_dtypes(include=[np.number]).columns)
            x_col = st.selectbox("Select Log2 Fold Change Column", cols)
            y_col = st.selectbox("Select P-value Column", cols)
            
            if x_col and y_col:
                cutoffs = {
                    "p_value": st.slider("P-value cutoff (-log10)", 0.0, 10.0, 1.3),
                    "fold_change": st.slider("Fold change cutoff", 0.0, 5.0, 1.0)
                }
                
                fig = viz.create_interactive_volcano(
                    data,
                    x_col,
                    y_col,
                    "Gene Name" if "Gene Name" in data.columns else None,
                    cutoffs
                )
                st.plotly_chart(fig, use_container_width=True)

    # PCA Analysis Tab
    with tab4:
        st.header("PCA Analysis")
        if len(st.session_state.processed_data) > 0:
            selected_dataset = st.selectbox(
                "Select Dataset for PCA",
                list(st.session_state.processed_data.keys()),
                key="pca_dataset"
            )
            
            data = st.session_state.processed_data[selected_dataset]
            numeric_data = data.select_dtypes(include=[np.number])
            
            if not numeric_data.empty:
                fig = viz.create_pca_plot(
                    numeric_data,
                    np.array([0.5, 0.3]),  # Example variance ratios
                    {"PC1": "First Principal Component", "PC2": "Second Principal Component"},
                    data.index
                )
                st.plotly_chart(fig, use_container_width=True)

    # Heatmap Tab
    with tab5:
        st.header("Heatmap")
        if len(st.session_state.processed_data) > 0:
            selected_dataset = st.selectbox(
                "Select Dataset for Heatmap",
                list(st.session_state.processed_data.keys()),
                key="heatmap_dataset"
            )
            
            data = st.session_state.processed_data[selected_dataset]
            numeric_data = data.select_dtypes(include=[np.number])
            
            if not numeric_data.empty:
                cluster_rows = st.checkbox("Cluster Rows", value=True)
                cluster_cols = st.checkbox("Cluster Columns", value=True)
                
                fig = viz.create_heatmap(
                    numeric_data.values,
                    numeric_data.index,
                    numeric_data.columns
                )
                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload one or more datasets to begin analysis.")

# Footer
st.markdown("""
---
Created with ❤️ using Streamlit | [Documentation](https://github.com/yourusername/proteomics-analysis)
""")