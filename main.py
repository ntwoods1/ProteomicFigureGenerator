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

# Configure Streamlit page
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
- Data validation and preprocessing
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

# Initialize session state
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}
if 'processing_params' not in st.session_state:
    st.session_state.processing_params = {}
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = {}


# Data processing options
if uploaded_files:
    with st.sidebar.expander("1. Dataset Structure", expanded=True):
        st.write("Analyze dataset structure")

        if st.button("Analyze Dataset"):
            for file in uploaded_files:
                try:
                    # Load data
                    if file.name.endswith('.csv'):
                        data = pd.read_csv(file)
                    else:
                        data = pd.read_excel(file)

                    # Analyze dataset structure
                    dataset_info = dp.analyze_dataset_structure(data)

                    st.write(f"### Dataset: {file.name}")
                    st.write("#### Summary")
                    st.write(f"- Number of cell lines: {dataset_info['summary']['num_cell_lines']}")
                    st.write(f"- Number of conditions: {dataset_info['summary']['num_conditions']}")
                    st.write(f"- Number of quantity columns: {dataset_info['summary']['num_quantity_columns']}")

                    st.write("#### Cell Lines")
                    st.write(", ".join(dataset_info['cell_lines']))

                    st.write("#### Treatment Conditions")
                    st.write(", ".join(dataset_info['conditions']))

                    st.write("#### Sample Groups and Replicates")
                    for group, replicates in dataset_info['replicates'].items():
                        st.write(f"- {group}: {len(replicates)} replicates")

                    # Store analysis results
                    st.session_state.dataset_info[file.name] = dataset_info

                except Exception as e:
                    st.error(f"Error analyzing {file.name}: {str(e)}")

    with st.sidebar.expander("2. Peptide-based Filtering", expanded=True):
        st.write("Filter proteins based on the number of identified peptides")
        min_peptides = st.number_input(
            "Minimum number of peptides required",
            min_value=1,
            max_value=10,
            value=1,
            help="Only keep proteins with at least this many peptides identified"
        )

        if st.button("Apply Peptide Filter"):
            for file in uploaded_files:
                try:
                    # Load data
                    if file.name.endswith('.csv'):
                        data = pd.read_csv(file)
                    else:
                        data = pd.read_excel(file)

                    # Store raw data
                    st.session_state.datasets[file.name] = data

                    # Apply peptide-based filtering
                    try:
                        filtered_data, filter_stats = dp.filter_by_peptide_count(
                            data,
                            min_peptides=min_peptides
                        )

                        # Store processed data and parameters
                        st.session_state.processed_data[file.name] = filtered_data
                        st.session_state.processing_params[file.name] = {
                            "peptide_filter_stats": filter_stats
                        }

                        # Show processing summary
                        st.sidebar.success(f"Successfully processed: {file.name}")
                        st.sidebar.write("### Filtering Summary")
                        st.sidebar.write(f"- Total proteins: {filter_stats['total_proteins']}")
                        st.sidebar.write(f"- Proteins passing filter: {filter_stats['proteins_passing_filter']}")
                        st.sidebar.write(f"- Proteins removed: {filter_stats['proteins_removed']}")
                        st.sidebar.write(f"- Maximum peptides found: {filter_stats['max_peptides_found']}")

                    except ValueError as ve:
                        st.sidebar.error(f"Error in peptide filtering: {str(ve)}")

                except Exception as e:
                    st.sidebar.error(f"Error processing {file.name}: {str(e)}")

    with st.sidebar.expander("3. Data Preprocessing", expanded=True):
        st.subheader("1. Filtering Options")
        min_detection_rate = st.slider(
            "Minimum Detection Rate",
            0.0, 1.0, 0.5,
            help="Minimum fraction of non-missing values required for each protein"
        )
        min_samples = st.number_input(
            "Minimum Number of Samples",
            min_value=1,
            max_value=10,
            value=2,
            help="Minimum number of samples where protein must be detected"
        )

        st.subheader("2. Missing Value Handling")
        missing_value_method = st.selectbox(
            "Missing Value Method",
            ["knn", "mean", "median", "constant"],
            help="Method for imputing missing values"
        )
        min_valid_values = st.slider(
            "Minimum Valid Values Ratio",
            0.0, 1.0, 0.5,
            help="Minimum ratio of valid values required to keep a protein"
        )

        st.subheader("3. Normalization")
        normalization_method = st.selectbox(
            "Normalization Method",
            ["none", "log2", "zscore", "median", "loess"],
            help="Method for data normalization"
        )

        # Add row centering options
        enable_row_centering = st.checkbox(
            "Enable Row Centering",
            value=False,
            help="Apply row-wise centering after normalization"
        )

        if enable_row_centering:
            center_method = st.radio(
                "Centering Method",
                ["zscore", "scale100"],
                help="zscore: Center on 0 with SD=1, scale100: Center on 100"
            )
        else:
            center_method = None

        st.subheader("4. Quality Control")
        outlier_method = st.selectbox(
            "Outlier Detection Method",
            ["zscore", "iqr"],
            help="Method for detecting outliers"
        )
        outlier_threshold = st.slider(
            "Outlier Threshold",
            1.0, 5.0, 3.0,
            help="Threshold for outlier detection"
        )

        # Process button
        if st.sidebar.button("Process Data"):
            for file in uploaded_files:
                try:
                    # Load data
                    if file.name.endswith('.csv'):
                        data = pd.read_csv(file)
                    else:
                        data = pd.read_excel(file)

                    # Store raw data
                    st.session_state.datasets[file.name] = data

                    # Validate data
                    validation_results = dp.validate_data(data)

                    if validation_results["valid"]:
                        # 1. Filter proteins
                        filtered_data, filter_stats = dp.filter_proteins(
                            data,
                            min_detection_rate=min_detection_rate,
                            min_samples=min_samples
                        )

                        # 2. Handle missing values
                        cleaned_data = dp.handle_missing_values(
                            filtered_data,
                            method=missing_value_method,
                            min_valid_values=min_valid_values
                        )

                        # 3. Normalize data
                        if normalization_method != "none":
                            processed_data = dp.normalize_data(
                                cleaned_data,
                                method=normalization_method,
                                center_scale=enable_row_centering,
                                center_method=center_method if enable_row_centering else None,
                                quantity_only=True
                            )
                        else:
                            processed_data = cleaned_data

                        # 4. Calculate quality metrics
                        qc_metrics = dp.calculate_quality_metrics(processed_data)

                        # Store processed data and parameters
                        st.session_state.processed_data[file.name] = processed_data
                        st.session_state.processing_params[file.name] = {
                            "filter_stats": filter_stats,
                            "qc_metrics": qc_metrics
                        }

                        st.sidebar.success(f"Successfully processed: {file.name}")

                        # Show processing summary
                        st.sidebar.write("### Processing Summary")
                        st.sidebar.write(f"- Original proteins: {filter_stats['total_proteins']}")
                        st.sidebar.write(f"- Filtered proteins: {filter_stats['filtered_proteins']}")
                        st.sidebar.write(f"- Removed proteins: {filter_stats['removed_proteins']}")

                    else:
                        st.sidebar.error(f"Validation failed for {file.name}:")
                        for error in validation_results["errors"]:
                            st.sidebar.error(f"- {error}")

                except Exception as e:
                    st.sidebar.error(f"Error processing {file.name}: {str(e)}")

    # Main content area with tabs
    if st.session_state.processed_data:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Data Overview",
            "Quality Control",
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

                st.write("### Processing Parameters")
                if selected_dataset in st.session_state.processing_params:
                    st.json(st.session_state.processing_params[selected_dataset])

                st.write("### Dataset Structure Analysis")
                if selected_dataset in st.session_state.dataset_info:
                    dataset_info = st.session_state.dataset_info[selected_dataset]
                    st.write("#### Summary")
                    st.write(f"- Number of cell lines: {dataset_info['summary']['num_cell_lines']}")
                    st.write(f"- Number of conditions: {dataset_info['summary']['num_conditions']}")
                    st.write(f"- Number of quantity columns: {dataset_info['summary']['num_quantity_columns']}")

                    st.write("#### Cell Lines")
                    st.write(", ".join(dataset_info['cell_lines']))

                    st.write("#### Treatment Conditions")
                    st.write(", ".join(dataset_info['conditions']))

                    st.write("#### Sample Groups and Replicates")
                    for group, replicates in dataset_info['replicates'].items():
                        st.write(f"- {group}: {len(replicates)} replicates")


        # Quality Control Tab
        with tab2:
            st.header("Quality Control")
            if selected_dataset:
                data = st.session_state.processed_data[selected_dataset]
                if "qc_metrics" in st.session_state.processing_params[selected_dataset]:
                    qc_metrics = st.session_state.processing_params[selected_dataset]["qc_metrics"]

                    # Display QC metrics
                    st.write("### Quality Metrics")
                    st.write("#### Missing Values")
                    missing_values_df = pd.DataFrame({
                        "Missing Values": qc_metrics["missing_values_per_sample"],
                        "Percentage": qc_metrics["missing_values_percentage"]
                    })
                    st.dataframe(missing_values_df)

                    # Detect and display outliers
                    outliers = dp.detect_outliers(
                        data,
                        method=outlier_method,
                        threshold=outlier_threshold
                    )

                    st.write("### Outlier Detection")
                    outlier_summary = outliers.sum().to_frame("Number of Outliers")
                    st.dataframe(outlier_summary)


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
Created with ❤️ using Streamlit
""")