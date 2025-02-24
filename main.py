import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from upsetplot import UpSet, from_contents
import tempfile
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.cluster.hierarchy import linkage, dendrogram
from utils.data_processing import (
    analyze_dataset_structure, 
    calculate_cv_table, 
    handle_missing_values,
    normalize_data
)

# Set page configuration
st.set_page_config(
    page_title="Proteomics Analysis",
    page_icon="🧬",
    layout="wide"
)

# Initialize session state for data caching
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = {}

# Title
st.title("Proteomic Data Analysis")

# Data Processing Options in sidebar
st.sidebar.header("Data Processing Options")

# Missing Values Handling
st.sidebar.subheader("Missing Values")
missing_values_method = st.sidebar.selectbox(
    "How to handle missing values?",
    options=["constant", "mean", "median", "knn", "half_min"],
    help="Method to handle missing values in the dataset. 'half_min' uses 1/2 of the row minimum value."
)

min_valid_values = st.sidebar.slider(
    "Minimum % of valid values required",
    min_value=0,
    max_value=100,
    value=50,
    help="Filter out proteins with too many missing values"
)

# Normalization Options
st.sidebar.subheader("Normalization")
normalization_method = st.sidebar.selectbox(
    "Normalization method",
    options=["none", "log2", "zscore", "median", "loess"],
    help="Method to normalize the data. LOESS performs local regression smoothing."
)

# Centering Options
if normalization_method != "none":
    apply_centering = st.sidebar.checkbox(
        "Apply row centering",
        value=False,
        help="Center the data row-wise after normalization"
    )

    if apply_centering:
        center_method = st.sidebar.selectbox(
            "Centering method",
            options=["zscore", "scale100"],
            help="Method to center the rows"
        )

# CV Threshold
cv_threshold = st.sidebar.slider(
    "CV% threshold",
    min_value=0,
    max_value=100,
    value=20,
    help="Maximum allowed Coefficient of Variation percentage"
)

# File Upload
st.sidebar.header("Upload Datasets")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more datasets (Excel format)",
    accept_multiple_files=True,
    type=["xlsx"]
)

# Function to extract gene names from the Description column
def extract_gene_name(description):
    if pd.isna(description):
        return None
    if "GN=" in description:
        try:
            return description.split("GN=")[1].split()[0]
        except IndexError:
            return None
    return None

# Function to create cache key
def get_cache_key(file_name, processing_params):
    return f"{file_name}_{processing_params['missing_method']}_{processing_params['min_valid']}_{processing_params['norm_method']}_{processing_params['center']}_{processing_params['center_method']}"

# Placeholder for datasets
datasets = {}
dataset_structures = {}

if uploaded_files:
    # Process each uploaded file
    for uploaded_file in uploaded_files:
        try:
            # Create a processing status container
            status_container = st.empty()
            progress_bar = st.progress(0)

            status_container.text(f"Reading {uploaded_file.name}...")
            data = pd.read_excel(uploaded_file)
            progress_bar.progress(20)

            if "Description" in data.columns:
                status_container.text("Extracting gene names...")
                data["Gene Name"] = data["Description"].apply(extract_gene_name)
            progress_bar.progress(30)

            # Create cache key for current processing parameters
            processing_params = {
                'missing_method': missing_values_method,
                'min_valid': min_valid_values,
                'norm_method': normalization_method,
                'center': apply_centering if normalization_method != "none" else False,
                'center_method': center_method if normalization_method != "none" and apply_centering else "none"
            }
            cache_key = get_cache_key(uploaded_file.name, processing_params)

            # Check if we have cached results
            if cache_key in st.session_state['processed_data']:
                status_container.text("Using cached processed data...")
                processed_data = st.session_state['processed_data'][cache_key]
                progress_bar.progress(100)
            else:
                # Handle missing values
                status_container.text("Handling missing values...")
                if missing_values_method == "half_min":
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    filtered_data = data.copy()

                    # Process in chunks for better performance
                    chunk_size = max(1, len(filtered_data) // 10)
                    for i in range(0, len(filtered_data), chunk_size):
                        chunk = filtered_data.iloc[i:i+chunk_size]
                        for idx in chunk.index:
                            row_data = filtered_data.loc[idx, numeric_cols]
                            if not row_data.isnull().all():
                                min_val = row_data.min()
                                filtered_data.loc[idx, numeric_cols] = row_data.fillna(min_val / 2)
                        progress = 40 + (i // chunk_size) * 2
                        progress_bar.progress(min(60, progress))

                    valid_counts = filtered_data[numeric_cols].notna().sum(axis=1)
                    filtered_data = filtered_data[valid_counts >= (len(numeric_cols) * min_valid_values/100)]
                else:
                    filtered_data = handle_missing_values(
                        data,
                        method=missing_values_method,
                        min_valid_values=min_valid_values/100
                    )
                progress_bar.progress(60)

                # Apply normalization
                status_container.text("Applying normalization...")
                if normalization_method != "none":
                    try:
                        normalized_data = normalize_data(
                            filtered_data,
                            method=normalization_method,
                            center_scale=apply_centering,
                            center_method=center_method if apply_centering else None
                        )
                    except Exception as e:
                        st.error(f"Error during normalization: {str(e)}")
                        normalized_data = filtered_data.copy()
                else:
                    normalized_data = filtered_data.copy()
                progress_bar.progress(80)

                # Analyze dataset structure
                status_container.text("Analyzing dataset structure...")
                try:
                    structure = analyze_dataset_structure(normalized_data)
                    dataset_structures[uploaded_file.name] = structure
                except Exception as e:
                    st.warning(f"Could not analyze structure of {uploaded_file.name}: {str(e)}")
                progress_bar.progress(90)

                # Cache the processed data
                processed_data = {
                    'original': data,
                    'filtered': filtered_data,
                    'normalized': normalized_data
                }
                st.session_state['processed_data'][cache_key] = processed_data
                progress_bar.progress(100)

            # Store the processed data
            datasets[uploaded_file.name] = processed_data
            status_container.empty()
            progress_bar.empty()
            st.success(f"Successfully processed {uploaded_file.name}")

        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")
            continue

    # Tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Volcano Plot", "PCA", "Heat Map"])

    # Data Overview Tab
    with tab1:
        st.header("Data Overview")
        dataset_name = st.selectbox(
            "Select a dataset to view",
            options=list(datasets.keys())
        )

        if dataset_name:
            selected_data = datasets[dataset_name]['normalized']
            original_data = datasets[dataset_name]['original']

            # Display dataset structure information
            if dataset_name in dataset_structures:
                structure = dataset_structures[dataset_name]

                st.subheader("Dataset Structure")
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Cell Lines Detected:**")
                    st.write(", ".join(structure["cell_lines"]) if structure["cell_lines"] else "None detected")

                    st.write("**Conditions/Treatments Detected:**")
                    st.write(", ".join(structure["conditions"]) if structure["conditions"] else "None detected")

                with col2:
                    st.write("**Summary Statistics:**")
                    st.write(f"- Number of Cell Lines: {structure['summary']['num_cell_lines']}")
                    st.write(f"- Number of Conditions: {structure['summary']['num_conditions']}")
                    st.write(f"- Number of Quantity Columns: {structure['summary']['num_quantity_columns']}")

                # Calculate and display CV for replicate groups
                cv_results = calculate_cv_table(selected_data, structure)

                st.subheader("Coefficient of Variation Analysis")
                for group in structure["replicates"].keys():
                    with st.expander(f"CV Analysis for {group}"):
                        # Filter CV results for this group
                        group_cv = cv_results[[col for col in cv_results.columns if col.startswith(f"CV_{group}")]]
                        if not group_cv.empty:
                            # Count proteins above threshold
                            above_threshold = (group_cv > cv_threshold).sum().sum()
                            total_proteins = len(group_cv)

                            st.write(f"**CV Statistics for {group}:**")
                            st.write(f"- Total proteins: {total_proteins}")
                            st.write(f"- Proteins with CV > {cv_threshold}%: {above_threshold}")
                            st.write(f"- Percentage above threshold: {(above_threshold/total_proteins*100):.1f}%")

                            # Display CV distribution plot
                            fig = px.histogram(
                                group_cv,
                                nbins=50,
                                title=f"CV Distribution for {group}",
                                labels={'value': 'CV%', 'count': 'Number of Proteins'}
                            )
                            fig.add_vline(x=cv_threshold, line_dash="dash", line_color="red")
                            st.plotly_chart(fig)

                # Display filtering summary
                st.subheader("Filtering Summary")
                st.write(f"- Original number of proteins: {len(original_data)}")
                st.write(f"- Proteins after filtering: {len(selected_data)}")
                st.write(f"- Proteins removed: {len(original_data) - len(selected_data)}")

                # Display replicate groups
                st.subheader("Replicate Groups")
                for group, replicate_cols in structure["replicates"].items():
                    with st.expander(f"Group: {group}"):
                        st.write("Replicate columns:")
                        for col in replicate_cols:
                            st.write(f"- {col}")

            st.subheader("Data Preview")
            st.dataframe(selected_data.head(10))

            st.subheader("Basic Statistics")
            st.write(selected_data.describe())

    # Volcano Plot Tab
    with tab2:
        st.header("Volcano Plot")
        dataset_name = st.selectbox(
            "Select a dataset for the Volcano Plot",
            options=list(datasets.keys()),
            key="volcano_dataset"
        )

        if dataset_name:
            selected_data = datasets[dataset_name]['normalized']
            # Column selection for volcano plot
            columns = selected_data.columns
            log2fc_col = st.selectbox("Select Log2 Fold Change Column", options=["Select a column"] + list(columns))
            pvalue_col = st.selectbox("Select P-value Column", options=["Select a column"] + list(columns))

            if log2fc_col != "Select a column" and pvalue_col != "Select a column":
                try:
                    # Prepare data for volcano plot
                    selected_data[log2fc_col] = pd.to_numeric(selected_data[log2fc_col], errors='coerce')
                    selected_data[pvalue_col] = pd.to_numeric(selected_data[pvalue_col], errors='coerce')
                    selected_data = selected_data.dropna(subset=[log2fc_col, pvalue_col])

                    if not selected_data.empty:
                        selected_data["-log10(p-value)"] = -np.log10(selected_data[pvalue_col])

                        # Significance thresholds
                        pval_cutoff = st.slider("-log10(p-value) cutoff", 0.0, 10.0, 1.3, 0.1)
                        log2fc_cutoff = st.slider("Log2 Fold Change cutoff", 0.0, 5.0, 1.0, 0.1)

                        # Generate interactive volcano plot
                        fig = px.scatter(
                            selected_data,
                            x=log2fc_col,
                            y="-log10(p-value)",
                            hover_name="Gene Name" if "Gene Name" in selected_data.columns else None,
                            hover_data={log2fc_col: True, "-log10(p-value)": True},
                            title="Volcano Plot"
                        )

                        # Add cutoff lines
                        fig.add_hline(y=pval_cutoff, line_dash="dash", line_color="red")
                        fig.add_vline(x=log2fc_cutoff, line_dash="dash", line_color="blue")
                        fig.add_vline(x=-log2fc_cutoff, line_dash="dash", line_color="blue")

                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error generating volcano plot: {e}")

    # PCA Tab
    with tab3:
        st.header("Principal Component Analysis")
        dataset_name = st.selectbox(
            "Select dataset for PCA",
            options=list(datasets.keys()),
            key="pca_dataset"
        )

        if dataset_name:
            data = datasets[dataset_name]['normalized']
            numeric_cols = data.select_dtypes(include=[np.number]).columns

            selected_columns = st.multiselect(
                "Select columns for PCA",
                options=numeric_cols
            )

            if len(selected_columns) >= 2:
                try:
                    X = data[selected_columns].dropna()
                    if not X.empty:
                        pca = PCA()
                        X_pca = pca.fit_transform(X)

                        # Create PCA plot
                        pca_df = pd.DataFrame(
                            X_pca[:, :2],
                            columns=['PC1', 'PC2']
                        )

                        fig = px.scatter(
                            pca_df,
                            x='PC1',
                            y='PC2',
                            title='PCA Plot',
                            labels={
                                'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'
                            }
                        )
                        st.plotly_chart(fig)

                except Exception as e:
                    st.error(f"Error performing PCA: {e}")

    # Heat Map Tab
    with tab4:
        st.header("Heat Map")
        dataset_name = st.selectbox(
            "Select dataset for heat map",
            options=list(datasets.keys()),
            key="heatmap_dataset"
        )

        if dataset_name:
            data = datasets[dataset_name]['normalized']
            numeric_cols = data.select_dtypes(include=[np.number]).columns

            selected_columns = st.multiselect(
                "Select columns for heat map",
                options=numeric_cols,
                key="heatmap_columns"
            )

            if len(selected_columns) >= 2:
                try:
                    correlation_matrix = data[selected_columns].corr()
                    fig = px.imshow(
                        correlation_matrix,
                        title='Correlation Heat Map',
                        labels=dict(color="Correlation")
                    )
                    st.plotly_chart(fig)

                except Exception as e:
                    st.error(f"Error creating heat map: {e}")

else:
    st.info("Please upload one or more datasets to begin analysis")