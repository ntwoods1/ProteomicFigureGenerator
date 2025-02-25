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
    normalize_data,
    filter_by_peptide_count  
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
if 'cv_results' not in st.session_state:
    st.session_state['cv_results'] = {}
if 'dataset_structures' not in st.session_state:
    st.session_state['dataset_structures'] = {}

# Title and File Upload Section
st.title("Proteomic Data Analysis")

# File Upload in main area
st.header("Upload Datasets")
uploaded_files = st.file_uploader(
    "Upload one or more datasets (Excel format)",
    accept_multiple_files=True,
    type=["xlsx"]
)

# Data Processing Options in sidebar - Reordered according to processing pipeline
st.sidebar.header("Data Processing Pipeline")

# 1. Peptide Count Filter
st.sidebar.subheader("1. Peptide Count Filter")
min_peptides = st.sidebar.number_input(
    "Minimum number of peptides required",
    min_value=1,
    max_value=10,
    value=2,
    help="Filter out proteins identified by fewer peptides than this threshold"
)

# 2. Valid Values Filter
st.sidebar.subheader("2. Valid Values Filter")
min_valid_values = st.sidebar.slider(
    "Minimum % of valid values required",
    min_value=0,
    max_value=100,
    value=50,
    help="Filter out proteins with too many missing values in PG.Quantity columns"
)

filter_by_group = st.sidebar.checkbox(
    "Apply valid values filter within each replicate group",
    value=False,
    help="If checked, the valid values filter will be applied separately to each group of replicates"
)

# 2. CV Threshold
st.sidebar.subheader("2. CV Threshold")
cv_threshold = st.sidebar.slider(
    "CV% threshold",
    min_value=0,
    max_value=100,
    value=20,
    help="Maximum allowed Coefficient of Variation percentage"
)

# 3. Missing Values Handling
st.sidebar.subheader("3. Missing Values")
missing_values_method = st.sidebar.selectbox(
    "How to handle missing values?",
    options=["none", "constant", "mean", "median", "knn", "half_min"],
    help="Method to handle missing values in the dataset. Only applies to PG.Quantity columns."
)

# 4. Normalization Options
st.sidebar.subheader("4. Normalization")
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
    param_str = "_".join([
        f"{k}:{str(v)}" for k, v in sorted(processing_params.items())
    ])
    return f"{file_name}_{param_str}"

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

            # Create cache key
            processing_params = {
                'min_valid': min_valid_values,
                'cv_threshold': cv_threshold,
                'missing_method': missing_values_method,
                'norm_method': normalization_method,
                'center': apply_centering if normalization_method != "none" else False,
                'center_method': center_method if normalization_method != "none" and apply_centering else "none",
                'min_peptides': min_peptides,
                'filter_by_group': filter_by_group
            }
            cache_key = get_cache_key(uploaded_file.name, processing_params)

            # Check cache
            if cache_key in st.session_state['processed_data']:
                status_container.text("Using cached processed data...")
                datasets[uploaded_file.name] = st.session_state['processed_data'][cache_key]
                progress_bar.progress(100)
                continue

            # Store original dataset and get quantity columns
            quantity_cols = [col for col in data.columns if col.endswith("PG.Quantity")]
            if not quantity_cols:
                st.error(f"No quantitative columns (PG.Quantity) found in {uploaded_file.name}")
                continue

            processed_data = {
                'original': data.copy(),
                'peptide_filtered': None,
                'cv_filtered': None,
                'missing_handled': None,
                'normalized': None
            }

            # 1. Filter by peptide count
            status_container.text("Filtering by peptide count...")
            peptide_filtered_data, peptide_stats = filter_by_peptide_count(data, min_peptides=min_peptides)
            processed_data['peptide_filtered'] = peptide_filtered_data
            progress_bar.progress(30)

            # Add peptide filtering statistics
            st.write(f"Peptide count filter: {peptide_stats['proteins_removed']} proteins removed")

            # 2. Calculate CV on peptide-filtered data
            status_container.text("Calculating CV on filtered data...")
            structure = analyze_dataset_structure(peptide_filtered_data)
            st.session_state['dataset_structures'][uploaded_file.name] = structure
            dataset_structures[uploaded_file.name] = structure
            cv_results = calculate_cv_table(peptide_filtered_data, structure)
            progress_bar.progress(40)

            # 3. Apply valid values filter (only on PG.Quantity columns)
            status_container.text("Applying valid values filter to quantitative columns...")
            if filter_by_group:
                filtered_data = handle_missing_values(
                    peptide_filtered_data,
                    method="none",  # We're just filtering here, not imputing
                    min_valid_values=min_valid_values/100,
                    by_group=True,
                    replicate_groups=structure["replicates"]
                )
            else:
                # Global filtering across all quantity columns
                valid_counts = peptide_filtered_data[quantity_cols].notna().sum(axis=1)
                valid_mask = valid_counts >= (len(quantity_cols) * min_valid_values/100)
                filtered_data = peptide_filtered_data[valid_mask].copy()

            progress_bar.progress(50)

            # Add filtering statistics with more detail
            n_after_peptide = len(peptide_filtered_data)
            n_after_valid = len(filtered_data)
            st.write("Valid values filter statistics:")
            st.write(f"- Proteins before filter: {n_after_peptide}")
            st.write(f"- Proteins after filter: {n_after_valid}")
            st.write(f"- Proteins removed: {n_after_peptide - n_after_valid}")
            if filter_by_group:
                st.write("(Filtering applied within each replicate group)")
            else:
                st.write(f"(Global filtering across all {len(quantity_cols)} quantity columns)")


            # 4. Apply CV threshold filter
            status_container.text("Applying CV threshold filter...")
            cv_mask = pd.Series(False, index=filtered_data.index)  # Start with all False

            # Calculate CV for each group and filter
            for group in structure["replicates"].keys():
                group_cv = cv_results[[col for col in cv_results.columns if col.startswith(f"CV_{group}")]]
                if not group_cv.empty:
                    # A protein passes if its CV is <= threshold for this group
                    group_mask = (group_cv <= cv_threshold).any(axis=1)
                    cv_mask |= group_mask  # Use OR instead of AND - pass in any group

            # Show CV filtering statistics
            n_before_cv = len(filtered_data)
            final_filtered_data = filtered_data[cv_mask].copy()
            n_after_cv = len(final_filtered_data)
            st.write(f"CV filter statistics:")
            st.write(f"- Proteins before CV filter: {n_before_cv}")
            st.write(f"- Proteins after CV filter: {n_after_cv}")
            st.write(f"- Proteins removed: {n_before_cv - n_after_cv}")

            processed_data['cv_filtered'] = final_filtered_data.copy()
            progress_bar.progress(70)

            # 4. Handle missing values if method is not "none"
            if missing_values_method != "none" and quantity_cols:
                status_container.text("Handling missing values...")
                if missing_values_method == "half_min":
                    for idx in final_filtered_data.index:
                        row_data = final_filtered_data.loc[idx, quantity_cols]
                        if not row_data.isnull().all():
                            min_val = row_data.min()
                            final_filtered_data.loc[idx, quantity_cols] = row_data.fillna(min_val / 2)
                else:
                    quantity_data = handle_missing_values(
                        final_filtered_data[quantity_cols],
                        method=missing_values_method,
                        min_valid_values=min_valid_values/100
                    )
                    final_filtered_data[quantity_cols] = quantity_data

            processed_data['missing_handled'] = final_filtered_data.copy()
            progress_bar.progress(85)

            # 5. Apply normalization if selected
            status_container.text("Applying normalization...")
            if normalization_method != "none":
                try:
                    normalized_data = normalize_data(
                        final_filtered_data,
                        method=normalization_method,
                        center_scale=apply_centering,
                        center_method=center_method if apply_centering else None,
                        quantity_only=True
                    )
                except Exception as e:
                    st.error(f"Error during normalization: {str(e)}")
                    normalized_data = final_filtered_data.copy()
            else:
                normalized_data = final_filtered_data.copy()

            processed_data['normalized'] = normalized_data
            progress_bar.progress(95)

            # Store the processed data
            processed_data = {
                'original': data.copy(),
                'peptide_filtered': peptide_filtered_data,
                'valid_filtered': filtered_data,
                'cv_filtered': final_filtered_data,
                'normalized': normalized_data,
                'stats': {
                    'original_count': len(data),
                    'peptide_filtered_count': len(peptide_filtered_data),
                    'valid_filtered_count': len(filtered_data),
                    'cv_filtered_count': len(final_filtered_data),
                    'final_count': len(normalized_data),
                    'peptide_stats': peptide_stats,
                    'filter_params': {
                        'min_peptides': min_peptides,
                        'min_valid_values': min_valid_values,
                        'filter_by_group': filter_by_group,
                        'cv_threshold': cv_threshold
                    }
                }
            }


            datasets[uploaded_file.name] = processed_data
            dataset_structures[uploaded_file.name] = structure
            st.session_state['processed_data'][cache_key] = processed_data
            st.session_state['cv_results'][uploaded_file.name] = cv_results
            st.session_state['dataset_structures'][uploaded_file.name] = structure

            progress_bar.progress(100)
            status_container.empty()
            progress_bar.empty()
            st.success(f"Successfully processed {uploaded_file.name}")

        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")
            continue

    # Display tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Volcano Plot", "PCA", "Heat Map"])

    # Data Overview Tab
    with tab1:
        st.header("Data Overview")
        dataset_name = st.selectbox(
            "Select a dataset to view",
            options=list(datasets.keys())
        )

        if dataset_name:
            processed_data = datasets[dataset_name]
            original_data = processed_data['original']
            peptide_filtered_data = processed_data['peptide_filtered']
            filtered_data = processed_data['valid_filtered']
            final_filtered_data = processed_data['cv_filtered']
            final_data = processed_data['normalized']

            if dataset_name in dataset_structures:
                structure = dataset_structures[dataset_name]
                cv_results = st.session_state['cv_results'][dataset_name]
                stats = datasets[dataset_name]['stats']

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

                # Show CV analysis from original data
                st.subheader("Coefficient of Variation Analysis (Original Data)")
                for group in structure["replicates"].keys():
                    with st.expander(f"CV Analysis for {group}"):
                        group_cv = cv_results[[col for col in cv_results.columns if col.startswith(f"CV_{group}")]]
                        if not group_cv.empty:
                            below_threshold = (group_cv <= cv_threshold).sum().sum()
                            total_proteins = len(group_cv)

                            st.write(f"**CV Statistics for {group}:**")
                            st.write(f"- Total proteins: {total_proteins}")
                            st.write(f"- Proteins with CV ≤ {cv_threshold}%: {below_threshold}")
                            st.write(f"- Percentage below threshold: {(below_threshold/total_proteins*100):.1f}%")

                            fig = px.histogram(
                                group_cv,
                                nbins=50,
                                title=f"CV Distribution for {group}",
                                labels={'value': 'CV%', 'count': 'Number of Proteins'}
                            )
                            fig.add_vline(x=cv_threshold, line_dash="dash", line_color="red",
                                            annotation_text=f"CV threshold ({cv_threshold}%)")
                            st.plotly_chart(fig)

                # Display filtering summary
                st.subheader("Filtering Summary")
                st.write(f"- Original number of proteins: {stats['original_count']}")
                st.write(f"- Proteins after peptide filter (min. {stats['filter_params']['min_peptides']} peptides): {stats['peptide_filtered_count']}")

                if stats['filter_params']['filter_by_group']:
                    st.write(f"- Proteins after valid values filter ({stats['filter_params']['min_valid_values']}% within each group): {stats['valid_filtered_count']}")
                else:
                    st.write(f"- Proteins after valid values filter ({stats['filter_params']['min_valid_values']}% across all columns): {stats['valid_filtered_count']}")

                st.write(f"- Proteins after CV filter (CV ≤ {stats['filter_params']['cv_threshold']}%): {stats['cv_filtered_count']}")
                st.write(f"- Final number of proteins: {stats['final_count']}")

                # Add detailed filtering statistics in expandable sections
                with st.expander("Detailed Filtering Statistics"):
                    st.write("**Peptide Filter Details**")
                    st.write(f"- Proteins removed: {stats['original_count'] - stats['peptide_filtered_count']}")

                    st.write("\n**Valid Values Filter Details**")
                    st.write(f"- Proteins removed: {stats['peptide_filtered_count'] - stats['valid_filtered_count']}")
                    if stats['filter_params']['filter_by_group']:
                        st.write("- Filter applied within each replicate group")
                    else:
                        st.write(f"- Filter applied across all {len(quantity_cols)} quantity columns")

                    st.write("\n**CV Filter Details**")
                    st.write(f"- Proteins removed: {stats['valid_filtered_count'] - stats['cv_filtered_count']}")


                # Display replicate groups
                st.subheader("Replicate Groups")
                for group, replicate_cols in structure["replicates"].items():
                    with st.expander(f"Group: {group}"):
                        st.write("Replicate columns:")
                        for col in replicate_cols:
                            st.write(f"- {col}")

            st.subheader("Data Preview")
            st.dataframe(final_data.head(10))

            st.subheader("Basic Statistics")
            st.write(final_data.describe())

    # Volcano Plot Tab
    with tab2:
        st.header("Volcano Plot")

        # Dataset selection with default empty option
        dataset_options = [""] + list(datasets.keys())
        dataset_name = st.selectbox(
            "Select a dataset for Volcano Plot",
            options=dataset_options,
            key="volcano_dataset"
        )

        if dataset_name and dataset_name in datasets:
            selected_data = datasets[dataset_name]['normalized']

            # Get structure from session state
            if dataset_name in st.session_state['dataset_structures']:
                structure = st.session_state['dataset_structures'][dataset_name]
            else:
                # If structure not in session state, regenerate it
                try:
                    structure = analyze_dataset_structure(selected_data)
                    st.session_state['dataset_structures'][dataset_name] = structure
                except Exception as e:
                    st.error(f"Error analyzing dataset structure: {str(e)}")
                    structure = None

            if structure:
                # Get unique groups from replicate structure
                all_groups = list(structure["replicates"].keys())

                if all_groups:
                    # Group selection for comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        group1 = st.selectbox(
                            "Select first group",
                            options=all_groups,
                            key="volcano_group1"
                        )
                    with col2:
                        remaining_groups = [g for g in all_groups if g != group1] if group1 else all_groups
                        group2 = st.selectbox(
                            "Select second group",
                            options=remaining_groups,
                            key="volcano_group2"
                        )

                    if group1 and group2:
                        try:
                            # Get the quantity columns for each group
                            group1_cols = [col for col in structure["replicates"][group1] if col.endswith("PG.Quantity")]
                            group2_cols = [col for col in structure["replicates"][group2] if col.endswith("PG.Quantity")]

                            if group1_cols and group2_cols:
                                # Calculate fold change and p-values
                                group1_data = selected_data[group1_cols]
                                group2_data = selected_data[group2_cols]

                                # Calculate log2 fold change
                                log2fc = np.log2(group2_data.mean(axis=1) / group1_data.mean(axis=1))

                                # Calculate p-values using t-test
                                p_values = []
                                for idx in selected_data.index:
                                    g1_values = group1_data.loc[idx].dropna()
                                    g2_values = group2_data.loc[idx].dropna()
                                    if len(g1_values) >= 2 and len(g2_values) >= 2:
                                        _, p_val = ttest_ind(g1_values, g2_values)
                                        p_values.append(p_val)
                                    else:
                                        p_values.append(np.nan)

                                # Create DataFrame for volcano plot
                                volcano_data = pd.DataFrame({
                                    'log2FoldChange': log2fc,
                                    '-log10(p-value)': -np.log10(p_values),
                                    'Gene Name': selected_data['Gene Name'] if 'Gene Name' in selected_data.columns else selected_data.index
                                })
                                volcano_data = volcano_data.replace([np.inf, -np.inf], np.nan).dropna()

                                # Significance thresholds
                                pval_cutoff = st.slider("-log10(p-value) cutoff", 0.0, 10.0, 1.3, 0.1)
                                log2fc_cutoff = st.slider("Log2 Fold Change cutoff", 0.0, 5.0, 1.0, 0.1)

                                # Generate interactive volcano plot
                                fig = px.scatter(
                                    volcano_data,
                                    x='log2FoldChange',
                                    y='-log10(p-value)',
                                    hover_name=volcano_data['Gene Name'],
                                    hover_data={
                                        'log2FoldChange': ':.2f',
                                        '-log10(p-value)': ':.2f'
                                    },
                                    title=f"Volcano Plot: {group2} vs {group1}"
                                )

                                # Add cutoff lines
                                fig.add_hline(y=pval_cutoff, line_dash="dash", line_color="red")
                                fig.add_vline(x=log2fc_cutoff, line_dash="dash", line_color="blue")
                                fig.add_vline(x=-log2fc_cutoff, line_dash="dash", line_color="blue")

                                # Color points based on significance
                                significant = (volcano_data['-log10(p-value)'] >= pval_cutoff) & \
                                            (abs(volcano_data['log2FoldChange']) >= log2fc_cutoff)

                                # Update point colors
                                fig.update_traces(
                                    marker=dict(
                                        color=significant.map({True: 'red', False: 'gray'}),
                                        size=8
                                    )
                                )

                                st.plotly_chart(fig, use_container_width=True)

                                # Display summary statistics
                                total_proteins = len(volcano_data)
                                sig_proteins = significant.sum()
                                st.write(f"Total proteins analyzed: {total_proteins}")
                                st.write(f"Significantly different proteins: {sig_proteins}")
                                st.write(f"Percentage significant: {(sig_proteins/total_proteins*100):.1f}%")

                            else:
                                st.warning("Selected groups don't have quantitative data for comparison")
                        except Exception as e:
                            st.error(f"Error generating volcano plot: {str(e)}")
                else:
                    st.warning("No replicate groups found in the dataset")
            else:
                st.error("Dataset structure information not found")
        else:
            if dataset_name:
                st.error("Selected dataset not found")
            else:
                st.info("Please select a dataset to create a volcano plot")

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