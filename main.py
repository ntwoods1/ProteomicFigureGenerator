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
import io
from utils.data_processing import (
    analyze_dataset_structure, 
    calculate_cv_table, 
    handle_missing_values,
    normalize_data,
    filter_by_peptide_count  
)
from itertools import combinations

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
if 'volcano_comparisons' not in st.session_state:
    st.session_state['volcano_comparisons'] = {}
if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = 0
if 'filtering_stats' not in st.session_state:
    st.session_state['filtering_stats'] = {}
if 'pca_selections' not in st.session_state:
    st.session_state['pca_selections'] = {}

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
                'valid_filtered': None,
                'cv_filtered': None,
                'normalized': None,
                'stats': {}
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

            processed_data['valid_filtered'] = filtered_data
            progress_bar.progress(50)


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

            # Store the processed data with updated stats
            processed_data['stats'] = {
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
            st.session_state['filtering_stats'][uploaded_file.name] = processed_data['stats']

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
    tabs = ["Data Overview", "Volcano Plot", "PCA", "Heat Map"]
    active_tab = st.radio("Select Analysis", tabs, horizontal=True, key="analysis_tabs", index=st.session_state['active_tab'])
    st.session_state['active_tab'] = tabs.index(active_tab)

    # Display appropriate content based on active tab
    if active_tab == "Data Overview":
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

    elif active_tab == "Volcano Plot":
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
                    # Add new comparison button
                    if st.button("Add New Comparison"):
                        comparison_key = f"comparison_{len(st.session_state['volcano_comparisons'])}"
                        st.session_state['volcano_comparisons'][comparison_key] = {
                            'group1': None,
                            'group2': None,
                            'significant_up': set(),
                            'significant_down': set()
                        }

                    # Display all comparisons
                    for comp_key, comp_data in st.session_state['volcano_comparisons'].items():
                        with st.expander(f"Volcano Plot - {comp_key}", expanded=True):
                            col1, col2 = st.columns(2)
                            with col1:
                                group1 = st.selectbox(
                                    "Select first group",
                                    options=all_groups,
                                    key=f"{comp_key}_group1"
                                )
                            with col2:
                                remaining_groups = [g for g in all_groups if g != group1] if group1 else all_groups
                                group2 = st.selectbox(
                                    "Select second group",
                                    options=remaining_groups,
                                    key=f"{comp_key}_group2"
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
                                            'Gene Name': selected_data['Gene Name'] if 'Gene Name' in selected_data.columns else selected_data.index,
                                            'Description': selected_data['Description'] if 'Description' in selected_data.columns else '',
                                            'Mean1': group1_data.mean(axis=1),
                                            'Mean2': group2_data.mean(axis=1)
                                        })
                                        volcano_data = volcano_data.replace([np.inf, -np.inf], np.nan).dropna()

                                        # Significance thresholds
                                        pval_cutoff = st.slider("-log10(p-value) cutoff", 0.0, 10.0, 1.3, 0.1, key=f"{comp_key}_pval")
                                        log2fc_cutoff = st.slider("Log2 Fold Change cutoff", 0.0, 5.0, 1.0, 0.1, key=f"{comp_key}_fc")

                                        # Generate interactive Plotly volcano plot
                                        fig_plotly = px.scatter(
                                            volcano_data,
                                            x='log2FoldChange',
                                            y='-log10(p-value)',
                                            hover_name=volcano_data['Gene Name'],
                                            hover_data={
                                                'log2FoldChange': ':.2f',
                                                '-log10(p-value)': ':.2f',
                                                'Mean1': ':.2f',
                                                'Mean2': ':.2f',
                                                'Description': True
                                            },
                                            title=f"Volcano Plot: {group2} vs {group1}"
                                        )

                                        # Add cutoff lines
                                        fig_plotly.add_hline(y=pval_cutoff, line_dash="dash", line_color="red")
                                        fig_plotly.add_vline(x=log2fc_cutoff, line_dash="dash", line_color="blue")
                                        fig_plotly.add_vline(x=-log2fc_cutoff, line_dash="dash", line_color="blue")

                                        # Color points based on significance
                                        significant = (volcano_data['-log10(p-value)'] >= pval_cutoff)
                                        significant_up = significant & (volcano_data['log2FoldChange'] >= log2fc_cutoff)
                                        significant_down = significant & (volcano_data['log2FoldChange'] <= -log2fc_cutoff)

                                        # Create color array
                                        marker_colors = ['red' if up else 'blue' if down else 'gray' 
                                                                for up, down in zip(significant_up, significant_down)]

                                        # Add protein labels input
                                        proteins_to_label = st.text_area(
                                            "Enter protein names to label (one per line)",
                                            help="Enter gene names (one per line) to add labels on the plot",
                                            key=f"{comp_key}_proteins"
                                        ).strip().split('\n')

                                        # Filter out empty lines
                                        proteins_to_label = [p.strip() for p in proteins_to_label if p.strip()]

                                        # Update Plotly markers
                                        fig_plotly.update_traces(
                                            marker=dict(
                                                color=marker_colors,
                                                size=8
                                            )
                                        )

                                        # Add labels for selected proteins in Plotly
                                        if proteins_to_label:
                                            for protein in proteins_to_label:
                                                mask = volcano_data['Gene Name'].str.contains(protein, case=False, na=False)
                                                if mask.any():
                                                    for idx in volcano_data[mask].index:
                                                        gene_name = volcano_data.loc[idx, 'Gene Name']
                                                        x = volcano_data.loc[idx, 'log2FoldChange']
                                                        y = volcano_data.loc[idx, '-log10(p-value)']

                                                        fig_plotly.add_annotation(
                                                            x=x,
                                                            y=y,
                                                            text=gene_name,
                                                            showarrow=True,
                                                            arrowhead=2,
                                                            arrowsize=1,
                                                            arrowwidth=2,
                                                            ax=20,
                                                            ay=-30
                                                        )

                                        # Display interactive Plotly plot
                                        st.plotly_chart(fig_plotly, use_container_width=True)

                                        # Generate static matplotlib plot for SVG export
                                        fig_mpl, ax = plt.subplots(figsize=(12, 8))

                                        # Plot points
                                        scatter = ax.scatter(
                                            volcano_data['log2FoldChange'],
                                            volcano_data['-log10(p-value)'],
                                            c=marker_colors,
                                            s=80,
                                            alpha=0.6
                                        )

                                        # Add labels for selected proteins in Matplotlib
                                        if proteins_to_label:
                                            for protein in proteins_to_label:
                                                mask = volcano_data['Gene Name'].str.contains(protein, case=False, na=False)
                                                if mask.any():
                                                    for idx in volcano_data[mask].index:
                                                        gene_name = volcano_data.loc[idx, 'Gene Name']
                                                        x = volcano_data.loc[idx, 'log2FoldChange']
                                                        y = volcano_data.loc[idx, '-log10(p-value)']
                                                        ax.annotate(
                                                            gene_name, 
                                                            (x, y),
                                                            xytext=(5, 5), 
                                                            textcoords='offset points',
                                                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8),
                                                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                                                        )

                                        # Add cutoff lines
                                        ax.axhline(y=pval_cutoff, color='red', linestyle='--', alpha=0.5)
                                        ax.axvline(x=log2fc_cutoff, color='blue', linestyle='--', alpha=0.5)
                                        ax.axvline(x=-log2fc_cutoff, color='blue', linestyle='--', alpha=0.5)

                                        # Labels and title
                                        ax.set_xlabel('log2 Fold Change')
                                        ax.set_ylabel('-log10(p-value)')
                                        ax.set_title(f'Volcano Plot: {group2} vs {group1}')

                                        # Add grid
                                        ax.grid(True, alpha=0.3)

                                        # Download buttons
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            # Download HTML (Interactive Plotly)
                                            html_buffer = fig_plotly.to_html()
                                            st.download_button(
                                                label="Download Interactive Plot (HTML)",
                                                data=html_buffer,
                                                file_name=f"volcano_plot_{group2}_vs_{group1}.html",
                                                mime="text/html"
                                            )
                                        with col2:
                                            # Save SVG (Static matplotlib)
                                            buffer = io.BytesIO()
                                            fig_mpl.savefig(buffer, format='svg', bbox_inches='tight')
                                            buffer.seek(0)
                                            st.download_button(
                                                label="Download Plot as SVG",
                                                data=buffer,
                                                file_name=f"volcano_plot_{group2}_vs_{group1}.svg",
                                                mime="image/svg+xml"
                                            )
                                        with col3:
                                            csv_buffer = volcano_data.to_csv(index=True)
                                            st.download_button(
                                                label="Download Results as CSV",
                                                data=csv_buffer,
                                                file_name=f"volcano_data_{group2}_vs_{group1}.csv",
                                                mime="text/csv"
                                            )

                                        plt.close(fig_mpl)

                                        # Store significant proteins
                                        st.session_state['volcano_comparisons'][comp_key]['significant_up'] = set(volcano_data[significant_up]['Gene Name'])
                                        st.session_state['volcano_comparisons'][comp_key]['significant_down'] = set(volcano_data[significant_down]['Gene Name'])

                                        # Display detailed results for labeled proteins
                                        if proteins_to_label:
                                            st.subheader("Details for labeled proteins:")
                                            for protein in proteins_to_label:
                                                mask = volcano_data['Gene Name'].str.contains(protein, case=False, na=False)
                                                if mask.any():
                                                    matching_proteins = volcano_data[mask]
                                                    for _, row in matching_proteins.iterrows():
                                                        st.write(f"**Gene Name:** {row['Gene Name']}")
                                                        if 'Description' in row:
                                                            st.write(f"**Description:** {row['Description']}")
                                                        st.write(f"**Log2 Fold Change:** {row['log2FoldChange']:.2f}")
                                                        st.write(f"**-log10(p-value):** {row['-log10(p-value)']:.2f}")
                                                        st.write(f"**Mean {group1}:** {row['Mean1']:.2f}")
                                                        st.write(f"**Mean {group2}:** {row['Mean2']:.2f}")
                                                        st.write("---")
                                                else:
                                                    st.warning(f"No proteins found matching '{protein}'")

                                except Exception as e:
                                    st.error(f"Error generating volcano plot: {str(e)}")

                    # Add UpSet plot after all comparisons
                    if len(st.session_state['volcano_comparisons']) > 1:
                        st.markdown("---")  # Visual separator
                        st.header("Overlap Analysis")

                        try:
                            # Create dictionariesfor storing gene sets
                            up_sets = {}
                            down_sets ={}

                            # Collect genes from all comparisons
                            for comp_key, comp_data in st.session_state['volcano_comparisons'].items():
                                if comp_data['significant_up']:
                                    up_sets[comp_key] = list(comp_data['significant_up'])
                                if comp_data['significant_down']:
                                    down_sets[comp_key] = list(comp_data['significant_down'])

                            # Generate overlaps visualizations
                            if up_sets:
                                st.subheader("Up-regulated Proteins")
                                st.write(f"Overlap analysis of up-regulated proteins across {len(up_sets)} comparisons")

                                # Create figure
                                fig_up = plt.figure(figsize=(12, 6))
                                upset = UpSet(from_contents(up_sets))
                                upset.plot(fig=fig_up)
                                st.pyplot(fig_up)

                                # Create download buttons
                                col1, col2 = st.columns(2)
                                with col1:
                                    # Save SVG
                                    buffer = io.BytesIO()
                                    fig_up.savefig(buffer, format='svg', bbox_inches='tight')
                                    buffer.seek(0)
                                    st.download_button(
                                        label="Download Plot as SVG",
                                        data=buffer,
                                        file_name="upset_plot_upregulated.svg",
                                        mime="image/svg+xml"
                                    )
                                with col2:
                                    # Create protein list with group memberships
                                    protein_data = []
                                    all_proteins = set().union(*up_sets.values())
                                    for protein in all_proteins:
                                        groups = [group for group, proteins in up_sets.items() if protein in proteins]
                                        protein_data.append({
                                            'Protein': protein,
                                            'Groups': ';'.join(groups),
                                            'Number_of_Groups': len(groups)
                                        })
                                    protein_df = pd.DataFrame(protein_data)
                                    protein_csv = protein_df.to_csv(index=False)
                                    st.download_button(
                                        label="Download Protein List",
                                        data=protein_csv,
                                        file_name="upregulated_proteins.csv",
                                        mime="text/csv"
                                    )
                                plt.close(fig_up)

                            if down_sets:
                                st.subheader("Down-regulated Proteins")
                                st.write(f"Overlap analysis of down-regulated proteins across {len(down_sets)} comparisons")

                                # Create figure
                                fig_down = plt.figure(figsize=(12, 6))
                                upset = UpSet(from_contents(down_sets))
                                upset.plot(fig=fig_down)
                                st.pyplot(fig_down)

                                # Create download buttons
                                col1, col2 = st.columns(2)
                                with col1:
                                    # Save SVG
                                    buffer = io.BytesIO()
                                    fig_down.savefig(buffer, format='svg', bbox_inches='tight')
                                    buffer.seek(0)
                                    st.download_button(
                                        label="Download Plot as SVG",
                                        data=buffer,
                                        file_name="upset_plot_downregulated.svg",
                                        mime="image/svg+xml"
                                    )
                                with col2:
                                    # Create protein list with group memberships
                                    protein_data = []
                                    all_proteins = set().union(*down_sets.values())
                                    for protein in all_proteins:
                                        groups = [group for group, proteins in down_sets.items() if protein in proteins]
                                        protein_data.append({
                                            'Protein': protein,
                                            'Groups': ';'.join(groups),
                                            'Number_of_Groups': len(groups)
                                        })
                                    protein_df = pd.DataFrame(protein_data)
                                    protein_csv = protein_df.to_csv(index=False)
                                    st.download_button(
                                        label="Download Protein List",
                                        data=protein_csv,
                                        file_name="downregulated_proteins.csv",
                                        mime="text/csv"
                                    )
                                plt.close(fig_down)

                        except Exception as e:
                            st.error(f"Error generating overlap analysis: {str(e)}")

                else:
                    st.warning("No replicate groups found in the dataset")
            else:
                st.error("Dataset structure information not found")
        else:
            if dataset_name:
                st.error("Selected dataset not found")
            else:
                st.info("Please select a dataset to create a volcano plot")

    elif active_tab == "PCA":
        st.header("PCA Analysis")
        dataset_name = st.selectbox(
            "Select dataset for PCA",
            options=list(datasets.keys()),
            key="pca_dataset"
        )

        if dataset_name:
            data = datasets[dataset_name]['normalized']

            # Ensure we have the dataset structure
            try:
                if dataset_name in st.session_state['dataset_structures']:
                    structure = st.session_state['dataset_structures'][dataset_name]
                else:
                    structure = analyze_dataset_structure(data)
                    st.session_state['dataset_structures'][dataset_name] = structure
            except Exception as e:
                st.error(f"Error analyzing dataset structure: {str(e)}")
                st.stop()

            # Get replicate groups
            replicate_groups = list(structure["replicates"].keys())

            # Allow selection of replicate groups
            selected_groups = st.multiselect(
                "Select replicate groups for PCA",
                options=replicate_groups,
                default=replicate_groups[:3] if len(replicate_groups) > 2 else replicate_groups
            )

            # Custom names for selected groups
            group_names = {}
            if selected_groups:
                st.write("Enter custom names for selected groups (leave blank to use original names):")
                cols = st.columns(len(selected_groups))
                for idx, group in enumerate(selected_groups):
                    with cols[idx]:
                        custom_name = st.text_input(f"Name for {group}", value=group, key=f"custom_name_{idx}")
                        group_names[group] = custom_name

                # Get columns for selected groups
                selected_columns = []
                group_to_columns = {}  # Map to track which columns belong to which group
                for group in selected_groups:
                    group_cols = [col for col in structure["replicates"][group] 
                                if col.endswith("PG.Quantity")]
                    selected_columns.extend(group_cols)
                    group_to_columns[group] = group_cols

                if len(selected_columns) >= 2:
                    try:
                        # Prepare data for PCA
                        X = data[selected_columns].dropna()
                        if not X.empty:
                            # Create a DataFrame with samples as rows and proteins as columns
                            pca_input = pd.DataFrame(index=X.index)
                            sample_groups = []  # To store group information for each sample
                            sample_names = []   # To store sample names

                            for col in selected_columns:
                                # Extract sample name and group
                                sample_name = col.split("]")[1].split(".PG.Quantity")[0].strip()
                                group = next(g for g in selected_groups if col in structure["replicates"][g])

                                # Add data to PCA input
                                pca_input[sample_name] = X[col]
                                sample_groups.append(group_names[group])
                                sample_names.append(sample_name)

                            # Transpose so samples are rows and proteins are columns
                            pca_input = pca_input.T

                            # Perform PCA
                            pca = PCA()
                            pca_result = pca.fit_transform(pca_input)

                            # Create DataFrame for plotting
                            plot_df = pd.DataFrame({
                                'PC1': pca_result[:, 0],
                                'PC2': pca_result[:, 1],
                                'Group': sample_groups,
                                'Sample': sample_names
                            })

                            # Create interactive scatter plot with template for dark theme
                            fig_plotly = px.scatter(
                                plot_df,
                                x='PC1',
                                y='PC2',
                                color='Group',
                                hover_name='Sample',
                                title='PCA Score Plot',
                                labels={
                                    'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                                    'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'
                                },
                                template='plotly_dark'  # Use dark template for Plotly
                            )

                            # Update layout for dark theme
                            fig_plotly.update_layout(
                                height=600,
                                legend_title="Sample Groups",
                                plot_bgcolor='black',
                                paper_bgcolor='black',
                                font=dict(color='white'),
                                title_font_color='white',
                                legend_font_color='white'
                            )

                            # Update axes for dark theme
                            fig_plotly.update_xaxes(gridcolor='rgba(128, 128, 128, 0.2)', zerolinecolor='rgba(128, 128, 128, 0.2)')
                            fig_plotly.update_yaxes(gridcolor='rgba(128, 128, 128, 0.2)', zerolinecolor='rgba(128, 128, 128, 0.2)')

                            # Display interactive plot
                            st.plotly_chart(fig_plotly, use_container_width=True)

                            # Create static matplotlib plots for SVG export with white background
                            fig_mpl = plt.figure(figsize=(20, 8))
                            # Create a wider right subplot for legend space
                            gs = fig_mpl.add_gridspec(1, 2, width_ratios=[1, 1.2])
                            ax1 = fig_mpl.add_subplot(gs[0])
                            ax2 = fig_mpl.add_subplot(gs[1])

                            # Score plot
                            groups = plot_df['Group'].unique()
                            colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
                            for group, color in zip(groups, colors):
                                mask = plot_df['Group'] == group
                                ax1.scatter(
                                    plot_df.loc[mask, 'PC1'],
                                    plot_df.loc[mask, 'PC2'],
                                    c=[color],
                                    label=group,
                                    alpha=0.7
                                )

                            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
                            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
                            ax1.set_title('PCA Score Plot')
                            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                            ax1.grid(True, alpha=0.3)
                            ax1.set_facecolor('white')

                            # Loading plot
                            loadings = pca.components_.T
                            loading_df = pd.DataFrame(
                                loadings[:, :2],
                                columns=['PC1', 'PC2'],
                                index=pca_input.columns
                            )

                            # Create arrows for loadings
                            for i in range(len(loading_df)):
                                ax2.arrow(
                                    0, 0,
                                    loading_df.iloc[i, 0],
                                    loading_df.iloc[i, 1],
                                    color='red',
                                    alpha=0.5
                                )
                                if i < 10:  # Only label top 10 loadings for clarity
                                    ax2.text(
                                        loading_df.iloc[i, 0] * 1.15,
                                        loading_df.iloc[i, 1] * 1.15,
                                        loading_df.index[i],
                                        color='green',
                                        ha='center',
                                        va='center'
                                    )

                            ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
                            ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
                            ax2.set_title('PCA Loading Plot')
                            ax2.grid(True, alpha=0.3)
                            ax2.set_facecolor('white')

                            # Set axis limits to make the plot more symmetric
                            max_val = max(
                                abs(loading_df['PC1'].max()),
                                abs(loading_df['PC1'].min()),
                                abs(loading_df['PC2'].max()),
                                abs(loading_df['PC2'].min())
                            )
                            ax2.set_xlim(-max_val * 1.2, max_val * 1.2)
                            ax2.set_ylim(-max_val * 1.2, max_val * 1.2)

                            # Set figure background to white
                            fig_mpl.patch.set_facecolor('white')
                            plt.tight_layout()

                            # Download buttons
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                # Download HTML (Interactive Plotly)
                                html_buffer = fig_plotly.to_html(include_plotlyjs=True, full_html=True)
                                st.download_button(
                                    label="Download Interactive Plot (HTML)",
                                    data=html_buffer,
                                    file_name="pca_score_plot.html",
                                    mime="text/html"
                                )
                            with col2:
                                # Save SVG (Static matplotlib)
                                buffer = io.BytesIO()
                                fig_mpl.savefig(buffer, format='svg', bbox_inches='tight')
                                buffer.seek(0)
                                st.download_button(
                                    label="Download Plots as SVG",
                                    data=buffer,
                                    file_name="pca_plots.svg",
                                    mime="image/svg+xml"
                                )
                            with col3:
                                # Download data as CSV
                                csv_data = pd.concat([
                                    plot_df,
                                    pd.DataFrame({
                                        'Loading_PC1': loading_df['PC1'],
                                        'Loading_PC2': loading_df['PC2']
                                    }, index=loading_df.index)
                                ], axis=1)
                                csv_buffer = csv_data.to_csv(index=True)
                                st.download_button(
                                    label="Download PCA Results as CSV",
                                    data=csv_buffer,
                                    file_name="pca_results.csv",
                                    mime="text/csv"
                                )

                            plt.close(fig_mpl)

                            # Store current PCA selections
                            st.session_state['pca_selections'][dataset_name] = {
                                'selected_groups': selected_groups,
                                'group_names': group_names
                            }

                            # Display explained variance ratios
                            st.write("### Explained Variance Ratios")
                            explained_var_df = pd.DataFrame({
                                'Principal Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                                'Explained Variance Ratio': pca.explained_variance_ratio_,
                                'Cumulative Variance Ratio': np.cumsum(pca.explained_variance_ratio_)
                            })
                            st.dataframe(explained_var_df)

                    except Exception as e:
                        st.error(f"Error performing PCA: {str(e)}")
                    else:
                        st.warning("Please select groups with at least 2 quantity columns for PCA")
                else:
                    st.warning("Please select at least one replicate group")

    elif active_tab == "Heat Map":
        st.header("Heat Map")
        dataset_name = st.selectbox(
            "Select dataset for heat map",
            options=list(datasets.keys())
        )

        if dataset_name:
            data = datasets[dataset_name]['normalized']
            structure = st.session_state['dataset_structures'][dataset_name]

            # Get replicate groups
            replicate_groups = list(structure["replicates"].keys())

            # Group renaming functionality
            st.subheader("Replicate Group Names")
            group_names = {}
            cols = st.columns(3)  # Create 3 columns for compact display
            for i, group in enumerate(replicate_groups):
                col_idx = i % 3
                with cols[col_idx]:
                    group_names[group] = st.text_input(
                        f"Name for {group}",
                        value=group,
                        key=f"group_name_{group}"
                    )

            # Allow selection of replicate groups
            selected_groups = st.multiselect(
                "Select replicate groups for heat map",
                options=replicate_groups,
                default=replicate_groups[:2] if len(replicate_groups) > 1 else replicate_groups
            )

            if selected_groups:
                # Get quantity columns for selected groups
                selected_columns = []
                for group in selected_groups:
                    selected_columns.extend([col for col in structure["replicates"][group] 
                                          if col.endswith("PG.Quantity")])

                if len(selected_columns) >= 2:
                    # Add slider for number of proteins
                    n_proteins = st.slider(
                        "Number of proteins to display",
                        min_value=5,
                        max_value=100,
                        value=50,
                        step=5
                    )

                    try:
                        # Prepare data for heat map
                        heatmap_data = data[selected_columns].copy()

                        # Calculate scores for protein selection
                        scores = pd.DataFrame(index=heatmap_data.index)

                        # Calculate intra-group variation (want this to be low)
                        intra_cv = pd.DataFrame(index=heatmap_data.index)
                        for group in selected_groups:
                            group_cols = [col for col in structure["replicates"][group] 
                                        if col.endswith("PG.Quantity")]
                            group_data = heatmap_data[group_cols]
                            cv = (group_data.std(axis=1) / group_data.mean(axis=1)) * 100
                            intra_cv[group] = cv

                        # Average intra-group CV (lower is better)
                        scores['intra_cv'] = intra_cv.mean(axis=1)

                        # Calculate inter-group variation (want this to be high)
                        group_means = pd.DataFrame(index=heatmap_data.index)
                        for group in selected_groups:
                            group_cols = [col for col in structure["replicates"][group] 
                                        if col.endswith("PG.Quantity")]
                            group_means[group] = heatmap_data[group_cols].mean(axis=1)

                        # Calculate inter-group variation score
                        scores['inter_cv'] = (group_means.std(axis=1) / group_means.mean(axis=1)) * 100

                        # Final score: high inter-group variation and low intra-group variation
                        scores['final_score'] = scores['inter_cv'] / (scores['intra_cv'] + 1)  # Add 1 to avoid division by zero

                        # Select top proteins based on score
                        top_proteins = scores.nlargest(n_proteins, 'final_score').index

                        # Prepare final data for plotting
                        plot_data = heatmap_data.loc[top_proteins]
                        plot_data_means = group_means.loc[top_proteins]

                        # Create column labels that show sample names
                        column_labels = []
                        for col in plot_data.columns:
                            sample_name = col.split("]")[1].split(".PG.Quantity")[0].strip()
                            for group in selected_groups:
                                if col in structure["replicates"][group]:
                                    column_labels.append(f"{sample_name}\n({group_names[group]})")
                                    break

                        # Create row labels (gene names if available)
                        if 'Gene Name' in data.columns:
                            row_labels = data.loc[top_proteins, 'Gene Name']
                        else:
                            row_labels = top_proteins

                        # Create two tabs for different heatmap views
                        heatmap_tab1, heatmap_tab2 = st.tabs(["Detailed Heatmap", "Group Average Heatmap"])

                        with heatmap_tab1:
                            # Create detailed clustermap
                            plt.figure(figsize=(12, 8))
                            g1 = sns.clustermap(
                                plot_data,
                                cmap='RdBu_r',
                                center=0,
                                robust=True,
                                xticklabels=column_labels,
                                yticklabels=row_labels,
                                dendrogram_ratio=(.1, .2),
                                cbar_pos=(0.02, .2, .03, .4),
                                figsize=(15, 10),
                                row_cluster=True,
                                col_cluster=True
                            )

                            # Adjust y-axis labels after creation
                            g1.ax_heatmap.set_yticklabels(
                                g1.ax_heatmap.get_yticklabels(),
                                fontsize=8,
                                rotation=0
                            )

                            # Increase spacing between labels
                            g1.fig.subplots_adjust(left=0.3)  # Increase left margin for labels

                            # Rotate x-axis labels
                            plt.setp(g1.ax_heatmap.get_xticklabels(), rotation=45, ha='right')

                            # Show plot in Streamlit
                            st.pyplot(g1.figure)
                            plt.close('all')

                        with heatmap_tab2:
                            # Create averaged clustermap
                            plt.figure(figsize=(12, 8))
                            g2 = sns.clustermap(
                                plot_data_means,
                                cmap='RdBu_r',
                                center=0,
                                robust=True,
                                xticklabels=[group_names[group] for group in selected_groups],
                                yticklabels=row_labels,
                                dendrogram_ratio=(.1, .2),
                                cbar_pos=(0.02, .2, .03, .4),
                                figsize=(12, 10),
                                row_cluster=True,
                                col_cluster=True
                            )

                            # Adjust y-axis labels after creation
                            g2.ax_heatmap.set_yticklabels(
                                g2.ax_heatmap.get_yticklabels(),
                                fontsize=8,
                                rotation=0
                            )

                            # Increase spacing between labels
                            g2.fig.subplots_adjust(left=0.3)  # Increase left margin for labels

                            # Rotate x-axis labels
                            plt.setp(g2.ax_heatmap.get_xticklabels(), rotation=45, ha='right')

                            # Show plot in Streamlit
                            st.pyplot(g2.figure)
                            plt.close('all')

                        # Download buttons
                        col1, col2, col3 = st.columns(3)

                        # Create buffers for both plots
                        buf1 = io.BytesIO()
                        g1.savefig(buf1, format='svg', bbox_inches='tight')
                        buf1.seek(0)

                        buf2 = io.BytesIO()
                        g2.savefig(buf2, format='svg', bbox_inches='tight')
                        buf2.seek(0)

                        with col1:
                            st.download_button(
                                label="Download Detailed Heatmap (SVG)",
                                data=buf1,
                                file_name="detailed_heatmap.svg",
                                mime="image/svg+xml"
                            )
                        with col2:
                            st.download_button(
                                label="Download Group Average Heatmap (SVG)",
                                data=buf2,
                                file_name="group_average_heatmap.svg",
                                mime="image/svg+xml"
                            )
                        with col3:
                            # Prepare CSV with additional information
                            csv_data = pd.concat([
                                plot_data,
                                plot_data_means,
                                scores.loc[top_proteins],
                                data.loc[top_proteins, 'Gene Name'] if 'Gene Name' in data.columns else pd.Series(index=top_proteins)
                            ], axis=1)
                            csv_buffer = csv_data.to_csv(index=True)
                            st.download_button(
                                label="Download Data as CSV",
                                data=csv_buffer,
                                file_name="heatmap_data.csv",
                                mime="text/csv"
                            )

                        plt.close('all')

                    except Exception as e:
                        st.error(f"Error generating heat map: {str(e)}")
                else:
                    st.warning("Please select groups with at least 2 samples")
            else:
                st.warning("Please select at least one replicate group")

    else:
        st.info("Please upload one or more datasets to begin analysis")