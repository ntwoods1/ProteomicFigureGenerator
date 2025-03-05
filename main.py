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
from scipy.stats import ttest_ind, stats
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

def apply_multiple_testing_correction(p_values, method='bonferroni'):
    """Apply multiple testing correction to p-values."""
    from scipy import stats
    import numpy as np
    
    if method == 'bonferroni':
        # Bonferroni correction
        return np.minimum(p_values * len(p_values), 1.0)
    elif method == 'fdr':
        # Benjamini-Hochberg FDR
        ranked_p_values = stats.rankdata(p_values)
        fdr = p_values * len(p_values) / ranked_p_values
        fdr[fdr > 1] = 1  # Cap at 1
        return fdr
    else:
        return p_values

def calculate_significance_matrix(data, groups, structure, alpha=0.05):
    """Calculate statistical significance between groups for each protein."""
    from scipy import stats
    import numpy as np
    
    # Get all pairs of groups
    group_pairs = [(g1, g2) for i, g1 in enumerate(groups) for g2 in groups[i+1:]]
    
    # Initialize results dictionary
    significance = {}
    
    # For each protein
    for protein in data.index:
        significance[protein] = {}
        
        # For each pair of groups
        for g1, g2 in group_pairs:
            # Get quantity columns for each group
            g1_cols = [col for col in structure["replicates"][g1] if col.endswith("PG.Quantity")]
            g2_cols = [col for col in structure["replicates"][g2] if col.endswith("PG.Quantity")]
            
            # Get values for each group
            g1_values = data.loc[protein, g1_cols].dropna()
            g2_values = data.loc[protein, g2_cols].dropna()
            
            # Perform t-test if we have enough values
            if len(g1_values) >= 2 and len(g2_values) >= 2:
                _, p_val = stats.ttest_ind(g1_values, g2_values)
                significance[protein][(g1, g2)] = p_val < alpha
            else:
                significance[protein][(g1, g2)] = False
                
    return significance

def add_significance_markers(g, significance_data, ax, groups, group_means=None):
    """Add significance markers to the heatmap."""
    import numpy as np
    
    # If using group means, we'll mark directly on the heatmap
    if group_means is not None:
        for protein_idx, protein in enumerate(significance_data.keys()):
            for (g1, g2), is_significant in significance_data[protein].items():
                if is_significant:
                    g1_idx = groups.index(g1)
                    g2_idx = groups.index(g2)
                    # Add asterisk between significant pairs
                    ax.text(g1_idx, protein_idx, '*', 
                           ha='center', va='center',
                           color='black', fontweight='bold')
                    ax.text(g2_idx, protein_idx, '*',
                           ha='center', va='center',
                           color='black', fontweight='bold')
    else:
        # For detailed heatmap, add markers to the group labels
        for protein_idx, protein in enumerate(significance_data.keys()):
            significant_groups = set()
            for (g1, g2), is_significant in significance_data[protein].items():
                if is_significant:
                    significant_groups.add(g1)
                    significant_groups.add(g2)
            
            if significant_groups:
                # Add marker to the protein label
                current_label = ax.get_yticklabels()[protein_idx].get_text()
                ax.get_yticklabels()[protein_idx].set_text(f"{current_label} *")

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
    tabs = ["Data Overview", "Volcano Plot", "PCA", "Heat Map", "Custom Protein Heatmap", "Protein Bar Charts"]
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
                                        # Create dictionaries for storing gene sets
                                        up_sets = {}
                                        down_sets = {}

                                        # Collect genes from all comparisons
                                        for comp_key, comp_data in st.session_state['volcano_comparisons'].items():
                                            if comp_data['significant_up']:
                                                up_sets[comp_key] = list(comp_data['significant_up'])
                                            if comp_data['significant_down']:
                                                down_sets[comp_key] = list(comp_data['significant_down'])

                                        if up_sets:
                                            st.subheader("Up-regulated Proteins")
                                            st.write(f"Overlap analysis of up-regulated proteins across {len(up_sets)} comparisons")

                                            # Check if we need to regenerate the plot
                                            cache_key = f"upset_up_{dataset_name}"
                                            if cache_key not in st.session_state:
                                                # Create figure for up-regulated proteins
                                                fig_up = plt.figure(figsize=(12, 6))
                                                up_data = from_contents(up_sets)
                                                upset = UpSet(up_data)
                                                upset.plot(fig=fig_up)

                                                # Store in session state
                                                buf = io.BytesIO()
                                                fig_up.savefig(buf, format='svg', bbox_inches='tight')
                                                buf.seek(0)

                                                # Create protein list
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

                                                st.session_state[cache_key] = {
                                                    'figure': fig_up,
                                                    'svg_buffer': buf,
                                                    'protein_data': protein_df
                                                }
                                                plt.close(fig_up)

                                            # Display cached figure
                                            st.pyplot(st.session_state[cache_key]['figure'])

                                            # Create download buttons
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.download_button(
                                                    label="Download Plot as SVG",
                                                    data=st.session_state[cache_key]['svg_buffer'],
                                                    file_name="upset_plot_upregulated.svg",
                                                    mime="image/svg+xml",
                                                    key=f"up_svg_{dataset_name}"
                                                )
                                            with col2:
                                                protein_csv = st.session_state[cache_key]['protein_data'].to_csv(index=False)
                                                st.download_button(
                                                    label="Download Protein List",
                                                    data=protein_csv,
                                                    file_name="upregulated_proteins.csv",
                                                    mime="text/csv",
                                                    key=f"up_csv_{dataset_name}"
                                                )

                                        if down_sets:
                                            st.subheader("Down-regulated Proteins")
                                            st.write(f"Overlap analysis of down-regulated proteins across {len(down_sets)} comparisons")

                                            # Check if we need to regenerate the plot
                                            cache_key = f"upset_down_{dataset_name}"
                                            if cache_key not in st.session_state:
                                                # Create figure for down-regulated proteins
                                                fig_down = plt.figure(figsize=(12, 6))
                                                down_data = from_contents(down_sets)
                                                upset = UpSet(down_data)
                                                upset.plot(fig=fig_down)

                                                # Store in session state
                                                buf = io.BytesIO()
                                                fig_down.savefig(buf, format='svg', bbox_inches='tight')
                                                buf.seek(0)

                                                # Create protein list
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

                                                st.session_state[cache_key] = {
                                                    'figure': fig_down,
                                                    'svg_buffer': buf,
                                                    'protein_data': protein_df
                                                }
                                                plt.close(fig_down)

                                            # Display cached figure
                                            st.pyplot(st.session_state[cache_key]['figure'])

                                            # Create download buttons
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.download_button(
                                                    label="Download Plot as SVG",
                                                    data=st.session_state[cache_key]['svg_buffer'],
                                                    file_name="upset_plot_downregulated.svg",
                                                    mime="image/svg+xml",
                                                    key=f"down_svg_{dataset_name}"
                                                )
                                            with col2:
                                                protein_csv = st.session_state[cache_key]['protein_data'].to_csv(index=False)
                                                st.download_button(
                                                    label="Download Protein List",
                                                    data=protein_csv,
                                                    file_name="downregulated_proteins.csv",
                                                    mime="text/csv",
                                                    key=f"down_csv_{dataset_name}"
                                                )

                                    except Exception as e:
                                        st.error(f"Error generating overlap analysis: {str(e)}")

                                else:
                                    st.warning("No replicate groups found in the dataset")
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

                        # Calculate group means for each protein
                        group_means = pd.DataFrame(index=heatmap_data.index)
                        for group in selected_groups:
                            group_cols = [col for col in structure["replicates"][group] 
                                        if col.endswith("PG.Quantity")]
                            group_means[group] = heatmap_data[group_cols].mean(axis=1)

                        # Calculate fold changes between all pairs of groups
                        fold_changes = []
                        for g1, g2 in combinations(selected_groups, 2):
                            fc = np.abs(np.log2(group_means[g2] / group_means[g1]))
                            fold_changes.append(fc)

                        # Use maximum fold change as part of the score
                        scores['max_fold_change'] = pd.concat(fold_changes, axis=1).max(axis=1)

                        # Calculate F-statistic and p-value using ANOVA
                        f_stats = []
                        p_values = []
                        for protein in heatmap_data.index:
                            group_data = []
                            for group in selected_groups:
                                group_cols = [col for col in structure["replicates"][group] 
                                            if col.endswith("PG.Quantity")]
                                group_data.append(heatmap_data.loc[protein, group_cols])
                            try:
                                f_stat, p_val = stats.f_oneway(*group_data)
                                f_stats.append(f_stat)
                                p_values.append(p_val)
                            except:
                                f_stats.append(0)
                                p_values.append(1)

                        scores['f_statistic'] = f_stats
                        scores['p_value'] = p_values
                        scores['-log10_p'] = -np.log10(scores['p_value'].clip(1e-10, 1))

                        # Calculate intra-group variation
                        intra_cv = pd.DataFrame(index=heatmap_data.index)
                        for group in selected_groups:
                            group_cols = [col for col in structure["replicates"][group] 
                                        if col.endswith("PG.Quantity")]
                            group_data = heatmap_data[group_cols]
                            cv = (group_data.std(axis=1) / group_data.mean(axis=1)) * 100
                            intra_cv[group] = cv

                        # Average intra-group CV (lower is better)
                        scores['intra_cv'] = intra_cv.mean(axis=1)

                        # Final score: combine statistical significance, fold change, and reproducibility
                        # Higher score for:
                        # - Lower p-values (higher -log10_p)
                        # - Higher fold changes
                        # - Lower intra-group variation
                        scores['final_score'] = (
                            scores['-log10_p'] * 
                            scores['max_fold_change'] * 
                            (100 / (scores['intra_cv'] + 10))  # Add 10 to avoid division by very small numbers
                        )

                        # Add minimum fold change filter
                        min_fold_change = 0.5  # log2 fold change threshold
                        scores.loc[scores['max_fold_change'] < min_fold_change, 'final_score'] = 0

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

                        # Calculate figure size using the specified formula
                        # width is fixed at 10, height is dynamic based on number of proteins (n)
                        figure_height = 10 + (n_proteins/10) - 1

                        # Add clustering toggles
                        col1, col2 = st.columns(2)
                        with col1:
                            cluster_rows = st.toggle("Cluster proteins", value=True)
                        with col2:
                            cluster_cols = st.toggle("Cluster samples", value=True)

                        # Add significance testing toggle
                        col3, col4 = st.columns(2)
                        with col3:
                            show_significance = st.toggle("Show statistical significance", value=True)
                        with col4:
                            significance_alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)

                        # Calculate significance if enabled
                        if show_significance:
                            significance_results = calculate_significance_matrix(
                                plot_data, selected_groups, structure, alpha=significance_alpha
                            )

                        with heatmap_tab1:
                            g1 = sns.clustermap(
                                plot_data,
                                cmap='RdBu_r',
                                center=0,
                                robust=True,
                                xticklabels=column_labels,
                                yticklabels=row_labels,
                                dendrogram_ratio=(.1, .2),
                                cbar_pos=(0.02, .2, .03, .4),
                                figsize=(10, figure_height),
                                row_cluster=cluster_rows,
                                col_cluster=cluster_cols
                            )

                            # Adjust y-axis labels after creation
                            g1.ax_heatmap.set_yticklabels(
                                g1.ax_heatmap.get_yticklabels(),
                                fontsize=8,
                                rotation=0
                            )

                            # Increase spacing between labels
                            g1.fig.subplots_adjust(left=0.3)

                            # Rotate x-axis labels
                            plt.setp(g1.ax_heatmap.get_xticklabels(), rotation=45, ha='right')

                            # Add significance markers if enabled
                            if show_significance:
                                add_significance_markers(g1, significance_results, g1.ax_heatmap, 
                                                      selected_groups)

                            # Show plot in Streamlit
                            st.pyplot(g1.figure)
                            plt.close('all')

                        with heatmap_tab2:
                            g2 = sns.clustermap(
                                plot_data_means,
                                cmap='RdBu_r',
                                center=0,
                                robust=True,
                                xticklabels=[group_names[group] for group in selected_groups],
                                yticklabels=row_labels,
                                dendrogram_ratio=(.1, .2),
                                cbar_pos=(0.02, .2, .03, .4),
                                figsize=(10, figure_height),
                                row_cluster=cluster_rows,
                                col_cluster=cluster_cols
                            )

                            # Adjust y-axis labels after creation
                            g2.ax_heatmap.set_yticklabels(
                                g2.ax_heatmap.get_yticklabels(),
                                fontsize=8,
                                rotation=0
                            )

                            # Increase spacing between labels
                            g2.fig.subplots_adjust(left=0.3)

                            # Rotate x-axis labels
                            plt.setp(g2.ax_heatmap.get_xticklabels(), rotation=45, ha='right')

                            # Add significance markers if enabled
                            if show_significance:
                                add_significance_markers(g2, significance_results, g2.ax_heatmap, 
                                                      selected_groups, group_means=True)

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

    elif active_tab == "Custom Protein Heatmap":
        st.header("Custom Protein Heatmap")

        # Dataset selection
        dataset_name = st.selectbox(
            "Select a dataset",
            options=list(datasets.keys()),
            key="custom_heatmap_dataset"
        )

        if dataset_name and dataset_name in datasets:
            data = datasets[dataset_name]['normalized']

            if dataset_name in st.session_state['dataset_structures']:
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
                            key=f"custom_group_name_{group}"
                        )

                # Allow selection of replicate groups
                selected_groups = st.multiselect(
                    "Select replicate groups for heat map",
                    options=replicate_groups,
                    default=replicate_groups[:2] if len(replicate_groups) >= 2 else replicate_groups
                )

                if selected_groups:
                    # Get protein list from user
                    protein_input = st.text_area(
                        "Enter protein names (one per line)",
                        help="Enter gene names or protein IDs, one per line. These will be used to find matching proteins in the dataset."
                    )

                    if protein_input:
                        # Process protein list
                        protein_list = [p.strip() for p in protein_input.split('\n') if p.strip()]

                        # Find matching proteins in the dataset
                        if 'Gene Name' in data.columns:
                            matches = data[data['Gene Name'].str.contains('|'.join(protein_list), case=False, na=False)]
                        else:
                            matches = data[data.index.str.contains('|'.join(protein_list), case=False, na=False)]

                        if not matches.empty:
                            st.write(f"Found {len(matches)} matching proteins")

                            # Prepare heatmap data
                            heatmap_data = pd.DataFrame()
                            for group in selected_groups:
                                group_cols = [col for col in structure["replicates"][group] 
                                            if col.endswith("PG.Quantity")]
                                if group_cols:
                                    heatmap_data = pd.concat([heatmap_data, matches[group_cols]], axis=1)

                            if not heatmap_data.empty:
                                # Calculate group means for averaged heatmap
                                group_means = pd.DataFrame(index=heatmap_data.index)
                                for group in selected_groups:
                                    group_cols = [col for col in structure["replicates"][group] 
                                                if col.endswith("PG.Quantity")]
                                    group_means[group] = heatmap_data[group_cols].mean(axis=1)

                                # Create column labels
                                column_labels = []
                                for col in heatmap_data.columns:
                                    sample_name = col.split("]")[1].split(".PG.Quantity")[0].strip()
                                    for group in selected_groups:
                                        if col in structure["replicates"][group]:
                                            column_labels.append(f"{sample_name}\n({group_names[group]})")
                                            break

                                # Create row labels (gene names if available)
                                if 'Gene Name' in matches.columns:
                                    row_labels = matches['Gene Name']
                                else:
                                    row_labels = matches.index

                                # Create two tabs for different heatmap views
                                heatmap_tab1, heatmap_tab2 = st.tabs(["Detailed Heatmap", "Group Average Heatmap"])

                                # Calculate figure size using the specified formula
                                figure_height = 10 + (len(matches)/10) - 1

                                # Add clustering toggles
                                col1, col2 = st.columns(2)
                                with col1:
                                    cluster_rows = st.toggle("Cluster proteins", value=True)
                                with col2:
                                    cluster_cols = st.toggle("Cluster samples", value=True)

                                # Add significance testing toggle
                                col3, col4 = st.columns(2)
                                with col3:
                                    show_significance = st.toggle("Show statistical significance", value=True)
                                with col4:
                                    significance_alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)

                                # Calculate significance if enabled
                                if show_significance:
                                    significance_results = calculate_significance_matrix(
                                        heatmap_data, selected_groups, structure, alpha=significance_alpha
                                    )

                                with heatmap_tab1:
                                    g1 = sns.clustermap(
                                        heatmap_data,
                                        cmap='RdBu_r',
                                        center=0,
                                        robust=True,
                                        xticklabels=column_labels,
                                        yticklabels=row_labels,
                                        dendrogram_ratio=(.1, .2),
                                        cbar_pos=(0.02, .2, .03, .4),
                                        figsize=(10, figure_height),
                                        row_cluster=cluster_rows,
                                        col_cluster=cluster_cols
                                    )

                                    # Adjust y-axis labels
                                    g1.ax_heatmap.set_yticklabels(
                                        g1.ax_heatmap.get_yticklabels(),
                                        fontsize=8,
                                        rotation=0
                                    )

                                    # Increase spacing between labels
                                    g1.fig.subplots_adjust(left=0.3)

                                    # Rotate x-axis labels
                                    plt.setp(g1.ax_heatmap.get_xticklabels(), rotation=45, ha='right')

                                    # Add significance markers if enabled
                                    if show_significance:
                                        add_significance_markers(g1, significance_results, g1.ax_heatmap, 
                                                             selected_groups)

                                    # Show plot in Streamlit
                                    st.pyplot(g1.figure)
                                    plt.close('all')

                                with heatmap_tab2:
                                    g2 = sns.clustermap(
                                        group_means,
                                        cmap='RdBu_r',
                                        center=0,
                                        robust=True,
                                        xticklabels=[group_names[group] for group in selected_groups],
                                        yticklabels=row_labels,
                                        dendrogram_ratio=(.1, .2),
                                        cbar_pos=(0.02, .2, .03, .4),
                                        figsize=(10, figure_height),
                                        row_cluster=cluster_rows,
                                        col_cluster=cluster_cols
                                    )

                                    # Adjust y-axis labels
                                    g2.ax_heatmap.set_yticklabels(
                                        g2.ax_heatmap.get_yticklabels(),
                                        fontsize=8,
                                        rotation=0
                                    )

                                    # Increase spacing between labels
                                    g2.fig.subplots_adjust(left=0.3)

                                    # Rotate x-axis labels
                                    plt.setp(g2.ax_heatmap.get_xticklabels(), rotation=45, ha='right')

                                    # Add significance markers if enabled
                                    if show_significance:
                                        add_significance_markers(g2, significance_results, g2.ax_heatmap, 
                                                             selected_groups, group_means=True)

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
                                        file_name="custom_detailed_heatmap.svg",
                                        mime="image/svg+xml"
                                    )
                                with col2:
                                    st.download_button(
                                        label="Download Group Average Heatmap (SVG)",
                                        data=buf2,
                                        file_name="custom_group_average_heatmap.svg",
                                        mime="image/svg+xml"
                                    )
                                with col3:
                                    pass

    elif active_tab == "Protein Bar Charts":
        st.header("Protein Bar Charts")

        # Dataset selection
        dataset_name = st.selectbox(
            "Select a dataset",
            options=list(datasets.keys()),
            key="bar_chart_dataset"
        )

        if dataset_name and dataset_name in datasets:
            data = datasets[dataset_name]['normalized']

            if dataset_name in st.session_state['dataset_structures']:
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
                            key=f"bar_group_name_{group}"
                        )

                # Allow selection of replicate groups
                selected_groups = st.multiselect(
                    "Select replicate groups for bar charts",
                    options=replicate_groups,
                    default=replicate_groups[:2] if len(replicate_groups) >= 2 else replicate_groups
                )

                if selected_groups:
                    # Statistical analysis options
                    st.subheader("Statistical Analysis Options")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        error_bar_type = st.selectbox(
                            "Error bar type",
                            options=["Standard Deviation", "Standard Error of Mean"],
                            key="error_bar_type"
                        )
                    
                    with col2:
                        if len(selected_groups) >= 3:
                            stat_test = st.selectbox(
                                "Statistical test",
                                options=["T-test vs Control", "ANOVA"],
                                key="stat_test"
                            )
                        else:
                            stat_test = "T-test vs Control"

                    with col3:
                        show_replicates = st.toggle("Show individual replicates", value=False)

                    with col4:
                        show_fold_change = st.toggle("Show fold change vs control", value=False)

                    # Define use_log2 with a default value
                    use_log2 = False 
                    if show_fold_change:
                        use_log2 = st.toggle("Use log2 fold change", value=False)

                    # Multiple testing correction selection
                    multiple_testing = st.selectbox(
                        "Multiple testing correction",
                        options=["None", "Bonferroni", "FDR"],
                        help="Bonferroni is more conservative, FDR (False Discovery Rate) is less stringent",
                        key="multiple_testing"
                    )

                    if stat_test == "T-test vs Control":
                        control_group = st.selectbox(
                            "Select control group",
                            options=selected_groups,
                            key="control_group"
                        )

                    # Get protein list from user
                    protein_input = st.text_area(
                        "Enter protein names (one per line)",
                        help="Enter gene names or protein IDs, one per line. These will be used to find matching proteins in the dataset."
                    )

                    if protein_input:
                        # Process protein list
                        protein_list = [p.strip() for p in protein_input.split('\n') if p.strip()]

                        # Find matching proteins in the dataset
                        if 'Gene Name' in data.columns:
                            # Use exact matching instead of partial matching
                            matches = data[data['Gene Name'].isin(protein_list)]
                        else:
                            # Use exact matching for index
                            matches = data[data.index.isin(protein_list)]

                        if not matches.empty:
                            st.write(f"Found {len(matches)} matching proteins")
                            
                            # Add information about which proteins were not found
                            found_proteins = matches['Gene Name'].tolist() if 'Gene Name' in matches.columns else matches.index.tolist()
                            not_found = [p for p in protein_list if p not in found_proteins]
                            if not_found:
                                st.warning(f"Could not find the following proteins: {', '.join(not_found)}")

                            # Initialize statistics table and multiple testing correction data
                            stats_data = []
                            all_p_values = []
                            p_value_indices = []  # To keep track of which comparison each p-value belongs to

                            # Create bar plots for each protein
                            for idx, protein in matches.iterrows():
                                protein_name = protein['Gene Name'] if 'Gene Name' in matches.columns else idx
                                
                                # Prepare data for plotting
                                plot_data = []
                                errors = []
                                replicate_data = []
                                
                                # Get control values if using fold change
                                if show_fold_change and stat_test == "T-test vs Control":
                                    control_cols = [col for col in structure["replicates"][control_group] 
                                                  if col.endswith("PG.Quantity")]
                                    control_values = pd.to_numeric(protein[control_cols], errors='coerce').dropna()
                                    control_mean = float(control_values.mean()) if len(control_values) >= 1 else 1.0

                                for group in selected_groups:
                                    group_cols = [col for col in structure["replicates"][group] 
                                                if col.endswith("PG.Quantity")]
                                    # Convert to numeric and handle any non-numeric values
                                    group_values = pd.to_numeric(protein[group_cols], errors='coerce').dropna()
                                    
                                    if len(group_values) >= 2:
                                        if show_fold_change and stat_test == "T-test vs Control":
                                            values = group_values / control_mean
                                            if use_log2:
                                                values = np.log2(values)
                                            mean_value = float(values.mean())
                                            if error_bar_type == "Standard Deviation":
                                                error = float(values.std())
                                            else:  # Standard Error of Mean
                                                error = float(values.std() / np.sqrt(len(values)))
                                        else:
                                            mean_value = float(group_values.mean())
                                            if error_bar_type == "Standard Deviation":
                                                error = float(group_values.std())
                                            else:  # Standard Error of Mean
                                                error = float(group_values.std() / np.sqrt(len(group_values)))
                                        
                                        plot_data.append(mean_value)
                                        errors.append(error)
                                        if show_replicates:
                                            if show_fold_change and stat_test == "T-test vs Control":
                                                replicate_values = group_values / control_mean
                                                if use_log2:
                                                    replicate_values = np.log2(replicate_values)
                                                replicate_data.append(replicate_values)
                                            else:
                                                replicate_data.append(group_values)
                                    else:
                                        plot_data.append(0.0)
                                        errors.append(0.0)
                                        if show_replicates:
                                            replicate_data.append([])

                                # Create bar plot
                                fig, ax = plt.subplots(figsize=(10, 6))
                                bars = ax.bar(range(len(selected_groups)), plot_data)
                                
                                # Add error bars
                                ax.errorbar(range(len(selected_groups)), plot_data, yerr=errors, 
                                          fmt='none', color='black', capsize=5)

                                # Add individual replicate points if enabled
                                if show_replicates:
                                    for i, replicates in enumerate(replicate_data):
                                        if len(replicates) > 0:
                                            # Add random jitter to x positions
                                            x_jitter = np.random.uniform(-0.2, 0.2, size=len(replicates))
                                            ax.scatter([i + j for j in x_jitter], replicates,
                                                     color='black', alpha=0.5, zorder=3)

                                # Calculate and add statistical significance
                                if stat_test == "T-test vs Control" and len(selected_groups) >= 2:
                                    control_idx = selected_groups.index(control_group)
                                    control_cols = [col for col in structure["replicates"][control_group] 
                                                  if col.endswith("PG.Quantity")]
                                    control_values = pd.to_numeric(protein[control_cols], errors='coerce').dropna()

                                    for i, group in enumerate(selected_groups):
                                        if group != control_group:
                                            group_cols = [col for col in structure["replicates"][group] 
                                                        if col.endswith("PG.Quantity")]
                                            group_values = pd.to_numeric(protein[group_cols], errors='coerce').dropna()
                                            
                                            if len(control_values) >= 2 and len(group_values) >= 2:
                                                t_stat, p_val = stats.ttest_ind(control_values, group_values)
                                                # Calculate fold change for statistics
                                                fold_change = float(group_values.mean() / control_values.mean())
                                                if use_log2:
                                                    fold_change = np.log2(fold_change)

                                                stats_data.append({
                                                    'Protein': protein_name,
                                                    'Control': group_names[control_group],
                                                    'Test Group': group_names[group],
                                                    'Test Type': 'T-test',
                                                    'P-value': p_val,
                                                    'Fold Change': fold_change,
                                                    'Fold Change Type': 'log2' if use_log2 else 'regular',
                                                    'Significant': p_val < 0.05
                                                })
                                                
                                elif stat_test == "ANOVA" and len(selected_groups) >= 3:
                                    # Perform one-way ANOVA
                                    groups_for_anova = []
                                    for group in selected_groups:
                                        group_cols = [col for col in structure["replicates"][group] 
                                                    if col.endswith("PG.Quantity")]
                                        group_values = pd.to_numeric(protein[group_cols], errors='coerce').dropna()
                                        groups_for_anova.append(group_values)
                                    
                                    if all(len(g) >= 2 for g in groups_for_anova):
                                        f_stat, p_val = stats.f_oneway(*groups_for_anova)
                                        
                                        # Store p-value for multiple testing correction
                                        all_p_values.append(p_val)
                                        p_value_indices.append((protein_name, "ANOVA"))
                                        
                                        # Add to statistics table
                                        stats_data.append({
                                            'Protein': protein_name,
                                            'Groups': ', '.join([group_names[g] for g in selected_groups]),
                                            'Test Type': 'ANOVA',
                                            'P-value': p_val
                                        })

                                # Apply multiple testing correction if selected
                                if multiple_testing != "None" and all_p_values:
                                    adjusted_p_values = apply_multiple_testing_correction(
                                        np.array(all_p_values), 
                                        method=multiple_testing.lower()
                                    )
                                    
                                    # Update statistics with adjusted p-values
                                    for i, (protein_name, group) in enumerate(p_value_indices):
                                        adjusted_p = adjusted_p_values[i]
                                        # Update the corresponding entry in stats_data
                                        for stat in stats_data:
                                            if stat['Protein'] == protein_name and (
                                                (stat['Test Type'] == 'T-test' and stat['Test Group'] == group_names[group]) or
                                                (stat['Test Type'] == 'ANOVA' and group == "ANOVA")
                                            ):
                                                stat['Adjusted P-value'] = adjusted_p
                                                stat['Adjustment Method'] = multiple_testing
                                                stat['Significant'] = adjusted_p < 0.05

                                # Customize plot
                                ax.set_xticks(range(len(selected_groups)))
                                ax.set_xticklabels([group_names[g] for g in selected_groups], rotation=45, ha='right')
                                if show_fold_change and stat_test == "T-test vs Control":
                                    ax.set_ylabel('log2 Fold Change vs Control' if use_log2 else 'Fold Change vs Control')
                                    # Add horizontal line at y=0 for log2 or y=1 for regular fold change
                                    ax.axhline(y=0 if use_log2 else 1, color='gray', linestyle='--', alpha=0.5)
                                else:
                                    ax.set_ylabel('Normalized Intensity')
                                ax.set_title(f'{protein_name}')

                                # Show plot
                                st.pyplot(fig)
                                plt.close()

                                # Add download button for the plot
                                buf = io.BytesIO()
                                fig.savefig(buf, format='svg', bbox_inches='tight')
                                buf.seek(0)
                                st.download_button(
                                    label=f"Download {protein_name} plot (SVG)",
                                    data=buf,
                                    file_name=f"{protein_name}_bar_chart.svg",
                                    mime="image/svg+xml"
                                )

                            # Create and display statistics table
                            if stats_data:
                                st.subheader("Statistical Analysis Results")
                                stats_df = pd.DataFrame(stats_data)
                                st.dataframe(stats_df)

                                # Add download button for statistics
                                csv = stats_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Statistics (CSV)",
                                    data=csv,
                                    file_name="statistical_analysis.csv",
                                    mime="text/csv"
                                )
                        else:
                            st.warning("No matching proteins found in the dataset")
            else:
                st.error("Dataset structure information not found")
        else:
            if dataset_name:
                st.error("Selected dataset not found")
            else:
                st.info("Please select a dataset to create bar charts")

    else:
        st.info("Please upload one or more datasets to begin analysis")
