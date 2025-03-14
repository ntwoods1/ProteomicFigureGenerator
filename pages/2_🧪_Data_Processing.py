import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.visualizations import Visualizer
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Processing", page_icon="🧪")

st.header("Data Processing")

if "data" not in st.session_state or st.session_state.data is None:
    st.warning(
        "Please upload and process your data first in the Data Upload page.")
else:
    df = st.session_state.data.copy()
    protein_col = st.session_state.protein_col
        
    # Store reference to original full data if available (from file upload)
    if "original_full_data" not in st.session_state:
        # If not already stored, the current data is all we have
        st.session_state.original_full_data = df.copy()

    # Display original data stats
    st.subheader("Original Data")
    st.write(f"Number of proteins: {df.shape[0]}")
    st.write(
        f"Number of samples: {df.shape[1] - 1}")  # -1 for the protein column

    # Create tabs for each processing step
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Peptide Count Filter", "2. Valid Values Filter", "3. CV Threshold",
        "4. Normalization"
    ])

    # Initialize processing state if not exists
    if "filtered_data" not in st.session_state:
        st.session_state.filtered_data = df.copy()

    with tab1:
        st.subheader("Peptide Count Filter")
        st.write("""
        Filter proteins based on peptide count. Proteins with low peptide counts 
        are often less reliable.
        """)

        # Get all sample columns from group selections
        group_selections = st.session_state.get('group_selections', {})
        sample_cols = []
        for group, cols in group_selections.items():
            sample_cols.extend(cols)
        
        # Look for corresponding peptide count columns in original full data
        peptide_cols = []
        peptide_indicators = ["PG.NrOfPrecursorsMeasured", "PG.NrOfStrippedSequencesIdentified", 
                              "NrOfPrecursors", "NumberOfPeptides", "PeptideCount"]
        
        # Use original full data for peptide column lookup
        original_df = st.session_state.original_full_data
        
        # First, try to match based on sample names
        for col in sample_cols:
            if "PG.Quantity" in col:
                # Extract the base part of the sample column name (everything before .PG.Quantity)
                base_name = col.rsplit(".PG.Quantity", 1)[0]
                
                # Look for exact matches with the correct suffix
                precursor_col = f"{base_name}.PG.NrOfPrecursorsMeasured"
                sequence_col = f"{base_name}.PG.NrOfStrippedSequencesIdentified"
                
                if precursor_col in original_df.columns:
                    peptide_cols.append(precursor_col)
                elif sequence_col in original_df.columns:
                    peptide_cols.append(sequence_col)
        
        # If no peptide columns were found, try to find any peptide-related columns
        if not peptide_cols:
            # Try to identify peptide count columns by common patterns
            all_columns = df.columns.tolist()
            for col in all_columns:
                for indicator in peptide_indicators:
                    if indicator in col:
                        peptide_cols.append(col)
                        break
                
                # Also check for other common patterns
                if "peptide" in col.lower() or "precursor" in col.lower() or "sequence" in col.lower():
                    peptide_cols.append(col)
            
            # Remove duplicates
            peptide_cols = list(set(peptide_cols))

        if not peptide_cols:
            st.warning(
                "No peptide count columns detected. Skipping this filter.")
        else:
            st.success(f"Found {len(peptide_cols)} peptide count columns.")
            
            min_peptides = st.slider("Minimum number of peptides:", 1, 10, 2)
            
            # Add option for how to apply the filter
            filter_method = st.selectbox(
                "Filter method:",
                ["Any (at least one sample must meet threshold)", 
                 "All (all samples must meet threshold)",
                 "Average (average peptide count must meet threshold)"],
                index=0
            )

            if st.button("Apply Peptide Filter"):
                filtered_df = DataProcessor.filter_by_peptide_count(
                    st.session_state.filtered_data, 
                    peptide_cols, 
                    min_peptides, 
                    filter_method,
                    original_df=st.session_state.original_full_data)
                st.session_state.filtered_data = filtered_df
                st.success(
                    f"Filtered data now has {filtered_df.shape[0]} proteins")

    with tab2:
        st.subheader("Valid Values Filter")
        st.write("""
        Filter proteins based on the percentage of valid (non-missing) values in each group.
        After filtering, remaining missing values can be imputed using various methods.
        """)

        group_selections = st.session_state.get('group_selections', {})
        if not group_selections:
            st.warning(
                "No sample groups defined. Please configure groups in the Data Upload page."
            )
        else:
            min_valid_values = st.slider(
                "Minimum percentage of valid values per group:",
                0,
                100,
                70,
                help=
                "Proteins with fewer valid values than this threshold in any group will be filtered out"
            )

            # Add imputation options
            st.subheader("Missing Value Imputation")
            imputation_method = st.selectbox(
                "Select imputation method for remaining missing values:",
                ["None", "Mean", "Median", "Min", "KNN", "Group-wise Mean"],
                help="""
                None: Keep missing values as they are
                Mean: Replace with mean of each column
                Median: Replace with median of each column
                Min: Replace with minimum/2 of each row
                KNN: K-nearest neighbors imputation
                Group-wise Mean: Replace with mean of each group
                """)

            col1, col2 = st.columns(2)
            with col1:
                filter_button = st.button("1. Apply Valid Values Filter")
            with col2:
                impute_button = st.button("2. Apply Imputation")

            if filter_button:
                filtered_df = DataProcessor.filter_by_valid_values(
                    st.session_state.filtered_data,
                    group_selections,
                    min_valid_values / 100  # Convert to fraction
                )
                st.session_state.filtered_data = filtered_df
                st.success(
                    f"Filtered data now has {filtered_df.shape[0]} proteins")

                # Display missing values statistics
                missing_counts = filtered_df.isna().sum().sum()
                total_cells = filtered_df.size
                if missing_counts > 0:
                    st.info(
                        f"There are still {missing_counts} missing values ({missing_counts/total_cells:.2%} of data). Use imputation to fill these values."
                    )

            if impute_button:
                if imputation_method != "None":
                    imputed_df = DataProcessor.impute_missing_values(
                        st.session_state.filtered_data, imputation_method,
                        group_selections
                        if imputation_method == "Group-wise Mean" else None)
                    st.session_state.filtered_data = imputed_df

                    # Check if imputation was successful
                    remaining_missing = imputed_df.isna().sum().sum()
                    if remaining_missing == 0:
                        st.success(
                            f"Successfully imputed all missing values using {imputation_method} method"
                        )
                    else:
                        # Apply one final imputation with zeros for any remaining values
                        numeric_cols = imputed_df.select_dtypes(
                            include=[np.number]).columns
                        imputed_df[numeric_cols] = imputed_df[
                            numeric_cols].fillna(0)
                        st.session_state.filtered_data = imputed_df

                        st.success(
                            f"Imputation completed with {imputation_method} method. Any remaining missing values were filled with zeros."
                        )
                else:
                    st.info("No imputation applied.")

    with tab3:
        st.subheader("CV Threshold")
        st.write("""
        Filter proteins based on coefficient of variation (CV) within each group.
        Proteins with high CV are often more variable and potentially less reliable.
        """)

        group_selections = st.session_state.get('group_selections', {})
        if not group_selections:
            st.warning(
                "No sample groups defined. Please configure groups in the Data Upload page."
            )
        else:
            # Always use original data for CV visualization
            if "original_full_data" in st.session_state:
                display_df = st.session_state.original_full_data
                st.info("Showing CV histogram for original unprocessed data.")
            else:
                display_df = st.session_state.filtered_data
                st.warning("Original data not available. Showing current data.")
            
            # CV Slider (using fraction 0-1 for internal calculations, but displaying as percentage)
            max_cv = st.slider(
                "Maximum coefficient of variation (%):",
                0,
                100,
                20,
                help=
                "Proteins with higher CV than this threshold in any group will be filtered out"
            )
            
            # Convert percentage to fraction for visualization
            cv_cutoff = max_cv / 100
            
            # Display CV Histogram
            st.subheader("CV Distribution")
            
            # Generate the CV histogram using matplotlib (more reliable for exports)
            from utils.visualizations import Visualizer
            import matplotlib.pyplot as plt
            
            # Display information about the current data
            st.info(f"Displaying CV distribution for {len(display_df)} proteins across {len(group_selections)} groups.")
            
            # Generate the CV histogram using matplotlib (more reliable for exports)
            matplotlib_fig, cv_values_by_group = Visualizer.create_cv_histogram_matplotlib(display_df, group_selections, cv_cutoff)
            
            # Also generate the original plotly version for interactive display
            plotly_fig = Visualizer.create_cv_histogram(display_df, group_selections, cv_cutoff)
            
            # Display plotly version for interactive viewing
            st.plotly_chart(plotly_fig)
            
            # Get SVG data from matplotlib figure
            svg_data = Visualizer.get_matplotlib_svg(matplotlib_fig)
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                # Download button for SVG
                st.download_button(
                    label="Download Histogram as SVG",
                    data=svg_data,
                    file_name="cv_histogram.svg",
                    mime="image/svg+xml"
                )
            with col2:
                # Generate CV table for all proteins
                cv_table = Visualizer.generate_cv_table(display_df, group_selections)
                
                # Download button for CV table
                st.download_button(
                    label="Download CV Table",
                    data=cv_table.to_csv(index=False),
                    file_name="cv_values.csv",
                    mime="text/csv"
                )
            
            # Add explanation
            st.markdown("""
            **About the CV Histogram:**
            - The Coefficient of Variation (CV) is calculated as standard deviation divided by mean for each protein across samples in a group
            - Lower CV values indicate more consistent measurements across replicates
            - The red dashed line shows your selected CV cutoff value
            - Proteins to the right of the red line would be filtered out by the CV filter
            """)
            
            # Calculate percentage of proteins that would be filtered at current cutoff
            filtered_count = 0
            total_proteins = len(display_df)
            
            for group_name, columns in group_selections.items():
                if columns:
                    group_data = display_df[columns]
                    means = group_data.mean(axis=1).abs()
                    stds = group_data.std(axis=1)
                    
                    # Calculate CV and handle potential division by zero
                    mask = (means > 0) & (~means.isna()) & (~stds.isna())
                    cv_values = pd.Series(np.nan, index=means.index)
                    cv_values.loc[mask] = stds.loc[mask] / means.loc[mask]
                    
                    filtered_count += (cv_values > cv_cutoff).sum()
                    
            # Show percentage that would be filtered (if at least one group has a high CV)
            if total_proteins > 0:
                st.write(f"At {cv_cutoff:.2f} cutoff, approximately {filtered_count/total_proteins:.1%} of proteins would be filtered out.")
            
            # Add the apply button after the visualization
            if st.button("Apply CV Filter"):
                filtered_df = DataProcessor.filter_by_cv(
                    st.session_state.filtered_data,
                    group_selections,
                    cv_cutoff
                )
                st.session_state.filtered_data = filtered_df
                st.success(
                    f"Filtered data now has {filtered_df.shape[0]} proteins")

    with tab4:
        st.subheader("Normalization")
        st.write("""
        Normalize data to adjust for technical variation between samples.
        """)

        # Keep a copy of the pre-normalized data if it doesn't exist yet
        if "pre_normalized_data" not in st.session_state:
            st.session_state.pre_normalized_data = st.session_state.filtered_data.copy()

        norm_method = st.selectbox("Select normalization method:", [
            "None", "Log2", "Median", "Z-score", "Quantile", "Total Intensity"
        ],
                                   help="""
            None: No normalization
            Log2: Log2 transformation
            Median: Subtract median of each sample
            Z-score: (x - mean) / std for each sample
            Quantile: Quantile normalization across samples
            Total Intensity: Divide by sum of intensities for each sample
            """)

        if st.button("Apply Normalization"):
            # Reset to pre-normalized data before applying new normalization
            st.session_state.filtered_data = st.session_state.pre_normalized_data.copy()
            
            if norm_method != "None":
                numeric_cols = st.session_state.filtered_data.select_dtypes(
                    include=[np.number]).columns

                filtered_df = DataProcessor.normalize_data(
                    st.session_state.filtered_data, norm_method, numeric_cols)
                st.session_state.filtered_data = filtered_df
                st.success(f"Data normalized using {norm_method} method")
            else:
                st.success("Normalization removed, using original filtered data")

    # Display final processed data
    st.subheader("Processed Data Preview")
    st.dataframe(st.session_state.filtered_data.head())

    # Save processed data to session state for next steps
    if st.button("Save Processed Data"):
        st.session_state.data = st.session_state.filtered_data
        st.success("Processed data saved successfully!")

        # Download processed data
        csv = st.session_state.filtered_data.to_csv(index=False)
        st.download_button(label="Download processed data",
                           data=csv,
                           file_name="processed_proteomics_data.csv",
                           mime="text/csv")
