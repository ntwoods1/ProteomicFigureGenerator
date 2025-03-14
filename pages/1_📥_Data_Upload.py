import streamlit as st
import pandas as pd
from utils.data_processor import DataProcessor

st.set_page_config(page_title="Data Upload", page_icon="📥")

st.header("Data Upload and Validation")

uploaded_file = st.file_uploader(
    "Upload your proteomics data file (CSV or Excel)",
    type=['csv', 'xlsx']
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        # Check for PG.Genes column
        if 'PG.Genes' not in df.columns:
            st.error("Required column 'PG.Genes' not found in the dataset")
        else:
            # Get numeric columns for quantitative data selection
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

            # Use a form to batch all inputs
            with st.form("sample_group_form"):
                st.subheader("Configure Sample Groups")

                # Sample group configuration
                num_groups = st.number_input("Number of sample groups", min_value=1, max_value=10, value=2)

                # Container for group selections
                group_selections = {}

                # Create selection boxes for each group
                for i in range(num_groups):
                    st.subheader(f"Sample Group {i+1}")
                    group_name = st.text_input(f"Group {i+1} Name", value=f"Group {i+1}")
                    selected_cols = st.multiselect(
                        f"Select quantitative columns for {group_name}",
                        options=numeric_cols,
                        key=f"group_{i}"
                    )
                    group_selections[group_name] = selected_cols

                # Submit button for the form
                submitted = st.form_submit_button("Process Data")

                if submitted:
                    # Check if all groups have selections
                    if all(len(cols) > 0 for cols in group_selections.values()):
                        # Store group information in session state
                        st.session_state.group_selections = group_selections
                        st.session_state.protein_col = 'PG.Genes'

                        # Create a subset of the data with selected columns
                        selected_cols = ['PG.Genes'] + [col for group in group_selections.values() for col in group]
                        
                        # Include peptide count columns if they exist
                        peptide_cols = [col for col in df.columns if 'peptide' in col.lower()]
                        for col in peptide_cols:
                            if col not in selected_cols:
                                selected_cols.append(col)
                                
                        processed_df = df[selected_cols].copy()

                        # Store processed data in session state
                        st.session_state.data = processed_df
                        st.session_state.filtered_data = processed_df.copy()  # For processing page
                        st.session_state.original_full_data = df.copy()  # Store the full original data
                        st.session_state.show_download = True
                        
                        # Add a success message with next steps
                        st.success("Data processed successfully! Go to the Data Processing page for filtering and normalization.")
                    else:
                        st.error("Please select at least one column for each group")

            # Show preview and download button outside the form
            if hasattr(st.session_state, 'show_download') and st.session_state.show_download:
                st.success("Data processed successfully!")
                st.write("Preview of processed data:")
                st.dataframe(st.session_state.data.head())

                # Download processed data
                csv = st.session_state.data.to_csv(index=False)
                st.download_button(
                    label="Download processed data",
                    data=csv,
                    file_name="processed_data.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")