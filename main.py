import os
import sys
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    from utils import data_processing as dp

    # Log successful imports
    logger.info("All required packages imported successfully")

except Exception as e:
    logger.error(f"Error during imports: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)

try:
    # Configure Streamlit page
    st.set_page_config(
        page_title="Proteomics Data Analysis",
        page_icon="🧬",
        layout="wide"
    )

    # Title and introduction
    st.title("🧬 Proteomics Data Analysis Platform")
    st.markdown("""
    This platform provides comprehensive tools for proteomics data analysis, including:
    - Data validation and preprocessing
    - Statistical analysis
    - Interactive visualizations
    - Publication-ready figure export
    """)

    # Initialize session state
    if 'datasets' not in st.session_state:
        st.session_state.datasets = {}
    if 'dataset_info' not in st.session_state:
        st.session_state.dataset_info = {}

    # File Upload Section
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more datasets (Excel format)",
        accept_multiple_files=True,
        type=["xlsx", "csv"]
    )

    if uploaded_files:
        try:
            # Load and store datasets
            for uploaded_file in uploaded_files:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        data = pd.read_csv(uploaded_file)
                    else:
                        data = pd.read_excel(uploaded_file)

                    # Store raw data
                    st.session_state.datasets[uploaded_file.name] = data

                    # Show basic data preview
                    st.write(f"### Dataset: {uploaded_file.name}")
                    st.write("#### Preview")
                    st.dataframe(data.head())

                    # Show basic statistics
                    st.write("#### Basic Statistics")
                    st.write(f"Number of rows: {len(data)}")
                    st.write(f"Number of columns: {len(data.columns)}")

                    logger.info(f"Successfully loaded dataset: {uploaded_file.name}")

                except Exception as e:
                    logger.error(f"Error loading {uploaded_file.name}: {str(e)}")
                    st.error(f"Error loading {uploaded_file.name}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")
            st.error(f"Error processing files: {str(e)}")

    else:
        st.info("Please upload one or more datasets to get started.")

except Exception as e:
    st.error(f"Application error: {str(e)}")
    logger.error(f"Application error: {str(e)}")
    logger.error(traceback.format_exc())