import streamlit as st
import pandas as pd
import logging

# Configure basic logging (improved from edited code)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

logger = logging.getLogger(__name__)

try:
    logger.info("Starting Streamlit application")

    # Configure the page (retaining original configuration)
    st.set_page_config(
        page_title="Proteomics Analysis",
        page_icon="🧬",
        layout="wide"
    )
    logger.info("Page configuration set")

    # Title (simplified from edited code)
    st.title("Proteomics Data Analysis")
    st.write("Upload your proteomics data file to begin analysis.")

    # File uploader (simplified from edited code)
    file = st.file_uploader("Choose a file", type=['xlsx', 'csv'])

    if file is not None:
        try:
            logger.info(f"Processing file: {file.name}")

            # Read the file (retaining original code's handling of .csv and .xlsx)
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            # Display basic info (simplified from edited code)
            st.write("### Data Preview")
            st.dataframe(df.head())

            st.write("### Basic Statistics")
            st.write(f"Number of rows: {len(df)}")
            st.write(f"Number of columns: {len(df.columns)}")
            #Show column names
            st.write("#### Columns")
            st.write(df.columns.tolist())


            logger.info("File processed successfully")

        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            st.error(f"Error processing file: {str(e)}")

    else:
        st.info("Please upload a file to begin.")

    logger.info("Application running")

except Exception as e:
    st.error(f"Application error: {str(e)}")
    logger.error(f"Application error: {str(e)}")
    #Retaining original more detailed error logging.
    import traceback
    logger.error(traceback.format_exc())