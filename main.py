import streamlit as st
import logging
import sys

# Configure logging (from edited code)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

try:
    logger.info("Starting Streamlit application")

    st.set_page_config(
        page_title="Proteomics Analysis",
        page_icon="🧬",
        layout="wide"
    )
    logger.info("Page configuration set")

    st.title("Proteomics Analysis")
    st.write("Hello! The server is running.")

    logger.info("Application running")

except Exception as e:
    error_msg = f"Critical application error: {str(e)}" 
    logger.error(error_msg)
    logger.error(sys.exc_info()) 
    st.error(error_msg)