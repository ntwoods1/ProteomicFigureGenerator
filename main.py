import streamlit as st
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('streamlit.log')
    ]
)
logger = logging.getLogger(__name__)

try:
    logger.debug("Starting streamlit application")
    print("Debug: Application starting", flush=True)  # Direct print for immediate feedback

    logger.debug("Setting page config")
    st.set_page_config(
        page_title="Proteomics Analysis",
        page_icon="🧬",
        layout="wide"
    )
    logger.debug("Page config set successfully")

    st.write("### Debug: Streamlit Initialization Test")
    st.write("If you can see this message, Streamlit is working correctly.")

    # Title
    st.title("Proteomic Data Analysis")
    logger.debug("Title rendered")

    # Basic content
    st.write("Welcome to the Proteomics Analysis tool!")
    logger.debug("Welcome message rendered")

    logger.info("Application initialized successfully")

except Exception as e:
    error_msg = f"Critical error during startup: {str(e)}"
    logger.error(error_msg)
    logger.error("Exception details:", exc_info=True)
    print(f"Error: {error_msg}", flush=True)  # Direct print for immediate feedback
    st.error(error_msg)