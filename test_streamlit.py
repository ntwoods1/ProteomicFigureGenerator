import streamlit as st
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

try:
    logger.info("Starting Streamlit application...")

    st.write("Hello")  # Absolutely minimal test

    logger.info("Application components rendered successfully")

except Exception as e:
    logger.error(f"Error in Streamlit application: {str(e)}", exc_info=True)
    st.error(f"Application error: {str(e)}")