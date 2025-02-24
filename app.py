import os
from flask import Flask, jsonify
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.route('/')
def home():
    return jsonify({'status': 'Server is running'})

if __name__ == '__main__':
    try:
        # Use port 8080 instead since 5000 is in use
        app.run(host='0.0.0.0', port=8080)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)