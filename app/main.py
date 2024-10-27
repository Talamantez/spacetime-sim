import os
from flask import Flask, render_template, jsonify
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get the absolute path to the template directory
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))

app = Flask(__name__,
           template_folder=template_dir,
           static_folder=static_dir)

logger.debug(f"Template directory: {template_dir}")
logger.debug(f"Static directory: {static_dir}")
logger.debug(f"Available templates: {os.listdir(template_dir) if os.path.exists(template_dir) else 'Directory not found'}")

# Sample spacetimes for testing UI
SPACETIMES = {
    "Schwarzschild": "Black Hole",
    "Anti-de Sitter": "AdS",
    "Gödel": "Time Machine"
}

@app.route('/health')
def health():
    return 'OK', 200

@app.route('/')
def index():
    try:
        return render_template('index.html', spacetimes=SPACETIMES.keys())
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        return str(e), 500

@app.route('/simulate', methods=['POST'])
def simulate():
    """Temporary simulation endpoint that just returns success"""
    return jsonify({
        'status': 'success',
        'message': 'Simulation framework coming soon!',
        'metric_tensor': np.eye(4).tolist()  # Identity matrix for now
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)