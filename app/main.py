from flask import Flask, render_template, jsonify
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__,
           template_folder='templates',
           static_folder='static')

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
    return render_template('index.html', spacetimes=SPACETIMES.keys())

@app.route('/simulate', methods=['POST'])
def simulate():
    """Temporary simulation endpoint that just returns success"""
    return jsonify({
        'status': 'success',
        'message': 'Simulation framework coming soon!',
        'metric_tensor': np.eye(4).tolist()  # Identity matrix for now
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)