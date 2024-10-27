import os
import numpy as np
from flask import Flask, render_template, jsonify, request
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

class SpacetimeSimulator:
    def __init__(self, name, metric_tensor_func):
        self.name = name
        self.metric_tensor_func = metric_tensor_func
        logger.debug(f"Created spacetime: {name}")

    def metric_tensor(self, coordinates):
        try:
            return self.metric_tensor_func(coordinates)
        except Exception as e:
            logger.error(f"Error computing metric tensor: {e}")
            return np.eye(4)  # Return Minkowski space if computation fails

def create_preset_spacetimes():
    def schwarzschild_metric(coordinates):
        t, x, y, z = coordinates[:4]
        r = np.sqrt(x**2 + y**2 + z**2)
        Rs = 2  # Schwarzschild radius
        factor = 1 - Rs / (r + Rs)  # Avoid division by zero
        return np.diag([-factor, 1/factor, r**2, r**2 * np.sin(np.arccos(z/(r+1e-10)))**2])

    return {
        "Schwarzschild": SpacetimeSimulator("Schwarzschild", schwarzschild_metric),
    }

# Global variable to store spacetimes
SPACETIMES = create_preset_spacetimes()

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

@app.route('/metric_tensor', methods=['POST'])
def get_metric():
    """Test endpoint for metric tensor calculation"""
    try:
        spacetime_name = request.form['spacetime']
        coordinates = np.array([0, 1, 0, 0])  # Test coordinates
        
        if spacetime_name not in SPACETIMES:
            return jsonify({'error': 'Invalid spacetime selected'}), 400
        
        spacetime = SPACETIMES[spacetime_name]
        metric = spacetime.metric_tensor(coordinates).tolist()
        
        return jsonify({
            'status': 'success',
            'metric_tensor': metric,
            'coordinates': coordinates.tolist()
        })
    
    except Exception as e:
        logger.error(f"Error in metric_tensor route: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)