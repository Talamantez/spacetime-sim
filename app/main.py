import os
import numpy as np
from flask import Flask, render_template, jsonify, request
import logging

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log startup sequence
logger.debug("Starting application initialization")

# Get the absolute path to the template directory
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))

logger.debug(f"Template directory: {template_dir}")
logger.debug(f"Static directory: {static_dir}")

app = Flask(__name__,
           template_folder=template_dir,
           static_folder=static_dir)

logger.debug("Flask app created")

class SpacetimeSimulator:
    def __init__(self, name, metric_tensor_func):
        logger.debug(f"Initializing spacetime: {name}")
        self.name = name
        self.metric_tensor_func = metric_tensor_func

    def metric_tensor(self, coordinates):
        logger.debug(f"Computing metric tensor for {self.name} at {coordinates}")
        try:
            result = self.metric_tensor_func(coordinates)
            logger.debug(f"Metric tensor computed successfully")
            return result
        except Exception as e:
            logger.error(f"Error computing metric tensor: {e}")
            return np.eye(4)

def create_preset_spacetimes():
    logger.debug("Creating preset spacetimes")
    
    def schwarzschild_metric(coordinates):
        logger.debug(f"Computing Schwarzschild metric at {coordinates}")
        t, x, y, z = coordinates[:4]
        r = np.sqrt(x**2 + y**2 + z**2)
        Rs = 2  # Schwarzschild radius
        factor = 1 - Rs / (r + Rs)  # Avoid division by zero
        return np.diag([-factor, 1/factor, r**2, r**2 * np.sin(np.arccos(z/(r+1e-10)))**2])

    spacetimes = {
        "Schwarzschild": SpacetimeSimulator("Schwarzschild", schwarzschild_metric),
    }
    logger.debug("Preset spacetimes created successfully")
    return spacetimes

# Global variable to store spacetimes
logger.debug("Initializing global spacetimes")
SPACETIMES = create_preset_spacetimes()
logger.debug("Global spacetimes initialized")

@app.route('/health')
def health():
    logger.debug("Health check requested")
    return 'OK', 200

@app.route('/')
def index():
    logger.debug("Index route requested")
    try:
        logger.debug(f"Available spacetimes: {list(SPACETIMES.keys())}")
        return render_template('index.html', spacetimes=SPACETIMES.keys())
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        return str(e), 500

@app.route('/metric_tensor', methods=['POST'])
def get_metric():
    logger.debug("Metric tensor calculation requested")
    try:
        spacetime_name = request.form['spacetime']
        radius = float(request.form.get('radius', 3.0))
        logger.debug(f"Parameters: spacetime={spacetime_name}, radius={radius}")
        
        coordinates = np.array([0, radius, np.pi/2, 0])
        
        if spacetime_name not in SPACETIMES:
            logger.error(f"Invalid spacetime: {spacetime_name}")
            return jsonify({'error': 'Invalid spacetime selected'}), 400
        
        spacetime = SPACETIMES[spacetime_name]
        metric = spacetime.metric_tensor(coordinates).tolist()
        
        logger.debug("Metric tensor calculated successfully")
        return jsonify({
            'status': 'success',
            'metric_tensor': metric,
            'coordinates': coordinates.tolist()
        })
    
    except Exception as e:
        logger.error(f"Error in metric_tensor route: {e}")
        return jsonify({'error': str(e)}), 500

logger.debug("Routes configured")

if __name__ == '__main__':
    logger.debug("Starting Flask development server")
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

logger.debug("Application initialization complete")