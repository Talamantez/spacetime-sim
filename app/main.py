import os
from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
from scipy.integrate import odeint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, writers
import time
import logging
import glob


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__,
           template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates')),
           static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static')))


# Log the template folder path
logger.debug(f"Template folder path: {app.template_folder}")
logger.debug(f"Current working directory: {os.getcwd()}")
logger.debug(f"Directory contents: {os.listdir('.')}")
if os.path.exists('templates'):
    logger.debug(f"Templates directory contents: {os.listdir('templates')}")


# Configure for Railway
PORT = os.environ.get('PORT', 5000)
HOST = '0.0.0.0'

# Ensure upload directory exists
app.config['UPLOAD_FOLDER'] = 'static/outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create an empty .gitkeep file in the outputs directory
with open(os.path.join(app.config['UPLOAD_FOLDER'], '.gitkeep'), 'w') as f:
    pass
class SpacetimeSimulator:
    def __init__(self, name, metric_tensor_func, dimensions=4):
        self.name = name
        self.metric_tensor_func = metric_tensor_func
        self.dimensions = dimensions
        self.dt = 0.1

    def metric_tensor(self, coordinates):
        return self.metric_tensor_func(coordinates)

    def christoffel_symbols(self, coordinates):
        h = 1e-5
        gamma = np.zeros((self.dimensions, self.dimensions, self.dimensions))
        try:
            for mu in range(self.dimensions):
                for nu in range(self.dimensions):
                    for sigma in range(self.dimensions):
                        dg_mu = (self.metric_tensor(coordinates + h*np.eye(self.dimensions)[mu])
                                 - self.metric_tensor(coordinates - h*np.eye(self.dimensions)[mu])) / (2*h)
                        dg_nu = (self.metric_tensor(coordinates + h*np.eye(self.dimensions)[nu])
                                 - self.metric_tensor(coordinates - h*np.eye(self.dimensions)[nu])) / (2*h)
                        dg_sigma = (self.metric_tensor(coordinates + h*np.eye(self.dimensions)[sigma])
                                    - self.metric_tensor(coordinates - h*np.eye(self.dimensions)[sigma])) / (2*h)
                        g = self.metric_tensor(coordinates)
                        g_inv = np.linalg.inv(g)
                        for rho in range(self.dimensions):
                            gamma[mu][nu][sigma] += 0.5 * g_inv[mu][rho] * (dg_nu[rho][sigma] + dg_sigma[rho][nu] - dg_sigma[nu][rho])
        except np.linalg.LinAlgError:
            pass
        return gamma

    def geodesic_equation(self, state, t):
        x = state[:self.dimensions]
        v = state[self.dimensions:]
        dxdt = v
        dvdt = -np.einsum('ijk,j,k->i', self.christoffel_symbols(x), v, v)
        return np.concatenate([dxdt, dvdt])

    def simulate_geodesic(self, initial_position, initial_velocity, num_steps):
        initial_state = np.concatenate([initial_position, initial_velocity])
        t = np.linspace(0, num_steps * self.dt, num_steps)
        solution = odeint(self.geodesic_equation, initial_state, t)
        return solution

    def generate_simulation_plot(self, initial_positions, initial_velocities, num_steps):
        # Use timestamp for unique filename
        timestamp = int(time.time())
        output_file = f"static/outputs/{self.name}_{timestamp}.mp4"
        output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), output_file)
        
        fig = plt.figure(figsize=(15, 10))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)

        geodesics = [self.simulate_geodesic(pos, vel, num_steps) 
                    for pos, vel in zip(initial_positions, initial_velocities)]

        def update(frame):
            ax1.clear()
            ax1.set_xlim(-10, 10)
            ax1.set_ylim(-10, 10)
            ax1.set_zlim(-10, 10)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.set_title(f'{self.name} Spacetime (t={frame*self.dt:.2f})')

            for geodesic in geodesics:
                ax1.plot(geodesic[:frame, 1], geodesic[:frame, 2], geodesic[:frame, 3])
                ax1.scatter(geodesic[frame, 1], geodesic[frame, 2], geodesic[frame, 3], s=50)

            ax2.clear()
            ax2.set_title('Metric Tensor Components')
            coords = geodesics[0][frame, :4]
            metric = self.metric_tensor(coords)
            im = ax2.imshow(metric, cmap='coolwarm', interpolation='nearest')
            plt.colorbar(im, ax=ax2)

        anim = FuncAnimation(fig, update, frames=num_steps, interval=50, blit=False)
        writer = writers['ffmpeg'](fps=15, metadata=dict(artist='SpacetimeSimulator'), bitrate=1800)
        
        try:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the animation
            anim.save(output_path, writer=writer)
            plt.close()
            
            # Return the relative path for the web server
            return output_file
            
        except Exception as e:
            logger.error(f"Error generating simulation: {str(e)}")
            plt.close()
            raise e


def create_preset_spacetimes():
    def schwarzschild_metric(coordinates):
        t, x, y, z = coordinates[:4]
        r = np.sqrt(x**2 + y**2 + z**2)
        Rs = 2  # Schwarzschild radius
        factor = 1 - Rs / (r + Rs)  # Avoid division by zero
        return np.diag([-factor, 1/factor, r**2, r**2 * np.sin(np.arccos(z/(r+1e-10)))**2])

    def ads_metric(coordinates):
        t, x, y, z = coordinates[:4]
        L = 1  # AdS radius
        r2 = x**2 + y**2 + z**2
        factor = L**2 / (L**2 + r2)
        return np.diag([-1/factor, factor, factor, factor])

    def godel_metric(coordinates):
        t, x, y, z = coordinates[:4]
        a = 1  # Gödel parameter
        g00 = -1
        g11 = g22 = 1
        g33 = -1/2 * np.exp(2*a*x)
        g03 = g30 = np.exp(a*x)
        metric = np.diag([g00, g11, g22, g33])
        metric[0, 3] = metric[3, 0] = g03
        return metric

    return {
        "Schwarzschild": SpacetimeSimulator("Schwarzschild", schwarzschild_metric),
        "Anti-de Sitter": SpacetimeSimulator("Anti-de Sitter", ads_metric),
        "Gödel": SpacetimeSimulator("Gödel", godel_metric)
    }

# Global variable to store spacetimes
SPACETIMES = create_preset_spacetimes()


@app.route('/')
def index():
    logger.debug("Handling index route")
    template_list = app.jinja_loader.list_templates()
    logger.debug(f"Available templates: {template_list}")
    return render_template('index.html', spacetimes=SPACETIMES.keys())

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        spacetime_name = request.form['spacetime']
        num_steps = int(request.form.get('num_steps', 200))
        
        # Get initial conditions from form
        pos_x = float(request.form.get('pos_x', 5))
        pos_y = float(request.form.get('pos_y', 0))
        pos_z = float(request.form.get('pos_z', 0))
        
        vel_x = float(request.form.get('vel_x', 0))
        vel_y = float(request.form.get('vel_y', 0.5))
        vel_z = float(request.form.get('vel_z', 0))
        
        if spacetime_name not in SPACETIMES:
            return jsonify({'error': 'Invalid spacetime selected'}), 400
        
        spacetime = SPACETIMES[spacetime_name]
        
        # Use the form values for initial conditions
        initial_positions = [
            np.array([0, pos_x, pos_y, pos_z])
        ]
        initial_velocities = [
            np.array([1, vel_x, vel_y, vel_z])
        ]
        
        output_file = spacetime.generate_simulation_plot(
            initial_positions, initial_velocities, num_steps)
        
        # Get current metric tensor for visualization
        metric = spacetime.metric_tensor(initial_positions[0]).tolist()
        
        return jsonify({
            'status': 'success',
            'video_path': '/' + output_file,
            'metric_tensor': metric
        })
    
    except Exception as e:
        logger.error(f"Error in simulate route: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/custom_spacetime', methods=['POST'])
def create_custom_spacetime():
    try:
        name = request.form['name']
        metric_func_str = request.form['metric_function']
        
        # Create the metric function from the string
        metric_func = eval(f"lambda coordinates: {metric_func_str}")
        
        # Create and store the new spacetime
        SPACETIMES[name] = SpacetimeSimulator(name, metric_func)
        
        return jsonify({
            'status': 'success',
            'message': f'Created new spacetime: {name}'
        })
    
    except Exception as e:
        logger.error(f"Error creating custom spacetime: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup_old_files():
    """Clean up simulation files older than 1 hour"""
    try:
        # Get all mp4 files in the outputs directory
        output_dir = os.path.join(app.static_folder, 'outputs')
        files = glob.glob(os.path.join(output_dir, '*.mp4'))
        
        # Current time
        now = datetime.now()
        
        # Delete files older than 1 hour
        for f in files:
            file_time = datetime.fromtimestamp(os.path.getctime(f))
            if now - file_time > timedelta(hours=1):
                try:
                    os.remove(f)
                    logger.debug(f"Deleted old file: {f}")
                except OSError as e:
                    logger.error(f"Error deleting {f}: {e}")
        
        return jsonify({'status': 'success', 'message': 'Cleanup completed'})
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')