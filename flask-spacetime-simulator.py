import os
from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
from scipy.integrate import odeint
import matplotlib
matplotlib.use('Agg')  # Required for headless environments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, writers
from datetime import datetime

app = Flask(__name__)

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{app.config['UPLOAD_FOLDER']}/{self.name}_{timestamp}.mp4"
        
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
        anim.save(output_file, writer=writer)
        plt.close()
        
        return output_file

def create_preset_spacetimes():
    def schwarzschild_metric(coordinates):
        t, x, y, z = coordinates[:4]
        r = np.sqrt(x**2 + y**2 + z**2)
        Rs = 2  # Schwarzschild radius
        factor = 1 - Rs / (r + Rs)
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
    return render_template('index.html', spacetimes=SPACETIMES.keys())

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        spacetime_name = request.form['spacetime']
        num_steps = int(request.form.get('num_steps', 200))
        
        if spacetime_name not in SPACETIMES:
            return jsonify({'error': 'Invalid spacetime selected'}), 400
        
        spacetime = SPACETIMES[spacetime_name]
        
        # Default initial conditions
        initial_positions = [
            np.array([0, 5, 0, 0]),
            np.array([0, -5, 0, 0]),
            np.array([0, 0, 5, 0])
        ]
        initial_velocities = [
            np.array([1, 0, 0.5, 0]),
            np.array([1, 0, -0.5, 0]),
            np.array([1, -0.5, 0, 0])
        ]
        
        output_file = spacetime.generate_simulation_plot(
            initial_positions, initial_velocities, num_steps)
        
        return jsonify({
            'status': 'success',
            'video_path': output_file
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/custom_spacetime', methods=['POST'])
def create_custom_spacetime():
    try:
        name = request.form['name']
        metric_func_str = request.form['metric_function']
        
        # Create the metric function from the string
        # Note: This is potentially dangerous in a production environment
        metric_func = eval(f"lambda coordinates: {metric_func_str}")
        
        # Create and store the new spacetime
        SPACETIMES[name] = SpacetimeSimulator(name, metric_func)
        
        return jsonify({
            'status': 'success',
            'message': f'Created new spacetime: {name}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)