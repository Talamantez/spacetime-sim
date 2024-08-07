import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, writers
from scipy.integrate import odeint
import cv2
import os
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox


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
            # Return zero Christoffel symbols if we encounter a singular matrix
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

    def run_simulation(self, initial_positions, initial_velocities, num_steps, output_file):
        fig = plt.figure(figsize=(15, 10))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)

        geodesics = [self.simulate_geodesic(pos, vel, num_steps) for pos, vel in zip(initial_positions, initial_velocities)]

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


class SpacetimeDefinitionGUI:
    def __init__(self, master, callback):
        self.master = master
        self.callback = callback
        master.title("Define New Spacetime")

        self.name_label = ttk.Label(master, text="Spacetime Name:")
        self.name_entry = ttk.Entry(master)

        self.metric_label = ttk.Label(master, text="Metric Tensor Function:")
        self.metric_text = tk.Text(master, height=10, width=50)

        self.submit_button = ttk.Button(master, text="Create Spacetime", command=self.create_spacetime)

        self.name_label.pack()
        self.name_entry.pack()
        self.metric_label.pack()
        self.metric_text.pack()
        self.submit_button.pack()

    def create_spacetime(self):
        name = self.name_entry.get()
        if not name:
            messagebox.showerror("Error", "Please enter a name for the spacetime.")
            return
        metric_func_str = self.metric_text.get("1.0", tk.END).strip()
        if not metric_func_str:
            messagebox.showerror("Error", "Please enter a metric tensor function.")
            return
        try:
            metric_func = eval(f"lambda coordinates: {metric_func_str}")
            new_spacetime = SpacetimeSimulator(name, metric_func)
            self.callback(new_spacetime)
            self.master.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Invalid metric tensor function: {str(e)}")

class ExperimentSetupGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Spacetime Experiment Setup")
        self.spacetimes = []
        self.selected_spacetimes = []

        self.create_widgets()

    def create_widgets(self):
        self.spacetime_frame = ttk.Frame(self.master)
        self.spacetime_frame.pack(pady=10)

        self.create_spacetime_button = ttk.Button(self.spacetime_frame, text="Create New Spacetime", command=self.open_spacetime_creator)
        self.create_spacetime_button.pack(side=tk.LEFT, padx=5)

        self.add_preset_button = ttk.Button(self.spacetime_frame, text="Add Preset Spacetime", command=self.add_preset_spacetime)
        self.add_preset_button.pack(side=tk.LEFT, padx=5)

        self.spacetime_listbox = tk.Listbox(self.master, selectmode=tk.MULTIPLE, width=50)
        self.spacetime_listbox.pack(pady=10)

        self.run_button = ttk.Button(self.master, text="Run Simulation", command=self.run_simulation)
        self.run_button.pack(pady=10)

    def open_spacetime_creator(self):
        creator_window = tk.Toplevel(self.master)
        SpacetimeDefinitionGUI(creator_window, self.add_spacetime)

    def add_spacetime(self, spacetime):
        self.spacetimes.append(spacetime)
        self.spacetime_listbox.insert(tk.END, spacetime.name)

    def add_preset_spacetime(self):
        presets = {
            "Schwarzschild": lambda coordinates: np.diag([-1, 1/(1 - 2/max(coordinates[1], 2)), coordinates[1]**2, coordinates[1]**2 * np.sin(coordinates[2])**2]),
            "Anti-de Sitter": lambda coordinates: np.diag([-1, 1, np.cosh(coordinates[1])**2, np.cosh(coordinates[1])**2 * np.sin(coordinates[2])**2]),
            "Gödel": lambda coordinates: np.array([[-1, 0, coordinates[1], 0], [0, 1, 0, 0], [coordinates[1], 0, coordinates[1]**2 - 1, 0], [0, 0, 0, 1]])
        }
        
        preset_window = tk.Toplevel(self.master)
        preset_window.title("Add Preset Spacetime")
        
        for name in presets:
            ttk.Button(preset_window, text=name, command=lambda n=name: self.add_spacetime(SpacetimeSimulator(n, presets[n]))).pack()

    def run_simulation(self):
        selected_indices = self.spacetime_listbox.curselection()
        self.selected_spacetimes = [self.spacetimes[i] for i in selected_indices]
        if not self.selected_spacetimes:
            messagebox.showwarning("No Selection", "Please select at least one spacetime for the simulation.")
            return
        self.master.quit()
def create_spacetimes():
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

    return [
        SpacetimeSimulator("Schwarzschild", schwarzschild_metric),
        SpacetimeSimulator("Anti-de Sitter", ads_metric),
        SpacetimeSimulator("Gödel", godel_metric)
    ]

def analyze_geodesics(video_files):
    def extract_trajectories(video_file):
        cap = cv2.VideoCapture(video_file)
        trajectories = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            centers = [cv2.moments(c) for c in contours]
            centers = [(int(m['m10']/m['m00']), int(m['m01']/m['m00'])) for m in centers if m['m00'] != 0]
            trajectories.append(centers)
        cap.release()
        return trajectories

    all_trajectories = [extract_trajectories(video) for video in video_files]
    
    # Ensure all trajectories have the same number of particles
    min_particles = min(min(len(frame) for frame in traj) for traj in all_trajectories)
    all_trajectories = [[frame[:min_particles] for frame in traj] for traj in all_trajectories]
    
    # Convert to numpy arrays for easier manipulation
    all_trajectories = [np.array(traj) for traj in all_trajectories]
    
    # Compute differences between trajectories
    differences = []
    for i in range(len(all_trajectories)):
        for j in range(i+1, len(all_trajectories)):
            # Compute MSE across all frames and particles
            mse = np.mean((all_trajectories[i] - all_trajectories[j])**2)
            differences.append((video_files[i], video_files[j], mse))
    
    return differences

def create_summary_video(video_files, differences, output_file):
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    plt.subplots_adjust(hspace=0.4)

    def update(frame):
        for ax in axs.flatten():
            ax.clear()

        for i, video_file in enumerate(video_files):
            cap = cv2.VideoCapture(video_file)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()
            if ret:
                axs[i//2, i%2].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axs[i//2, i%2].set_title(os.path.basename(video_file))
            cap.release()

        axs[1, 1].axis('off')
        axs[1, 1].text(0.1, 0.9, "Geodesic Differences (MSE):", fontsize=12, fontweight='bold')
        for i, (v1, v2, diff) in enumerate(differences):
            axs[1, 1].text(0.1, 0.8-i*0.1, f"{os.path.basename(v1)} vs {os.path.basename(v2)}: {diff:.2f}", fontsize=10)

    anim = FuncAnimation(fig, update, frames=200, interval=50)
    writer = writers['ffmpeg'](fps=15, metadata=dict(artist='SpacetimeSimulator'), bitrate=1800)
    anim.save(output_file, writer=writer)

def main():
    root = tk.Tk()
    experiment_setup = ExperimentSetupGUI(root)
    root.mainloop()

    if not experiment_setup.selected_spacetimes:
        print("No spacetimes selected. Exiting.")
        return

    initial_positions = [np.array([0, 5, 0, 0]), np.array([0, -5, 0, 0]), np.array([0, 0, 5, 0])]
    initial_velocities = [np.array([1, 0, 0.5, 0]), np.array([1, 0, -0.5, 0]), np.array([1, -0.5, 0, 0])]
    
    video_files = []
    for spacetime in experiment_setup.selected_spacetimes:
        output_file = f"{spacetime.name}_simulation.mp4"
        spacetime.run_simulation(initial_positions, initial_velocities, 200, output_file)
        video_files.append(output_file)
        print(f"Generated video for {spacetime.name}")

    differences = analyze_geodesics(video_files)
    create_summary_video(video_files, differences, "spacetime_comparison.mp4")
    print("Generated summary video: spacetime_comparison.mp4")

if __name__ == "__main__":
    main()