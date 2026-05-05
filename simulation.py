import numpy as np
import matplotlib.pyplot as plt

def generate_chirikov_plot(K=0.5):
    print(f"Simulating Chirikov Standard Map with K = {K}...")
    n_particles = 100
    n_iterations = 1000

    # Generate random starting positions
    theta = np.random.uniform(0, 2 * np.pi, n_particles)
    p = np.random.uniform(0, 2 * np.pi, n_particles)

    theta_all = []
    p_all = []

    # Iterate the map equations
    for _ in range(n_iterations):
        p = np.mod(p + K * np.sin(theta), 2 * np.pi)
        theta = np.mod(theta + p, 2 * np.pi)
        
        theta_all.extend(theta)
        p_all.extend(p)

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.scatter(theta_all, p_all, s=0.2, c='midnightblue', marker='.')
    plt.title(f"Magnetic Topology (Chirikov Map, K = {K})", fontsize=14)
    plt.xlabel("Poloidal Angle (θ)", fontsize=12)
    plt.ylabel("Radial Flux (p)", fontsize=12)
    plt.xlim(0, 2 * np.pi)
    plt.ylim(0, 2 * np.pi)
    plt.grid(True, alpha=0.3)
    
    # Save the file automatically
    filename = f"tokamak_simulation_K{str(K).replace('.', '')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved as {filename}")

# Run the simulation for the three different states
generate_chirikov_plot(0.5)   # Tearing Mode / Islands
generate_chirikov_plot(0.97)  # Chirikov Overlap Threshold
generate_chirikov_plot(1.5)   # Global Stochasticity (Disruption)
