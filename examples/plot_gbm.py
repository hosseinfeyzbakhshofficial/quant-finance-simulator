import matplotlib.pyplot as plt
from src.processes.gbm import simulate_gbm

def run_example():
    """
    Generate and plot multiple Geometric Brownian Motion (GBM) simulation paths.
    """
    # Parameters for the simulation
    s0 = 100
    mu = 0.1
    sigma = 0.2
    t_horizon = 1
    dt_step = 0.01
    num_paths = 5

    plt.figure(figsize=(10, 6))

    # Generate multiple independent paths using different seeds
    for i in range(num_paths):
        path = simulate_gbm(S0=s0, mu=mu, sigma=sigma, T=t_horizon, dt=dt_step, seed=i)
        plt.plot(path, label=f"Path {i+1}")

    plt.title("Geometric Brownian Motion (GBM) - Simulated Paths", fontsize=14)
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Asset Price", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_example()