import numpy as np

from src.processes.gbm import simulate_gbm


def monte_carlo_gbm(S0, mu, sigma, T, dt, n_simulations):
    results = []

    for _ in range(n_simulations):
        path = simulate_gbm(S0, mu, sigma, T, dt)
        results.append(path[-1])  # final value

    return np.array(results)
