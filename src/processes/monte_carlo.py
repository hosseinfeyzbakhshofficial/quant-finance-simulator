"""
Vectorized Monte Carlo simulation of Geometric Brownian Motion asset paths.

This module exposes a single canonical implementation, MonteCarloSimulator,
used both for generating full simulated price-path matrices and for
computing terminal asset values only.

get_final_gbm_values is a thin convenience wrapper around the class for
callers that only need terminal prices and do not want to materialize the
full path matrix. It exists for API convenience, not as a second
implementation: it delegates all path generation to MonteCarloSimulator.
"""

from __future__ import annotations

import numpy as np


class MonteCarloSimulator:
    """Vectorized Monte Carlo engine for Geometric Brownian Motion paths."""

    def __init__(self, S0: float, mu: float, sigma: float, T: float, steps: int):
        if S0 <= 0:
            raise ValueError("S0 must be strictly positive.")
        if sigma < 0:
            raise ValueError("sigma cannot be negative.")
        if T <= 0:
            raise ValueError("T must be strictly positive.")
        if steps <= 0:
            raise ValueError("steps must be a strictly positive integer.")

        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.steps = steps
        self.dt = T / steps

    def generate_paths(self, num_simulations: int, seed: int | None = None) -> np.ndarray:
        """
        Generate simulated GBM price paths using the exact log-normal solution.

        Returns an array of shape (num_simulations, steps + 1), where column 0
        is the initial price S0 for every simulated path.
        """
        if num_simulations <= 0:
            raise ValueError("num_simulations must be a strictly positive integer.")

        rng = np.random.default_rng(seed)
        z = rng.standard_normal(size=(num_simulations, self.steps))

        drift = (self.mu - 0.5 * self.sigma ** 2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * z
        log_returns = drift + diffusion

        # Vectorized cumulative sum along the time axis (no explicit Python loop).
        log_paths = np.cumsum(log_returns, axis=1)
        paths = self.S0 * np.exp(log_paths)

        initial_column = np.full((num_simulations, 1), self.S0)
        return np.hstack([initial_column, paths])


def get_final_gbm_values(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    dt: float,
    n_simulations: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Return only the terminal prices of a Monte Carlo simulation.

    Convenience wrapper for callers that do not need the intermediate path
    values. Internally delegates to MonteCarloSimulator so that GBM path
    generation has exactly one implementation in the codebase.
    """
    steps = max(1, round(T / dt))
    simulator = MonteCarloSimulator(S0=S0, mu=mu, sigma=sigma, T=T, steps=steps)
    paths = simulator.generate_paths(num_simulations=n_simulations, seed=seed)
    return paths[:, -1]