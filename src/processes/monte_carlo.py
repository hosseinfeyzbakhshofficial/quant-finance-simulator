import numpy as np


def monte_carlo_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    dt: float,
    n_simulations: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Vectorized Monte Carlo Geometric Brownian Motion (GBM) simulation.

    Parameters
    ----------
    S0 : float
        Initial asset price.
    mu : float
        Expected return (drift).
    sigma : float
        Volatility.
    T : float
        Time to maturity in years.
    dt : float
        Time step size.
    n_simulations : int
        Number of simulated paths.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Matrix of shape (n_simulations, n_steps + 1) containing simulated paths.
    """
    if seed is not None:
        np.random.seed(seed)

    # Number of time steps
    steps = int(T / dt)

    # Random shocks matrix (Vectorized over all simulations and steps)
    shocks = np.random.normal(0, 1, size=(n_simulations, steps)) * np.sqrt(dt)

    # Drift and diffusion components
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * shocks

    # Cumulative log returns
    log_returns = drift + diffusion
    cumulative_returns = np.cumsum(log_returns, axis=1)

    # GBM simulation paths
    paths = S0 * np.exp(cumulative_returns)

    # Prepend the initial asset price to each simulation path
    paths = np.column_stack([np.full(n_simulations, S0), paths])

    return paths


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
    Simulates GBM paths and returns only the final asset price for each path.
    Optimized to eliminate loops and prevent redundant memory allocation.
    """
    paths = monte_carlo_gbm(S0, mu, sigma, T, dt, n_simulations, seed)
    return paths[:, -1]
