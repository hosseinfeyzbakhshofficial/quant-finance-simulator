import logging
import numpy as np

logger = logging.getLogger(__name__)


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    dt: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate a single Geometric Brownian Motion (GBM) path using vectorization.

    Parameters
    ----------
    S0 : float
        Initial asset price (must be positive).
    mu : float
        Expected return (drift rate).
    sigma : float
        Volatility (must be non-negative).
    T : float
        Total time horizon in years (must be positive).
    dt : float
        Time step size (must be positive).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Simulated GBM price path array of shape (steps + 1,).
    """
    logger.debug("Starting single-path GBM simulation")

    # Input validation based on financial and physical constraints
    if S0 <= 0:
        raise ValueError("Initial price S0 must be strictly positive.")
    if sigma < 0:
        raise ValueError("Volatility sigma cannot be negative.")
    if T <= 0 or dt <= 0:
        raise ValueError("Total time T and time step dt must be positive.")

    if seed is not None:
        np.random.seed(seed)

    steps = int(T / dt)
    logger.info(f"Simulating GBM: S0={S0}, mu={mu}, sigma={sigma}, steps={steps}")

    # Vectorized generation of random shocks (Wiener process increments: dW ~ N(0, dt))
    shocks = np.random.normal(0, 1, size=steps) * np.sqrt(dt)

    # Compute drift (with Itô calculus correction) and diffusion components
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * shocks

    # Cumulative sum of log returns to eliminate the sequential Python loop
    log_returns = drift + diffusion
    cumulative_returns = np.cumsum(log_returns)

    # Reconstruct the price path and prepend the initial asset price
    prices = S0 * np.exp(cumulative_returns)
    prices = np.insert(prices, 0, S0)

    logger.debug("GBM simulation completed successfully")
    return prices