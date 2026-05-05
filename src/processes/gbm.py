# import
import numpy as np
from src.utils.logger import setup_logger

import logging

logger = logging.getLogger(__name__)
# We use NumPy for: random numbers, arrays,  math functions

# Function definition
# Parameters: S0 Initial value (e.g., 100k € startup value), mu Drift (average growth rate), 
# sigma Volatility (risk, randomness), T Total time (e.g., 1 year), dt Time step (e.g., 1/365 for daily), 
# seed Fix randomness (VERY important for testing)
# This code """  Simulate a Geometric Brownian Motion (GBM) path.  ...  """
# This is documentation inside the code 
# Why it matters: Users understand how to use the function, Professor gives points for this, Tools can auto-generate docs from this
def simulate_gbm(S0, mu, sigma, T, dt, seed=None):
    """
    Simulate a Geometric Brownian Motion (GBM) path.

    Parameters
    ----------
    S0 : float
        Initial value
    mu : float
        Drift (expected return)
    sigma : float
        Volatility (risk)
    T : float
        Total time (years)
    dt : float
        Time step
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Simulated GBM path
    """

    logger.info("Starting GBM simulation")

    #Input validation
    if S0 <= 0:
        raise ValueError("S0 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if T <= 0 or dt <= 0:
        raise ValueError("T and dt must be positive")
    
    logger.info(f"S0={S0}, mu={mu}, sigma={sigma}, T={T}, dt={dt}")
    # For S0 Why? GBM assumes positive values (log-normal process)   If S0 ≤ 0: math breaks, log undefined
    # For sigma Volatility = standard deviation   Cannot be negative (physics + statistics)
    # For T Time must move forward        Negative or zero time step = nonsense physically
    # Random seed
    if seed is not None:
        np.random.seed(seed)
    # Why? Makes simulation reproducible 
    # Without seed: every run → different result, With seed: same input → same output
    # This is CRITICAL for: testing, debugging, scientific reproducibility
    # Number of steps
    try:
        steps = int(T / dt)
    except Exception as e:
        logger.error(f"Error computing steps: {e}")
        raise
    logger.debug(f"Generated {steps} steps")
    # Example: T = 1 year dt = 1/365  → steps = 365  Converts continuous time → discrete simulation 
    # Memory allocation
    prices = np.empty(steps + 1)
    prices[0] = S0
    # Why empty instead of zeros? empty = faster (does NOT initialize values)
    # In this code prices[0] = S0 You must initialize manually
    # Performance: zeros → safer but slower   empty → faster but requires care
    # Instead of You were doing randomness with loop and one by one Now: You generate all randomness at once
    # Physics meaning: This is your discrete version of: dWt ∼ N(0,dt)
    # Vectorized random shocks
    shocks = np.random.normal(0, 1, size=steps) * np.sqrt(dt)
    logger.debug(f"First shock: {shocks[0]}")

    # Simulation loop
    try:
        for t in range(1, steps + 1):
            prices[t] = prices[t - 1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * shocks[t - 1]
            )
    except Exception as e:
        logger.exception("Simulation failed")
        raise

    
    logger.info("Simulation completed")
    # Return
    return prices 
# engine that generates the step-by-step evolution of the system over time
# The loop iterates through each time step (from 1 to steps). In each iteration, 
# it calculates the next value of the system (prices[t]) based on the previous value (prices[t-1]) and a random "shock."
# Deterministic drift term This is NOT just drift The -0.5 σ² term comes from: Itô correction
# Physical interpretation: Because randomness is multiplicative: mean behavior shifts
# Stochastic term sigma * shocks[t - 1] This is the random shock that adds variability to the path. shocks ~ N(0, dt), scaled by volatility
# Why exponential term? GBM is a multiplicative process: S(t) = S(0) * exp(...) This ensures positivity and captures compounding effects
# Output: type: np.ndarray  shape: (steps + 1,)