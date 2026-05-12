import numpy as np
from src.processes.gbm import simulate_gbm


def monte_carlo_gbm(
    S0,
    mu,
    sigma,
    T,
    dt,
    n_simulations,
):
    """
    Run multiple GBM simulations.

    Parameters
    ----------
    S0 : float
        Initial value
    mu : float
        Drift
    sigma : float
        Volatility
    T : float
        Total time
    dt : float
        Time step
    n_simulations : int
        Number of Monte Carlo paths

    Returns
    -------
    np.ndarray
        Matrix of simulated paths
        shape = (n_simulations, steps + 1)
    """

    all_paths = []

    for i in range(n_simulations):

        path = simulate_gbm(
            S0,
            mu,
            sigma,
            T,
            dt,
        )

        all_paths.append(path)

    return np.array(all_paths)