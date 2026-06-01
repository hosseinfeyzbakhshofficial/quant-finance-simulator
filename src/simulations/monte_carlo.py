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
    Vectorized Monte Carlo GBM simulation.

    Returns
    -------
    np.ndarray
        Matrix of shape:
        (n_simulations, n_steps + 1)
    """

    if seed is not None:
        np.random.seed(seed)

    # number of time steps
    steps = int(T / dt)

    # random shocks matrix
    shocks = np.random.normal(0, 1, size=(n_simulations, steps)) * np.sqrt(dt)

    # drift term
    drift = (mu - 0.5 * sigma**2) * dt

    # diffusion term
    diffusion = sigma * shocks

    # cumulative log returns
    log_returns = drift + diffusion

    cumulative_returns = np.cumsum(log_returns, axis=1)

    # GBM formula
    paths = S0 * np.exp(cumulative_returns)

    # add initial price column
    paths = np.column_stack([np.full(n_simulations, S0), paths])

    return paths
