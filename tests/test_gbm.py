import time
import numpy as np
import pytest
from src.processes.gbm import simulate_gbm

def test_gbm_output_length():
    """
    STORY: Verify the discretization of the time grid in Geometric Brownian Motion.
    EXPECTED: For T=1 and dt=0.01, the output array must contain exactly (T/dt) + 1 points.
    """
    result = simulate_gbm(100, 0.1, 0.2, T=1, dt=0.01)
    expected_length = int(1 / 0.01) + 1
    assert len(result) == expected_length

def test_gbm_initial_value():
    """
    STORY: Check that the simulation path correctly originates from the given spot price.
    EXPECTED: The first element of the simulated array must exactly equal S0.
    """
    S0 = 123
    result = simulate_gbm(S0, 0.1, 0.2, T=1, dt=0.01)
    assert result[0] == S0

def test_gbm_positive_values():
    """
    STORY: Validate the theoretical non-negativity property of Geometric Brownian Motion.
    EXPECTED: Due to the exponential nature of GBM, all simulated prices must be strictly greater than zero.
    """
    result = simulate_gbm(100, 0.1, 0.2, T=1, dt=0.01)
    assert np.all(result > 0)

def test_gbm_reproducibility():
    """
    STORY: Ensure that providing a fixed pseudo-random number generator seed yields identical paths.
    EXPECTED: Two separate runs with the same seed must produce identical numpy arrays.
    """
    r1 = simulate_gbm(100, 0.1, 0.2, T=1, dt=0.01, seed=42)
    r2 = simulate_gbm(100, 0.1, 0.2, T=1, dt=0.01, seed=42)
    assert np.allclose(r1, r2)

def test_gbm_invalid_input():
    """
    STORY: Test input validation and exception handling for unphysical parameters (Edge Cases).
    EXPECTED: Negative prices, negative volatilities, or invalid time steps must raise a ValueError.
    """
    with pytest.raises(ValueError):
        simulate_gbm(-100, 0.1, 0.2, T=1, dt=0.01)

    with pytest.raises(ValueError):
        simulate_gbm(100, 0.1, -0.2, T=1, dt=0.01)

    with pytest.raises(ValueError):
        simulate_gbm(100, 0.1, 0.2, T=-1, dt=0.01)

    with pytest.raises(ValueError):
        simulate_gbm(100, 0.1, 0.2, T=1, dt=-0.01)

def test_gbm_shape():
    """
    STORY: Ensure the output is a 1D NumPy array with the correct shape dimension.
    """
    result = simulate_gbm(100, 0.1, 0.2, T=1, dt=0.01)
    assert result.shape == (101,)

def test_gbm_no_explosion():
    """
    STORY: Verify stability over long time horizons.
    EXPECTED: The path values must remain finite numbers and not overflow/explode to infinity.
    """
    result = simulate_gbm(100, 0.1, 0.2, T=5, dt=0.01)
    assert np.all(np.isfinite(result))

def test_gbm_variance_positive():
    """
    STORY: Verify stochastic behavior across multiple independent simulations.
    EXPECTED: Different seeds must generate different terminal prices, yielding a positive variance.
    """
    sims = [simulate_gbm(100, 0.1, 0.2, 1, 0.01, seed=i)[-1] for i in range(20)]
    assert np.var(sims) > 0

def test_gbm_mean_growth():
    """
    STORY: Verify the deterministic limit of the GBM process when volatility is set to zero.
    EXPECTED: With sigma=0, the asset must grow deterministically at the rate of mu: S_T = S_0 * e^(mu * T).
    """
    S0 = 100
    mu = 0.1
    sigma = 0.0
    result = simulate_gbm(S0, mu, sigma, T=1, dt=0.01, seed=42)
    expected = S0 * np.exp(mu * 1)
    assert np.isclose(result[-1], expected, rtol=0.01)

def test_gbm_convergence():
    """
    STORY: Test that the ensemble average of the paths trends in accordance with positive drift.
    """
    paths = [simulate_gbm(100, 0.1, 0.2, 1, 0.01, seed=i)[-1] for i in range(100)]
    assert np.mean(paths) > 100

def test_gbm_speed():
    """
    STORY: Performance test to ensure the simulation algorithm is optimized.
    EXPECTED: Simulating a high-density path (1000 steps) must execute in under 1.0 second.
    """
    start = time.time()
    simulate_gbm(100, 0.1, 0.2, 1, 0.001)
    assert time.time() - start < 1