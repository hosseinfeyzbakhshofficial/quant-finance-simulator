import time

import numpy as np
import pytest

from src.processes.gbm import simulate_gbm


def test_gbm_output_length():
    result = simulate_gbm(100, 0.1, 0.2, T=1, dt=0.01)
    expected_length = int(1 / 0.01) + 1
    assert len(result) == expected_length


def test_gbm_initial_value():
    S0 = 123
    result = simulate_gbm(S0, 0.1, 0.2, T=1, dt=0.01)
    assert result[0] == S0


def test_gbm_positive_values():
    result = simulate_gbm(100, 0.1, 0.2, T=1, dt=0.01)
    assert np.all(result > 0)


def test_gbm_reproducibility():
    r1 = simulate_gbm(100, 0.1, 0.2, T=1, dt=0.01, seed=42)
    r2 = simulate_gbm(100, 0.1, 0.2, T=1, dt=0.01, seed=42)
    assert np.allclose(r1, r2)


def test_gbm_invalid_input():
    with pytest.raises(ValueError):
        simulate_gbm(-100, 0.1, 0.2, T=1, dt=0.01)

    with pytest.raises(ValueError):
        simulate_gbm(100, 0.1, -0.2, T=1, dt=0.01)

    with pytest.raises(ValueError):
        simulate_gbm(100, 0.1, 0.2, T=-1, dt=0.01)

    with pytest.raises(ValueError):
        simulate_gbm(100, 0.1, 0.2, T=1, dt=-0.01)


def test_gbm_shape():
    result = simulate_gbm(100, 0.1, 0.2, T=1, dt=0.01)
    assert result.shape == (101,)


def test_gbm_no_explosion():
    result = simulate_gbm(100, 0.1, 0.2, T=5, dt=0.01)
    assert np.all(np.isfinite(result))


def test_gbm_variance_positive():
    sims = [simulate_gbm(100, 0.1, 0.2, 1, 0.01, seed=i)[-1] for i in range(20)]
    assert np.var(sims) > 0


def test_gbm_mean_growth():
    S0 = 100
    mu = 0.1
    sigma = 0.0

    result = simulate_gbm(S0, mu, sigma, T=1, dt=0.01, seed=42)

    expected = S0 * np.exp(mu * 1)

    assert np.isclose(result[-1], expected, rtol=0.01)


def test_gbm_convergence():
    paths = [simulate_gbm(100, 0.1, 0.2, 1, 0.01, seed=i)[-1] for i in range(100)]
    assert np.mean(paths) > 100


def test_gbm_speed():
    start = time.time()
    simulate_gbm(100, 0.1, 0.2, 1, 0.001)
    assert time.time() - start < 1
