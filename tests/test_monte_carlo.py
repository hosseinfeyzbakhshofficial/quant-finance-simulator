import pytest
import numpy as np
from src.processes.monte_carlo import (
    MonteCarloSimulator,
    get_final_gbm_values,
)

def test_monte_carlo_simulation_shape():
    """
    STORY: Verify that the simulator returns a matrix containing all paths and steps.
    EXPECTED: Matrix shape must strictly equal (num_simulations, num_steps).
    """
    sim = MonteCarloSimulator(S0=100.0, mu=0.05, sigma=0.2, T=1.0, steps=100)
    paths = sim.generate_paths(num_simulations=50)
    
    assert isinstance(paths, np.ndarray)
    assert paths.shape == (50, 101)  # Includes initial spot price step

def test_monte_carlo_invalid_parameters():
    """
    STORY: Ensure the simulation engine restricts unphysical or mathematically unstable parameters.
    EXPECTED: Raises ValueError for negative stock prices, negative volatilities, or zero time horizons.
    """
    with pytest.raises(ValueError):
        MonteCarloSimulator(S0=-100.0, mu=0.05, sigma=0.2, T=1.0, steps=100)
        
    with pytest.raises(ValueError):
        MonteCarloSimulator(S0=100.0, mu=0.05, sigma=-0.2, T=1.0, steps=100)

def test_monte_carlo_reproducibility():
    """
    STORY: Check if setting a deterministic seed ensures exact reproducibility of random paths.
    EXPECTED: Two simulation runs utilizing identical seeds must produce matching matrices.
    """
    sim = MonteCarloSimulator(S0=100.0, mu=0.05, sigma=0.2, T=1.0, steps=50)
    paths_1 = sim.generate_paths(num_simulations=10, seed=42)
    paths_2 = sim.generate_paths(num_simulations=10, seed=42)
    
    np.testing.assert_array_almost_equal(paths_1, paths_2) 

def test_get_final_gbm_values():
    """
    STORY: Verify extraction of terminal GBM prices.
    """

    values = get_final_gbm_values(
        S0=100,
        mu=0.05,
        sigma=0.2,
        T=1,
        dt=0.01,
        n_simulations=5,
        seed=42,
    )

    assert isinstance(values, np.ndarray)

    assert len(values) == 5

def test_monte_carlo_edge_cases_and_errors():
    """
    STORY: Cover invalid T, invalid steps, and invalid num_simulations.
    """
    # Test negative or zero time horizon (Line 28)
    with pytest.raises(ValueError):
        MonteCarloSimulator(S0=100.0, mu=0.05, sigma=0.2, T=-1.0, steps=100)
    
    # Test negative number of steps (Line 30)
    with pytest.raises(ValueError):
        MonteCarloSimulator(S0=100.0, mu=0.05, sigma=0.2, T=1.0, steps=-5)

    # Test zero or negative number of simulations (Line 47)
    sim = MonteCarloSimulator(S0=100.0, mu=0.05, sigma=0.2, T=1.0, steps=100)
    with pytest.raises(ValueError):
        sim.generate_paths(num_simulations=0)