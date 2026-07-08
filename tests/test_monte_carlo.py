import pytest
import numpy as np
from src.processes.monte_carlo import MonteCarloSimulator

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
