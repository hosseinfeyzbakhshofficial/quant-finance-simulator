import pytest
import numpy as np
import matplotlib
# Use non-interactive backend to prevent GUI windows from appearing during testing
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.analysis.visualization import plot_simulation_paths, plot_price_histogram

def test_plot_simulation_paths_returns_figure():
    """
    STORY: Verify that the path visualization module creates a valid plot.
    EXPECTED: The function must return a matplotlib Figure instance and close cleanly.
    """
    dummy_paths = np.random.normal(loc=100, scale=5, size=(10, 50))
    fig = plot_simulation_paths(dummy_paths)
    
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_price_histogram_returns_figure():
    """
    STORY: Validate histogram distribution plots for terminal option asset prices.
    EXPECTED: Generates a valid Figure without throwing matrix or dimension errors.
    """
    dummy_terminal_prices = np.random.normal(loc=105, scale=10, size=500)
    fig = plot_price_histogram(dummy_terminal_prices, bins=30)
    
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
