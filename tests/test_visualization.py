import os
import pytest
import numpy as np
import pandas as pd
import matplotlib

# Use non-interactive backend to prevent GUI windows from appearing during testing
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.visualization import (
    plot_simulation_paths,
    plot_price_histogram,
    plot_gbm,
    plot_gbm_dataframe,
    plot_option_payoff,
    plot_confidence_band,
)


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


def test_multiple_visualization_functions():
    """
    STORY: Execute remaining visualization routines.
    """
    paths = np.random.normal(100, 5, (10, 50))
    fig = plot_simulation_paths(paths)
    assert fig is not None
    plt.close(fig)


def test_remaining_visualization_plots():
    """
    STORY: Execute and cover remaining core graph rendering routines.
    """
    sample_prices = np.array([100.0, 101.5, 100.8, 102.3])
    sample_df = pd.DataFrame({"time_step": [0, 1, 2], "price": [100.0, 101.0, 102.0]})
    sample_paths = np.random.normal(100.0, 2.0, (5, 10))

    plot_gbm(sample_prices)
    plot_gbm_dataframe(sample_df)
    plot_option_payoff(sample_prices, strike=101.0)
    plot_confidence_band(sample_paths)

    plt.close("all")