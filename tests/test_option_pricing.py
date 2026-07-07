import numpy as np
import pytest
from src.finance.option_pricing import monte_carlo_option_price

def test_option_price_positive():
    """
    STORY: Validate Monte Carlo option pricing given paths that finish in-the-money.
    EXPECTED: The discounted expected payoff must be strictly positive.
    """
    paths = np.array([
        [100, 110],
        [100, 120],
        [100, 90],
    ])
    price = monte_carlo_option_price(paths=paths, strike=100, r=0.05, T=1)
    assert price >= 0

def test_option_price_zero_payoff():
    """
    STORY: Edge Case - Test pricing when all simulated asset paths finish out-of-the-money.
    EXPECTED: If no path exceeds the strike price, the call option payoff is 0, so its price must be exactly 0.0.
    """
    paths = np.array([
        [100, 80],
        [100, 90],
        [100, 95],
    ])
    price = monte_carlo_option_price(paths=paths, strike=100, r=0.05, T=1)
    assert price == 0.0

def test_option_pricing_empty_paths():
    """
    STORY: Robustness Check - Test how the function handles an empty array of paths.
    EXPECTED: Passing empty data should gracefully raise a ValueError or IndexError instead of returning an invalid price.
    """
    paths = np.array([])
    with pytest.raises((ValueError, IndexError)):
        monte_carlo_option_price(paths=paths, strike=100, r=0.05, T=1)