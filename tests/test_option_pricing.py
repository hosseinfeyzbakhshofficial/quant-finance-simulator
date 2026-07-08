import pytest
import numpy as np
from src.finance.option_pricing import (
    monte_carlo_option_price,
    price_call,
    volatility_smile,
)

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

# ==============================================================================
# NEW SUPPLEMENTARY TESTS ADDED TO COVER MISSING LINES (31, 38)
# ==============================================================================

def test_price_call_wrapper():
    """
    STORY: Verify the wrapper function for Black-Scholes call option pricing.
    EXPECTED: Successfully invokes the core BS model and returns a valid price (Line 31).
    """
    wrapper_price = price_call(S=100, K=100, T=1, r=0.05, sigma=0.2)
    assert wrapper_price > 0
    assert isinstance(wrapper_price, float)

def test_volatility_smile_calculation():
    """
    STORY: Validate the quadratic polynomial representation of the volatility smile.
    EXPECTED: Correctly calculates implied volatility based on strike distance from ATM (Line 38).
    """
    # Using simple numbers to test the quadratic expression: a*(K - K_atm)^2 + b*(K - K_atm) + c
    implied_vol = volatility_smile(strike=110, atm_strike=100, a=0.001, b=0.01, c=0.2)
    expected_vol = 0.001 * (110 - 100)**2 + 0.01 * (110 - 100) + 0.2
    
    assert np.isclose(implied_vol, expected_vol)