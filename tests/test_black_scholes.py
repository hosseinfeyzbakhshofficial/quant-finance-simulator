import pytest
import numpy as np
from src.finance.black_scholes import (
    black_scholes_call,
    call_delta,
    call_gamma,
    call_theta,
    call_vega,
)

def test_call_price_positive():
    """
    STORY: Ensure the Black-Scholes Call price is positive under standard conditions.
    EXPECTED: The option value must be strictly greater than zero for an at-the-money option.
    """
    price = black_scholes_call(S0=100, K=100, T=1, r=0.05, sigma=0.2)
    assert price > 0

def test_call_price_increases_with_spot():
    """
    STORY: Validate the monotonicity of the option price with respect to the asset price.
    EXPECTED: As S0 increases from 90 to 110, the call option price must increase.
    """
    p1 = black_scholes_call(90, 100, 1, 0.05, 0.2)
    p2 = black_scholes_call(110, 100, 1, 0.05, 0.2)
    assert p2 > p1

def test_delta_bounds():
    """
    STORY: Verify that the Delta of a standard Call option respects theoretical bounds.
    EXPECTED: Delta must always fall within the interval [0, 1].
    """
    delta = call_delta(100, 100, 1, 0.05, 0.2)
    assert 0 <= delta <= 1

def test_gamma_positive():
    """
    STORY: Verify that Gamma is always positive for a long vanilla call option.
    EXPECTED: Gamma represents the acceleration of Delta and must be strictly positive.
    """
    gamma = call_gamma(100, 100, 1, 0.05, 0.2)
    assert gamma > 0

def test_vega_positive():
    """
    STORY: Verify that Vega is positive, meaning option value increases with volatility.
    EXPECTED: An increase in sigma must always lead to an increase or stable option price.
    """
    vega = call_vega(100, 100, 1, 0.05, 0.2)
    assert vega > 0

def test_theta_finite():
    """
    STORY: Ensure Theta is a finite number under standard simulation parameters.
    EXPECTED: Theta should represent time decay as a valid finite floating point number.
    """
    theta = call_theta(100, 100, 1, 0.05, 0.2)
    assert np.isfinite(theta)

def test_black_scholes_asset_price_zero():
    """
    STORY: Edge Case - Test option pricing when the underlying asset price drops to 0.
    EXPECTED: A call option on a completely worthless asset must have a price of exactly 0.0.
    """
    price = black_scholes_call(S0=0, K=100, T=1, r=0.05, sigma=0.2)
    assert price == 0.0

def test_greeks_at_asset_price_zero():
    """
    STORY: Edge Case - Ensure Greeks behave gracefully when the underlying asset price is 0.
    EXPECTED: Delta, Gamma, and Vega should resolve to 0.0 instead of math errors or infinity.
    """
    assert call_delta(S0=0, K=100, T=1, r=0.05, sigma=0.2) == 0.0
    assert call_gamma(S0=0, K=100, T=1, r=0.05, sigma=0.2) == 0.0
    assert call_vega(S0=0, K=100, T=1, r=0.05, sigma=0.2) == 0.0

def test_black_scholes_invalid_inputs():
    """
    STORY: Robustness & Exception Handling - Test validation for unphysical parameters.
    EXPECTED: Negative prices, negative strikes, negative time, or negative volatilities must raise a ValueError.
    """
    with pytest.raises(ValueError):
        black_scholes_call(S0=-100, K=100, T=1, r=0.05, sigma=0.2)
        
    with pytest.raises(ValueError):
        black_scholes_call(S0=100, K=-100, T=1, r=0.05, sigma=0.2)
        
    with pytest.raises(ValueError):
        black_scholes_call(S0=100, K=100, T=-1, r=0.05, sigma=0.2)
        
    with pytest.raises(ValueError):
        black_scholes_call(S0=100, K=100, T=1, r=0.05, sigma=-0.2)

def test_black_scholes_zero_volatility():
    """
    STORY: Edge Case - Test pricing in a deterministic world where volatility is exactly 0.
    EXPECTED: The option value should converge cleanly without division-by-zero errors.
    """
    price = black_scholes_call(S0=110, K=100, T=1, r=0.05, sigma=0.0)
    assert price >= 0 and np.isfinite(price)