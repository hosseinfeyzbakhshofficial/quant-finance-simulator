import numpy as np

from src.finance.black_scholes import (
    black_scholes_call,
    call_delta,
    call_gamma,
    call_theta,
    call_vega,
)


def test_call_price_positive():
    price = black_scholes_call(
        S0=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.2,
    )

    assert price > 0


def test_call_price_increases_with_spot():
    p1 = black_scholes_call(90, 100, 1, 0.05, 0.2)
    p2 = black_scholes_call(110, 100, 1, 0.05, 0.2)

    assert p2 > p1


def test_delta_bounds():
    delta = call_delta(100, 100, 1, 0.05, 0.2)

    assert 0 <= delta <= 1


def test_gamma_positive():
    gamma = call_gamma(100, 100, 1, 0.05, 0.2)

    assert gamma > 0


def test_vega_positive():
    vega = call_vega(100, 100, 1, 0.05, 0.2)

    assert vega > 0


def test_theta_finite():
    theta = call_theta(100, 100, 1, 0.05, 0.2)

    assert np.isfinite(theta)
