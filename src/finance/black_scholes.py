import numpy as np
from scipy.stats import norm


def black_scholes_call(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Black-Scholes analytical formula for European call option.
    """

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return call_price


def call_delta(S0, K, T, r, sigma):
    """
    Delta of European call option.
    """

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    return norm.cdf(d1)


def call_gamma(S0, K, T, r, sigma):
    """
    Gamma of European call option.
    """

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    return norm.pdf(d1) / (S0 * sigma * np.sqrt(T))


def call_vega(S0, K, T, r, sigma):
    """
    Vega of European call option.
    """

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    return S0 * norm.pdf(d1) * np.sqrt(T)


def call_theta(S0, K, T, r, sigma):
    """
    Theta of European call option.
    """

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    theta = -(S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(
        -r * T
    ) * norm.cdf(d2)

    return theta
