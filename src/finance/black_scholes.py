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
    Black-Scholes analytical formula for a European call option.
    """
    if S0 < 0 or K < 0 or T < 0 or sigma < 0:
        raise ValueError("Inputs (S0, K, T, sigma) cannot be negative.")
        
    if S0 == 0:
        return 0.0
        
    if T <= 0 or sigma <= 0:
        return float(max(0.0, S0 - K))
        
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return float(call_price)


def call_delta(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the Delta of a European call option.
    """
    if S0 < 0 or K < 0 or T < 0 or sigma < 0:
        raise ValueError("Inputs (S0, K, T, sigma) cannot be negative.")
        
    if S0 == 0:
        return 0.0
        
    if T <= 0 or sigma <= 0:
        return 1.0 if S0 > K else 0.0
        
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1))


def call_gamma(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the Gamma of a European call option.
    """
    if S0 < 0 or K < 0 or T < 0 or sigma < 0:
        raise ValueError("Inputs (S0, K, T, sigma) cannot be negative.")
        
    if S0 == 0 or T <= 0 or sigma <= 0:
        return 0.0
        
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    return float(gamma)


def call_vega(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the Vega of a European call option.
    """
    if S0 < 0 or K < 0 or T < 0 or sigma < 0:
        raise ValueError("Inputs (S0, K, T, sigma) cannot be negative.")
        
    if S0 == 0 or T <= 0 or sigma <= 0:
        return 0.0
        
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega = S0 * norm.pdf(d1) * np.sqrt(T)
    return float(vega)


def call_theta(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the Theta of a European call option.
    """
    if S0 < 0 or K < 0 or T < 0 or sigma < 0:
        raise ValueError("Inputs (S0, K, T, sigma) cannot be negative.")
        
    if S0 == 0 or T <= 0 or sigma <= 0:
        return 0.0
        
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    theta = -(S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    return float(theta)