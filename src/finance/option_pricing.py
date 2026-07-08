import numpy as np
from src.finance.black_scholes import black_scholes_call

def monte_carlo_option_price(paths: np.ndarray, strike: float, r: float, T: float) -> float:
    """
    Calculate European Call option price using simulated Monte Carlo paths.
    """
    if paths is None or paths.size == 0:
        raise ValueError("Paths array cannot be empty.")
    
    # Extract final prices at maturity (last column of the matrix)
    final_prices = paths[:, -1]
    
    # Calculate payoff for a standard Call Option
    payoffs = np.maximum(final_prices - strike, 0)
    
    # Discount the average expected payoff using the risk-free rate
    discounted_price = np.mean(payoffs) * np.exp(-r * T)
    return float(discounted_price)

def price_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Wrapper for Black-Scholes call option pricing.
    """
    return black_scholes_call(S, K, T, r, sigma)

def volatility_smile(strike: float, atm_strike: float, a: float, b: float, c: float) -> float:
    """
    Quadratic polynomial representation of the volatility smile.
    """
    return a * (strike - atm_strike)**2 + b * (strike - atm_strike) + c