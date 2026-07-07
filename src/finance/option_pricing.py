import numpy as np
from src.finance.black_scholes import black_scholes_call


def european_call_payoff(final_prices: np.ndarray, strike: float) -> np.ndarray:
    """
    Compute payoff of a European call option.
    """
    return np.maximum(final_prices - strike, 0)


def monte_carlo_option_price(
    paths: np.ndarray,
    strike: float,
    r: float,
    T: float,
) -> float:
    """
    Price a European call option using Monte Carlo simulation paths.
    """
    final_prices = paths[:, -1]
    payoffs = european_call_payoff(final_prices, strike)
    discounted_price = np.exp(-r * T) * np.mean(payoffs)
    return float(discounted_price)


def price_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Wrapper function for Black-Scholes call option pricing.
    """
    return black_scholes_call(S0=S, K=K, T=T, r=r, sigma=sigma)


def volatility_smile(strike: float, atm_strike: float, a: float, b: float, c: float) -> float:
    """
    Compute implied volatility using a quadratic volatility smile model.
    """
    return a * (strike - atm_strike) ** 2 + b * (strike - atm_strike) + c