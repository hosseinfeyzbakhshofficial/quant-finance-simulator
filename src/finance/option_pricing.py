import numpy as np


def european_call_payoff(final_prices, strike):
    """
    Compute payoff of a European call option.

    Parameters
    ----------
    final_prices : np.ndarray
        Final simulated prices

    strike : float
        Strike price

    Returns
    -------
    np.ndarray
        Option payoffs
    """

    return np.maximum(final_prices - strike, 0)


def monte_carlo_option_price(
    paths,
    strike,
    r,
    T,
):
    """
    Price a European call option using Monte Carlo.

    Parameters
    ----------
    paths : np.ndarray
        Monte Carlo simulated paths

    strike : float
        Strike price

    r : float
        Risk-free interest rate

    T : float
        Time to maturity

    Returns
    -------
    float
        Estimated option price
    """

    final_prices = paths[:, -1]

    payoffs = european_call_payoff(
        final_prices,
        strike
    )

    discounted_price = np.exp(-r * T) * np.mean(payoffs)

    return discounted_price