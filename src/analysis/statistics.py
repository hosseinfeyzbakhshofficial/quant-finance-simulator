import numpy as np
import pandas as pd


def estimate_statistics(results: np.ndarray) -> dict:
    """
    Compute basic statistics of Geometric Brownian Motion (GBM) simulation output.
    """
    return {
        "mean": float(np.mean(results)),
        "variance": float(np.var(results)),
        "min": float(np.min(results)),
        "max": float(np.max(results)),
    }


def create_dataframe(results: np.ndarray) -> pd.DataFrame:
    """
    Convert simulation output into a formatted pandas DataFrame.
    """
    return pd.DataFrame({"time_step": range(len(results)), "price": results})


def estimate_mc_statistics(paths: np.ndarray) -> dict:
    """
    Compute statistics on the final asset prices across all Monte Carlo paths.
    """
    final_prices = paths[:, -1]
    return {
        "mean_final_price": float(np.mean(final_prices)),
        "std_final_price": float(np.std(final_prices)),
        "min_final_price": float(np.min(final_prices)),
        "max_final_price": float(np.max(final_prices)),
    }


def value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate the Value at Risk (VaR) for a given confidence level.
    """
    percentile = (1.0 - confidence) * 100.0
    return float(np.percentile(returns, percentile))


def expected_shortfall(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate the Expected Shortfall (Conditional VaR) below the VaR threshold.
    """
    returns_arr = np.array(returns)
    var = value_at_risk(returns_arr, confidence)
    under_var = returns_arr[returns_arr <= var]
    return float(under_var.mean()) if under_var.size > 0 else 0.0


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.05, trading_days: int = 252) -> float:
    """
    Calculate the annualized Sharpe Ratio for daily or periodic returns.
    """
    returns_arr = np.array(returns)
    period_rf = risk_free_rate / trading_days
    excess_returns = returns_arr - period_rf
    
    std = np.std(excess_returns)
    if std == 0:
        return 0.0
        
    return float((np.mean(excess_returns) / std) * np.sqrt(trading_days))


def portfolio_value(positions: np.ndarray) -> float:
    """
    Calculate total portfolio value by summing all open asset positions.
    """
    return float(np.sum(positions))