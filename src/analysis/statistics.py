import numpy as np
import pandas as pd


def estimate_statistics(
    results: np.ndarray
) -> dict:
    """
    Compute basic statistics of GBM simulation output.
    """

    return {
        "mean": np.mean(results),
        "variance": np.var(results),
        "min": np.min(results),
        "max": np.max(results),
    }


def create_dataframe(
    results: np.ndarray
) -> pd.DataFrame:
    """
    Convert simulation output into pandas DataFrame.
    """

    df = pd.DataFrame({
        "time_step": range(len(results)),
        "price": results
    })

    return df


def estimate_mc_statistics(
    paths: np.ndarray
) -> dict:
    """
    Compute statistics on Monte Carlo final prices.
    """

    final_prices = paths[:, -1]

    return {
        "mean_final_price": np.mean(final_prices),
        "std_final_price": np.std(final_prices),
        "min_final_price": np.min(final_prices),
        "max_final_price": np.max(final_prices),
    }