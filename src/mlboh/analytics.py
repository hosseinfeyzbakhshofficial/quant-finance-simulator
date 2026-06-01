import numpy as np


def sharpe_ratio(
    returns,
    risk_free_rate=0.05,
):
    excess = returns - risk_free_rate / 252

    return (np.mean(excess) / np.std(excess)) * np.sqrt(252)
