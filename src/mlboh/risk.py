import numpy as np


def value_at_risk(
    returns,
    confidence=0.95,
):
    percentile = (1 - confidence) * 100

    return np.percentile(
        returns,
        percentile,
    )


def expected_shortfall(
    returns,
    confidence=0.95,
):
    returns = np.array(returns)

    var = value_at_risk(
        returns,
        confidence,
    )

    return returns[returns <= var].mean()


def sharpe_ratio(
    returns,
    risk_free_rate=0.0,
):
    returns = np.array(returns)

    excess_returns = returns - risk_free_rate

    std = np.std(excess_returns)

    if std == 0:
        return 0.0

    return np.mean(excess_returns) / std
