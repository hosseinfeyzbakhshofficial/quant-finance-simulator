from src.mlboh.risk import (
    sharpe_ratio,
    value_at_risk,
)


def test_var_negative():
    returns = [
        -0.05,
        -0.03,
        -0.02,
        0.01,
        0.02,
    ]

    var = value_at_risk(
        returns,
        confidence=0.95,
    )

    assert var < 0


def test_sharpe_finite():
    returns = [
        0.01,
        0.02,
        -0.01,
        0.03,
        0.01,
    ]

    sr = sharpe_ratio(returns)

    assert sr != 0
