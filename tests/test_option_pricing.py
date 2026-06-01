import numpy as np

from src.finance.option_pricing import (
    monte_carlo_option_price,
)


def test_option_price_positive():
    paths = np.array(
        [
            [100, 110],
            [100, 120],
            [100, 90],
        ]
    )

    price = monte_carlo_option_price(
        paths=paths,
        strike=100,
        r=0.05,
        T=1,
    )

    assert price >= 0


def test_option_price_zero_payoff():
    paths = np.array(
        [
            [100, 80],
            [100, 90],
            [100, 95],
        ]
    )

    price = monte_carlo_option_price(
        paths=paths,
        strike=100,
        r=0.05,
        T=1,
    )

    assert price == 0
