from src.finance.black_scholes import (
    black_scholes_call,
)


def price_call(
    S,
    K,
    T,
    r,
    sigma,
):
    return black_scholes_call(
        S0=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
    )
