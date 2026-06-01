def volatility_smile(
    strike,
    atm_strike,
    a,
    b,
    c,
):
    return a * (strike - atm_strike) ** 2 + b * (strike - atm_strike) + c
