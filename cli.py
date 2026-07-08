"""
Command-line interface for Black Sigma Terminal.

Exposes the quantitative engine (GBM simulation, Black-Scholes pricing and
Greeks, Monte Carlo option pricing, and risk metrics) directly from the
terminal, without launching the Streamlit dashboard.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from src.processes.gbm import simulate_gbm
from src.processes.monte_carlo import MonteCarloSimulator
from src.finance.black_scholes import (
    black_scholes_call,
    call_delta,
    call_gamma,
    call_vega,
    call_theta,
)
from src.finance.option_pricing import monte_carlo_option_price
from src.analysis.performance import sharpe_ratio, value_at_risk


def _add_market_args(subparser: argparse.ArgumentParser, include_strike: bool = False) -> None:
    """Attach the market-parameter arguments shared by most subcommands."""
    subparser.add_argument("--S0", type=float, default=100.0, help="Initial asset price")
    subparser.add_argument("--mu", type=float, default=0.05, help="Expected return (drift)")
    subparser.add_argument("--sigma", type=float, default=0.2, help="Volatility")
    subparser.add_argument("--T", type=float, default=1.0, help="Time to maturity, in years")
    subparser.add_argument("--r", type=float, default=0.05, help="Risk-free rate")
    if include_strike:
        subparser.add_argument("--K", type=float, default=100.0, help="Strike price")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="black-sigma",
        description="Black Sigma Terminal - quantitative finance CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate_parser = subparsers.add_parser("simulate", help="Run a single GBM asset-path simulation")
    _add_market_args(simulate_parser)
    simulate_parser.add_argument("--dt", type=float, default=0.01, help="Time step size")
    simulate_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    price_parser = subparsers.add_parser("price", help="Price a European call with Black-Scholes")
    _add_market_args(price_parser, include_strike=True)

    greeks_parser = subparsers.add_parser("greeks", help="Compute Delta, Gamma, Vega and Theta for a call")
    _add_market_args(greeks_parser, include_strike=True)

    mc_parser = subparsers.add_parser("montecarlo", help="Price a European call with Monte Carlo simulation")
    _add_market_args(mc_parser, include_strike=True)
    mc_parser.add_argument("--steps", type=int, default=252, help="Number of time steps per path")
    mc_parser.add_argument("--n-simulations", type=int, default=10000, help="Number of simulated paths")
    mc_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    risk_parser = subparsers.add_parser(
        "risk", help="Simulate a path and report the Sharpe ratio and VaR of its returns"
    )
    _add_market_args(risk_parser)
    risk_parser.add_argument("--dt", type=float, default=0.01, help="Time step size")
    risk_parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level for VaR")
    risk_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    return parser


def run_simulate(args: argparse.Namespace) -> None:
    prices = simulate_gbm(args.S0, args.mu, args.sigma, args.T, dt=args.dt, seed=args.seed)
    print(f"Simulated {len(prices)} points. Terminal price: {prices[-1]:.2f}")


def run_price(args: argparse.Namespace) -> None:
    price = black_scholes_call(S0=args.S0, K=args.K, T=args.T, r=args.r, sigma=args.sigma)
    print(f"Black-Scholes call price: {price:.4f}")


def run_greeks(args: argparse.Namespace) -> None:
    kwargs = dict(S0=args.S0, K=args.K, T=args.T, r=args.r, sigma=args.sigma)
    print(f"Delta: {call_delta(**kwargs):.4f}")
    print(f"Gamma: {call_gamma(**kwargs):.4f}")
    print(f"Vega:  {call_vega(**kwargs):.4f}")
    print(f"Theta: {call_theta(**kwargs):.4f}")


def run_montecarlo(args: argparse.Namespace) -> None:
    simulator = MonteCarloSimulator(S0=args.S0, mu=args.mu, sigma=args.sigma, T=args.T, steps=args.steps)
    paths = simulator.generate_paths(num_simulations=args.n_simulations, seed=args.seed)
    price = monte_carlo_option_price(paths=paths, strike=args.K, r=args.r, T=args.T)
    print(f"Monte Carlo call price ({args.n_simulations} paths): {price:.4f}")


def run_risk(args: argparse.Namespace) -> None:
    prices = simulate_gbm(args.S0, args.mu, args.sigma, args.T, dt=args.dt, seed=args.seed)
    returns = np.diff(np.log(prices))
    sr = sharpe_ratio(returns, risk_free_rate=args.r)
    var = value_at_risk(returns, confidence=args.confidence)
    print(f"Sharpe ratio: {sr:.4f}")
    print(f"Value at Risk ({args.confidence:.0%}): {var:.4f}")


COMMANDS = {
    "simulate": run_simulate,
    "price": run_price,
    "greeks": run_greeks,
    "montecarlo": run_montecarlo,
    "risk": run_risk,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    handler = COMMANDS[args.command]
    try:
        handler(args)
    except ValueError as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()