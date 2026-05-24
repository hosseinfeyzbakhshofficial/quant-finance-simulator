import argparse
import logging

import config

from src.utils.logger import setup_logger

from src.processes.gbm import simulate_gbm

from src.analysis.statistics import (
    estimate_statistics,
    create_dataframe,
    estimate_mc_statistics,
)

from src.analysis.visualization import (
    plot_gbm_dataframe,
    plot_mc_paths,
    plot_final_price_distribution,
    plot_option_payoff,
    plot_confidence_band,
)

from src.analysis.performance import benchmark_simulation

from src.simulations.monte_carlo import monte_carlo_gbm

from src.finance.option_pricing import (
    monte_carlo_option_price,
)

from src.finance.black_scholes import (
    black_scholes_call,
    call_delta,
    call_gamma,
    call_vega,
    call_theta,
)

from src.utils.exporter import export_results


def main():

    # Argument parser

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument("--s0", type=float, default=config.S0)
    parser.add_argument("--mu", type=float, default=config.MU)
    parser.add_argument("--sigma", type=float, default=config.SIGMA)

    parser.add_argument("--T", type=float, default=config.T)
    parser.add_argument("--dt", type=float, default=config.DT)

    parser.add_argument("--paths", type=int, default=config.N_SIMULATIONS)

    parser.add_argument("--strike", type=float, default=config.STRIKE)

    parser.add_argument("--rate", type=float, default=config.RISK_FREE_RATE)
    

    args = parser.parse_args()

    S0 = args.s0
    MU = args.mu
    SIGMA = args.sigma
    
    T = args.T
    DT = args.dt
    
    N_SIMULATIONS = args.paths
    
    STRIKE = args.strike
    
    RISK_FREE_RATE = args.rate

    

    # Logging

    level = logging.DEBUG if args.debug else logging.INFO

    logger = setup_logger(level)

    logger.info("Running simulation...")

    # Single GBM Simulation

    result = simulate_gbm(
    S0,
    MU,
    SIGMA,
    T=T,
    dt=DT,
    seed=config.SEED
)

    # GBM Statistics

    stats = estimate_statistics(result)

    print("\n===== GBM Statistics =====")
    print(stats)

    # DataFrame + Plot

    df = create_dataframe(result)

    plot_gbm_dataframe(df)

    # Benchmark

    # benchmark_simulation()

    # Monte Carlo Simulation

    mc_paths = monte_carlo_gbm(
    S0=S0,
    mu=RISK_FREE_RATE,
    sigma=SIGMA,
    T=T,
    dt=DT,
    n_simulations=N_SIMULATIONS,
)

    print("\n===== Monte Carlo Simulation =====")

    print("Monte Carlo matrix shape:", mc_paths.shape)

    print("Number of simulations:", mc_paths.shape[0])

    print("Number of time steps:", mc_paths.shape[1])

    # Monte Carlo Statistics

    mc_stats = estimate_mc_statistics(mc_paths)

    print("\n===== Monte Carlo Statistics =====")

    print(mc_stats)

    # Monte Carlo Option Pricing

    option_price = monte_carlo_option_price(
        paths=mc_paths,
        strike=config.STRIKE,
        r=RISK_FREE_RATE,
        T=config.T,
    )

    print("\n===== European Call Option Price =====")

    print(f"{option_price:.4f}")

    # Black-Scholes Analytical Price

    bs_price = black_scholes_call(
        S0=config.S0,
        K=config.STRIKE,
        T=config.T,
        r=RISK_FREE_RATE,
        sigma=config.SIGMA,
    )

    print("\n===== Black-Scholes Price =====")

    print(f"{bs_price:.4f}")

    # Greeks
    delta = call_delta(
    S0,
    STRIKE,
    T,
    RISK_FREE_RATE,
    SIGMA,
)
    gamma = call_gamma(
    S0,
    STRIKE,
    T,
    RISK_FREE_RATE,
    SIGMA,
)
    vega = call_vega(
    S0,
    STRIKE,
    T,
    RISK_FREE_RATE,
    SIGMA,
)
    theta = call_theta(
    S0,
    STRIKE,
    T,
    RISK_FREE_RATE,
    SIGMA,
)
    print("\n===== Greeks =====")
    
    print(f"Delta: {delta:.6f}")
    print(f"Gamma: {gamma:.6f}")
    print(f"Vega : {vega:.6f}")
    print(f"Theta: {theta:.6f}")

    # Pricing Error

    error = abs(option_price - bs_price)

    print("\n===== Pricing Error =====")

    print(f"{error:.6f}")

    # Visualization

    plot_mc_paths(mc_paths, n_paths=20)

    plot_final_price_distribution(mc_paths)

    plot_option_payoff(
        mc_paths[:, -1],
        strike=STRIKE
    )

    plot_confidence_band(mc_paths)

    # Export Results

    export_results({
        "gbm_statistics": stats,
        "mc_statistics": mc_stats,
        "option_price_mc": float(option_price),
        "option_price_black_scholes": float(bs_price),
        "pricing_error": float(error),
    })


if __name__ == "__main__":
    main()