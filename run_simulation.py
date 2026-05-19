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

from src.utils.exporter import export_results


def main():

    # argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # set logging level
    level = logging.DEBUG if args.debug else logging.INFO

    logger = setup_logger(level)

    logger.info("Running simulation...")

    # Single GBM Simulation
    mc_paths = monte_carlo_gbm(
        S0=config.S0,
        mu=config.MU,
        sigma=config.SIGMA,
        T=config.T,
        dt=config.DT,
        n_simulations=config.N_SIMULATIONS,
    )

    result = simulate_gbm(
    config.S0,
    config.MU,
    config.SIGMA,
    T=config.T,
    dt=config.DT,
    seed=config.SEED
)

    # statistics
    stats = estimate_statistics(result)

    print("\n===== GBM Statistics =====")
    print(stats)

    # dataframe
    df = create_dataframe(result)
    # plot single GBM path
    plot_gbm_dataframe(df)

    #benchmark_simulation()

    # Monte Carlo simulation
    mc_paths = monte_carlo_gbm(
        S0=config.S0,
        mu=config.MU,
        sigma=config.SIGMA,
        T=config.T,
        dt=config.DT,
        n_simulations=config.N_SIMULATIONS,
    )

    print("\n===== Monte Carlo Simulation =====")

    print("Monte Carlo matrix shape:", mc_paths.shape)

    print("Number of simulations:", mc_paths.shape[0])

    print("Number of time steps:", mc_paths.shape[1])

    # Monte Carlo statistics
    mc_stats = estimate_mc_statistics(mc_paths)

    print("\n===== Monte Carlo Statistics =====")

    print(mc_stats)

    # European Call Option Pricing

    option_price = monte_carlo_option_price(
    paths=mc_paths,
    strike=config.STRIKE,
    r=config.RISK_FREE_RATE,
    T=config.T,
)

    print("\n===== European Call Option Price =====")

    print(f"{option_price:.4f}")

    # Visualization

    plot_mc_paths(mc_paths, n_paths=20)
    
    # final price distribution
    plot_final_price_distribution(mc_paths)
    
    # option payoff
    plot_option_payoff(
    mc_paths[:, -1],
    strike=config.STRIKE
)
    # confidence band
    plot_confidence_band(mc_paths)
    
    # final price distribution
    plot_final_price_distribution(mc_paths)

    # option payoff
    plot_option_payoff(
        mc_paths[:, -1],
        strike=config.STRIKE
    )
    
    # confidence band
    plot_confidence_band(mc_paths)

    # JSON Export
    export_results({
    "gbm_statistics": stats,
    "mc_statistics": mc_stats,
    "option_price": float(option_price),
})

if __name__ == "__main__":
    main()