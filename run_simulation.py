import argparse
import logging

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
    result = simulate_gbm(
        S0=100,
        mu=0.1,
        sigma=0.2,
        T=1,
        dt=0.01,
        seed=42
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
        S0=100,
        mu=0.1,
        sigma=0.2,
        T=1,
        dt=0.01,
        n_simulations=1000,
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
        strike=100,
        r=0.05,
        T=1,
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
    strike=100
)
    # confidence band
    plot_confidence_band(mc_paths)
    
    # final price distribution
    plot_final_price_distribution(mc_paths)

    # option payoff
    plot_option_payoff(
        mc_paths[:, -1],
        strike=100
    )
    
    # confidence band
    plot_confidence_band(mc_paths)

if __name__ == "__main__":
    main()