import argparse
import logging

from src.utils.logger import setup_logger

from src.processes.gbm import simulate_gbm

from src.analysis.statistics import (
    estimate_statistics,
    create_dataframe,
)

from src.analysis.visualization import (
    plot_gbm_dataframe,
)

from src.analysis.performance import benchmark_simulation

from src.simulations.monte_carlo import monte_carlo_gbm


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

    # simulate
    result = simulate_gbm(
        100,
        0.1,
        0.2,
        T=1,
        dt=0.01,
        seed=42
    )

    # stats
    stats = estimate_statistics(result)

    print("Statistics:", stats)

    # dataframe + plot
    df = create_dataframe(result)

    plot_gbm_dataframe(df)

    # benchmark
    benchmark_simulation()

    # Monte Carlo simulation
    mc_paths = monte_carlo_gbm(
        S0=100,
        mu=0.1,
        sigma=0.2,
        T=1,
        dt=0.01,
        n_simulations=1000,
    )

    print(mc_paths.shape)

    print("Monte Carlo matrix shape:", mc_paths.shape)

    print("Number of simulations:", mc_paths.shape[0])

    print("Number of time steps:", mc_paths.shape[1])


if __name__ == "__main__":
    main()