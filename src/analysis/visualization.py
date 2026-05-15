import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_gbm(results):
    """
    Plot GBM path using seaborn
    """

    df = pd.DataFrame({
        "time": range(len(results)),
        "price": results
    })

    sns.lineplot(data=df, x="time", y="price")

    plt.title("GBM Simulation")
    plt.xlabel("Time Step")
    plt.ylabel("Price")

    plt.grid(True)

    plt.show()


def plot_gbm_dataframe(df):
    """
    Plot GBM dataframe
    """

    plt.figure(figsize=(8, 5))

    sns.lineplot(
        data=df,
        x="time_step",
        y="price"
    )

    plt.title("GBM Simulation")

    plt.xlabel("Time Step")

    plt.ylabel("Price")

    plt.grid(True)

    plt.show()


def plot_mc_paths(paths, n_paths=20):
    """
    Plot multiple Monte Carlo GBM paths.
    """

    plt.figure(figsize=(12, 6))

    for i in range(min(n_paths, len(paths))):
        plt.plot(
            paths[i],
            alpha=0.5,
            linewidth=1
        )

    plt.title("Monte Carlo GBM Paths")

    plt.xlabel("Time Step")
    plt.ylabel("Price")

    plt.grid(True)

    plt.show()

def plot_final_price_distribution(paths):
    """
    Plot histogram of final Monte Carlo prices.
    """

    final_prices = paths[:, -1]

    plt.figure(figsize=(10, 6))

    sns.histplot(
        final_prices,
        bins=30,
        kde=True
    )

    plt.title("Distribution of Final Prices")
    plt.xlabel("Final Price")
    plt.ylabel("Frequency")

    plt.grid(True)

    plt.show()

def plot_option_payoff(final_prices, strike):
    """
    Plot European call option payoff.
    """

    payoff = np.maximum(final_prices - strike, 0)

    plt.figure(figsize=(10, 6))

    plt.scatter(
        final_prices,
        payoff,
        alpha=0.5
    )

    plt.title("European Call Option Payoff")

    plt.xlabel("Final Price")
    plt.ylabel("Payoff")

    plt.grid(True)

    plt.show()

def plot_confidence_band(paths):
    """
    Plot mean path with confidence interval band.
    """

    mean_path = np.mean(paths, axis=0)

    std_path = np.std(paths, axis=0)

    upper = mean_path + std_path

    lower = mean_path - std_path

    plt.figure(figsize=(12, 6))

    plt.plot(
        mean_path,
        label="Mean Path"
    )

    plt.fill_between(
        range(len(mean_path)),
        lower,
        upper,
        alpha=0.3,
        label="±1 Std Dev"
    )

    plt.title("Monte Carlo Mean Path with Confidence Band")

    plt.xlabel("Time Step")
    plt.ylabel("Price")

    plt.legend()

    plt.grid(True)

    plt.show()