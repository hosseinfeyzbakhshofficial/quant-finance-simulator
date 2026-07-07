import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_gbm(results):
    """Plot GBM path using seaborn"""
    df = pd.DataFrame({"time": range(len(results)), "price": results})
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="time", y="price", color="#00e676")
    plt.title("GBM Simulation Path")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.grid(True, linestyle="--", alpha=0.5)
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/gbm_path.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_gbm_dataframe(df):
    """Plot GBM dataframe"""
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="time_step", y="price", color="#3b82f6")
    plt.title("GBM Simulation - Dataframe Analytics")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.grid(True, linestyle="--", alpha=0.5)
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/gbm_dataframe.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_mc_paths(paths: np.ndarray):
    """Plot multiple Monte Carlo simulation paths"""
    plt.figure(figsize=(10, 6))
    for i in range(min(100, len(paths))):
        plt.plot(paths[i], alpha=0.2, linewidth=1)
    plt.title(f"Monte Carlo Simulation ({len(paths)} Paths)")
    plt.xlabel("Time Steps")
    plt.ylabel("Asset Price")
    plt.grid(True, linestyle="--", alpha=0.3)
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/monte_carlo_paths.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_gbm_final_distribution(final_prices):
    """Plot distribution of final price states"""
    plt.figure(figsize=(10, 6))
    sns.histplot(final_prices, bins=30, kde=True, color="#3b82f6")
    plt.title("Distribution of Terminal Prices")
    plt.xlabel("Final Price")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/gbm_final_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_option_payoff(final_prices, strike):
    """Plot European call option payoff Architecture"""
    payoff = np.maximum(final_prices - strike, 0)
    plt.figure(figsize=(10, 6))
    plt.scatter(final_prices, payoff, alpha=0.5, color="#ef4444", s=15)
    plt.title("European Call Option Payoff Profile")
    plt.xlabel("Final Price")
    plt.ylabel("Payoff")
    plt.grid(True, linestyle="--", alpha=0.5)
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/option_payoff.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_confidence_band(paths):
    """Plot mean path with 95% confidence interval band"""
    mean_path = np.mean(paths, axis=0)
    std_path = np.std(paths, axis=0)
    time_steps = np.arange(len(mean_path))
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, mean_path, color="#00e676", label="Mean Trajectory", linewidth=2)
    plt.fill_between(time_steps, mean_path - 1.96 * std_path, mean_path + 1.96 * std_path, 
                     color="rgba(0, 230, 118, 0.15)", alpha=0.15, label="95% Confidence Band")
    plt.title("GBM Expectation Pathway with Volatility Bands")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/confidence_band.png", dpi=300, bbox_inches="tight")
    plt.close()