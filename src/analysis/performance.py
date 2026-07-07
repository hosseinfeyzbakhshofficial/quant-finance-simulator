import time
from typing import List, Union

import numpy as np

from src.processes.gbm import simulate_gbm


def value_at_risk(returns: Union[List[float], np.ndarray], confidence: float = 0.95) -> float:
    """
    Calculate the Value at Risk (VaR) of a returns array at a given confidence level.
    """
    if isinstance(returns, list):
        returns = np.array(returns)
    
    # Calculate the percentile corresponding to 1 - confidence
    return float(np.percentile(returns, (1 - confidence) * 100))


def sharpe_ratio(returns: Union[List[float], np.ndarray], risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe Ratio of a returns array.
    """
    if isinstance(returns, list):
        returns = np.array(returns)
        
    excess_returns = returns - risk_free_rate
    std_dev = np.std(excess_returns)
    
    if std_dev == 0:
        return 0.0
        
    return float(np.mean(excess_returns) / std_dev)


def benchmark_simulation():
    """
    Benchmark the execution time of the GBM simulation.
    """
    start = time.time()
    
    simulate_gbm(100, 0.1, 0.2, T=1, dt=0.0001, seed=42)
    
    end = time.time()
    
    print(f"Execution time: {end - start:.6f} seconds")