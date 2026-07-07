import pytest
import numpy as np
from src.analysis.performance import sharpe_ratio, value_at_risk

def test_var_negative():
    """
    STORY: Verify that Value at Risk (VaR) correctly identifies losses.
    EXPECTED: The 95% VaR for a distribution containing negative returns 
              must be a negative value representing the threshold loss.
    """
    returns = [-0.05, -0.03, -0.02, 0.01, 0.02]
    var = value_at_risk(returns, confidence=0.95)
    assert var < 0

def test_var_edge_case_zero_returns():
    """
    STORY: Test VaR behavior with zero risk/returns.
    EXPECTED: If all returns are zero, VaR at any confidence level must be 0.0.
    """
    returns = [0.0, 0.0, 0.0, 0.0]
    var = value_at_risk(returns, confidence=0.99)
    assert var == 0.0

def test_sharpe_finite():
    """
    STORY: Verify the Sharpe Ratio calculation under normal conditions.
    EXPECTED: Given positive excess returns, the Sharpe Ratio should be a non-zero finite number.
    """
    returns = [0.01, 0.02, -0.01, 0.03, 0.01]
    sr = sharpe_ratio(returns)
    assert sr != 0 and np.isfinite(sr)

def test_sharpe_zero_variance():
    """
    STORY: Test Sharpe Ratio when there is no volatility in returns (Edge Case).
    EXPECTED: When standard deviation is zero, it should return 0.0 instead of throwing a DivisionByZero error.
    """
    returns = [0.02, 0.02, 0.02, 0.02]
    sr = sharpe_ratio(returns, risk_free_rate=0.01)
    assert sr == 0.0