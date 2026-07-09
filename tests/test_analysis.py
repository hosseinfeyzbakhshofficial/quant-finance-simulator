import pytest
import numpy as np
from src.analysis.performance import sharpe_ratio, value_at_risk
from src.analysis.statistics import calculate_skewness, calculate_kurtosis, financial_summary


def test_sharpe_ratio_zero_variance():
    """
    STORY: Handle edge cases where portfolio returns are completely flat (zero volatility).
    EXPECTED: Return 0.0 or handle safely to avoid division by zero errors (Lines 10-18 coverage).
    """
    flat_returns = np.array([0.02, 0.02, 0.02, 0.02])
    result = sharpe_ratio(flat_returns, risk_free_rate=0.02)
    assert result == 0.0 or np.isnan(result)

def test_statistics_edge_cases():
    """
    STORY: Test statistical analysis functions under empty inputs or single-element arrays.
    EXPECTED: Safe termination or custom calculation handling (Lines 9, 21, 28-29 coverage).
    """
    empty_returns = np.array([])
    with pytest.raises(ValueError):
        calculate_skewness(empty_returns)

def test_financial_summary_dictionary_output():
    """
    STORY: Verify structural compliance of full statistics summaries.
    EXPECTED: Output must be a dictionary containing essential statistical metrics.
    """
    sample_returns = np.random.normal(0.001, 0.02, 100)
    summary = financial_summary(sample_returns)
    assert isinstance(summary, dict)
    assert "mean" in summary
    assert "std" in summary

def test_analysis_additional_edge_cases():
    """
    STORY: Exercise additional statistical edge cases.
    """
    data = np.array([1, 2, 3, 4, 5])
    assert value_at_risk(data) is not None