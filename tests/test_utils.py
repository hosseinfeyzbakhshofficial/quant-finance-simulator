import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, mock_open

# Import utility functions from your src package
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config
from src.utils.cache_manager import save_cache, load_cache
from src.utils.exporter import export_results

def test_logger_flow():
    """
    STORY: Verify the logging module correctly initializes and returns a logger.
    EXPECTED: Logger instance is not None and has the expected module name.
    """
    logger = setup_logger("test_logger")
    assert logger is not None
    assert logger.name == "test_logger"

@patch("builtins.open", new_callable=mock_open, read_data="simulation:\n  steps: 100\n  paths: 1000")
def test_config_loader_success(mock_file):
    """
    STORY: Test valid YAML configuration parsing using mocked file access.
    EXPECTED: The config dictionary is loaded properly with matching keys.
    """
    config = load_config("fake_config.yaml")
    assert isinstance(config, dict)
    assert "simulation" in config

def test_config_loader_file_not_found():
    """
    STORY: Verify exception handling when a configuration file is missing.
    EXPECTED: Raises FileNotFoundError when attempting to open a non-existent path.
    """
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_path.yaml")

@patch("pickle.dump")
@patch("builtins.open", new_callable=mock_open)
def test_cache_manager_save_and_load(mock_file, mock_pickle):
    """
    STORY: Validate the serialization and saving capabilities of the cache manager.
    EXPECTED: File streams are correctly opened in binary write mode.
    ```"""
    data = {"spot": 100.0, "vol": 0.2}
    save_cache(data, "test_cache.pkl")
    mock_file.assert_called_with("test_cache.pkl", "wb")

@patch("pandas.DataFrame.to_csv")
def test_exporter_dataframe(mock_to_csv):
    """
    STORY: Test exporting simulation data matrix directly into a CSV file format.
    EXPECTED: Triggers pandas data frame export pipeline with correct parameters.
    """
    dummy_data = {"Path_1": [100, 102, 104], "Path_2": [100, 98, 97]}
    df = pd.DataFrame(dummy_data)
    export_results(df, "output.csv")
    mock_to_csv.assert_called_once()
