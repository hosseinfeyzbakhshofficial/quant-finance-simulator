import os
import pickle
from typing import Any


def save_simulation(data, filename="cache/mc_cache.pkl"):
    os.makedirs("cache", exist_ok=True)

    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_simulation(filename="cache/mc_cache.pkl"):
    if not os.path.exists(filename):
        return None

    with open(filename, "rb") as f:
        return pickle.load(f)


def save_cache(data: Any, filepath: str) -> None:
    """
    Interface wrapper to save data matching test suite expectations.
    """
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_cache(filepath: str) -> Any:
    """
    Interface wrapper to load data matching test suite expectations.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cache file not found: {filepath}")
    with open(filepath, "rb") as f:
        return pickle.load(f)