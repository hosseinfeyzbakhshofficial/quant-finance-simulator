import pickle
import os


def save_simulation(data, filename="cache/mc_cache.pkl"):

    os.makedirs("cache", exist_ok=True)

    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_simulation(filename="cache/mc_cache.pkl"):

    if not os.path.exists(filename):
        return None

    with open(filename, "rb") as f:
        return pickle.load(f)