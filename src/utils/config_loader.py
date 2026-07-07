import yaml

def load_config(path="config.yaml") -> dict:
    """
    Load project configuration parameters from a YAML file.
    """
    try:
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        return config if config is not None else {}
    except FileNotFoundError:
        # Fallback to default dictionary if file is missing
        return {
            "market": {"S0": 100.0, "r": 0.05},
            "process": {"mu": 0.10, "sigma": 0.20, "T": 1.0, "dt": 0.01},
            "simulation": {"n_simulations": 10000}
        }
