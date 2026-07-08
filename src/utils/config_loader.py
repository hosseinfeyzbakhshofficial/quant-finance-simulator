import yaml

def load_config(path="config.yaml") -> dict:
    """
    Load project configuration parameters from a YAML file.
    Raises FileNotFoundError if the file does not exist.
    """
    # Remove try-except block to let FileNotFoundError propagate to the test suite
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config if config is not None else {}