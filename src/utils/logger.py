import logging

def setup_logger(name: str = __name__, level=logging.INFO):
    """
    Initializes and returns a logger instance with the given name.
    """
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(name)