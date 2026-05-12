import time
from src.processes.gbm import simulate_gbm


def benchmark_simulation():
    start = time.time()

    simulate_gbm(
        100,
        0.1,
        0.2,
        T=1,
        dt=0.0001,
        seed=42
    )

    end = time.time()

    print(f"Execution time: {end - start:.6f} seconds")