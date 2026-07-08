# Black Sigma Terminal

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![MIT License](https://img.shields.io/badge/License-MIT-green)
![Pytest](https://img.shields.io/badge/Tested%20with-Pytest-success)
![Code Style](https://img.shields.io/badge/Code%20Style-Black-black)
![Linter](https://img.shields.io/badge/Linter-Ruff-orange)
![Streamlit](https://img.shields.io/badge/Deployed%20on-Streamlit-red)

An institutional-grade quantitative finance workstation for stochastic asset path simulation, analytical option valuation, risk analysis, and real-time market data broadcasting.

---

## 🎯 What the Software Does

Black Sigma Terminal is designed to simulate, price, and analyze European-style contingent claims under uncertain market regimes. The platform bridges continuous-time financial mathematics with interactive computational tools:

1. **Stochastic Asset Simulation:** Simulates continuous price trajectories of an underlying asset using a **Geometric Brownian Motion (GBM)** stochastic differential equation:
   $$dS_t = \mu S_t dt + \sigma S_t dW_t$$

2. **Derivative Valuation Engine:** - Computes empirical option prices for European Call options using an advanced **Monte Carlo Simulation** pipeline over tens of thousands of generated hypotheses.
   - Provides exact baseline comparisons by evaluating the closed-form analytical solutions derived from the **Black-Scholes-Merton Pricing Model**.

3. **Risk Sensitivity Auditing (The Greeks):** Quantifies localized exposure and portfolio risk profiles by calculating first and second-order partial derivatives of the option price formula:
   - **Delta ($\Delta$):** Absolute price sensitivity.
   - **Gamma ($\Gamma$):** Acceleration of Delta.
   - **Vega ($\nu$):** Sensitivity to shifts in underlying asset volatility.
   - **Theta ($\Theta$):** Deterministic time-decay rate.

4. **Statistical Risk Infrastructure:** Evaluates distributional risk parameters including Value at Risk (VaR), Expected Shortfall (Conditional VaR), and annualized Sharpe Ratios across dynamic pricing paths.

---

## ⚙️ How It Is Implemented

The software is engineered with a strict emphasis on performance, decoupled concerns, and production-grade software architecture:

### 1. High-Performance NumPy Vectorization
To eliminate the severe performance limits of native Python `for` loops, the entire simulation matrix is vectorized. 
- Random Wiener increments ($dW$) are sampled simultaneously into a massive multi-dimensional matrix of size `(n_simulations, steps)`.
- Asset path generation uses highly optimized C-level matrix operations (`np.cumsum` along the temporal axis), allowing the calculation of hundreds of thousands of pathways in milliseconds without blocking the execution thread.
- The 3D Volatility Surface rendering grid has been fully vectorized using matrix operations to guarantee instant recalculation upon metric updates.

### 2. Decoupled Modular Architecture
The system discards monolithic script layouts in favor of a professional Object-Oriented/Functional layer split:
- `src/processes/`: Pure mathematical definitions of stochastic differential equations and path generation (`gbm.py`, `monte_carlo.py`).
- `src/finance/`: Analytical option pricing logic, risk sensitivities, and deterministic financial wrappers (`black_scholes.py`, `option_pricing.py`).
- `src/analysis/`: Statistical calculations, performance benchmarks, and non-blocking Matplotlib visualization routines.
- `src/utils/`: Safe YAML I/O configuration management and robust session snapshot serialization.

### 3. Asynchronous Dashboard Design
The web frontend is built via Streamlit. To simulate a continuous real-time trading floor experience, the platform hooks into active application session states to run an isolated rendering loop. This updates the underlying live asset price feeds and dynamically recalculates the entire pricing and Greeks matrix every second without causing user interface lags.
---

## 🏗 Design Philosophy

The project follows modern software engineering principles with a strong emphasis on modularity, maintainability, and separation of concerns. The architecture is intentionally divided into independent analytical components, making each module reusable, testable, and easy to extend.

### Separation of Concerns

- **`src/processes/`** — Vectorized stochastic processes and Geometric Brownian Motion simulations.
- **`src/finance/`** — Black-Scholes analytical pricing and Monte Carlo option valuation.
- **`src/analysis/`** — Statistical analysis, visualization, and performance evaluation.
- **`src/utils/`** — Configuration management, caching, exporting, and logging utilities.

This modular design enables comprehensive unit testing while keeping mathematical models independent from visualization and user interface components.

---

## 📂 Project Structure

```text
quant-finance-simulator
├── .github/workflows/       # Automated CI testing pipeline
├── cache/                  # Serialized simulation snapshots
├── plots/                  # Exported institutional chart assets
├── src/                    # Core Analytical Source Code
│   ├── analysis/           # Statistical risk, benchmarking & plots
│   │   ├── performance.py
│   │   ├── statistics.py
│   │   └── visualization.py
│   ├── finance/            # Black-Scholes models & option pricing engines
│   │   ├── black_scholes.py
│   │   └── option_pricing.py
│   ├── processes/          # Vectorized GBM SDE simulation routines
│   │   ├── gbm.py
│   │   └── monte_carlo.py
│   └── utils/              # Configuration, caching & logging utilities
│       ├── cache_manager.py
│       ├── config_loader.py
│       ├── exporter.py
│       └── logger.py
├── tests/                  # Unified Pytest testing suite
├── app.py                  # Live Streamlit Interactive Workstation
├── config.yaml             # Decoupled project parameter infrastructure
├── requirements.txt        # Production dependency manifest
└── pyproject.toml          # Tooling configuration specification (Ruff/Black)
```
---

## Quick Start

### Clone the repository

```bash
git clone https://github.com/hosseinfeyzbakhshofficial/quant-finance-simulator.git
cd quant-finance-simulator
```

### Initialize virtual environment

```bash
python -m venv .venv
```

### Activate environment (Windows)

```bash
.venv\Scripts\activate
```

### Activate environment (Linux / macOS)

```bash
source .venv/bin/activate
```

### Install required dependencies

```bash
pip install -r requirements.txt
```

---

## Launching the Terminal Dashboard

```bash
streamlit run app.py
```

---

## Execution of Automated Verification Suite

```bash
python -m pytest tests/ -v
```

---


## 💡 Examples of Usage

While the Streamlit dashboard provides a full interactive graphical interface, the core quantitative engine is decoupled and can be used programmatically in your own Python scripts or Jupyter Notebooks.

### 1. Programmatic API Usage
You can directly import the vectorized stochastic models and analytical pricing functions into your custom pipelines:

```python
import numpy as np
from src.processes.gbm import simulate_gbm
from src.finance.black_scholes import black_scholes_call

# 1. Simulate Asset Path (Geometric Brownian Motion)
prices = simulate_gbm(
    S0=100.0,      # Initial price
    mu=0.05,       # Drift
    sigma=0.20,    # Volatility
    T=1.0,         # Time to maturity (years)
    dt=0.01,       # Time step
    seed=42        # For reproducibility
)
print(f"Terminal Asset Price: ${prices[-1]:.2f}")

# 2. Calculate Analytical Option Price (Black-Scholes)
call_price = black_scholes_call(
    S0=100.0,
    K=100.0,       # Strike price
    T=1.0,
    r=0.05,        # Risk-free rate
    sigma=0.20
)
print(f"Theoretical European Call Price: ${call_price:.4f}")
```
### 2. Programmatic API Usage
The repository includes pre-built procedural examples to demonstrate the engine's capabilities. You can run the visualization example directly from your terminal:

```Bash
python examples/plot_gbm.py
```
---
## Code Quality Assurance

The codebase enforces industry-standard validation hooks before integrations.

### Ruff (Linting & Style Analysis)

```bash
ruff check .
```

### Black (Deterministic Formatting)

```bash
black .
```

---

## License

This project is open-source software licensed under the MIT License.

See the `LICENSE` file for details.

---

## Author

**Hossein Feyzbakhsh**

M.Sc. Physics Student (Materials Physics and Nanoscience)

University of Bologna