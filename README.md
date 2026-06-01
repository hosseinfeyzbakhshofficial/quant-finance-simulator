# Black Sigma Terminal

### Institutional Quantitative Finance Platform

Black Sigma Terminal is a quantitative finance platform built in Python for option pricing, Monte Carlo simulation, risk analysis, and financial visualization.

The project combines an interactive Streamlit dashboard with the MLBOH quantitative analytics library, providing a modular environment for financial modeling and computational finance research.

---

## Live Demo

https://quant-finance-simulator.streamlit.app/

---

## Key Features

### Quantitative Finance

- Geometric Brownian Motion (GBM) simulation
- Monte Carlo asset path generation
- European option pricing
- Black-Scholes analytical pricing
- Greeks calculation
  - Delta
  - Gamma
  - Vega
  - Theta
- Risk analytics
- Volatility analysis

### Visualization

- Monte Carlo path visualization
- Terminal price distributions
- Confidence bands
- Option payoff diagrams
- Statistical summaries

### Software Engineering

- Streamlit Dashboard
- GitHub Actions CI
- Docker Support
- Pytest Test Suite
- Ruff Linting
- Black Formatting
- Pre-commit Hooks
- MIT License

---

## Project Architecture

```text
Black Sigma Terminal
│
├── Streamlit Frontend
│
├── MLBOH Quant Library
│   ├── Pricing
│   ├── Risk
│   ├── Analytics
│   ├── Calibration
│   ├── Portfolio
│   └── Reporting
│
├── Simulation Engine
│   ├── GBM
│   └── Monte Carlo
│
└── Financial Analytics
    ├── Black-Scholes
    ├── Option Pricing
    └── Risk Metrics
```

---

## Project Structure

```text
quant-finance-simulator
│
├── app.py
├── cli.py
├── run_simulation.py
│
├── src
│   ├── finance
│   │   ├── black_scholes.py
│   │   └── option_pricing.py
│   │
│   ├── processes
│   │   ├── gbm.py
│   │   └── monte_carlo.py
│   │
│   ├── simulations
│   │   └── monte_carlo.py
│   │
│   ├── analysis
│   │   ├── performance.py
│   │   ├── statistics.py
│   │   └── visualization.py
│   │
│   ├── utils
│   │   ├── cache_manager.py
│   │   ├── config_loader.py
│   │   ├── exporter.py
│   │   └── logger.py
│   │
│   └── mlboh
│       ├── pricing.py
│       ├── risk.py
│       ├── analytics.py
│       ├── calibration.py
│       ├── portfolio.py
│       ├── reporting.py
│       └── constants.py
│
├── tests
│   ├── test_gbm.py
│   ├── test_black_scholes.py
│   ├── test_option_pricing.py
│   └── test_risk.py
│
└── .github/workflows
    └── python.yml
```

---

## MLBOH Quant Library

The project includes a modular quantitative finance package named **MLBOH**.

Example:

```python
from src.mlboh import *

from src.mlboh.risk import value_at_risk
from src.mlboh.pricing import *
```

The library provides reusable building blocks for:

- Pricing
- Risk Management
- Portfolio Analytics
- Calibration
- Reporting
- Financial Statistics

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/quant-finance-simulator.git

cd quant-finance-simulator
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Streamlit Application

```bash
streamlit run app.py
```

---

## Running Tests

```bash
pytest tests/
```

Current test coverage includes:

- GBM simulation
- Black-Scholes pricing
- Option pricing
- Risk metrics

---

## Code Quality

### Ruff

```bash
ruff check .
```

### Black

```bash
black .
```

### Pre-commit

```bash
pre-commit run --all-files
```

---

## Docker

Build image:

```bash
docker build -t black-sigma .
```

Run container:

```bash
docker run -p 8501:8501 black-sigma
```

---

## Continuous Integration

GitHub Actions automatically:

- Installs dependencies
- Runs tests
- Verifies code quality
- Validates project integrity

Workflow file:

```text
.github/workflows/python.yml
```

---

## License

This project is distributed under the MIT License.

See:

```text
LICENSE
```

for details.

---

## Author

Hossein Feyzbakhsh

Physics Graduate Student

University of Bologna

Quantitative Finance • Computational Modeling • Scientific Computing

---

## Future Development

Planned future improvements:

- Advanced portfolio optimization
- Additional stochastic processes
- Volatility surface calibration
- Expanded risk analytics
- Documentation website
- PyPI package distribution

---

### Black Sigma Terminal

Institutional Quantitative Finance Platform
