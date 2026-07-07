import argparse
import sys
from src.processes.gbm import simulate_gbm
from src.finance.option_pricing import price_call

def main():
    """
    Command Line Interface for Black Sigma Terminal.
    """
    parser = argparse.ArgumentParser(description="Black Sigma Terminal - Quantitative CLI")
    
    # Define CLI execution modes
    parser.add_argument("--simulate", action="store_true", help="Run a single GBM asset simulation")
    parser.add_argument("--price", action="store_true", help="Calculate Black-Scholes call option price")
    
    # Define financial parameters
    parser.add_argument("--S0", type=float, default=100.0, help="Initial asset price")
    parser.add_argument("--mu", type=float, default=0.05, help="Expected return (drift)")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility")
    parser.add_argument("--T", type=float, default=1.0, help="Time to maturity (years)")
    parser.add_argument("--K", type=float, default=100.0, help="Strike price")
    parser.add_argument("--r", type=float, default=0.05, help="Risk-free rate")
    
    args = parser.parse_args()

    # Route logic based on user argument
    if args.simulate:
        print(f"🚀 Simulating GBM with S0={args.S0}, mu={args.mu}, sigma={args.sigma}, T={args.T}")
        prices = simulate_gbm(args.S0, args.mu, args.sigma, args.T, dt=0.01, seed=42)
        print(f"✅ Simulation completed successfully. Final price: ${prices[-1]:.2f}")
        
    elif args.price:
        print(f"📊 Pricing European Call Option using Black-Scholes...")
        price = price_call(S=args.S0, K=args.K, T=args.T, r=args.r, sigma=args.sigma)
        print(f"✅ Calculated Call Option Price: ${price:.4f}")
        
    else:
        # If no argument is passed, show help
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()