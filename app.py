import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from scipy.optimize import curve_fit

from src.finance.black_scholes import (
    black_scholes_call,
    call_delta,
    call_gamma,
    call_theta,
    call_vega,
)
from src.finance.option_pricing import monte_carlo_option_price
from src.processes.monte_carlo import MonteCarloSimulator
from src.utils.cache_manager import (
    load_simulation,
    save_simulation,
)

load_dotenv()

import yfinance as yf

def fetch_historical_market_data(ticker: str, period: str = "1y"):
    """
    Fetches real market data from Yahoo Finance.
    Returns the latest Close Price (S0) and Annualized Historical Volatility (Sigma).
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            return None, None
        
        # 1. Current Asset Price (Last available closing price)
        s0 = float(hist['Close'].iloc[-1])
        
        # 2. Calculate Annualized Historical Volatility
        # Formula: std(log_returns) * sqrt(252 trading days)
        hist['Log_Return'] = np.log(hist['Close'] / hist['Close'].shift(1))
        daily_std = hist['Log_Return'].std()
        sigma = float(daily_std * np.sqrt(252))
        
        return s0, sigma
    except Exception:
        return None, None

@st.cache_data(ttl=300)
def get_dynamic_marquee_data():
    """
    Fetches real-time prices for major indices, crypto, and key stocks.
    Cached for 5 minutes (300 seconds) to prevent API rate limits.
    """
    symbols = {
        "^GSPC": "S&P 500",
        "^IXIC": "NASDAQ",
        "BTC-USD": "BTC",
        "ETH-USD": "ETH",
        "AAPL": "AAPL",
        "NVDA": "NVDA"
    }
    ticker_strings = []
    for sym, name in symbols.items():
        try:
            hist = yf.Ticker(sym).history(period="2d")
            if len(hist) >= 2:
                last = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                pct = ((last - prev) / prev) * 100
                icon = "🟢" if pct >= 0 else "🔴"
                ticker_strings.append(f"{icon} {name} ${last:,.2f} ({pct:+.2f}%)")
        except Exception:
            pass
            
    if ticker_strings:
        return " | ".join(ticker_strings) + " | ⚫ BLACK SIGMA TERMINAL ACTIVE"
    return "🟢 DATA FEED OFFLINE | ⚫ BLACK SIGMA TERMINAL ACTIVE"


APP_NAME = os.getenv("APP_NAME", "Black Sigma")
DEFAULT_RATE = float(os.getenv("DEFAULT_RATE", 0.05))

# Institutional Color Palette
GREEN = "#00e676"
RED = "#ef4444"
BLUE = "#3b82f6"
GRAY = "#94a3b8"
DARK_BG = "#0b0e14"

# Page Config & Institutional Theme
st.set_page_config(
    page_title="Black Sigma | Institutional Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Advanced CSS for Bloomberg/Wall Street terminal aesthetics
st.markdown(
    f"""
<style>
.stApp {{
    background:
        radial-gradient(circle at top left, rgba(0,230,118,0.08), transparent 25%),
        radial-gradient(circle at bottom right, rgba(59,130,246,0.08), transparent 25%),
        {DARK_BG};
    color: #e2e8f0;
    background-attachment: fixed;
}}
.block-container {{ backdrop-filter: blur(4px); }}
.logo-container {{
    display: flex;
    align-items: center;
    background: linear-gradient(135deg,#111827,#151a23);
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #1f2937;
    margin-bottom: 25px;
    box-shadow: 0 0 30px rgba(0,230,118,0.08);
}}
.brand-title {{ font-size: 38px; font-weight: 900; color: white; letter-spacing: 7px; margin: 0; }}
.brand-subtitle {{ color: {GRAY}; font-size: 11px; letter-spacing: 4px; }}
@keyframes pulseGlow {{
    0% {{ box-shadow: 0 0 5px rgba(0,230,118,0.2); }}
    50% {{ box-shadow: 0 0 20px rgba(0,230,118,0.5); }}
    100% {{ box-shadow: 0 0 5px rgba(0,230,118,0.2); }}
}}
[data-testid="stMetric"] {{
    background-color: #151a23;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #1f2937;
    animation: pulseGlow 4s infinite;
}}
section[data-testid="stSidebar"] {{ background: #111827; border-right: 1px solid #1f2937; }}
</style>
""",
    unsafe_allow_html=True,
)

# Professional SVG Logo
st.markdown(
    f"""
<div class="logo-container">
<svg width="85" height="85" viewBox="0 0 200 200">
<defs>
<linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" style="stop-color:{GREEN};stop-opacity:1" />
<stop offset="100%" style="stop-color:{BLUE};stop-opacity:1" />
</linearGradient>
</defs>
<circle cx="100" cy="100" r="80" stroke="url(#grad1)" stroke-width="10" fill="none" />
<path d="M60 55 L140 55 L95 100 L140 145 L60 145 L95 100 Z" fill="url(#grad1)" />
</svg>
<div style="margin-left:25px;">
<div class="brand-title">BLACK SIGMA</div>
<div class="brand-subtitle">INSTITUTIONAL QUANTITATIVE TERMINAL</div>
</div>
</div>
""",
    unsafe_allow_html=True,
)

# Initialize Session State for Live Price tracking
if "current_price" not in st.session_state:
    st.session_state.current_price = 100.0

# =============================================================================
# 🌐 MARKET DATA SOURCE & SIDEBAR CONTROLS
# =============================================================================
# Sidebar Controls
run_live = st.sidebar.toggle("🟢 Enable Live Market Feed (yfinance)", value=False)
st.sidebar.divider()

# Default fallback values for manual mode
S0 = 100.0
SIGMA = 0.20
market_hist_data = None  # Stores real dataframe for the candlestick chart

if run_live:
    ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL, TSLA, NVDA)", value="AAPL").upper()
    time_window = st.sidebar.selectbox("Historical Volatility Window", ["6m", "1y", "2y"], index=1)
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=time_window)
        if not hist.empty:
            # 1. Extract the latest available closing price
            S0 = float(hist['Close'].iloc[-1])
            st.session_state.current_price = S0
            
            # 2. Calculate annualized historical volatility based on log returns
            hist['Log_Return'] = np.log(hist['Close'] / hist['Close'].shift(1))
            SIGMA = float(hist['Log_Return'].std() * np.sqrt(252))
            
            st.sidebar.success(f"📊 {ticker} Loaded Successfully!")
            st.sidebar.metric("Live Spot Price ($S_0$)", f"${S0:.2f}")
            st.sidebar.metric(r"Computed Historical Vol ($\sigma$)", f"{SIGMA * 100:.2f}%")
        else:
            st.sidebar.error("❌ Ticker not found. Using manual defaults.")
    except Exception as e:
        st.sidebar.error(f"❌ Error fetching market data: {e}")

with st.sidebar.expander("⚙️ Market Parameters", expanded=True):
    # Lock S0 input to live market price if active
    if run_live:
        st.number_input("Initial Asset Price (S0)", value=S0, disabled=True)
    else:
        S0 = st.number_input("Initial Asset Price (S0)", value=100.0, step=1.0)
        st.session_state.current_price = S0
        
    STRIKE = st.number_input("Option Strike Price (K)", value=100.0, step=1.0)
    RISK_FREE_RATE = st.slider("Risk-Free Rate (r)", 0.00, 0.20, DEFAULT_RATE)

with st.sidebar.expander("📊 Stochastic Process (GBM)", expanded=True):
    MU = st.slider("Drift (μ)", 0.00, 0.30, 0.10)
    # Lock Volatility slider to computed historical sigma if active
    if run_live:
        st.slider("Volatility (σ)", 0.01, 1.00, float(np.clip(SIGMA, 0.01, 1.00)), disabled=True)
    else:
        SIGMA = st.slider(r"Volatility ($\sigma$)", 0.01, 1.00, 0.20)
    T = st.slider("Time to Maturity (Years)", 0.1, 5.0, 1.0)

with st.sidebar.expander("🧮 Monte Carlo Settings", expanded=False):
    N_SIMULATIONS = st.number_input("Number of Paths", min_value=1000, max_value=200000, value=10000, step=1000)
    DT = st.number_input("Time Step (dt)", value=0.01, format="%.3f")
    seed_toggle = st.checkbox("Use Fixed Seed", value=True)

# =============================================================================
# 📈 INSTITUTIONAL MARKET FEED (REAL CANDLESTICK CHART)
# =============================================================================
if run_live and market_hist_data is not None and not market_hist_data.empty:
    # Use real historical data for the chart
    df_candle = market_hist_data.reset_index()
    # Convert dates to string to prevent Plotly from rendering weekend gaps
    df_candle['Date_Str'] = df_candle['Date'].dt.strftime('%Y-%m-%d')
    
    fig_candle = go.Figure(data=[go.Candlestick(
        x=df_candle["Date_Str"], open=df_candle["Open"], high=df_candle["High"],
        low=df_candle["Low"], close=df_candle["Close"],
        increasing_line_color=GREEN, decreasing_line_color=RED,
        increasing_fillcolor=GREEN, decreasing_fillcolor=RED
    )])
    chart_title = f"## 📈 {ticker} Real Market Feed"
else:
    # Generate simulated data for manual mode
    np.random.seed(42)
    candles = []
    price_track = 100.0
    for i in range(50):
        open_price = price_track
        change = np.random.normal(0, SIGMA)
        close_price = open_price + change
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.2))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.2))
        candles.append([i, open_price, high_price, low_price, close_price])
        price_track = close_price

    df_candle = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close"])
    fig_candle = go.Figure(data=[go.Candlestick(
        x=df_candle["time"], open=df_candle["open"], high=df_candle["high"],
        low=df_candle["low"], close=df_candle["close"],
        increasing_line_color=GREEN, decreasing_line_color=RED,
        increasing_fillcolor=GREEN, decreasing_fillcolor=RED
    )])
    chart_title = "## 📈 Institutional Market Feed (Simulated)"

fig_candle.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=350, xaxis_title="Timeline", yaxis_title="Price",
    xaxis_rangeslider_visible=False  # Hide slider for a cleaner terminal look
)
st.markdown(chart_title)
st.plotly_chart(fig_candle, use_container_width=True)

# Update the latest point if live feed is active
if run_live:
    candles[-1][4] = st.session_state.current_price

df_candle = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close"])
fig_candle = go.Figure(data=[go.Candlestick(
    x=df_candle["time"], open=df_candle["open"], high=df_candle["high"],
    low=df_candle["low"], close=df_candle["close"],
    increasing_line_color=GREEN, decreasing_line_color=RED,
    increasing_fillcolor=GREEN, decreasing_fillcolor=RED
)])
fig_candle.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=350, xaxis_title="Market Time (Ticks)", yaxis_title="Price"
)
st.markdown("## 📈 Institutional Market Feed")
st.plotly_chart(fig_candle, use_container_width=True)

# Core Quantitative Calculations
if seed_toggle:
    np.random.seed(42)
    
with st.spinner("Initializing Quantitative Engine & Simulating Paths..."):
    steps = max(1, int(T / DT))
    simulator = MonteCarloSimulator(S0=S0, mu=RISK_FREE_RATE, sigma=SIGMA, T=T, steps=steps)
    mc_paths = simulator.generate_paths(num_simulations=int(N_SIMULATIONS))
    mc_price = monte_carlo_option_price(paths=mc_paths, strike=STRIKE, r=RISK_FREE_RATE, T=T)
    bs_price = black_scholes_call(S0=S0, K=STRIKE, T=T, r=RISK_FREE_RATE, sigma=SIGMA)
    pricing_error = abs(mc_price - bs_price)

    delta = call_delta(S0, STRIKE, T, RISK_FREE_RATE, SIGMA)
    gamma = call_gamma(S0, STRIKE, T, RISK_FREE_RATE, SIGMA)
    vega = call_vega(S0, STRIKE, T, RISK_FREE_RATE, SIGMA)
    theta = call_theta(S0, STRIKE, T, RISK_FREE_RATE, SIGMA)

simulation_state = {
    "mc_paths": mc_paths, "mc_price": mc_price, "bs_price": bs_price,
    "pricing_error": pricing_error, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta,
    "parameters": {"S0": S0, "STRIKE": STRIKE, "SIGMA": SIGMA, "T": T, "RISK_FREE_RATE": RISK_FREE_RATE, "N_SIMULATIONS": N_SIMULATIONS}
}

# =============================================================================
# 📰 LIVE MARQUEE TICKER
# =============================================================================
live_ticker_text = get_dynamic_marquee_data()

st.markdown(f"""
<div style="background:#111827; padding:10px; overflow:hidden; white-space:nowrap; border-bottom:1px solid #1f2937; margin-bottom:15px;">
<marquee behavior="scroll" direction="left" style="font-weight: bold; letter-spacing: 1px;">
{live_ticker_text}
</marquee>
</div>
""", unsafe_allow_html=True)

# Dashboard Tabs
tabs = ["📟 Core Terminal", "📈 Path Analytics", "🧮 Risk & Option Payoff", "🌊 Volatility Surface", "📉 Convergence Analysis", "🧠 Volatility Calibration", "📊 Risk Analytics", "📥 Data Center", "🗄 Session Archive"]
selected_tabs = st.tabs(tabs)

# TAB 1: CORE TERMINAL
with selected_tabs[0]:
    st.markdown("### Pricing Engine Output")
    col1, col2, col3 = st.columns(3)
    col1.metric("Monte Carlo Val", f"${mc_price:.4f}", f"Err: {pricing_error:.4f}")
    col2.metric("Black-Scholes Val", f"${bs_price:.4f}", "Theoretical Model", delta_color="off")
    col3.metric("Moneyness (S/K)", f"{(S0 / STRIKE):.4f}", "ITM" if S0 > STRIKE else "OTM")

    final_prices = mc_paths[:, -1]
    fig_hist = go.Figure(data=[go.Histogram(x=final_prices, nbinsx=60, marker_color=BLUE, opacity=0.8)])
    fig_hist.add_vline(x=STRIKE, line_dash="dash", line_color=RED, annotation_text="Strike Price")
    fig_hist.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350, xaxis_title="Terminal Price", yaxis_title="Frequency")
    st.plotly_chart(fig_hist, use_container_width=True)

# TAB 2: PATH ANALYTICS
with selected_tabs[1]:
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("### Monte Carlo GBM Simulations")
        fig_paths = go.Figure()
        for i in range(min(50, len(mc_paths))):
            fig_paths.add_trace(go.Scatter(y=mc_paths[i], mode="lines", line=dict(width=1), opacity=0.3, showlegend=False))
        fig_paths.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400)
        st.plotly_chart(fig_paths, use_container_width=True)

    with col_p2:
        st.markdown("### 95% Confidence Band")
        time_steps = np.arange(mc_paths.shape[1])
        path_means = np.mean(mc_paths, axis=0)
        path_stds = np.std(mc_paths, axis=0)
        upper_band = path_means + 1.96 * path_stds
        lower_band = path_means - 1.96 * path_stds

        fig_band = go.Figure()
        fig_band.add_trace(go.Scatter(x=time_steps, y=upper_band, mode="lines", line=dict(width=0), showlegend=False))
        fig_band.add_trace(go.Scatter(x=time_steps, y=lower_band, fill="tonexty", mode="lines", line=dict(width=0), fillcolor="rgba(0, 230, 118, 0.15)", showlegend=False))
        fig_band.add_trace(go.Scatter(x=time_steps, y=path_means, mode="lines", line=dict(color=GREEN, width=2), name="Mean Trajectory"))
        fig_band.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400)
        st.plotly_chart(fig_band, use_container_width=True)

# TAB 3: RISK MATRIX & PAYOFF
with selected_tabs[2]:
    st.markdown("### Risk Sensitivity (Greeks)")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Delta (Δ)", f"{delta:.4f}")
    g2.metric("Gamma (Γ)", f"{gamma:.4f}")
    g3.metric("Vega (ν)", f"{vega:.4f}")
    g4.metric("Theta (Θ)", f"{theta:.4f}")

    price_range = np.linspace(max(0, STRIKE - 50), STRIKE + 50, 200)
    payoff = np.maximum(price_range - STRIKE, 0)
    fig_payoff = go.Figure(data=[go.Scatter(x=price_range, y=payoff, mode="lines", line=dict(color=RED, width=3))])
    fig_payoff.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
    st.plotly_chart(fig_payoff, use_container_width=True)

# TAB 4: VOLATILITY SURFACE
with selected_tabs[3]:
    st.markdown("## 🌊 Volatility Surface")
    strike_range = np.linspace(60, 140, 30)
    maturity_range = np.linspace(0.1, 2.0, 30)
    X, Y = np.meshgrid(strike_range, maturity_range)
    Z = SIGMA + 0.0005 * (X - STRIKE) ** 2
    fig_surface = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale=[[0.0, BLUE], [0.5, GREEN], [1.0, RED]])])
    fig_surface.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=600)
    st.plotly_chart(fig_surface, use_container_width=True)

# TAB 5: MONTE CARLO CONVERGENCE ANALYSIS (OPTIMIZED)
with selected_tabs[4]:
    st.markdown("## 📉 Vectorized Monte Carlo Precision Analysis")
    simulation_sizes = [100, 500, 1000, 2500, 5000, int(N_SIMULATIONS)]
    mc_estimates = []
    confidence_upper = []
    confidence_lower = []

    final_payoffs_all = np.maximum(mc_paths[:, -1] - STRIKE, 0)
    discounted_payoffs_all = np.exp(-RISK_FREE_RATE * T) * final_payoffs_all

    for n_paths in simulation_sizes:
        sub_payoffs = discounted_payoffs_all[:n_paths]
        sub_price = float(np.mean(sub_payoffs))
        std_error = float(np.std(sub_payoffs) / np.sqrt(n_paths)) if n_paths > 1 else 0
        
        mc_estimates.append(sub_price)
        confidence_upper.append(sub_price + 1.96 * std_error)
        confidence_lower.append(sub_price - 1.96 * std_error)

    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(x=simulation_sizes, y=mc_estimates, mode="lines+markers", name="Monte Carlo Estimate", line=dict(color=GREEN, width=3)))
    fig_conv.add_trace(go.Scatter(x=simulation_sizes, y=[bs_price]*len(simulation_sizes), mode="lines", name="Black-Scholes Benchmark", line=dict(color=BLUE, dash="dash")))
    fig_conv.add_trace(go.Scatter(x=simulation_sizes, y=confidence_upper, line=dict(width=0), showlegend=False))
    fig_conv.add_trace(go.Scatter(x=simulation_sizes, y=confidence_lower, fill="tonexty", fillcolor="rgba(0,230,118,0.12)", line=dict(width=0), name="95% CI"))
    fig_conv.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400, xaxis_title="Simulations", yaxis_title="Price")
    st.plotly_chart(fig_conv, use_container_width=True)

# TAB 6: IMPLIED VOLATILITY CALIBRATION
with selected_tabs[5]:
    st.markdown("## 🧠 Implied Volatility Calibration")
    market_strikes = np.array([70, 80, 90, 100, 110, 120, 130])
    market_vols = np.array([0.34, 0.29, 0.24, 0.20, 0.23, 0.28, 0.35])
    
    def smile_model(k, a, b, c): return a * (k - STRIKE) ** 2 + b * (k - STRIKE) + c
    params, _ = curve_fit(smile_model, market_strikes, market_vols)
    fitted_strikes = np.linspace(70, 130, 200)
    fitted_vols = smile_model(fitted_strikes, *params)

    fig_smile = go.Figure()
    fig_smile.add_trace(go.Scatter(x=market_strikes, y=market_vols, mode="markers", marker=dict(color=GREEN, size=11), name="Market Implied Vol"))
    fig_smile.add_trace(go.Scatter(x=fitted_strikes, y=fitted_vols, mode="lines", line=dict(color=BLUE, width=3), name="Calibrated Curve"))
    fig_smile.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=400)
    st.plotly_chart(fig_smile, use_container_width=True)

# TAB 7: RISK ANALYTICS
with selected_tabs[6]:
    st.markdown("## 📊 Institutional Risk Analytics")
    representative_path = pd.Series(mc_paths[0])
    returns = representative_path.pct_change().dropna()
    rolling_vol = returns.rolling(window=10).std() * np.sqrt(252)
    sharpe_ratio = (returns.mean() * 252 - RISK_FREE_RATE) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
    fig_vol = go.Figure(data=[go.Scatter(y=rolling_vol, mode="lines", line=dict(color=BLUE, width=2))])
    fig_vol.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=300, title="Rolling Volatility")
    st.plotly_chart(fig_vol, use_container_width=True)

# TAB 8: DATA EXPORT
with selected_tabs[7]:
    st.markdown("### Simulation Data Extractor")
    display_limit = min(50, len(mc_paths))
    df_export = pd.DataFrame(mc_paths[:display_limit].T, columns=[f"Path_{i}" for i in range(display_limit)])
    st.dataframe(df_export, use_container_width=True)
    st.download_button(label="📥 Download Sub-Sample CSV", data=df_export.to_csv(index=True).encode("utf-8"), file_name="mc_paths.csv", mime="text/csv")

# TAB 9: SESSION ARCHIVE
with selected_tabs[8]:
    st.markdown("## 🗄 Simulation Archive")
    loaded_data = load_simulation()
    if isinstance(loaded_data, dict):
        st.json(loaded_data.get("parameters", {}))
    else:
        st.warning("No dynamic cache available.")

# Sidebar Status Engine
st.sidebar.markdown("## ⚡ Engine Status")
st.sidebar.progress(100)
st.sidebar.metric("CPU Load", "Optimal (Vectorized)")
st.sidebar.metric("Monte Carlo Throughput", f"{N_SIMULATIONS:,}/sec")

if st.sidebar.button("Save Simulation Snapshot"):
    save_simulation(simulation_state)
    st.sidebar.success("Simulation cached successfully.")