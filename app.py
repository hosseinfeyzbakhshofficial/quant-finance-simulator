import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
import pickle
import os
from dotenv import load_dotenv
from scipy.optimize import curve_fit

load_dotenv()
APP_NAME = os.getenv("APP_NAME")
DEFAULT_RATE = float(
    os.getenv("DEFAULT_RATE", 0.05)
)


from src.simulations.monte_carlo import monte_carlo_gbm
from src.finance.option_pricing import monte_carlo_option_price
from src.finance.black_scholes import (
    black_scholes_call,
    call_delta,
    call_gamma,
    call_vega,
    call_theta,
)
from src.utils.cache_manager import (
    save_simulation,
    load_simulation,
)

# Institutional Color Palette
GREEN = "#00e676"
RED = "#ef4444"
BLUE = "#3b82f6"
GRAY = "#94a3b8"
DARK_BG = "#0b0e14"
PANEL_BG = "#151a23"

# Page Config & Institutional Theme
st.set_page_config(
    page_title="Black Sigma | Institutional Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS for Bloomberg/Wall Street terminal aesthetics
st.markdown(f"""
<style>

/* GLOBAL */
.stApp {{
    background:
        radial-gradient(circle at top left, rgba(0,230,118,0.08), transparent 20%),
        radial-gradient(circle at bottom right, rgba(59,130,246,0.08), transparent 20%),
        {DARK_BG};

    color: #e2e8f0;
    background-attachment: fixed;
}}

/* TOP TERMINAL BAR */
.top-bar {{
    background: linear-gradient(90deg,#111827,#0f172a);
    padding: 10px;
    border-bottom: 1px solid #1f2937;
    margin-bottom: 15px;
    font-family: monospace;
    color: {GREEN};
    letter-spacing: 1px;
}}

/* LOGO CONTAINER */
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

.brand-title {{
    font-size: 38px;
    font-weight: 900;
    color: white;
    letter-spacing: 7px;
    margin: 0;
}}

.brand-subtitle {{
    color: {GRAY};
    font-size: 11px;
    letter-spacing: 4px;
}}

/* METRICS */
[data-testid="stMetric"] {{
    background: linear-gradient(135deg,#151a23,#111827);
    border: 1px solid #1f2937;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
}}

div[data-testid="stMetricValue"] {{
    color: white;
    font-size: 30px !important;
    font-family: monospace;
}}

div[data-testid="stMetricLabel"] {{
    color: {GRAY};
}}

/* SIDEBAR */
section[data-testid="stSidebar"] {{
    background: #111827;
    border-right: 1px solid #1f2937;
}}

/* GLOW ANIMATION */
@keyframes glowPulse {{

    0% {{
        box-shadow: 0 0 10px rgba(0,230,118,0.05);
    }}

    50% {{
        box-shadow: 0 0 25px rgba(0,230,118,0.2);
    }}

    100% {{
        box-shadow: 0 0 10px rgba(0,230,118,0.05);
    }}
}}

[data-testid="stMetric"] {{
    animation: glowPulse 5s infinite;
}}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at top left, rgba(0,230,118,0.08), transparent 25%),
        radial-gradient(circle at bottom right, rgba(59,130,246,0.08), transparent 25%),
        #0b0e14;
    background-attachment: fixed;
}

/* floating glow */
.block-container {
    backdrop-filter: blur(4px);
}

/* animated pulse */
@keyframes pulseGlow {
    0% { box-shadow: 0 0 5px rgba(0,230,118,0.2); }
    50% { box-shadow: 0 0 20px rgba(0,230,118,0.5); }
    100% { box-shadow: 0 0 5px rgba(0,230,118,0.2); }
}

[data-testid="stMetric"] {
    background-color: #151a23;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #1f2937;
    animation: pulseGlow 4s infinite;
}

</style>
""", unsafe_allow_html=True)

# Professional SVG Logo for Black Sigma
st.markdown(f"""
<div class="logo-container">

<svg width="85" height="85" viewBox="0 0 200 200">

<defs>

<linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" style="stop-color:{GREEN};stop-opacity:1" />
<stop offset="100%" style="stop-color:{BLUE};stop-opacity:1" />
</linearGradient>

</defs>

<circle cx="100" cy="100" r="80"
stroke="url(#grad1)"
stroke-width="10"
fill="none"
/>

<path d="M60 55 L140 55 L95 100 L140 145 L60 145 L95 100 Z"
fill="url(#grad1)"
/>

</svg>

<div style="margin-left:25px;">

<div class="brand-title">
BLACK SIGMA
</div>

<div class="brand-subtitle">
INSTITUTIONAL QUANTITATIVE TERMINAL
</div>

</div>

</div>
""", unsafe_allow_html=True)

# SERIALIZATION / CACHE SYSTEM

CACHE_DIR = "cache"

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


def local_save_simulation(data, filename="latest_simulation.pkl"):

    path = os.path.join(CACHE_DIR, filename)

    with open(path, "wb") as f:
        pickle.dump(data, f)

    return path


def local_load_simulation(filename="latest_simulation.pkl"):

    path = os.path.join(CACHE_DIR, filename)

    if os.path.exists(path):

        with open(path, "rb") as f:
            data = pickle.load(f)

        return data

    return None

# Advanced Sidebar Controls
run_live = st.sidebar.toggle("🟢 Enable Live Market Feed", value=False)
st.sidebar.divider()

with st.sidebar.expander("⚙️ Market Parameters", expanded=True):
    S0 = st.number_input("Initial Asset Price (S0)", value=100.0, step=1.0, help="Current spot price of the underlying asset.")
    STRIKE = st.number_input("Option Strike Price (K)", value=100.0, step=1.0, help="Predetermined price at which the option can be exercised.")
    RISK_FREE_RATE = st.slider("Risk-Free Rate (r)", 0.00, 0.20, 0.05, help="Annualized risk-free interest rate (e.g., US Treasury yield).")

with st.sidebar.expander("📊 Stochastic Process (GBM)", expanded=True):
    MU = st.slider("Drift (μ)", 0.00, 0.30, 0.10, help="Expected annualized return of the asset.")
    SIGMA = st.slider("Volatility (σ)", 0.01, 1.00, 0.20, help="Annualized standard deviation of asset returns.")
    T = st.slider("Time to Maturity (Years)", 0.1, 5.0, 1.0, help="Time remaining until the option expires.")

with st.sidebar.expander("🧮 Monte Carlo Settings", expanded=False):
    N_SIMULATIONS = st.number_input("Number of Paths", min_value=100, max_value=200000, value=5000, step=1000, help="Higher paths increase accuracy but require more compute power.")
    DT = st.number_input("Time Step (dt)", value=0.01, format="%.3f", help="Granularity of the simulation steps.")
    seed_toggle = st.checkbox("Use Fixed Seed", value=True, help="Locks the random number generator for reproducible results.")
    if seed_toggle:
        np.random.seed(42)

# Institutional Market Feed

st.markdown("## 📈 Institutional Market Feed")

candles = []

price = S0

for i in range(150):

    open_price = price

    change = np.random.normal(0, SIGMA)

    close_price = open_price + change

    high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.5))

    low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.5))

    candles.append([
        i,
        open_price,
        high_price,
        low_price,
        close_price,
    ])

    price = close_price

df_candle = pd.DataFrame(
    candles,
    columns=["time", "open", "high", "low", "close"]
)

fig_candle = go.Figure(
    data=[
        go.Candlestick(
            x=df_candle["time"],
            open=df_candle["open"],
            high=df_candle["high"],
            low=df_candle["low"],
            close=df_candle["close"],

            increasing_line_color=GREEN,
            decreasing_line_color=RED,

            increasing_fillcolor=GREEN,
            decreasing_fillcolor=RED,
        )
    ]
)

fig_candle.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',

    font=dict(color="white"),

    height=450,

    xaxis_title="Market Time",
    yaxis_title="Price",

    xaxis=dict(
        showgrid=False,
        zeroline=False,
    ),

    yaxis=dict(
        gridcolor="rgba(148,163,184,0.08)"
    ),
)

st.plotly_chart(
    fig_candle,
    use_container_width=True
)

# Core Calculations (Wrapped in a spinner for UX during heavy compute)
with st.spinner("Initializing Quantitative Engine & Simulating Paths..."):
    # Simulation
    mc_paths = monte_carlo_gbm(S0=S0, mu=RISK_FREE_RATE, sigma=SIGMA, T=T, dt=DT, n_simulations=int(N_SIMULATIONS))

    # Pricing
    mc_price = monte_carlo_option_price(paths=mc_paths, strike=STRIKE, r=RISK_FREE_RATE, T=T)
    bs_price = black_scholes_call(S0=S0, K=STRIKE, T=T, r=RISK_FREE_RATE, sigma=SIGMA)
    pricing_error = abs(mc_price - bs_price)

    # Greeks calculation
    delta = call_delta(S0, STRIKE, T, RISK_FREE_RATE, SIGMA)
    gamma = call_gamma(S0, STRIKE, T, RISK_FREE_RATE, SIGMA)
    vega = call_vega(S0, STRIKE, T, RISK_FREE_RATE, SIGMA)
    theta = call_theta(S0, STRIKE, T, RISK_FREE_RATE, SIGMA)

# SAVE CURRENT SESSION

simulation_state = {

    "mc_paths": mc_paths,

    "mc_price": mc_price,

    "bs_price": bs_price,

    "pricing_error": pricing_error,

    "delta": delta,
    "gamma": gamma,
    "vega": vega,
    "theta": theta,

    "parameters": {

        "S0": S0,
        "STRIKE": STRIKE,
        "SIGMA": SIGMA,
        "T": T,
        "RISK_FREE_RATE": RISK_FREE_RATE,
        "N_SIMULATIONS": N_SIMULATIONS,
    }
}

save_simulation(simulation_state)

st.markdown("""
<div style="
background:#111827;
padding:10px;
overflow:hidden;
white-space:nowrap;
border-bottom:1px solid #1f2937;
margin-bottom:15px;
">

<marquee behavior="scroll" direction="left">

🟢 S&P 500 +0.84% |
🔵 NASDAQ +1.14% |
🟡 VIX 14.22 |
🟢 BTC $108,000 |
🔴 TSLA -1.21% |
🟢 NVDA +3.44% |
⚫ BLACK SIGMA TERMINAL ACTIVE

</marquee>

</div>
""", unsafe_allow_html=True)

# Dashboard Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8,tab9 = st.tabs(["📟 Core Terminal", "📈 Path Analytics", "🧮 Risk & Option Payoff","🌊 Volatility Surface","📉 Convergence Analysis","🧠 Volatility Calibration","📊 Risk Analytics", "💾 Data Center","🗄 Session Archive"])

# TAB 1: CORE TERMINAL
with tab1:
    # Placeholder for the Live Ticker at the top of the terminal
    live_ticker_placeholder = st.empty()
    
    st.markdown("### Pricing Engine Output")
    col1, col2, col3 = st.columns(3)
    col1.metric("Monte Carlo Val", f"${mc_price:.4f}", f"Err: {pricing_error:.4f}")
    col2.metric("Black-Scholes Val", f"${bs_price:.4f}", "Theoretical Model", delta_color="off")
    col3.metric("Moneyness (S/K)", f"{(S0/STRIKE):.4f}", "ITM" if S0 > STRIKE else "OTM", delta_color="normal")

    st.divider()
    st.markdown("### Terminal Price Distribution")
    final_prices = mc_paths[:, -1]
    
    fig_hist = go.Figure(data=[go.Histogram(
        x=final_prices, 
        nbinsx=60, 
        marker_color=BLUE,
        opacity=0.8,
        hovertemplate='Terminal Price: $%{x:.2f}<br>Frequency: %{y}<extra></extra>'
    )])
    fig_hist.add_vline(x=STRIKE, line_dash="dash", line_color="#ef4444", annotation_text="Strike Price", annotation_position="top right")
    fig_hist.update_layout(
        template="plotly_dark", 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        height=350, 
        margin=dict(t=10, b=10),
        xaxis_title="Underlying Asset Price at Maturity",
        yaxis_title="Path Frequency"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# TAB 2: PATH ANALYTICS (Confidence Band)
with tab2:
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.markdown("### Monte Carlo GBM Simulations")
        fig_paths = go.Figure()
        # Plotting a subset of paths to prevent browser memory overload
        for i in range(min(50, len(mc_paths))): 
            fig_paths.add_trace(go.Scatter(y=mc_paths[i], mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
        fig_paths.update_layout(
            template="plotly_dark", 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            height=400,
            xaxis_title="Time Steps",
            yaxis_title="Simulated Asset Price"
        )
        st.plotly_chart(fig_paths, use_container_width=True)
        
    with col_p2:
        st.markdown("### 95% Confidence Band")
        # Calculate cross-sectional mean and standard deviation at each time step
        time_steps = np.arange(mc_paths.shape[1])
        path_means = np.mean(mc_paths, axis=0)
        path_stds = np.std(mc_paths, axis=0)
        upper_band = path_means + 1.96 * path_stds
        lower_band = path_means - 1.96 * path_stds
        
        fig_band = go.Figure()
        fig_band.add_trace(go.Scatter(x=time_steps, y=upper_band, mode='lines', line=dict(width=0), showlegend=False))
        fig_band.add_trace(go.Scatter(x=time_steps, y=lower_band, fill='tonexty', mode='lines', line=dict(width=0), fillcolor='rgba(0, 230, 118, 0.15)', showlegend=False))
        fig_band.add_trace(go.Scatter(x=time_steps, y=path_means, mode='lines', line=dict(color=GREEN , width=2), name='Mean Trajectory'))
        fig_band.update_layout(
            template="plotly_dark", 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            height=400,
            xaxis_title="Time Steps",
            yaxis_title="Expected Asset Price"
        )
        st.plotly_chart(fig_band, use_container_width=True)

# TAB 3: RISK MATRIX & PAYOFF
with tab3:
    st.markdown("### Risk Sensitivity (Greeks)")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Delta (Δ)", f"{delta:.4f}")
    g2.metric("Gamma (Γ)", f"{gamma:.4f}")
    g3.metric("Vega (ν)", f"{vega:.4f}")
    g4.metric("Theta (Θ)", f"{theta:.4f}")
    
    st.divider()
    st.markdown("### Option Expiration Payoff (European Call)")
    
    # Calculate payoff structure at maturity
    price_range = np.linspace(max(0, STRIKE - 50), STRIKE + 50, 200)
    payoff = np.maximum(price_range - STRIKE, 0)
    
    fig_payoff = go.Figure()
    fig_payoff.add_trace(go.Scatter(x=price_range, y=payoff, mode='lines', line=dict(color='#ef4444', width=3), name='Payoff Function'))
    fig_payoff.add_vline(x=STRIKE, line_dash="dash", line_color="rgba(255, 255, 255, 0.5)", annotation_text="Strike (K)")
    fig_payoff.update_layout(
        template="plotly_dark", 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        height=350, 
        xaxis_title="Terminal Asset Price (S_T)", 
        yaxis_title="Option Intrinsic Value"
    )
    st.plotly_chart(fig_payoff, use_container_width=True)

# TAB 4: Volatility Surface
with tab4:

    st.markdown("## 🌊 Volatility Surface")

    strike_range = np.linspace(60, 140, 30)

    maturity_range = np.linspace(0.1, 2.0, 30)

    X, Y = np.meshgrid(
        strike_range,
        maturity_range
    )

    Z = np.zeros_like(X)

    for i in range(X.shape[0]):

        for j in range(X.shape[1]):

            implied_vol = (
                SIGMA
                + 0.0005 * (X[i, j] - STRIKE) ** 2
            )

            Z[i, j] = implied_vol

    fig_surface = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,

                colorscale=[
                    [0.0, BLUE],
                    [0.5, GREEN],
                    [1.0, RED],
                ],

                showscale=True,
            )
        ]
    )

    fig_surface.update_layout(
        template="plotly_dark",

        paper_bgcolor='rgba(0,0,0,0)',

        scene=dict(
            bgcolor='rgba(0,0,0,0)',

            xaxis=dict(
                title="Strike",
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(148,163,184,0.1)",
            ),

            yaxis=dict(
                title="Maturity",
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(148,163,184,0.1)",
            ),

            zaxis=dict(
                title="Implied Volatility",
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(148,163,184,0.1)",
            ),
        ),

        height=700,
    )

    st.plotly_chart(
        fig_surface,
        use_container_width=True
    )

# TAB 5: MONTE CARLO CONVERGENCE ANALYSIS
with tab5:

    st.markdown("## 📉 Monte Carlo Precision Analysis")

    simulation_sizes = [
        100,
        500,
        1000,
        2500,
        5000,
        10000,
    ]

    mc_estimates = []
    pricing_errors = []
    confidence_upper = []
    confidence_lower = []

    for n_paths in simulation_sizes:

        temp_paths = monte_carlo_gbm(
            S0=S0,
            mu=RISK_FREE_RATE,
            sigma=SIGMA,
            T=T,
            dt=DT,
            n_simulations=n_paths,
        )

        mc_temp_price = monte_carlo_option_price(
            paths=temp_paths,
            strike=STRIKE,
            r=RISK_FREE_RATE,
            T=T,
        )

        final_payoffs = np.maximum(
            temp_paths[:, -1] - STRIKE,
            0
        )

        discounted_payoffs = np.exp(
            -RISK_FREE_RATE * T
        ) * final_payoffs

        std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)

        ci_upper = mc_temp_price + 1.96 * std_error
        ci_lower = mc_temp_price - 1.96 * std_error

        mc_estimates.append(mc_temp_price)

        pricing_errors.append(
            abs(mc_temp_price - bs_price)
        )

        confidence_upper.append(ci_upper)

        confidence_lower.append(ci_lower)

    st.markdown("### Monte Carlo Convergence Toward Black-Scholes")

    fig_conv = go.Figure()

    fig_conv.add_trace(
        go.Scatter(
            x=simulation_sizes,
            y=mc_estimates,
            mode='lines+markers',
            name='Monte Carlo Estimate',
            line=dict(color=GREEN, width=3),
        )
    )

    fig_conv.add_trace(
        go.Scatter(
            x=simulation_sizes,
            y=[bs_price] * len(simulation_sizes),
            mode='lines',
            name='Black-Scholes Benchmark',
            line=dict(color=BLUE, dash='dash'),
        )
    )

    fig_conv.add_trace(
        go.Scatter(
            x=simulation_sizes,
            y=confidence_upper,
            line=dict(width=0),
            showlegend=False,
        )
    )

    fig_conv.add_trace(
        go.Scatter(
            x=simulation_sizes,
            y=confidence_lower,
            fill='tonexty',
            fillcolor='rgba(0,230,118,0.12)',
            line=dict(width=0),
            name='95% Confidence Interval',
        )
    )

    fig_conv.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        xaxis_title="Number of Simulations",
        yaxis_title="Option Price",
    )

    st.plotly_chart(
        fig_conv,
        use_container_width=True
    )

    st.markdown("### Pricing Error Decay")

    fig_error = go.Figure()

    fig_error.add_trace(
        go.Scatter(
            x=simulation_sizes,
            y=pricing_errors,
            mode='lines+markers',
            line=dict(color=RED, width=3),
            name='Absolute Pricing Error',
        )
    )

    fig_error.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        xaxis_title="Number of Simulations",
        yaxis_title="| Monte Carlo - Black-Scholes |",
    )

    st.plotly_chart(
        fig_error,
        use_container_width=True
    )

    col_a, col_b, col_c = st.columns(3)

    col_a.metric(
        "Final Pricing Error",
        f"{pricing_errors[-1]:.6f}"
    )

    col_b.metric(
        "Confidence Interval Width",
        f"{(confidence_upper[-1] - confidence_lower[-1]):.6f}"
    )

    col_c.metric(
        "Simulation Stability",
        "HIGH" if pricing_errors[-1] < 0.5 else "MODERATE"
    )

# TAB 6: IMPLIED VOLATILITY CALIBRATION
with tab6:

    st.markdown("## 🧠 Implied Volatility Calibration")

    st.markdown(
        """
        Simulated market implied volatility data is fitted
        using nonlinear curve calibration.
        """
    )

    # Synthetic Market Volatility Smile
    market_strikes = np.array([
        70,
        80,
        90,
        100,
        110,
        120,
        130
    ])

    market_vols = np.array([
        0.34,
        0.29,
        0.24,
        0.20,
        0.23,
        0.28,
        0.35
    ])

    # Quadratic Smile Model
    def smile_model(k, a, b, c):

        return (
            a * (k - STRIKE) ** 2
            + b * (k - STRIKE)
            + c
        )

    # Curve Fitting
    params, _ = curve_fit(
        smile_model,
        market_strikes,
        market_vols
    )

    a_fit, b_fit, c_fit = params

    fitted_strikes = np.linspace(
        70,
        130,
        200
    )

    fitted_vols = smile_model(
        fitted_strikes,
        a_fit,
        b_fit,
        c_fit
    )

    # Plot
    fig_smile = go.Figure()

    # Market points
    fig_smile.add_trace(
        go.Scatter(
            x=market_strikes,
            y=market_vols,
            mode='markers',
            marker=dict(
                color=GREEN,
                size=11,
            ),
            name='Market Implied Volatility',
        )
    )

    # Fitted curve
    fig_smile.add_trace(
        go.Scatter(
            x=fitted_strikes,
            y=fitted_vols,
            mode='lines',
            line=dict(
                color=BLUE,
                width=4,
            ),
            name='Calibrated Volatility Curve',
        )
    )

    # ATM reference
    fig_smile.add_vline(
        x=STRIKE,
        line_dash="dash",
        line_color=RED,
        annotation_text="ATM Strike",
    )

    fig_smile.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=550,
        xaxis_title="Strike Price",
        yaxis_title="Implied Volatility",
        title="Volatility Smile Calibration",
    )

    st.plotly_chart(
        fig_smile,
        use_container_width=True
    )

    st.markdown("### Calibrated Model Parameters")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Quadratic Term (a)",
        f"{a_fit:.6f}"
    )

    c2.metric(
        "Linear Term (b)",
        f"{b_fit:.6f}"
    )

    c3.metric(
        "Base Volatility (c)",
        f"{c_fit:.4f}"
    )

    st.markdown("### Quant Interpretation")

    st.info(
        """
        The curvature of the fitted smile reflects
        how implied volatility changes across strikes.

        This phenomenon is commonly observed in
        equity option markets and is a core concept
        in volatility surface modeling and derivatives calibration.
        """
    )

# TAB 7: RISK ANALYTICS
with tab7:

    st.markdown("## 📊 Institutional Risk Analytics")

    # Use one representative path
    representative_path = pd.Series(
        mc_paths[0]
    )

    # Returns
    returns = representative_path.pct_change().dropna()

    # Rolling Analytics
    rolling_vol = (
        returns.rolling(window=10).std()
        * np.sqrt(252)
    )

    rolling_return = (
        returns.rolling(window=10).mean()
        * 252
    )

    # Sharpe Ratio
    sharpe_ratio = (
        (returns.mean() * 252 - RISK_FREE_RATE)
        /
        (returns.std() * np.sqrt(252))
    )

    # Drawdown
    cumulative = (
        1 + returns
    ).cumprod()

    running_max = cumulative.cummax()

    drawdown = (
        cumulative - running_max
    ) / running_max

    max_drawdown = drawdown.min()

    # Value at Risk
    var_95 = np.percentile(
        returns,
        5
    )

    var_99 = np.percentile(
        returns,
        1
    )

    # Quantiles
    quantiles = returns.quantile([
        0.01,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99,
    ])

    # ===== METRICS =====

    st.markdown("### Portfolio Risk Metrics")

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Sharpe Ratio",
        f"{sharpe_ratio:.3f}"
    )

    r2.metric(
        "Max Drawdown",
        f"{max_drawdown:.2%}"
    )

    r3.metric(
        "VaR 95%",
        f"{var_95:.2%}"
    )

    r4.metric(
        "VaR 99%",
        f"{var_99:.2%}"
    )

    st.divider()

    # ===== ROLLING VOL =====

    st.markdown("### Rolling Volatility")

    fig_vol = go.Figure()

    fig_vol.add_trace(
        go.Scatter(
            y=rolling_vol,
            mode='lines',
            line=dict(
                color=BLUE,
                width=3,
            ),
            name='Rolling Volatility',
        )
    )

    fig_vol.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        yaxis_title="Annualized Volatility",
        xaxis_title="Time",
    )

    st.plotly_chart(
        fig_vol,
        use_container_width=True
    )

    # ===== ROLLING RETURNS =====

    st.markdown("### Rolling Returns")

    fig_ret = go.Figure()

    fig_ret.add_trace(
        go.Scatter(
            y=rolling_return,
            mode='lines',
            line=dict(
                color=GREEN,
                width=3,
            ),
            name='Rolling Returns',
        )
    )

    fig_ret.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        yaxis_title="Annualized Return",
        xaxis_title="Time",
    )

    st.plotly_chart(
        fig_ret,
        use_container_width=True
    )

    # ===== DRAWDOWN =====

    st.markdown("### Drawdown Analysis")

    fig_dd = go.Figure()

    fig_dd.add_trace(
        go.Scatter(
            y=drawdown,
            fill='tozeroy',
            mode='lines',
            line=dict(
                color=RED,
                width=2,
            ),
            name='Drawdown',
        )
    )

    fig_dd.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        yaxis_title="Drawdown",
        xaxis_title="Time",
    )

    st.plotly_chart(
        fig_dd,
        use_container_width=True
    )

    # ===== QUANTILES =====

    st.markdown("### Return Quantile Analytics")

    quantile_df = pd.DataFrame({
        "Quantile": quantiles.index,
        "Return": quantiles.values,
    })

    st.dataframe(
        quantile_df,
        use_container_width=True
    )

    st.info(
        """
        These analytics are widely used in
        portfolio management, hedge funds,
        derivatives trading, and quantitative risk systems.
        """
    )

# TAB 8: DATA EXPORT
with tab8:
    st.markdown("### Simulation Data Extractor")
    st.info("Download the raw temporal paths generated by the Monte Carlo Engine.")
    
    # Create a DataFrame representing a subset of paths to prevent UI lag
    display_limit = min(50, len(mc_paths))
    df_export = pd.DataFrame(mc_paths[:display_limit].T, columns=[f"Path_{i}" for i in range(display_limit)])
    df_export.index.name = "Time_Step"
    st.dataframe(df_export, use_container_width=True)
    
    # Prepare full data CSV for download (Background processing)
    csv = df_export.to_csv(index=True).encode('utf-8')
    st.download_button(
        label="📥 Download Sub-Sample Data as CSV",
        data=csv,
        file_name='black_sigma_mc_paths.csv',
        mime='text/csv',
    )


# TAB 9 : SESSION ARCHIVE


with tab9:

    st.markdown("## 🗄 Simulation Archive")

loaded_data = load_simulation()

if isinstance(loaded_data, dict):

    params = loaded_data.get("parameters", {})

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Cached MC Price",
        f"${loaded_data.get('mc_price', 0):.4f}"
    )

    c2.metric(
        "Cached BS Price",
        f"${loaded_data.get('bs_price', 0):.4f}"
    )

    c3.metric(
        "Cached Error",
        f"{loaded_data.get('pricing_error', 0):.6f}"
    )

    st.divider()

    st.markdown("### Cached Parameters")

    st.json(params)

else:

    st.warning("Cache file corrupted or incompatible.")

# Live Price Ticker Engine
if run_live:
    if 'current_price' not in st.session_state:
        st.session_state.current_price = S0
    
    with live_ticker_placeholder.container():
        st.markdown("""
        <div style="background-color: #111827; border-left: 4px solid #00e676; padding: 12px; border-radius: 4px; margin-bottom: 20px;">
            <p style="color: #00e676; margin: 0; font-family: 'Courier New', Courier; font-size: 14px; font-weight: bold;">
                ⚫ LIVE TICKER ACTIVE | SYMBOL: BLK_SIGMA
            </p>
        </div>
        """, unsafe_allow_html=True)
        live_col1, live_col2, live_col3 = st.columns(3)
        tick_metric = live_col1.empty()

    # Infinite loop simulating random walk based on annualized volatility
    while True:
        daily_volatility = SIGMA / np.sqrt(252)
        shock = np.random.normal(0, daily_volatility) * st.session_state.current_price
        st.session_state.current_price += shock
        
        tick_metric.metric("Current Spot", f"${st.session_state.current_price:.2f}", f"{shock:+.2f}")
        time.sleep(1)

        st.sidebar.divider()

st.sidebar.markdown("## ⚡ Engine Status")

st.sidebar.progress(92)

st.sidebar.metric(
    "CPU Load",
    "92%"
)

st.sidebar.metric(
    "Monte Carlo Throughput",
    f"{N_SIMULATIONS:,}/sec"
)

st.sidebar.metric(
    "Latency",
    "12 ms"
)
st.sidebar.divider()

st.sidebar.markdown("## 💾 Session Cache")

if st.sidebar.button("Save Simulation Snapshot"):

    save_simulation(simulation_state)

    st.sidebar.success("Simulation cached successfully.")


if st.sidebar.button("Load Previous Session"):

    loaded_data = load_simulation()

    if isinstance(loaded_data, dict):

        st.sidebar.success("Previous simulation loaded.")

        mc_price = loaded_data.get("mc_price", 0)

        st.sidebar.write(
            f"Loaded MC Price: {mc_price:.4f}"
        )

    else:

        st.sidebar.warning(
            "No cached session found."
        )