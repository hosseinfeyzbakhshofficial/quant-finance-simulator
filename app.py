import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go

from src.simulations.monte_carlo import monte_carlo_gbm
from src.finance.option_pricing import monte_carlo_option_price
from src.finance.black_scholes import (
    black_scholes_call,
    call_delta,
    call_gamma,
    call_vega,
    call_theta,
)

# =========================================
# PAGE CONFIG & INSTITUTIONAL THEME
# =========================================
st.set_page_config(
    page_title="Black Sigma | Institutional Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS پیشرفته برای ظاهر بلومبرگ/وال‌استریت
st.markdown("""
    <style>
    /* Dark Deep Background */
    .stApp { background-color: #0b0e14; color: #e2e8f0; }
    
    /* Institutional Logo Design */
    .logo-container {
        display: flex;
        align-items: center;
        background: #151a23;
        padding: 20px 25px;
        border-bottom: 1px solid #1f2937;
        border-left: 4px solid #00e676; /* Institutional Green */
        margin-bottom: 25px;
    }
    .brand-text {
        margin-left: 20px;
    }
    .brand-title {
        font-size: 34px;
        font-weight: 800;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
        letter-spacing: 6px;
        margin: 0;
        line-height: 1;
    }
    .brand-subtitle {
        font-size: 11px;
        color: #94a3b8;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-top: 4px;
        font-weight: 600;
    }
    
    /* Metric Styling */
    div[data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
        color: #f8fafc;
    }
    </style>
""", unsafe_allow_html=True)

# لوگوی SVG حرفه ای Black Sigma
st.markdown("""
    <div class="logo-container">
        <svg width="55" height="55" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M20 20 L80 20 L50 50 L80 80 L20 80 L40 50 Z" fill="#00e676" opacity="0.9"/>
            <path d="M20 20 L40 50 L20 80 Z" fill="#00b259"/>
        </svg>
        <div class="brand-text">
            <div class="brand-title">BLACK SIGMA</div>
            <div class="brand-subtitle">Quantitative Research Terminal</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# =========================================
# ADVANCED SIDEBAR CONTROLS
# =========================================
run_live = st.sidebar.toggle("🟢 Enable Live Market Feed", value=False)
st.sidebar.markdown("---")

with st.sidebar.expander("⚙️ Market Parameters", expanded=True):
    S0 = st.number_input("Initial Asset Price (S0)", value=100.0, step=1.0)
    STRIKE = st.number_input("Option Strike Price (K)", value=100.0, step=1.0)
    RISK_FREE_RATE = st.slider("Risk-Free Rate (r)", 0.00, 0.20, 0.05)

with st.sidebar.expander("📊 Stochastic Process (GBM)", expanded=True):
    MU = st.slider("Drift (μ)", 0.00, 0.30, 0.10)
    SIGMA = st.slider("Volatility (σ)", 0.01, 1.00, 0.20)
    T = st.slider("Time to Maturity (Years)", 0.1, 5.0, 1.0)

with st.sidebar.expander("🧮 Monte Carlo Settings", expanded=False):
    N_SIMULATIONS = st.number_input("Number of Paths", min_value=100, max_value=100000, value=5000, step=1000)
    DT = st.number_input("Time Step (dt)", value=0.01, format="%.3f")
    seed_toggle = st.checkbox("Use Fixed Seed (Reproducibility)", value=True)
    if seed_toggle:
        np.random.seed(42) # برای نتایج ثابت در تحلیل

# =========================================
# CORE CALCULATIONS
# =========================================
# Simulation
mc_paths = monte_carlo_gbm(S0=S0, mu=RISK_FREE_RATE, sigma=SIGMA, T=T, dt=DT, n_simulations=int(N_SIMULATIONS))

# Pricing
mc_price = monte_carlo_option_price(paths=mc_paths, strike=STRIKE, r=RISK_FREE_RATE, T=T)
bs_price = black_scholes_call(S0=S0, K=STRIKE, T=T, r=RISK_FREE_RATE, sigma=SIGMA)
pricing_error = abs(mc_price - bs_price)

# Greeks
delta = call_delta(S0, STRIKE, T, RISK_FREE_RATE, SIGMA)
gamma = call_gamma(S0, STRIKE, T, RISK_FREE_RATE, SIGMA)
vega = call_vega(S0, STRIKE, T, RISK_FREE_RATE, SIGMA)
theta = call_theta(S0, STRIKE, T, RISK_FREE_RATE, SIGMA)

# =========================================
# DASHBOARD TABS
# =========================================
tab1, tab2, tab3, tab4 = st.tabs(["📟 Core Terminal", "📈 Path Analytics", "🧮 Risk & Option Payoff", "💾 Data Center"])

# ----------------- TAB 1: CORE TERMINAL -----------------
with tab1:
    # Live Ticker Placeholder
    live_ticker_placeholder = st.empty()
    
    st.markdown("### Pricing Engine Output")
    col1, col2, col3 = st.columns(3)
    col1.metric("Monte Carlo Val", f"${mc_price:.4f}", f"Err: {pricing_error:.4f}")
    col2.metric("Black-Scholes Val", f"${bs_price:.4f}", "Theoretical Model", delta_color="off")
    col3.metric("Moneyness (S/K)", f"{(S0/STRIKE):.2f}", "ITM" if S0 > STRIKE else "OTM", delta_color="normal")

    st.markdown("---")
    st.markdown("### Terminal Price Distribution")
    final_prices = mc_paths[:, -1]
    fig_hist = go.Figure(data=[go.Histogram(x=final_prices, nbinsx=60, marker_color='#3b82f6', opacity=0.8)])
    fig_hist.add_vline(x=STRIKE, line_dash="dash", line_color="#ef4444", annotation_text="Strike Price", annotation_position="top right")
    fig_hist.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig_hist, use_container_width=True)

# ----------------- TAB 2: PATH ANALYTICS (Confidence Band) -----------------
with tab2:
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.markdown("### Monte Carlo GBM Simulations")
        fig_paths = go.Figure()
        for i in range(min(50, len(mc_paths))): # Plotting 50 paths
            fig_paths.add_trace(go.Scatter(y=mc_paths[i], mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
        fig_paths.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig_paths, use_container_width=True)
        
    with col_p2:
        st.markdown("### 95% Confidence Band")
        # محاسبه میانگین و انحراف معیار در هر قدم زمانی
        time_steps = np.arange(mc_paths.shape[1])
        path_means = np.mean(mc_paths, axis=0)
        path_stds = np.std(mc_paths, axis=0)
        upper_band = path_means + 1.96 * path_stds
        lower_band = path_means - 1.96 * path_stds
        
        fig_band = go.Figure()
        fig_band.add_trace(go.Scatter(x=time_steps, y=upper_band, mode='lines', line=dict(width=0), showlegend=False))
        fig_band.add_trace(go.Scatter(x=time_steps, y=lower_band, fill='tonexty', mode='lines', line=dict(width=0), fillcolor='rgba(0, 230, 118, 0.2)', showlegend=False))
        fig_band.add_trace(go.Scatter(x=time_steps, y=path_means, mode='lines', line=dict(color='#00e676', width=2), name='Mean Path'))
        fig_band.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig_band, use_container_width=True)

# ----------------- TAB 3: RISK MATRIX & PAYOFF -----------------
with tab3:
    st.markdown("### Risk Sensitivity (Greeks)")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Delta (Δ)", f"{delta:.4f}")
    g2.metric("Gamma (Γ)", f"{gamma:.4f}")
    g3.metric("Vega (ν)", f"{vega:.4f}")
    g4.metric("Theta (Θ)", f"{theta:.4f}")
    
    st.markdown("---")
    st.markdown("### Option Expiration Payoff (European Call)")
    # شبیه سازی نمودار Payoff در سررسید
    price_range = np.linspace(max(0, STRIKE - 50), STRIKE + 50, 200)
    payoff = np.maximum(price_range - STRIKE, 0)
    
    fig_payoff = go.Figure()
    fig_payoff.add_trace(go.Scatter(x=price_range, y=payoff, mode='lines', line=dict(color='#ef4444', width=3), name='Payoff'))
    fig_payoff.add_vline(x=STRIKE, line_dash="dash", line_color="white", annotation_text="Strike (K)")
    fig_payoff.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, xaxis_title="Terminal Asset Price (S_T)", yaxis_title="Payoff")
    st.plotly_chart(fig_payoff, use_container_width=True)

# ----------------- TAB 4: DATA EXPORT -----------------
with tab4:
    st.markdown("### Simulation Data Extractor")
    st.info("Download the raw paths generated by the Monte Carlo Engine.")
    
    # ساخت دیتافریم برای 20 مسیر اول جهت نمایش
    df_export = pd.DataFrame(mc_paths[:20].T, columns=[f"Path_{i}" for i in range(20)])
    st.dataframe(df_export, use_container_width=True)
    
    csv = df_export.to_csv(index=True).encode('utf-8')
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name='black_sigma_mc_paths.csv',
        mime='text/csv',
    )

# =========================================
# LIVE PRICE TICKER ENGINE
# =========================================
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

    while True:
        shock = np.random.normal(0, SIGMA/np.sqrt(252)) * st.session_state.current_price
        st.session_state.current_price += shock
        tick_metric.metric("Current Spot", f"${st.session_state.current_price:.2f}", f"{shock:+.2f}")
        time.sleep(1)