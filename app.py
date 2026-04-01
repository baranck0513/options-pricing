import numpy as np
import plotly.graph_objects as go
import streamlit as st

from black_scholes import black_scholes, greeks, verify_put_call_parity

st.set_page_config(page_title="Black-Scholes Pricer", layout="wide")
st.title("Black-Scholes Option Pricer")

# sidebar
with st.sidebar:
    st.header("Parameters")

    current_spot_price = st.slider("Spot Price (S)", min_value=50.0, max_value=200.0, value=100.0, step=1.0)
    option_strike_price = st.slider("Strike Price (K)", min_value=50.0, max_value=200.0, value=105.0, step=1.0)
    years_to_expiry = st.slider("Time to Maturity (years)", min_value=0.1, max_value=2.0, value=1.0, step=0.05)
    annual_risk_free_rate = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.1) / 100
    annual_volatility = st.slider("Volatility (%)", min_value=1.0, max_value=100.0, value=20.0, step=0.5) / 100
    annual_dividend_yield = st.slider("Dividend Yield (%)", min_value=0.0, max_value=10.0, value=0.0, step=0.1) / 100
    option_type = st.radio("Option Type", ["call", "put"], horizontal=True)

# calculations 
call_price = black_scholes(current_spot_price, option_strike_price, years_to_expiry, annual_risk_free_rate, annual_volatility, "call", annual_dividend_yield)
put_price = black_scholes(current_spot_price, option_strike_price, years_to_expiry, annual_risk_free_rate, annual_volatility, "put", annual_dividend_yield)
option_delta, option_gamma, option_theta, option_vega = greeks(current_spot_price, option_strike_price, years_to_expiry, annual_risk_free_rate, annual_volatility, annual_dividend_yield, option_type)
parity_holds = verify_put_call_parity(current_spot_price, option_strike_price, years_to_expiry, annual_risk_free_rate, annual_volatility, annual_dividend_yield)

# option prices
st.subheader("Option Prices")
price_col, put_col, parity_col = st.columns(3)
price_col.metric("Call Price", f"{call_price:.4f}")
put_col.metric("Put Price", f"{put_price:.4f}")
parity_col.metric("Put-Call Parity", "Holds ✅" if parity_holds else "Violated ❌")

# greeks
st.subheader(f"Greeks ({option_type.capitalize()})")
delta_col, gamma_col, theta_col, vega_col = st.columns(4)
delta_col.metric("Delta (Δ)", f"{option_delta:.4f}", help="How much the option price changes per $1 move in spot price")
gamma_col.metric("Gamma (Γ)", f"{option_gamma:.4f}", help="Rate of change of delta with respect to spot price")
theta_col.metric("Theta (Θ) / day", f"{option_theta / 365:.4f}", help="Value lost per day due to time decay")
vega_col.metric("Vega (ν)", f"{option_vega:.4f}", help="How much the option price changes per 1-unit increase in volatility")

# payoff chart
st.subheader("Payoff & P&L at Expiry")

spot_prices_at_expiry = np.linspace(option_strike_price * 0.5, option_strike_price * 1.5, 300)

if option_type == "call":
    premium_paid = call_price
    payoff_at_expiry = np.maximum(spot_prices_at_expiry - option_strike_price, 0)
else:
    premium_paid = put_price
    payoff_at_expiry = np.maximum(option_strike_price - spot_prices_at_expiry, 0)

profit_and_loss = payoff_at_expiry - premium_paid

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=spot_prices_at_expiry, y=payoff_at_expiry,
    name="Payoff (excl. premium)",
    line=dict(color="royalblue", width=2, dash="dot"),
))

fig.add_trace(go.Scatter(
    x=spot_prices_at_expiry, y=profit_and_loss,
    name="P&L (after premium)",
    line=dict(color="mediumseagreen", width=2.5),
    fill="tozeroy",
    fillcolor="rgba(60, 179, 113, 0.08)",
))

# zero line
fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dash"))

# current spot price
fig.add_vline(
    x=current_spot_price,
    line=dict(color="crimson", width=1.5, dash="dash"),
    annotation_text=f"Spot: {current_spot_price:.0f}",
    annotation_position="top right",
)

# strike price
fig.add_vline(
    x=option_strike_price,
    line=dict(color="orange", width=1.5, dash="dash"),
    annotation_text=f"Strike: {option_strike_price:.0f}",
    annotation_position="top left",
)

fig.update_layout(
    xaxis_title="Spot Price at Expiry",
    yaxis_title="Profit / Loss",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=450,
    margin=dict(l=40, r=40, t=40, b=40),
)

st.plotly_chart(fig, use_container_width=True)

# breakeven
if option_type == "call":
    breakeven_price = option_strike_price + premium_paid
else:
    breakeven_price = option_strike_price - premium_paid

st.caption(f"Breakeven point: **{breakeven_price:.4f}** — the spot price at which the option exactly covers its cost")
