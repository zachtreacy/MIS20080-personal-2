#%%
#type: ignore

import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime, timedelta 
from scipy.optimize import minimize

st.set_page_config(page_title="MIS20080 - Project", layout="wide")
st.title("Stock Analysis & Portfolio Optimisation Dashboard")

#Code for portfolio optimisation part
if "portfolio_tickers" not in st.session_state:
    st.session_state.portfolio_tickers = []

# User input
ticker = st.text_input("Enter Stock Ticker","AAPL",key="analysis_ticker").upper()
if ticker == "":
    st.error("Please enter a valid ticker symbol.")
    st.stop()
add_to_portfolio = st.checkbox("Add this stock to portfolio", value=False)

if add_to_portfolio:
    try:
        data = yf.download(ticker, period="1d", progress=False)
        if data.empty:
            st.warning(f"Ticker {ticker} not found — not added.")
        elif ticker not in st.session_state.portfolio_tickers:
            st.session_state.portfolio_tickers.append(ticker)
            st.session_state.portfolio_tickers = sorted(set(st.session_state.portfolio_tickers))
            st.success(f"{ticker} added to portfolio.")
        else:
            st.info(f"{ticker} is already in your portfolio.")
    except Exception:
        st.warning(f"Ticker {ticker} not found — not added.")

portfolio_tickers = st.session_state.portfolio_tickers


# displaying company name in streamlit app and getting latest news link
tkr_obj = yf.Ticker(ticker)
try:
    ticker_info = tkr_obj.info or {}
except Exception:
    ticker_info = {}  # handle case where no info available

company_name = ticker_info.get("longName") or ticker_info.get("shortName") or ticker

if ticker_info:
    st.write(f"Company Name: {company_name}")
else:
    st.write("Company Name: Not Found")

# Industry info
def get_ticker_industry(ticker):
    """Safely get industry info; return 'Unknown' if ticker invalid."""
    try:
        ticker_info = yf.Ticker(ticker).info
        industry = ticker_info.get("industry", "Industry not found")
        return industry
    except Exception as e:
        st.error(f"Could not retrieve info for ticker '{ticker}'. Please check the symbol.")
        return "Unknown"

industry = get_ticker_industry(ticker)
if industry == "Unknown":
    st.stop()
st.write(f"Company Industry: {industry}")


# links to latest news regarding most recently added ticker to the portfolio
ft_query = company_name.replace(" ", "+")  # in order to correctly handle the URL
ft_search = f"https://www.ft.com/search?q={ft_query}"
st.caption(f"For the latest news from the Financial Times on {company_name}")
st.write(f"[Click here]({ft_search})")
st.caption("Note that no new articles may be available for the chosen stock.")

# Download data
SPX_data = yf.download(["^GSPC"], period="5y", auto_adjust=False, progress=False)
ticker_data = yf.download([ticker], period="5y", auto_adjust=False, progress=False)

if ticker_data is None or ticker_data.empty:
    st.error(f"Ticker {ticker} not found.")
    st.stop()

# Daily returns (ensure 1-D Series)
SPX_returns   = SPX_data["Adj Close"].pct_change().dropna().squeeze()
ticker_returns = ticker_data["Adj Close"].pct_change().dropna().squeeze()

# Cumulative returns
cumulative_SPX_returns    = (1 + SPX_returns).cumprod() - 1
cumulative_ticker_returns = (1 + ticker_returns).cumprod() - 1

# Plot cumulative returns
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=cumulative_ticker_returns.index,
    y=cumulative_ticker_returns,
    mode="lines",
    name=f"{ticker} Cumulative Returns",
    line=dict(color="orange")
))
fig.add_trace(go.Scatter(
    x=cumulative_SPX_returns.index,
    y=cumulative_SPX_returns,
    mode="lines",
    name="S&P 500 Cumulative Returns",
    line=dict(color="blue")
))
fig.update_layout(
    title=f"{ticker} vs S&P 500 Cumulative Returns",
    xaxis_title="Date",
    yaxis_title="Cumulative Returns",
    yaxis=dict(tickformat=".0%"),
    legend_title="Legend",
    template="plotly_white",
    hovermode="x unified",
    height=300,
)
st.plotly_chart(fig, use_container_width=True)

# 5-year Beta (align dates using a joined frame)
ret = pd.concat(
    [ticker_returns.rename("Ri"), SPX_returns.rename("Rm")],
    axis=1
).dropna()
Ri = ret["Ri"].values
Rm = ret["Rm"].values
N = len(ret)

Ri_mean = Ri.mean()
Rm_mean = Rm.mean()

Rm_var = Rm.var(ddof=1)
Covar = ((Ri - Ri_mean) * (Rm - Rm_mean)).sum() / (N - 1)
Beta = Covar / Rm_var
st.write(f"The 5-year Beta for {ticker} is {Beta:.3f}")

# Rolling Beta (60-day window)
window = 60
rolling_cov  = ret["Ri"].rolling(window).cov(ret["Rm"])
rolling_var  = ret["Rm"].rolling(window).var()
rolling_beta = (rolling_cov / rolling_var).dropna()

fig_beta = go.Figure()
fig_beta.add_trace(go.Scatter(
    x=rolling_beta.index,   # <-- use rolling_beta's index
    y=rolling_beta,
    mode="lines",
    name=f"{ticker} Rolling Beta ({window}-day window)",
    line=dict(color="orange")
))
fig_beta.add_hline(
    y=1.0, line_dash="dot",
    annotation_text="β = 1 (market)",
    annotation_position="top left"
)
fig_beta.update_layout(
    title=f"{ticker} Rolling Beta vs S&P 500 ({window}-day window)",
    xaxis_title="Date",
    yaxis_title="Beta (unitless)",
    template="plotly_white",
    hovermode="x unified",
    height=300,
)
st.plotly_chart(fig_beta, use_container_width=True)


# Colm (below your app)

st.subheader("📊 Competitor Comparison & Risk Metrics")

competitor = st.text_input("Enter Competitor's Stock Ticker", "NVDA").upper()
if competitor == "":
    st.error("Please enter a valid competitor ticker symbol.")
    st.stop()

# example
ticker_data = yf.download([ticker], period="5y", auto_adjust=False, progress=False)
competitor_data = yf.download([competitor], period="5y", auto_adjust=False, progress=False)

ticker_returns = ticker_data["Adj Close"].pct_change().dropna().squeeze()
competitor_returns = competitor_data["Adj Close"].pct_change().dropna().squeeze()

mean_ticker_returns = ticker_returns.mean()
mean_comp_returns = competitor_returns.mean()

ticker_std_daily = float(np.std(ticker_returns.values, ddof=1))
comp_std_daily = float(np.std(competitor_returns.values, ddof=1))
ticker_annual_std = ticker_std_daily * np.sqrt(252)
comp_annual_std = comp_std_daily * np.sqrt(252)

z_score = 1.645
ticker_vatr_daily = (z_score*ticker_std_daily) - mean_ticker_returns
ticker_vatr_annual = ticker_vatr_daily*np.sqrt(252)*100
comp_vatr_daily = (z_score*comp_std_daily) - mean_comp_returns
comp_vatr_annual = comp_vatr_daily*np.sqrt(252)*100

#Displaying in columns
col1, col2, col3 = st.columns(3)

col1.metric(label=f"{ticker} Mean Daily Return", value=f"{mean_ticker_returns:.4%}")
col1.metric(label=f"{competitor} Mean Daily Return", value=f"{mean_comp_returns:.4%}")

col2.metric(label=f"{ticker} Annual Std Dev", value=f"{ticker_annual_std:.2%}")
col2.metric(label=f"{competitor} Annual Std Dev", value=f"{comp_annual_std:.2%}")

col3.metric(label=f"{ticker} Annual VaR", value=f"{ticker_vatr_annual:.2f}%")
col3.metric(label=f"{competitor} Annual VaR", value=f"{comp_vatr_annual:.2f}%")

#Zach's part
st.subheader("Daily Returns & CAPM Analysis")

# ensure ticker_returns and SPX_returns are 1-D Series
if isinstance(ticker_returns, pd.DataFrame):
    ticker_returns = ticker_returns.iloc[:, 0].squeeze()
else:
    ticker_returns = ticker_returns.squeeze()

if isinstance(SPX_returns, pd.DataFrame):
    SPX_returns = SPX_returns.iloc[:, 0].squeeze()
else:
    SPX_returns = SPX_returns.squeeze()

# interactive sliders and moving average plots (only when data exists)
if ticker_returns.empty or SPX_returns.empty:
    st.warning("Not enough data to display interactive return charts.")
else:
    # determine max available days (limit slider to available data)
    max_days = min(len(ticker_returns), len(SPX_returns), 365)
    show_days = st.slider("Show Last N Days", min_value=1, max_value=max_days, value=max_days)
    ma_window = st.slider("Moving Average Length (Days)", min_value=0, max_value=60, value=30)

    # take tail of each series and align by date
    pf_display = ticker_returns.tail(show_days)
    bm_display = SPX_returns.tail(show_days)
    df_plot = pd.concat([pf_display.rename(ticker), bm_display.rename("S&P500")], axis=1).dropna()
    
    # add caption about data period used
    st.caption("Data period: 5 years (Adjusted Close)")

    # add moving averages if requested
    if ma_window > 0:
        df_plot[f"{ticker}_MA"] = df_plot[ticker].rolling(window=ma_window).mean()
        df_plot["S&P500_MA"] = df_plot["S&P500"].rolling(window=ma_window).mean()

    # portfolio vs benchmark chart using plotly (always plot main series; MAs optional)
    if ma_window > 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot[ticker], mode="lines", name=ticker,
            line=dict(color="orange", width=1)
        ))
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["S&P500"], mode="lines", name="S&P500",
            line=dict(color="blue", width=1)
        ))

        if ma_window > 0:
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot[f"{ticker}_MA"], mode="lines", name=f"{ticker} MA",
                line=dict(color="green", width=1)
            ))
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["S&P500_MA"], mode="lines", name="S&P500 MA",
                line=dict(color="purple", width=1)
            ))

        fig.update_layout(
            title=f"{ticker} vs S&P500 Daily Returns (last {show_days} days)",
            xaxis_title="Date",
            yaxis_title="Daily return",
            hovermode="x unified",
            height=600
        )
        fig.update_yaxes(tickformat=".2%")
        st.plotly_chart(fig, use_container_width=True)

        

# CAPM expected return calculations
Trading_days = 252  # assuming 252 trading days in a year

# subheader and formula
st.subheader("CAPM Model")
st.caption("CAPM Formula: E(Ri) = Rf + βi (E(Rm) - Rf)")

# risk-free rate input
rf = st.number_input("Choose Annual Risk-Free Rate (0 to 4%)",
                     min_value=0.0, max_value=0.04, value=0.02, step=0.001, format="%.4f")

# align asset and market returns on overlapping dates
combined = pd.concat([ticker_returns, SPX_returns], axis=1).dropna()
combined.columns = [ticker, "S&P500"]

#wrap in if statement to avoid errors if no overlapping data
if combined.empty:
    st.warning("Not enough overlapping returns to compute CAPM. Check data/period.")
else:
    asset = combined[ticker]
    market = combined["S&P500"]

    # market annual arithmetic return
    market_annual_arith = market.mean() * Trading_days

    # CAPM expected returns using Beta computed earlier
    capm_er_annual = rf + Beta * (market_annual_arith - rf)

    # convert rf to daily/quarter for CAPM on those scales
    rf_daily = (1 + rf) ** (1 / Trading_days) - 1
    rf_quarterly = (1 + rf) ** (1 / 4) - 1
    market_daily_arith = market.mean()
    market_quarterly_arith = market.mean() * (Trading_days // 4)

    capm_er_daily = rf_daily + Beta * (market_daily_arith - rf_daily)
    capm_er_quarterly = rf_quarterly + Beta * (market_quarterly_arith - rf_quarterly)

    # display neatly
    cols = st.columns(5)
    cols[0].metric("Beta", f"{Beta:.4f}")
    cols[1].metric("S&P 500 Annual return", f"{market_annual_arith:.2%}")
    cols[2].metric("CAPM Annual Return", f"{capm_er_annual:.2%}")
    cols[3].metric("CAPM Quarterly Return", f"{capm_er_quarterly:.2%}")
    cols[4].metric("CAPM Daily Return", f"{capm_er_daily:.4%}")

    # realized returns (annualized)
    asset_clean = asset.dropna()
    realized_arith_pf = float(asset_clean.mean() * Trading_days) if len(asset_clean) > 0 else float("nan")
    realized_geom_pf = float((np.prod(1.0 + asset_clean) ** (Trading_days / len(asset_clean))) - 1.0) if len(asset_clean) > 0 else float("nan")

    # create results DataFrame
    results_df = pd.DataFrame([{
        "Ticker": ticker,
        "Beta": Beta,
        "S&P 500 Annual Return": market_annual_arith,
        "CAPM Annual Expected Return": capm_er_annual,
        "Arithmetic Annual Expected Return": realized_arith_pf,
        "Geometric Annual Expected Return": realized_geom_pf
    }]).set_index("Ticker")

    # UI: show Bar Chart by default, allow Table
    view = st.radio("Show", ["Bar Chart", "Table"], index=0, horizontal=True)

    if view == "Bar Chart":
        bar_df = pd.DataFrame({
            "Return Metric": ["CAPM Annual Expected Return", "Arithmetic Annual Expected Return", "Geometric Annual Expected Return"],
            "Annual Percentage Return": [capm_er_annual, realized_arith_pf, realized_geom_pf]
        })
        fig = px.bar(bar_df, x="Return Metric", y="Annual Percentage Return",
                     text=bar_df["Annual Percentage Return"].apply(lambda v: f"{v:.2%}"),
                     title="CAPM vs Realized Annual Returns Chart",
                     )  # light grey bars
        fig.update_traces(textposition="outside")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(results_df.style.format({
            "Beta": "{:.4f}",
            "S&P 500 Annual Return": "{:.2%}",
            "CAPM Annual Expected Return": "{:.2%}",
            "Arithmetic Annual Expected Return": "{:.2%}",
            "Geometric Annual Expected Return": "{:.2%}"
        }))

#Jack part of project

# Stop if portfolio empty
if len(portfolio_tickers) <2:
    st.warning("Portfolio is empty. Add at least one ticker using the checkbox above.")
    st.stop()
    
end_date = datetime.today()
start_date = end_date - timedelta(days = 5*365)
print(start_date)



adj_close_df = pd.DataFrame()

for ticker in portfolio_tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    print(f"{ticker} columns:", data.columns)
    if 'Adj Close' in data.columns:
        adj_close_df[ticker] = data['Adj Close']
    else:
        adj_close_df[ticker] = data['Close']
print(adj_close_df) 

log_returns = np.log(adj_close_df/ adj_close_df.shift(1))

log_returns = log_returns.dropna()

cov_matrix = log_returns.cov()*252
print(cov_matrix)

def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @weights
    return np.sqrt(variance)

def expected_returns(weights, log_returns):
    return np.sum(log_returns.mean()*weights)*252

def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return(expected_returns(weights, log_returns) - risk_free_rate)/standard_deviation(weights, cov_matrix)

risk_free_rate = rf

def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

constraints = {'type':'eq','fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 1) for _ in range(len(portfolio_tickers))]

initial_weights = np.array([1/len(portfolio_tickers)]*len(portfolio_tickers))
print(initial_weights)

optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate), method='SLSQP', constraints=constraints, bounds=bounds)
optimal_weights = optimized_results.x

print('optimal_weights:')
for ticker, weight in zip(portfolio_tickers, optimal_weights):
    print(f"{ticker}:{weight:.4f}")

print()

optimal_portfolio_returns = expected_returns(optimal_weights, log_returns)
optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix , risk_free_rate)

st.write(f"Expected Annual Return of Optimized Portfolio: {optimal_portfolio_returns:.2%}")
st.write(f"Expected Volatility of Optimized Portfolio: {optimal_portfolio_volatility:.2%}")
st.write(f"Optimal Sharpe Ratio: {optimal_sharpe_ratio:.2f}")

fig = go.Figure(data=[
    go.Bar(
        x=portfolio_tickers,
        y=optimal_weights,
        text=[f"{w:.4f}" for w in optimal_weights],
        textposition="outside"
    )
])

fig.update_layout(
    title="Optimal Portfolio Weights",
    xaxis_title="Assets",
    yaxis_title="Optimal Weights",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
# %%
