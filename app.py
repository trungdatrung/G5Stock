import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta

# --- Configuration ---
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    .stMetric {
        background-color: transparent;
    }
    /* Dark mode adjustments would go here if needed, but Streamlit handles most */
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

@st.cache_data(ttl=3600*12)  # Cache data for 12 hours
def load_data(ticker_symbol, start_date, end_date):
    """
    Fetches historical stock data using yfinance.
    """
    try:
        df = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
        if df.empty:
            return None
        # Handle multi-index columns if yfinance returns them
        if isinstance(df.columns, pd.MultiIndex):
             # For some yfinance versions it returns (Price, Ticker)
             try:
                 df = df.xs(ticker_symbol, level=1, axis=1)
             except KeyError:
                 # Fallback if structure is different, keep simple
                 # Usually if single ticker, it might be just simple columns or (Price, Ticker)
                 # Another attempt: drop level
                 df.columns = df.columns.droplevel(1)
        
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def train_model(df, future_days):
    """
    Trains a Linear Regression model on the provided dataframe.
    Returns the model, the prepared data with predictions, and metrics.
    """
    data = df.copy()
    
    # Ensure Date column is datetime
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Create an ordinal date column for regression
    data['Date_Ordinal'] = data['Date'].map(pd.Timestamp.toordinal)
    
    X = data[['Date_Ordinal']]
    y = data['Close']
    
    # Train Model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict on training data (regression line)
    data['Prediction'] = model.predict(X)
    
    # Calculate Metrics
    r2 = model.score(X, y)
    mse = np.mean((data['Prediction'] - y) ** 2)
    
    # Future Predictions
    last_date = data['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=x) for x in range(1, future_days + 1)]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    future_preds = model.predict(future_ordinals)
    
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Prediction': future_preds
    })
    
    return model, data, future_df, r2, mse

# --- Sidebar ---
st.sidebar.header("Configuration")

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()

today = date.today()
start_default = today - timedelta(days=365)
date_range = st.sidebar.date_input(
    "Date Range",
    value=(start_default, today),
    max_value=today
)



# Time Unit Selection
time_unit = st.sidebar.selectbox("Prediction Unit", ["Days", "Months", "Years"])

if time_unit == "Days":
    horizon_val = st.sidebar.slider("Select Days", 1, 90, 30)
    future_days = horizon_val
elif time_unit == "Months":
    horizon_val = st.sidebar.slider("Select Months", 1, 24, 6)
    future_days = horizon_val * 30  # Approx
else:  # Years
    horizon_val = st.sidebar.slider("Select Years", 1, 10, 1)
    future_days = horizon_val * 365 # Approx

run_button = st.sidebar.button("Retrain Model")

st.sidebar.markdown("---")
st.sidebar.info("Note: This model uses simple Linear Regression independently on time. It does not account for volatility or other market factors.")

# --- Main App ---

st.title("📈 Stock Price Predictor")
st.markdown("Visualize historical trends and project future prices using Linear Regression.")

if not ticker:
    st.warning("Please enter a valid stock ticker.")
    st.stop()

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    st.error("Please select a valid date range.")
    st.stop()

with st.spinner(f"Fetching data for {ticker}..."):
    df = load_data(ticker, start_d, end_d)

if df is None or df.empty:
    st.error(f"No data found for {ticker}. Please check the symbol and try again.")
else:
    # Train Model
    model, df_with_pred, future_df, r2, mse = train_model(df, future_days)
    
    # --- Metrics Section ---
    current_price = df['Close'].iloc[-1]
    last_close_date = df['Date'].iloc[-1].strftime('%Y-%m-%d')
    predicted_future_price = future_df['Prediction'].iloc[-1]
    predicted_change = predicted_future_price - current_price
    pct_change = (predicted_change / current_price) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Latest Close", f"${current_price:.2f}", f"on {last_close_date}")
    with col2:
        st.metric(f"Prediction (+{horizon_val} {time_unit})", f"${predicted_future_price:.2f}", f"{predicted_change:+.2f} ({pct_change:+.2f}%)")
    with col3:
        st.metric("Model R² Score", f"{r2:.4f}", help="Explanation of variance (close to 1 is better, but watch for overfitting)")
    with col4:
        st.metric("MSE", f"{mse:.2f}", help="Mean Squared Error")

    # --- Plotting ---
    fig = go.Figure()

    # Historical Data
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        mode='lines',
        name='Historical Close',
        line=dict(color='#00ba38', width=2)
    ))

    # Regression Line (Historical)
    fig.add_trace(go.Scatter(
        x=df_with_pred['Date'], y=df_with_pred['Prediction'],
        mode='lines',
        name='Trend Line (Fit)',
        line=dict(color='#ffa500', width=2, dash='dash')
    ))

    # Future Prediction
    fig.add_trace(go.Scatter(
        x=future_df['Date'], y=future_df['Prediction'],
        mode='lines',
        name='Future Prediction',
        line=dict(color='#ff4b4b', width=2, dash='dot')
    ))

    fig.update_layout(
        title=f"Price History & Linear Regression Forecast for {ticker}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Data View ---
    with st.expander("See raw data"):
        st.write(df.tail())

    st.warning("⚠️ DISCLAIMER: This application is for educational purposes only. Do not use this for financial trading or investment advice.")

    # Model Logic Explanation
    st.markdown("### How it works")
    st.markdown("""
    1.  **Data Collection**: Fetches daily Close prices from Yahoo Finance.
    2.  **Preprocessing**: Converts dates to ordinal numbers (integer representation) to be used as the independent variable (X).
    3.  **Training**: Fits a **Linear Regression** model (`y = mx + c`) where:
        *   `y` is the Stock Price
        *   `x` is the Date
    4.  **Forecasting**: Extends the timeline by the chosen number of days and calculates `y` using the trained slope (`m`) and intercept (`c`).
    """)
