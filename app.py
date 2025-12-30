import streamlit as st
import logging
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import re

# Import local modules
from data import load_data
from model import train_model
from ui import render_metrics, plot_chart

# --- Configuration ---
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup basic logging for app.py
# Configure logging centrally here
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("Configuration")

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper().strip()

today = date.today()
start_default = today - timedelta(days=365*10)
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
    future_days = (today + relativedelta(months=horizon_val) - today).days
else:  # Years
    horizon_val = st.sidebar.slider("Select Years", 1, 10, 1)
    future_days = (today + relativedelta(years=horizon_val) - today).days

run_button = st.sidebar.button("Retrain Model")

st.sidebar.markdown("---")
st.sidebar.info("Note: This model uses simple Linear Regression independently on time. It does not account for volatility or other market factors.")

# --- Main App ---

st.title("📈 Stock Price Predictor")
st.markdown("Visualize historical trends and project future prices using Linear Regression.")

# Validation
# Relaxed validation to allow international tickers (e.g. ^GSPC, 0700.HK, BRK-B)
if not re.match(r'^[A-Z0-9^.-]{1,15}$', ticker):
    st.warning("Invalid ticker format. Please use alphanumeric characters, dots, hyphens, or carets.")
    st.stop()

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    st.error("Please select a valid date range.")
    st.stop()

# Execution
with st.spinner(f"Fetching data for {ticker}..."):
    df = load_data(ticker, start_d, end_d)

if df is None or df.empty:
    st.error(f"No data found for {ticker}. Please check the symbol and try again.")
else:
    try:
        # Train Model
        model, df_with_pred, future_df, r2, mse = train_model(df, future_days)
        
        # Prepare data for display
        current_price = df['Close'].iloc[-1]
        last_close_date = df['Date'].iloc[-1].strftime('%Y-%m-%d')
        predicted_future_price = future_df['Prediction'].iloc[-1]
        
        # Render UI
        render_metrics(
            current_price, 
            last_close_date, 
            predicted_future_price, 
            horizon_val, 
            time_unit, 
            r2, 
            mse
        )
        
        plot_chart(df, df_with_pred, future_df, ticker)
        
        # --- Data View ---
        with st.expander("See raw data"):
            st.write(df.tail())
            
    except Exception as e:
        st.error(f"An error occurred during model training: {e}")
        logger.error(f"App level error: {e}", exc_info=True)

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
