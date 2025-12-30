import yfinance as yf
import pandas as pd
import logging
import streamlit as st

# Configure logging
# logging.basicConfig removed - using app-level config
logger = logging.getLogger(__name__)

@st.cache_data(ttl=3600*12)
def load_data(ticker_symbol: str, start_date, end_date) -> pd.DataFrame:
    """
    Fetches historical stock data using yfinance.
    
    Args:
        ticker_symbol (str): The stock ticker (e.g., 'AAPL').
        start_date (date): Start date for data fetching.
        end_date (date): End date for data fetching.
        
    Returns:
        pd.DataFrame or None: DataFrame with stock data or None if fetch fails.
    """
    logger.info(f"Fetching data for {ticker_symbol} from {start_date} to {end_date}")
    try:
        df = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            logger.warning(f"No data found for ticker: {ticker_symbol}")
            return None
            
        # Handle multi-index columns if yfinance returns them
        if isinstance(df.columns, pd.MultiIndex):
             try:
                 df = df.xs(ticker_symbol, level=1, axis=1)
             except KeyError:
                 df.columns = df.columns.droplevel(1)
        
        df.reset_index(inplace=True)
        logger.info(f"Successfully loaded {len(df)} rows for {ticker_symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker_symbol}: {e}", exc_info=True)
        # Verify if it is an import error to give a hint
        if "No module named" in str(e):
             logger.error("It seems a dependency is missing.")
        return None
