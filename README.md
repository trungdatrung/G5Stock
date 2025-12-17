# G5Stock - Stock Price Predictor

A modern, interactive Streamlit application for predicting stock prices using Linear Regression.

## Features

*   **Real-time Data:** Fetches historical OHLC data from Yahoo Finance.
*   **Interactive Charts:** High-performance, interactive charts using Plotly.
*   **Linear Regression Model:** Predicts future stock prices based on historical trends.
*   **Flexible Prediction:** Choose prediction horizons in Days, Months, or Years.
*   **Metrics:** View Model R², Mean Squared Error (MSE), and predicted percentage change.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/trungdatrung/G5Stock.git
    cd G5Stock
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install streamlit yfinance plotly scikit-learn pandas numpy
    ```

## Usage

1.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

2.  **Open your browser:**
    Navigate to `http://localhost:8501` (or the URL displayed in the terminal).

3.  **Interact:**
    *   Enter a stock ticker (e.g., AAPL).
    *   Select a date range of historical data to train on.
    *   Choose a prediction unit (Days, Months, Years) and value.
    *   Click "Retrain Model" to update predictions.

## Disclaimer

**Not Financial Advice.** This tool is for educational purposes only. The Linear Regression model is a simple trend-following algorithm and does not account for market volatility, news, or complex economic factors. Do not make investment decisions based on these predictions.
