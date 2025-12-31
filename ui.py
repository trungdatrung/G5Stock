import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def render_metrics(current_price, last_close_date, predicted_future_price, horizon_val, time_unit, r2, mse):
    """
    Renders the metric cards for the dashboard.
    """
    predicted_change = predicted_future_price - current_price
    pct_change = (predicted_change / current_price) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Latest Close", 
            f"${current_price:.2f}", 
            f"on {last_close_date}"
        )
    with col2:
        st.metric(
            f"Prediction (+{horizon_val} {time_unit})", 
            f"${predicted_future_price:.2f}", 
            f"{predicted_change:+.2f} ({pct_change:+.2f}%)"
        )
    with col3:
        st.metric(
            "Model R² Score", 
            f"{r2:.4f}", 
            help="Explanation of variance (close to 1 is better, but watch for overfitting)"
        )
    with col4:
        st.metric(
            "MSE", 
            f"{mse:.2f}", 
            help="Mean Squared Error"
        )

def get_xaxis_layout(start_date, end_date):
    """
    Returns appropriate Plotly x-axis settings based on the time range duration.
    Rules:
    - range <= 1 day: Hours/Minutes (09:30, 10:00, 10:30)
    - 1 month >= range > 1 day: Days (Mar 1, Mar 5, Mar 10)
    - 1 year >= range > 1 month: Months (Jan, Feb, Mar)
    - range > 1 year: Years (2022, 2023, 2024)
    """
    # Ensure proper datetime types
    if not isinstance(start_date, pd.Timestamp):
        start_date = pd.Timestamp(start_date)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.Timestamp(end_date)
        
    duration = end_date - start_date
    
    xaxis_config = dict(
        title="Date",
        showgrid=True,
        zeroline=False
    )
    
    # 1. Less than or equal to 1 day
    if duration <= pd.Timedelta(days=1):
        xaxis_config.update(dict(
            tickformat="%H:%M",
            dtick=1800000.0, # 30 minutes in milliseconds
            tickmode="auto"
        ))
        
    # 2. More than 1 day: Allow Plotly to auto-scale ticks (daily/monthly/yearly)
    # We set a detailed format so that when daily ticks appear (e.g. on zoom), they are readable.
    else:
        xaxis_config.update(dict(
            tickformat="%b %d, %Y", # Jan 01, 2023
            tickmode="auto"
        ))
        
    return xaxis_config

def plot_chart(df: pd.DataFrame, df_with_pred: pd.DataFrame, future_df: pd.DataFrame, ticker: str):
    """
    Creates and displays the main Plotly chart.
    """
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
    
    # Determine overall range for axis formatting
    # Range covers historical start to future end
    full_start = df['Date'].min()
    full_end = future_df['Date'].max()
    
    xaxis_settings = get_xaxis_layout(full_start, full_end)

    layout_config = dict(
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
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis=xaxis_settings, # Apply dynamic settings
        dragmode="pan"
    )

    fig.update_layout(**layout_config)

    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
